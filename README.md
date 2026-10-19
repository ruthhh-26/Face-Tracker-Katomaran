# Intelligent Face Tracker with Auto-Registration & Unique Visitor Counting

Processes a video file or a live RTSP stream, detects and tracks people, recognises faces, auto-registers every new
visitor, logs each entry and exit (cropped image + timestamp + event type + ID) to the filesystem, `events.log` and
PostgreSQL, and counts **unique visitors**. People who never show a face (for example walking away from the camera) are
tracked too and counted separately.

- **Demo video:** _add your YouTube link here_
- **Sample output:** [`sample_output/`](sample_output/)
- **Planning:** [`docs/PLANNING.md`](docs/PLANNING.md)
- **Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Diagrams (8 slides):** [`Face_Tracker_Diagrams.pptx`](Face_Tracker_Diagrams.pptx)
- **Prompts used:** [`PROMPTS.md`](PROMPTS.md)

## Tech stack (as required)
| Module | Tech used |
|---|---|
| Face detection | **YOLOv8n-face** (ultralytics) |
| Face recognition | **InsightFace `buffalo_l` (ArcFace, 512-d)** - not the `face_recognition` library |
| Tracking | **ByteTrack** (ultralytics) on a YOLOv8n person detector |
| Backend & processing | Python 3.11 |
| Database | **PostgreSQL 16** (Docker) |
| Configuration | `config.json` (includes `detection_skip_frames`) |
| Logging | `logs/events.log` + `logs/entries|exits/YYYY-MM-DD/` + database |
| Camera input | video file (dev) / RTSP stream (interview) |

Extras beyond the brief: OSNet body re-identification (torchreid), a Streamlit dashboard, tuning tools.

## Setup
```bash
# 1. Python 3.11 virtual environment (InsightFace / onnxruntime have no wheels for 3.14)
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 2. PostgreSQL (listens on host port 5433)
docker compose up -d

# 3. Model weights live in models/ (included in this repo):
#    yolov8n-face.pt (face), yolov8n.pt (person), osnet_ain_x1_0.pt (body re-ID), osnet_mean.npy (feature centering)
#    InsightFace buffalo_l (~280 MB) is downloaded automatically on the first run (needs internet once).

# 4. Run on a video file or an RTSP stream
.venv\Scripts\python main.py --source "path\to\video.mp4"
.venv\Scripts\python main.py --source rtsp://user:pass@192.168.1.50:554/stream1
```
Ctrl+C stops cleanly (an EXIT is logged for everyone still in frame). Unique count at any time:
```bash
docker exec facetracker-db psql -U postgres -d facetracker -c "SELECT kind, COUNT(*) FROM faces GROUP BY kind;"
```

### Dashboard (frontend)
```bash
.venv\Scripts\streamlit run dashboard/app.py      # http://localhost:8501
```
Four tabs: **Live** (annotated stream, FPS, progress, visitor count, latest events), **Run** (upload a video, pick a sample
or enter an RTSP URL, then Start / Stop), **Results** (KPIs, charts, visitor and event tables with CSV export, latest
crops, `events.log` tail, and a **Report** tab that downloads a summary, the visitor and event CSVs, or a zip with the
crops) and **How it works** (the pipeline, the identity rules with the thresholds from `config.json`, and the tech stack).
The layout adapts to narrow screens.

## Sample `config.json`
The database password below is only the local Docker default from `docker-compose.yml`.
```json
{
  "source": "sample.mp4",
  "yolo_weights": "models/yolov8n-face.pt",
  "person_weights": "models/yolov8n.pt",
  "tracker_config": "bytetrack.yaml",
  "device": "cuda:0",
  "detection_skip_frames": 2,
  "detection_confidence": 0.35,
  "detection_imgsz": 1280,
  "person_confidence": 0.4,
  "person_imgsz": 960,
  "min_face_size": 28,
  "recognition_threshold": 0.45,
  "enable_reid": true,
  "reid_weights": "models/osnet_ain_x1_0.pt",
  "reid_mean": "models/osnet_mean.npy",
  "body_threshold": 0.70,
  "face_veto": 0.10,
  "continuity_frames": 150,
  "consolidate": true,
  "consolidate_body": 0.50,
  "min_hits_at_end": 2,
  "exit_grace_frames": 60,
  "log_dir": "logs",
  "save_annotated_video": "logs/annotated.mp4",
  "db": {"host": "localhost", "port": 5433, "name": "facetracker", "user": "postgres", "password": "postgres"}
}
```
`detection_skip_frames` is the frames skipped between detection cycles (the skip parameter the brief asks for). The full
list of settings, with comments, is in `src/config.py`.

## Architecture
```mermaid
flowchart LR
    SRC["Video file / RTSP"] --> CAP["capture.py<br/>reader thread + reconnect"]
    CAP --> CYC{"detection cycle?<br/>every skip+1 frames"}
    CYC -- no --> EXP["expire lost tracks"]
    CYC -- yes --> PT["PersonTracker<br/>YOLOv8n person + ByteTrack"]
    CYC -- yes --> FD["FaceDetector<br/>YOLOv8n-face"]
    PT --> EM["EventManager<br/>events.py"]
    FD --> EM
    EM --> ARC["ArcFace embedding<br/>InsightFace"]
    EM --> OSN["OSNet body embedding"]
    ARC --> MATCH["identity match<br/>face + body fusion, continuity, veto"]
    OSN --> MATCH
    MATCH --> EM
    EM --> IMG["crops<br/>logs/entries, logs/exits"]
    EM --> LOG["logs/events.log"]
    EM --> DB[("PostgreSQL<br/>faces, events, embeddings, track_spans")]
    DB --> CONS["end-of-run consolidation<br/>merge duplicates"]
    DB --> DASH["Streamlit dashboard"]
```
More detail (track lifecycle, identity rules, schema) in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## How a visitor is identified (new or old?)
1. **Face** (ArcFace cosine >= 0.45) is the strongest cue; the face is attached to the person box it fits best.
2. **Body** (OSNet, centred features) links a person seen from behind to the same person seen from the front.
3. **Fusion**: a stored visitor is matched if the face is strong, or the body is strong (>= 0.70), or both are moderate
   (face >= 0.30 and body >= 0.50). A clearly different face (< 0.10) vetoes any body match; a visitor visible on another
   track is never merged.
4. **Continuity**: a new track that starts near where, and soon after, another track ended is treated as the same person.
5. **Galleries**: each visitor keeps up to 8 diverse face and body embeddings in PostgreSQL, so people are also recognised
   across restarts and across videos.
6. **End of run**: duplicates are merged. Hard rule: two visitors seen on screen at the same time are never merged.
7. **Late arrivals**: someone who walks in during the last frames (>= 2 detections) is still counted.

Every decision is written to `events.log` (`MATCH`, `REGISTER`, `PROMOTE`, `MERGE_HEAD_BODY`, `CONSOLIDATE`, ...), with the
similarity values, so a count can be audited.

## Database
- `faces(face_id, kind 'face'|'body', embedding, first_seen, ...)` - one row per unique visitor
- `events(id, face_id, event_type 'entry'|'exit', ts, image_path, track_id)`
- `embeddings(face_id, kind, vec)` - the face and body galleries
- `track_spans(face_id, run_id, first_frame, last_frame)` - when each visitor was on screen (used by consolidation)

**Unique visitors = rows in `faces`**: visitors identified by face (`F####`) plus body-only visitors (`P####`, people who
never showed a usable face). The two are reported separately.

## Sample output and accuracy
Produced by `python tools/make_sample_output.py` (current pipeline, `config.json`):

| Video | People (counted by eye) | Unique visitors (ours) | by face | body-only |
|---|---|---|---|---|
| record_20250620_183903 | 11 | 12 | 10 | 2 |
| record_20250620_184302 | 8 | 8 | 6 | 2 |

Logs, entry/exit crops, `events.csv`, `visitors.csv` and a database dump are in [`sample_output/`](sample_output/).
Only two videos have been counted by hand so far, so **do not read this as a general accuracy figure**. Known
weaknesses: one person can still be split in two (a far body crop that never meets their face), and two people dressed
alike can be merged.

## Compute estimate
**Test machine:** Intel Core 5 205H (8 cores / 12 threads), 16 GB RAM, NVIDIA GeForce RTX 3050 6 GB Laptop GPU.
**Test video:** `record_20250620_183903.mp4` (1,502 frames, 2688x1520, 25 FPS), `detection_skip_frames=2`, no annotated
video written, same result on both devices (12 unique visitors: 10 by face, 2 body-only).

| Measured end to end | CPU (`"device": "cpu"`) | GPU (`"device": "cuda:0"`) |
|---|---|---|
| Throughput | **10.6 FPS** (148 s for the video) | **40.8 FPS** (44 s for the video) |
| Tracker process CPU use (share of the whole CPU) | ~66% | ~25% |
| Peak RAM (tracker process) | ~1.2 GB | ~1.9 GB |
| GPU memory | not used | ~200 MB above the desktop's baseline (~1.7 GB total in use) |

- With the GPU the pipeline is about **3.8x faster**, which is above the camera's 25 FPS, so a live stream is processed in
  real time without dropping frames. On CPU it is roughly 2.4x slower than real time, so a live stream would be
  processed at a reduced rate (the reader thread keeps only the latest frame).
- On the GPU, YOLOv8n-face, YOLOv8n (person + ByteTrack) and OSNet run on CUDA. **ArcFace (InsightFace) always runs on the
  CPU** (`CPUExecutionProvider` in `src/recognizer.py`); it is only invoked for the first few sightings of a track.
- GPU utilisation was low (the desktop itself already used about a third of the GPU), so the models are small enough that
  the CPU side (decoding a 2.7K video, drawing, bookkeeping) becomes the limit. Per-stage latency on the CPU run is about
  120 ms for face detection and 70 ms for person tracking per detection cycle (every 3rd frame), ~120 ms per ArcFace
  embedding and ~50 ms per OSNet embedding.
- If CUDA is not available, `main.py` logs a warning and falls back to the CPU automatically. Raise
  `detection_skip_frames` for less load on either device.
- To use the GPU, install a CUDA build of PyTorch, for example
  `pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

## Assumptions
- Faces >= `min_face_size` px are usable for recognition; for 2K/4K cameras `detection_imgsz` must be raised (1280+).
- A person not detected for more than `exit_grace_frames` has exited; ByteTrack's `track_buffer` (in detection cycles) is
  kept below `exit_grace_frames / (skip+1)` so a closed track is never revived.
- Body re-ID is appearance (clothing) based; thresholds were calibrated on the sample camera with
  `tools/calibrate_reid.py` - **re-run it on the actual camera**.
- Timestamps are wall-clock time of processing.
- Entry images are the face crop when a face was seen, otherwise the person crop.

## Tools
`tools/calibrate_reid.py` (centre and calibrate body features), `tools/record_video.py` + `tools/evaluate.py` +
`tools/ground_truth.json` (record a video once, replay identity settings in seconds on a separate database, compare with
hand counts), `tools/make_sample_output.py` (regenerate `sample_output/`).

---
This project is a part of a hackathon run by https://katomaran.com
