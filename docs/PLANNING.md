# AI-Assisted Planning

Built with Claude Code. The prompts that shaped the project are in [`../PROMPTS.md`](../PROMPTS.md).

## 1. Planning the app
Claude Code was given the hackathon problem statement and produced the plan below, which was then refined with real
footage and hand-counted videos.

**Pipeline:** capture (file / RTSP) -> YOLO detection every `skip+1` frames -> tracking -> ArcFace recognition ->
entry/exit state machine -> logs (images, `events.log`) and PostgreSQL -> unique count.

**Key decisions**
- YOLOv8n-face for faces (also gives 5 landmarks for alignment), InsightFace `buffalo_l` (ArcFace) for recognition,
  ByteTrack for tracking, PostgreSQL in Docker, `config.json` for settings.
- Python 3.11 virtual environment (InsightFace and onnxruntime have no wheels for 3.14).
- One state machine per tracked person guarantees exactly one entry and one exit per appearance.

**What changed after testing on the real videos**
- Faces were ~30 px in 2688x1520 footage, so YOLO input size was raised (`detection_imgsz`).
- People walking away show no face, so the tracked unit became the **person** (YOLO person detector + ByteTrack), with the
  face attached when visible and body-only visitors otherwise.
- OSNet body re-ID links back views to front views; its raw features needed centering and calibration.
- Identity became a fusion of face + body + continuity with a face veto, persisted galleries and an end-of-run merge with a
  "never on screen together" rule.

## 2. Feature list
- YOLO face detection with configurable frame skipping (`detection_skip_frames`)
- Person tracking (ByteTrack) so people without a visible face are still logged
- ArcFace recognition, automatic registration with a unique ID, metadata in PostgreSQL
- OSNet body re-ID, face + body fusion, continuity, multi-embedding galleries (also across videos)
- Exactly one entry and one exit per appearance, each with a cropped image, timestamp, type and ID
- Structured logging: `events.log` + `logs/entries|exits/<date>/` + database
- Unique visitor count (face and body-only reported separately)
- Late-arrival handling, end-of-run consolidation, graceful shutdown
- RTSP input with reconnect; video-file input
- Streamlit dashboard (live view, run control, results)
- Tuning tools (record once, replay settings, compare with hand counts)

## 3. Compute estimate
Measured on a CPU-only machine at 2688x1520 with `detection_skip_frames=2`: face detection ~120 ms and person tracking
~70 ms per detection cycle (every 3rd frame), ArcFace ~120 ms per face and OSNet ~50 ms per person (only a few times per
track), ~1 GB RAM, ~8-9 FPS end to end. GPU figures are not measured (no GPU on the dev machine); see the README table.
The main cost is the two YOLO models on detection frames; raising `detection_skip_frames` trades CPU for timing accuracy
of entries and exits.

## 4. Build order
1. Environment, PostgreSQL, config loader
2. Capture + detection + tracking, checked on a video
3. Face embeddings, registry and matching
4. Entry/exit state machine and logging (files, `events.log`, database)
5. RTSP reader, reconnect, graceful shutdown
6. Person tracking, body re-ID, identity fusion, consolidation
7. Dashboard, tuning tools, sample output, documentation

## 5. Verification
| Check | Result |
|---|---|
| Synthetic video (6 faces shown, blank, shown again) | 6 faces, 12 entries, 12 exits, count stayed 6 on re-appearance |
| `record_20250620_183903` (11 people counted by eye) | 12 visitors (10 face, 2 body-only) |
| `record_20250620_184302` (8 people counted by eye) | 8 visitors (6 face, 2 body-only) |
| Replay tool vs live run | identical counts (12 and 8) |
| Live RTSP | tested by the author with the dashboard |

Only two videos are hand-counted, so these are not general accuracy figures. Ideas that were tried and did not help
(BoT-SORT with ReID, motion-only merging) are recorded in `PROMPTS.md`.
