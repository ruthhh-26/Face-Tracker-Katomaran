# Prompts that shaped this project

Built with an AI coding assistant (AI-assisted workflow). Below are the prompts that **changed the design, found a real
bug or added a feature**, in the order they were asked. The prompts are written up in precise form: what was asked and
why is accurate, but the wording is condensed and rewritten, not a verbatim transcript. Routine prompts (run commands,
small questions) are left out.

For each prompt: **what it led to** and, where it matters, **what we measured**.

---

## 1. Planning

**P1 - Problem statement and architecture (4 screenshots):** "Act as a senior Python engineer. Here is the hackathon
problem statement. Design a modular, real-time face tracking and visitor-counting system for it. Constraints: YOLO for
face detection, InsightFace/ArcFace for embeddings (not the `face_recognition` library), ByteTrack or a similar tracker,
PostgreSQL for storage, frame skipping configurable in `config.json`, a video file for development and RTSP for the
interview. Deliver: the pipeline, a `src/` module layout, the DB schema, the `config.json` structure, resilience rules
(atomic file writes, DB retry, graceful shutdown), a build order and verification steps."
- Produced a plan: pipeline (capture -> YOLO face detect -> ByteTrack -> InsightFace ArcFace -> entry/exit state machine
  -> logs/DB), module split (`src/`), PostgreSQL schema, `config.json`, resilience rules, build order, verification steps.
- Decisions: Python 3.11 venv (InsightFace/onnxruntime wheels missing for 3.14), PostgreSQL in Docker, YOLOv8n-face,
  ByteTrack, InsightFace `buffalo_l`.

## 2. Real-footage failures that shaped the pipeline

**P2:** "In `record_20250620_183903` the toddler's face is never recognised. Inspect the detections, find the root cause
and fix it without hurting recognition of adults."
- Root cause found by inspecting detections: video is 2688x1520 and YOLO ran at 640 px, so the toddler's face shrank to
  ~8 px (30-45 px at full size, below `min_face_size`).
- Added `detection_imgsz` (now 1280), lowered `detection_confidence` and `min_face_size`.

**P3:** "People walking away from the camera show no face, so they are never captured. Propose and implement a way to track
and count them as well."
- Led to **person-level tracking**: COCO person detector (YOLOv8n) + ByteTrack is now the tracked unit; a face found inside
  a person box gives an `F####` identity, a person who never shows a face becomes a body-only `P####` visitor.

**P4:** "Add OSNet body re-identification so a back-view track and a later front-view track of the same person merge into
one visitor."
- Added OSNet-AIN x1.0 body re-ID so a back view and a later front view become one visitor (`PROMOTE`, `MERGE_HEAD_BODY`).

## 3. Making "new vs old visitor" trustworthy

**P5:** "Returning visitors are still registered as new unique visitors. Which additional signal can verify new vs.
returning? Add it, and log every decision so I can audit it."
- Multi-embedding galleries (face and body) stored in PostgreSQL (`embeddings` table) so people are recognised across
  restarts and across videos.
- Face + body **score fusion** with a face veto; every decision logged (`MATCH` / `REGISTER` with the similarities).
- Calibration (`tools/calibrate_reid.py`) exposed a real flaw: raw OSNet cosine between *different* people averaged 0.56
  (p95 0.74), so the old 0.75 threshold produced false merges. Fixed with feature centering + recalibrated thresholds.

**P6 (ground truth):** "`record_20250620_183903` has 11 people by manual count but the system reports 14. Inspect each
visitor's crops and the decision logs, find the causes and fix them."
- Inspecting every visitor's crops found a **bug**: a nearby adult's face was assigned to a small child (tightest-box rule).
  Replaced with a "fits the top of the box" rule with size checks.
- Added a **continuity** cue (a new track appearing where/when another just ended is probably the same person). Count 14 -> 13.

**P7:** "Implement both: (1) drop person boxes that wrap several people, (2) an end-of-run pass that merges duplicate
visitors but never merges two visitors seen on screen at the same time. Also confirm whether body-only visitors count
as unique visitors."
- `drop_union_boxes`, and end-of-run **consolidation** (`src/consolidate.py`): visitors seen at the same time are never
  merged, which allows looser thresholds inside one video. Count 13 -> 12 (toddler's two records merged).
- Body-only visitors do count as unique visitors: face (F) + body-only (P), reported by kind.

**P8:** "If a person's head and body are matched together in any frame, treat them as one visitor."
- A track first identified by body that later shows a matching face is merged into that visitor (`MERGE_HEAD_BODY`);
  body embeddings are now also taken when the person is closest/largest.

**P9 (ground truth, video 184302):** "This video has 8 people but we count 7; the 8th person (in the saree) only enters
in the last frames. Find out why she is dropped and fix it without letting noise through."
- Showed a person entering in the last frames was discarded as noise (2 detections). Added **late-arrival** handling at end
  of stream. Count 7 -> 8, matching the truth.

## 4. Product features

**P10:** "Build a frontend dashboard: KPIs, charts, visitor and event tables, a crop gallery and the `events.log` tail."
-> Streamlit dashboard.

**P11:** "From the dashboard I want to upload or pick a sample video, enter an RTSP URL, start and stop a run, and watch
the annotated stream live." -> **Run** tab (upload video / pick sample / RTSP URL, Start/Stop) and **Live** tab
(annotated frames, FPS, progress). Testing revealed a Windows file-lock bug that crashed the tracker while the dashboard
read the live frame; fixed with retry-and-skip.

## 5. Exploring better tracking (some ideas helped, some did not - kept honest)

**P12:** "A person should keep the same identity from start to end, even when crossing another person. Is there another
tracker or module that does this better? Measure it against ByteTrack." -> tested **BoT-SORT with ReID**: did *not*
help (13 vs 12 on video 3903), so ByteTrack stays the default (`botsort.yaml` kept as an option).

**P13:** "Is BoxMOT compatible with the mandatory stack in the problem statement?" -> checked it against the mandatory
stack (DeepSORT-family StrongSORT / ByteTrack are covered); installing it would force NumPy 2, so it was isolated and not
adopted.

**P14:** "Evaluate running the video forward and in reverse and intersecting the tracks where they meet." -> evaluated
(offline-only, 2x compute, cannot work on live RTSP).

**P15:** "When a person leaves view, predict where they could reappear in all eight directions, plus along their last
velocity, and use that as a matching cue for new tracks." -> built **prediction zones** that grow in all directions (plus
along last velocity). Measured: the motion-only merge made a wrong merge (toddler -> adult) and gave no count gain, so
`motion_only` is **off**; the zones still feed the continuity rule.

## 6. Measuring properly

**P16:** "Make tuning fast across several videos: record the models' output once, replay identity settings on a separate
DB, and compare against ground truth." -> `tools/record_video.py` + `tools/evaluate.py` (+ `tools/ground_truth.json`).
It reproduced the live results exactly (12 and 8). Its first grid result (`consolidate_body` 0.45 = zero error) was a
**false win**: `--verbose` showed it merged two different men and that cancelled a real duplicate. Lesson: a count alone
can hide wrong merges.

## 7. Delivery

**P17:** "Audit the project against the problem statement and list what is still pending for submission." -> compliance
audit against the brief (tech stack table, logs, README items, demo video, GitHub, sample output).

---

## Prompting patterns that worked
- **Give ground truth** ("this video has 11 people", "8 people, the 8th enters at the end"): it turned vague accuracy into
  measurable bugs (face-to-person assignment, late arrivals).
- **Ask for evidence, not guesses**: contact sheets of entry/exit crops and per-decision similarity logs made wrong merges
  visible.
- **Challenge each idea against the brief and the data** (BoT-SORT, BoxMOT, reverse video, motion prediction): ideas that
  did not help were measured and documented instead of shipped.
