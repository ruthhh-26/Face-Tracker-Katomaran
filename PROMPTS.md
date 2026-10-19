# Prompts that shaped this project

Built with Claude Code (AI-assisted workflow). Below are only the prompts that **changed the design, found a real bug or
added a feature**, in the order they were asked. Wording is the original, lightly cleaned for typos. Routine prompts
(run commands, small questions) are left out.

For each prompt: **what it led to** and, where it matters, **what we measured**.

---

## 1. Planning

**P1 - Problem statement (4 screenshots) + "how we can implement this in python ... we can use postgres DB"**
- Claude read the brief and produced a plan: pipeline (capture -> YOLO face detect -> ByteTrack -> InsightFace ArcFace ->
  entry/exit state machine -> logs/DB), module split (`src/`), PostgreSQL schema, `config.json`, resilience rules, build
  order, verification steps.
- Decisions: Python 3.11 venv (InsightFace/onnxruntime wheels missing for 3.14), PostgreSQL in Docker, YOLOv8n-face,
  ByteTrack, InsightFace `buffalo_l`.

## 2. Real-footage failures that shaped the pipeline

**P2 - "In record_20250620_183903 there is a baby in the video, it's not recognising the baby face"**
- Root cause found by inspecting detections: video is 2688x1520 and YOLO ran at 640 px, so the toddler's face shrank to
  ~8 px (30-45 px at full size, below `min_face_size`).
- Added `detection_imgsz` (now 1280), lowered `detection_confidence` and `min_face_size`.

**P3 - "It's not recognising the person who walks the opposite side, so that time we don't know the face and it's not
capturing the person. For that, what we can do?"**
- Led to **person-level tracking**: COCO person detector (YOLOv8n) + ByteTrack is now the tracked unit; a face found inside
  a person box gives an `F####` identity, a person who never shows a face becomes a body-only `P####` visitor.

**P4 - "Yes, add OSNet body re-identification to merge back-view and front-view tracks of the same person."**
- Added OSNet-AIN x1.0 body re-ID so a back view and a later front view become one visitor (`PROMOTE`, `MERGE_HEAD_BODY`).

## 3. Making "new vs old visitor" trustworthy

**P5 - "It's still taking the old visitors again and adding them as unique visitors. What can we do? Can we match something
else to verify if it is a new or old visitor?"**
- Multi-embedding galleries (face and body) stored in PostgreSQL (`embeddings` table) so people are recognised across
  restarts and across videos.
- Face + body **score fusion** with a face veto; every decision logged (`MATCH` / `REGISTER` with the similarities).
- Calibration (`tools/calibrate_reid.py`) exposed a real flaw: raw OSNet cosine between *different* people averaged 0.56
  (p95 0.74), so the old 0.75 threshold produced false merges. Fixed with feature centering + recalibrated thresholds.

**P6 - "In the sample video ending with 3903 has 11 unique people only but it says 14 unique people"** (ground truth)
- Inspecting every visitor's crops found a **bug**: a nearby adult's face was assigned to a small child (tightest-box rule).
  Replaced with a "fits the top of the box" rule with size checks.
- Added a **continuity** cue (a new track appearing where/when another just ended is probably the same person). Count 14 -> 13.

**P7 - "Do 1 and 2" (clean up merged boxes + merge duplicates using a 'never on screen together' rule)**
- `drop_union_boxes`, and end-of-run **consolidation** (`src/consolidate.py`): visitors seen at the same time are never
  merged, which allows looser thresholds inside one video. Count 13 -> 12 (toddler's two records merged).
- Also answered "are we taking the body count as unique visitor?": yes, face (F) + body-only (P), reported by kind.

**P8 - "If the body matches with the head and body in any frame, make it as 1"**
- A track first identified by body that later shows a matching face is merged into that visitor (`MERGE_HEAD_BODY`);
  body embeddings are now also taken when the person is closest/largest.

**P9 - "total 8 but at the last the 8th person in the saree just entered the frame at last"** (ground truth, video 184302)
- Showed a person entering in the last frames was discarded as noise (2 detections). Added **late-arrival** handling at end
  of stream. Count 7 -> 8, matching the truth.

## 4. Product features

**P10 - "Add frontend"** -> Streamlit dashboard (KPIs, charts, visitor/event tables, image gallery, `events.log` tail).

**P11 - "How do I add a sample file in the FE and how do I view live"** -> **Run** tab (upload video / pick sample / RTSP
URL, Start/Stop) and **Live** tab (annotated frames, FPS, progress). Testing revealed a Windows file-lock bug that crashed
the tracker while the dashboard read the live frame; fixed with retry-and-skip.

## 5. Exploring better tracking (some ideas helped, some did not - kept honest)

**P12 - "We track the person from start to end ... even if the person crosses another person we still know it is the same
person. For that can we use any other tool or module?"** -> tested **BoT-SORT with ReID**: did *not* help (13 vs 12 on
video 3903), so ByteTrack stays the default (`botsort.yaml` kept as an option).

**P13 - "Is BoxMOT OK with the problem statement requirement?"** -> checked it against the mandatory stack (DeepSORT-family
StrongSORT / ByteTrack are covered); installing it would force NumPy 2, so it was isolated and not adopted.

**P14 - Forward + reverse video, intersect where they meet** -> evaluated (offline-only, 2x compute, cannot work on live RTSP).

**P15 - "If we predict where the person will move next ... predict in all eight directions when there is no visual"** ->
built **prediction zones** that grow in all directions (plus along last velocity). Measured: the motion-only merge made a
wrong merge (toddler -> adult) and gave no count gain, so `motion_only` is **off**; the zones still feed the continuity rule.

## 6. Measuring properly

**P16 - "Can we tune with several videos?"** -> `tools/record_video.py` + `tools/evaluate.py` (+ `tools/ground_truth.json`):
record the models' output once, replay identity settings in seconds on a separate DB. It reproduced the live results
exactly (12 and 8). Its first grid result (`consolidate_body` 0.45 = zero error) was a **false win**: `--verbose` showed it
merged two different men and that cancelled a real duplicate. Lesson: a count alone can hide wrong merges.

## 7. Delivery

**P17 - "Have we done what they asked in the problem statement?"** and **"What is pending from the problem statement for the
upload?"** -> compliance audit against the brief (tech stack table, logs, README items, demo video, GitHub, sample output).

---

## Prompting patterns that worked
- **Give ground truth** ("this video has 11 people", "8 people, the 8th enters at the end"): it turned vague accuracy into
  measurable bugs (face-to-person assignment, late arrivals).
- **Ask for evidence, not guesses**: contact sheets of entry/exit crops and per-decision similarity logs made wrong merges
  visible.
- **Challenge each idea against the brief and the data** (BoT-SORT, BoxMOT, reverse video, motion prediction): ideas that
  did not help were measured and documented instead of shipped.
