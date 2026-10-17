"""Calibrate body re-ID: compute the mean OSNet feature (for centering) and measure how well cosine
similarity separates SAME person (same track, different time) from DIFFERENT people (same frame).

Usage:  python tools/calibrate_reid.py [video ...]
Writes models/osnet_mean.npy and prints similarity percentiles to help choose body_threshold / body_low_threshold.
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.detector import PersonTracker  # noqa: E402
from src.reid import BodyEmbedder  # noqa: E402

cfg = load_config("config.json")
videos = sys.argv[1:] or sorted(glob.glob("Sample/Video Datasets/*.mp4"))[::4][:6]
emb = BodyEmbedder(cfg["reid_weights"], cfg["reid_arch"], cfg["device"], mean_path=None)  # raw (uncentered)
MAX_FRAMES, STEP, EMBED_EVERY, MIN_H = 700, 3, 3, 100

samples = []   # (video_idx, frame, track_id, raw_feature)
for vi, v in enumerate(videos):
    trk = PersonTracker(cfg["person_weights"], cfg["tracker_config"], cfg["person_confidence"], cfg["device"], cfg["person_imgsz"])
    cap = cv2.VideoCapture(v)
    i = cycle = 0
    while i < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        if i % STEP == 0:
            cycle += 1
            for d in trk.track(frame):
                x1, y1, x2, y2 = d.box
                if cycle % EMBED_EVERY == 0 and (y2 - y1) >= MIN_H:
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size:
                        samples.append((vi, i, d.track_id, emb.embed(crop)))
        i += 1
    cap.release()
    print(f"{Path(v).name}: {len(samples)} crops so far", flush=True)

F = np.stack([s[3] for s in samples])
mean = F.mean(axis=0)
np.save("models/osnet_mean.npy", mean.astype(np.float32))


def center(x):
    c = x - mean
    return c / (np.linalg.norm(c, axis=-1, keepdims=True) + 1e-9)


C = center(F)
same, diff, raw_diff = [], [], []
for a in range(len(samples)):
    for b in range(a + 1, len(samples)):
        va, fa, ta, _ = samples[a]
        vb, fb, tb, _ = samples[b]
        if va == vb and ta == tb and abs(fa - fb) >= 30:
            same.append(float(C[a] @ C[b]))
        elif va == vb and fa == fb and ta != tb:
            diff.append(float(C[a] @ C[b]))
            raw_diff.append(float(F[a] @ F[b]))
        elif va != vb:
            diff.append(float(C[a] @ C[b]))      # different videos: mostly different people
pct = lambda x, q: float(np.percentile(x, q)) if len(x) else float("nan")  # noqa: E731
print(f"\ncrops={len(samples)}  same-person pairs={len(same)}  different-person pairs={len(diff)}")
print(f"RAW cosine, different people in same frame: median={pct(raw_diff, 50):.2f}  p95={pct(raw_diff, 95):.2f}")
print(f"CENTERED same person:       p5={pct(same, 5):.2f} p25={pct(same, 25):.2f} median={pct(same, 50):.2f}")
print(f"CENTERED different people:  median={pct(diff, 50):.2f} p95={pct(diff, 95):.2f} p99={pct(diff, 99):.2f}")
