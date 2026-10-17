"""Replay recorded videos through the identity logic and compare the visitor count with ground truth.

Replays use a SEPARATE database (facetracker_eval), so your real data is never touched.

Usage:
  python tools/evaluate.py                          # score the current config.json
  python tools/evaluate.py --set body_threshold=0.65 --set consolidate_body=0.55
  python tools/evaluate.py --grid tools/grid.json   # try every combination in the grid, best first

grid.json example:  {"body_threshold": [0.6, 0.7], "consolidate_body": [0.45, 0.5, 0.55]}
Prerequisites: videos recorded with tools/record_video.py and counts filled in tools/ground_truth.json
"""
import argparse
import itertools
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import psycopg2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402
from src.consolidate import apply_merges, consolidate  # noqa: E402
from src.db import Database  # noqa: E402
from src.detector import Detection  # noqa: E402
from src.events import EventManager  # noqa: E402
from src.gallery import MultiGallery  # noqa: E402


class _Rec:
    """Stands in for the ArcFace Recognizer: returns the recorded embedding."""
    def __init__(self):
        self.gallery = MultiGallery(512)
        self.cur = {}

    def embed(self, frame, box, kps, key=None):
        return self.cur["face_emb"][tuple(box)]


class _Body:
    """Stands in for the OSNet embedder."""
    def __init__(self):
        self.cur = {}

    def embed(self, crop, key=None):
        return self.cur["body"][key[1]]


class _Images:
    def save(self, kind, label, ts, crop):
        return ""


def eval_db(cfg):
    """Connect to (and create if missing) the separate evaluation database."""
    d = cfg["db"]
    admin = psycopg2.connect(host=d["host"], port=d["port"], dbname="postgres", user=d["user"], password=d["password"])
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname='facetracker_eval'")
        if not cur.fetchone():
            cur.execute("CREATE DATABASE facetracker_eval")
    admin.close()
    return Database(dict(d, name="facetracker_eval"))


def replay(cache: dict, cfg: dict, db) -> int:
    db._run(lambda cur: cur.execute("TRUNCATE events, faces, embeddings, track_spans RESTART IDENTITY CASCADE"))
    rec, body = _Rec(), _Body()
    cfg = dict(cfg, run_id="eval")
    gallery = MultiGallery(512, cfg["gallery_size"]) if cfg["enable_reid"] else None
    ev = EventManager(cfg, rec, db, _Images(), body if gallery is not None else None, gallery)
    w, h = cache["size"]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for fi in range(cache["total"]):
        cyc = cache["cycles"].get(fi)
        if cyc is None:
            ev.frame_idx = fi
            ev._expire(fi)
            continue
        rec.cur, body.cur = cyc, cyc
        persons = [Detection(tid, box, conf, None) for tid, box, conf in cyc["persons"]]
        faces = [Detection(-1, box, conf, kps) for box, kps, conf in cyc["faces"]]
        ev.update(frame, persons, faces, fi)
    ev.flush()
    if cfg["consolidate"]:
        apply_merges(db, consolidate(db, cfg))
    return db.count_unique()


def load_caches(gt):
    out = {}
    for name, truth in gt.items():
        if name.startswith("_") or truth is None:
            continue
        f = ROOT / "cache" / f"{name}.pkl"
        if f.exists():
            out[name] = (truth, pickle.loads(f.read_bytes()))
        else:
            print(f"  (skipping {name}: not recorded - run tools/record_video.py)")
    return out


def score(cfg, caches, db):
    res = {n: replay(c, cfg, db) for n, (t, c) in caches.items()}
    err = {n: res[n] - caches[n][0] for n in res}
    return res, err


def main():
    logging.getLogger("facetracker").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--set", action="append", default=[], help="override a setting, e.g. --set body_threshold=0.65")
    ap.add_argument("--grid", help="JSON file: {setting: [values]} - evaluates every combination")
    ap.add_argument("--gt", default="tools/ground_truth.json")
    ap.add_argument("--verbose", action="store_true", help="print every merge (CONSOLIDATE / MERGE_HEAD_BODY / MATCH lines)")
    a = ap.parse_args()
    if a.verbose:
        h = logging.StreamHandler()
        h.addFilter(lambda r: any(k in r.getMessage() for k in ("CONSOLIDATE merged", "MERGE_HEAD_BODY", "rule=motion", "LATE_ARRIVAL")))
        h.setFormatter(logging.Formatter("    %(message)s"))
        logging.getLogger("facetracker").addHandler(h)
        logging.getLogger("facetracker").setLevel(logging.INFO)

    cfg = load_config(a.config)
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    gt = json.loads((ROOT / a.gt).read_text(encoding="utf-8"))
    caches = load_caches(gt)
    if not caches:
        sys.exit("No labelled + recorded videos. Fill tools/ground_truth.json and run tools/record_video.py.")
    db = eval_db(cfg)

    if not a.grid:
        res, err = score(cfg, caches, db)
        print(f"{'video':28} {'truth':>5} {'ours':>5} {'error':>6}")
        for n in res:
            print(f"{n:28} {caches[n][0]:5d} {res[n]:5d} {err[n]:+6d}")
        print(f"\nvideos={len(res)}  total |error|={sum(abs(e) for e in err.values())}  "
              f"exact={sum(e == 0 for e in err.values())}/{len(res)}  bias={sum(err.values()):+d}")
        return

    grid = json.loads(Path(a.grid).read_text(encoding="utf-8"))
    keys = list(grid)
    rows = []
    for combo in itertools.product(*grid.values()):
        c = dict(cfg, **dict(zip(keys, combo)))
        res, err = score(c, caches, db)
        rows.append((sum(abs(e) for e in err.values()), sum(err.values()), dict(zip(keys, combo)), err))
        print(f"  tried {dict(zip(keys, combo))} -> total |error| {rows[-1][0]}", flush=True)
    rows.sort(key=lambda r: (r[0], abs(r[1])))
    print("\nBest settings (lowest total |error| across videos):")
    for tot, bias, params, err in rows[:8]:
        print(f"  total|err|={tot:2d} bias={bias:+d}  {params}  per-video={ {n[-6:]: e for n, e in err.items()} }")
    print("\nWith few labelled videos, prefer a setting that is good over a RANGE of values, not a lone best one.")


if __name__ == "__main__":
    main()
