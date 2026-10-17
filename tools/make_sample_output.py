"""Regenerate sample_output/: run the current pipeline on sample videos and collect logs, images and DB entries.

Usage:  python tools/make_sample_output.py ["Sample/Video Datasets/a.mp4" ...]
Needs PostgreSQL running (docker compose up -d). WARNING: empties the tracker database for each run.
For each video it writes sample_output/<video>/{logs/, events.csv, visitors.csv, db_dump.sql}
and a top-level sample_output/README.md comparing our count with tools/ground_truth.json.
"""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402
from src.db import Database  # noqa: E402

DEFAULT = ["Sample/Video Datasets/record_20250620_183903.mp4", "Sample/Video Datasets/record_20250620_184302.mp4"]


def reset(db):
    db._run(lambda cur: cur.execute("TRUNCATE events, faces, embeddings, track_spans RESTART IDENTITY CASCADE"))


def export(db, dest: Path):
    kinds = {i: k for i, k, _ in db.load_bodies()}
    label = lambda v: f"{'F' if kinds.get(v) == 'face' else 'P'}{v:04d}"  # noqa: E731
    ev = db._run(lambda cur: (cur.execute("SELECT id, face_id, event_type, ts, track_id, image_path FROM events ORDER BY id"),
                              cur.fetchall())[1])
    with open(dest / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "visitor", "event_type", "timestamp", "track_id", "image_path"])
        for i, v, t, ts, tr, p in ev:
            w.writerow([i, label(v), t, ts.isoformat(), tr, p])
    with open(dest / "visitors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["visitor", "kind", "entries", "exits", "first_event", "last_event"])
        for v in sorted(kinds):
            mine = [e for e in ev if e[1] == v]
            w.writerow([label(v), kinds[v], sum(e[2] == "entry" for e in mine), sum(e[2] == "exit" for e in mine),
                        min((e[3] for e in mine), default=""), max((e[3] for e in mine), default="")])
    dump = subprocess.run(["docker", "exec", "facetracker-db", "pg_dump", "-U", "postgres", "-d", "facetracker",
                           "--exclude-table-data=embeddings"], capture_output=True, text=True, encoding="utf-8")
    (dest / "db_dump.sql").write_text(dump.stdout, encoding="utf-8")
    return len(kinds), sum(1 for k in kinds.values() if k == "face"), len(ev)


def main():
    videos = sys.argv[1:] or DEFAULT
    cfg = load_config("config.json")
    truth = json.loads((ROOT / "tools" / "ground_truth.json").read_text(encoding="utf-8"))
    out = ROOT / "sample_output"
    shutil.rmtree(out, ignore_errors=True)
    rows = []
    tmp_cfg = ROOT / "config.sample_run.json"
    cfg_run = dict(json.loads((ROOT / "config.json").read_text(encoding="utf-8")), save_annotated_video="")
    tmp_cfg.write_text(json.dumps(cfg_run), encoding="utf-8")
    db = Database(cfg["db"])
    for v in videos:
        name = Path(v).stem
        print(f"== {name}", flush=True)
        reset(db)
        shutil.rmtree(ROOT / "logs", ignore_errors=True)
        subprocess.run([sys.executable, "main.py", "--config", str(tmp_cfg), "--source", v], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dest = out / name
        (dest / "logs").mkdir(parents=True)
        shutil.copy(ROOT / "logs" / "events.log", dest / "logs" / "events.log")
        for sub in ("entries", "exits"):
            shutil.copytree(ROOT / "logs" / sub, dest / "logs" / sub)
        total, faces, events = export(db, dest)
        rows.append((name, truth.get(name), total, faces, total - faces, events))
        print(f"   visitors={total} (face {faces}, body-only {total - faces}) events={events}", flush=True)
    reset(db)
    db.close()
    tmp_cfg.unlink(missing_ok=True)
    shutil.rmtree(ROOT / "logs", ignore_errors=True)

    lines = ["# Sample output", "",
             "Produced by `python tools/make_sample_output.py` with the current pipeline and `config.json`.", "",
             "| Video | People (counted by eye) | Unique visitors (ours) | by face | body-only | entry+exit events |",
             "|---|---|---|---|---|---|"]
    for n, t, tot, fc, bd, ev in rows:
        lines.append(f"| {n} | {t if t is not None else '?'} | {tot} | {fc} | {bd} | {ev} |")
    lines += ["", "Each video folder contains:", "",
              "- `logs/events.log` - tracking, embedding, registration, recognition, entry and exit events",
              "- `logs/entries/<date>/` and `logs/exits/<date>/` - cropped images (`F####` = identified by face, "
              "`P####` = tracked without a visible face)",
              "- `events.csv`, `visitors.csv` - the database rows (`image_path` is relative to that video's `logs/` parent)",
              "- `db_dump.sql` - PostgreSQL dump of the schema and data (embedding vectors omitted)", "",
              "Counts are honest: the tracker can still split or merge people (see README, Assumptions)."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("done ->", out)


if __name__ == "__main__":
    main()
