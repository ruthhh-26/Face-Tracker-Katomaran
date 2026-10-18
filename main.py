"""Intelligent Face Tracker: YOLO detection + ByteTrack + InsightFace recognition + PostgreSQL logging.

Usage:  python main.py [--config config.json] [--source video.mp4 | rtsp://...]
"""
import argparse
import json
import logging
import signal
import time
from pathlib import Path

import cv2
import torch

from src.capture import VideoSource
from src.config import load_config
from src.consolidate import apply_merges, consolidate
from src.db import Database
from src.detector import FaceDetector, PersonTracker
from src.events import EventManager
from src.logger import ImageStore, setup_logging
from src.recognizer import Recognizer
from src.gallery import MultiGallery
from src.reid import BodyEmbedder


def draw(frame, detections, tracks, unique, events):
    """Overlay person boxes, F-/P- IDs and the live unique-visitor count."""
    for d in detections:
        t = tracks.get(d.track_id)
        label = events.label(t) if t and t.face_id else f"track {d.track_id}"
        x1, y1, x2, y2 = d.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0) if t and t.face_id else (0, 165, 255), 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Unique visitors: {unique}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return frame


def _replace(tmp: Path, dest: Path):
    """Atomic replace; on Windows the target may be open in the dashboard, so retry briefly and then skip."""
    for _ in range(5):
        try:
            tmp.replace(dest)
            return
        except PermissionError:
            time.sleep(0.02)


def publish_live(vis, log_dir, status):
    """Write the latest annotated frame + status for the dashboard (atomic, so readers never see half a file)."""
    h, w = vis.shape[:2]
    small = cv2.resize(vis, (960, int(h * 960 / w))) if w > 960 else vis
    tmp = Path(log_dir) / "live.tmp.jpg"
    cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    _replace(tmp, Path(log_dir) / "live.jpg")
    sj = Path(log_dir) / "status.tmp.json"
    sj.write_text(json.dumps(status), encoding="utf-8")
    _replace(sj, Path(log_dir) / "status.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--source", help="override config source (file path or rtsp:// URL)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg["source"] = args.source
    cfg["run_id"] = time.strftime("%Y%m%d_%H%M%S")      # spans only overlap within one run
    log = setup_logging(cfg["log_dir"])
    if str(cfg["device"]).startswith("cuda") and not torch.cuda.is_available():
        log.warning("device=%s requested but CUDA is unavailable; falling back to cpu", cfg["device"])
        cfg["device"] = "cpu"
    log.info("START source=%s skip=%d", cfg["source"], cfg["detection_skip_frames"])

    db = Database(cfg["db"])
    rec = Recognizer(cfg["recognition_threshold"])
    face_det = FaceDetector(cfg["yolo_weights"], cfg["detection_confidence"], cfg["device"], cfg["detection_imgsz"])
    person_trk = PersonTracker(cfg["person_weights"], cfg["tracker_config"], cfg["person_confidence"],
                               cfg["device"], cfg["person_imgsz"])
    embedder = gallery = None
    if cfg["enable_reid"]:                          # OSNet body re-identification
        embedder = BodyEmbedder(cfg["reid_weights"], cfg["reid_arch"], cfg["device"], cfg.get("reid_mean"))
        gallery = MultiGallery(512, cfg["gallery_size"])
    events = EventManager(cfg, rec, db, ImageStore(cfg["log_dir"]), embedder, gallery)
    src = VideoSource(cfg["source"])

    writer = None
    if cfg["save_annotated_video"]:
        writer = cv2.VideoWriter(cfg["save_annotated_video"], cv2.VideoWriter_fourcc(*"mp4v"), src.fps, src.size)

    stop_flag = Path(cfg["log_dir"]) / "stop.flag"     # the dashboard's Stop button creates this file
    stop_flag.unlink(missing_ok=True)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))   # graceful Ctrl+C

    step = cfg["detection_skip_frames"] + 1
    frame_idx, last_dets, t0 = 0, [], time.time()
    try:
        while not stop["flag"]:
            frame = src.read()
            if frame is None:
                break
            if frame_idx % step == 0:               # detection cycle; other frames are skipped
                last_dets = person_trk.track(frame)
                events.update(frame, last_dets, face_det.detect(frame), frame_idx)
            else:
                events._expire(frame_idx)           # exits still fire on skipped frames
            if stop_flag.exists():
                log.info("Stop requested from dashboard")
                break
            live = cfg.get("live_preview", True) and frame_idx % 2 == 0
            if writer or cfg["show_preview"] or live:
                vis = draw(frame.copy(), last_dets, events.tracks, events.total_visitors, events)
                if writer:
                    writer.write(vis)
                if live:
                    publish_live(vis, cfg["log_dir"], {
                        "state": "running", "frame": frame_idx, "total": src.total, "source": str(cfg["source"]),
                        "fps": round(frame_idx / max(time.time() - t0, 1e-6), 1),
                        "visitors": events.total_visitors, "active": len(events.tracks)})
                if cfg["show_preview"]:
                    cv2.imshow("Face Tracker", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            frame_idx += 1
    finally:
        events.flush()                              # no exit event is ever lost
        if cfg["consolidate"]:                      # merge duplicates; people seen together are never merged
            try:
                merges = consolidate(db, cfg)
                apply_merges(db, merges)
                events.total_visitors = db.count_unique()
                log.info("CONSOLIDATE done: %d duplicate visitors merged", len(merges))
            except Exception:  # noqa: BLE001
                log.exception("consolidation failed (counts left unmerged)")
        src.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        fps = frame_idx / max(time.time() - t0, 1e-6)
        log.info("END frames=%d avg_fps=%.1f UNIQUE_VISITORS=%d (by kind: %s)", frame_idx, fps, db.count_unique(), db.count_by_kind())
        db.close()
        final = Path(cfg["log_dir"]) / "status.tmp.json"
        final.write_text(json.dumps({
            "state": "finished", "frame": frame_idx, "total": src.total, "source": str(cfg["source"]),
            "fps": round(fps, 1), "visitors": events.total_visitors, "active": 0}),
            encoding="utf-8")
        _replace(final, Path(cfg["log_dir"]) / "status.json")
        stop_flag.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
