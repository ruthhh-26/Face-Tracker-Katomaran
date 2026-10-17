"""Load and validate config.json."""
import json
from pathlib import Path

DEFAULTS = {
    "source": "sample.mp4",
    "yolo_weights": "models/yolov8n-face.pt",
    "person_weights": "models/yolov8n.pt",
    "person_confidence": 0.4,
    "person_imgsz": 960,
    "body_confirm_frames": 20,      # person detections without a usable face before a body-only (P) ID is assigned
    "min_hits_at_end": 2,           # at end of stream, a person seen this many times is still counted (just walked in)
    "min_body_hits": 10,            # shorter person tracks are treated as noise
    "enable_reid": True,
    "reid_weights": "models/osnet_ain_x1_0.pt",
    "reid_arch": "osnet_ain_x1_0",
    "reid_mean": "models/osnet_mean.npy",   # feature-centering vector from tools/calibrate_reid.py
    "body_threshold": 0.70,         # centered-OSNet similarity alone is enough (different people: p99 = 0.61)
    "face_low_threshold": 0.30,     # weaker face similarity, accepted only together with body_low_threshold
    "body_low_threshold": 0.50,
    "continuity_frames": 150,        # a new track starting within this many frames of another one ending, nearby, is likely the same person
    "continuity_body_threshold": 0.35,
    "lost_after_frames": 9,         # a track unseen this long is LOST (out of sight) and gets a prediction zone
    "zone_base": 0.6,               # prediction zone radius = height * (zone_base + zone_growth * frames_lost)
    "zone_growth": 0.012,
    "motion_only": False,           # accept a new track as the lost person when it is the only one that fits the zone
    "motion_body_floor": 0.0,       # ... and the body does not contradict
    "drop_union_boxes": True,       # ignore one person box that covers 2+ other people
    "consolidate": True,            # merge duplicate visitors at the end of a run (never merges people seen together)
    "consolidate_face": 0.40,       # looser than live matching: the never-together rule protects against false merges
    "consolidate_body": 0.50,
    "consolidate_pair_face": 0.25,
    "consolidate_pair_body": 0.40,
    "overlap_frames": 15,           # two visitors on screen together this many frames are different people
    "face_veto": 0.10,              # face similarity below this means different people, whatever the body says
    "gallery_size": 8,              # embeddings kept per visitor (face and body)
    "body_samples": 5,              # max body embeddings per track
    "body_sample_interval": 4,      # take one every N person detections
    "min_body_height": 80,          # px; shorter person crops are too small to embed
    "tracker_config": "bytetrack.yaml",
    "device": "cpu",
    "detection_skip_frames": 2,     # frames skipped between detection cycles
    "detection_confidence": 0.5,
    "detection_imgsz": 640,         # YOLO input size; raise (1280/1920) for small faces in 2K/4K video
    "min_face_size": 40,            # px; smaller faces are not embedded/registered
    "recognition_threshold": 0.45,  # cosine similarity needed to call two faces the same
    "embedding_samples": 3,         # embeddings averaged before identity decision
    "min_confirm_frames": 3,        # detections before a track counts as a real face
    "exit_grace_frames": 60,        # frames without detection before EXIT is logged
    "log_dir": "logs",
    "show_preview": False,
    "save_annotated_video": "",
    "db": {"host": "localhost", "port": 5433, "name": "facetracker",
           "user": "postgres", "password": "postgres"},
}


def load_config(path: str = "config.json") -> dict:
    cfg = dict(DEFAULTS)
    cfg["db"] = dict(DEFAULTS["db"])
    user = json.loads(Path(path).read_text(encoding="utf-8"))
    db = user.pop("db", {})
    cfg.update(user)
    cfg["db"].update(db)
    if cfg["detection_skip_frames"] < 0:
        raise ValueError("detection_skip_frames must be >= 0")
    return cfg
