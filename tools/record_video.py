"""Record the expensive per-video model outputs once (person tracks, faces, face + body embeddings) so identity
settings can be replayed and tuned in seconds (tools/evaluate.py).

Usage:  python tools/record_video.py "Sample/Video Datasets/xxx.mp4" [more videos ...]
Output: cache/<video name>.pkl
"""
import pickle
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.detector import FaceDetector, PersonTracker  # noqa: E402
from src.events import EventManager  # noqa: E402
from src.recognizer import Recognizer  # noqa: E402
from src.reid import BodyEmbedder  # noqa: E402

MIN_FACE, MIN_BODY_H = 20, 60          # record everything the tunable minimum sizes could ever need


def record(video: str, cfg: dict, out: Path):
    face_det = FaceDetector(cfg["yolo_weights"], cfg["detection_confidence"], cfg["device"], cfg["detection_imgsz"])
    person_trk = PersonTracker(cfg["person_weights"], cfg["tracker_config"], cfg["person_confidence"],
                               cfg["device"], cfg["person_imgsz"])
    rec = Recognizer(cfg["recognition_threshold"])
    body = BodyEmbedder(cfg["reid_weights"], cfg["reid_arch"], cfg["device"], cfg.get("reid_mean"))
    cap = cv2.VideoCapture(video)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    step, cycles, i = cfg["detection_skip_frames"] + 1, {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            persons, faces = person_trk.track(frame), face_det.detect(frame)
            body_emb = {}
            for p in persons:
                if p.box[3] - p.box[1] >= MIN_BODY_H:
                    crop = EventManager._crop(frame, p.box)
                    if crop.size:
                        body_emb[p.track_id] = body.embed(crop)
            face_emb = {tuple(f.box): rec.embed(frame, f.box, f.kps) for f in faces
                        if min(f.box[2] - f.box[0], f.box[3] - f.box[1]) >= MIN_FACE}
            cycles[i] = {"persons": [(p.track_id, tuple(p.box), p.conf) for p in persons],
                         "faces": [(tuple(f.box), f.kps, f.conf) for f in faces],
                         "body": body_emb, "face_emb": face_emb}
        i += 1
    cap.release()
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(pickle.dumps({"video": video, "size": size, "total": i, "step": step, "cycles": cycles}))
    print(f"recorded {video}: {i} frames, {len(cycles)} cycles -> {out}", flush=True)


if __name__ == "__main__":
    cfg = load_config("config.json")
    for v in sys.argv[1:]:
        record(v, cfg, Path("cache") / (Path(v).stem + ".pkl"))
