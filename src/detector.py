"""YOLO face detector (identity cue) and YOLO person detector + ByteTrack (the tracked entity)."""
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    track_id: int       # person track id (-1 for face detections)
    box: tuple          # x1,y1,x2,y2 (ints)
    conf: float
    kps: np.ndarray     # (5,2) facial landmarks, or None


class FaceDetector:
    """Plain per-frame face detection (tracking is done at person level)."""

    def __init__(self, weights: str, conf: float, device: str, imgsz: int = 640):
        self.model = YOLO(weights)
        self.conf, self.device, self.imgsz = conf, device, imgsz

    def detect(self, frame) -> list:
        res = self.model.predict(frame, conf=self.conf, device=self.device, imgsz=self.imgsz, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        kps = res.keypoints.xy.cpu().numpy() if res.keypoints is not None and res.keypoints.xy is not None else None
        out = []
        for i in range(len(boxes)):
            k = kps[i] if kps is not None and i < len(kps) and len(kps[i]) >= 5 else None
            out.append(Detection(-1, tuple(int(v) for v in boxes[i]), float(confs[i]), k))
        return out


class PersonTracker:
    """YOLO person detection (COCO class 0) + ByteTrack. Works even when the face is not visible."""

    def __init__(self, weights: str, tracker_cfg: str, conf: float, device: str, imgsz: int = 960):
        self.model = YOLO(weights)
        self.tracker_cfg, self.conf, self.device, self.imgsz = tracker_cfg, conf, device, imgsz

    def track(self, frame) -> list:
        res = self.model.track(frame, persist=True, tracker=self.tracker_cfg, classes=[0], conf=self.conf,
                               device=self.device, imgsz=self.imgsz, verbose=False)[0]
        if res.boxes is None or res.boxes.id is None:
            return []
        ids = res.boxes.id.int().cpu().tolist()
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        return [Detection(ids[i], tuple(int(v) for v in boxes[i]), float(confs[i]), None) for i in range(len(ids))]
