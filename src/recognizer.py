"""InsightFace (ArcFace w600k_r50 from buffalo_l) embeddings backed by a multi-embedding gallery."""
import logging
from types import SimpleNamespace

import numpy as np
from insightface.app import FaceAnalysis

from src.gallery import MultiGallery

log = logging.getLogger("facetracker")

# Relative 5-point layout inside a face box (eyes, nose, mouth corners); used only if YOLO gives no landmarks
_TEMPLATE = np.array([[0.30, 0.37], [0.70, 0.37], [0.50, 0.55], [0.34, 0.75], [0.66, 0.75]], dtype=np.float32)


class Recognizer:
    def __init__(self, threshold: float):
        self.threshold = threshold
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1)
        self.model = app.models["recognition"]
        self.gallery = MultiGallery(512)

    def embed(self, frame, box, kps, key=None) -> np.ndarray:
        """L2-normalised 512-d ArcFace embedding of the face at `box` (landmark-aligned)."""
        if kps is None:
            x1, y1, x2, y2 = box
            kps = np.array([[x1 + (x2 - x1) * px, y1 + (y2 - y1) * py] for px, py in _TEMPLATE], dtype=np.float32)
        face = SimpleNamespace(kps=np.asarray(kps, dtype=np.float32))
        emb = np.asarray(self.model.get(frame, face), dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)
