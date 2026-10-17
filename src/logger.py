"""events.log setup and structured image storage."""
import logging
from datetime import datetime
from pathlib import Path

import cv2


def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("facetracker")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(Path(log_dir) / "events.log", encoding="utf-8")
    sh = logging.StreamHandler()
    for h in (fh, sh):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


class ImageStore:
    """Saves face crops to logs/<kind>/YYYY-MM-DD/<label>_<time>.jpg."""

    def __init__(self, log_dir: str):
        self.root = Path(log_dir)

    def save(self, kind: str, label: str, ts: datetime, crop) -> str:
        folder = self.root / kind / ts.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{label}_{ts.strftime('%H%M%S_%f')[:-3]}.jpg"
        tmp = path.with_suffix(".tmp.jpg")
        cv2.imwrite(str(tmp), crop)
        tmp.replace(path)                      # atomic: never leave half-written images
        return str(path).replace("\\", "/")
