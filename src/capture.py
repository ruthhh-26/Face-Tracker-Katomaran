"""Video source: file or RTSP. RTSP uses a reader thread (latest frame only) + reconnect."""
import logging
import threading
import time

import cv2

log = logging.getLogger("facetracker")


class VideoSource:
    def __init__(self, source: str):
        self.source = source
        self.is_stream = str(source).lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.cap = self._open()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_stream else 0
        self._latest = None
        self._lock = threading.Lock()
        self._stop = False
        if self.is_stream:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        return cap

    def _reader(self):
        """Continuously drain the stream so processing never lags behind live video."""
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                log.warning("Stream interrupted, reconnecting...")
                self.cap.release()
                while not self._stop:
                    time.sleep(2)
                    try:
                        self.cap = self._open()
                        break
                    except RuntimeError:
                        continue
                continue
            with self._lock:
                self._latest = frame

    def read(self):
        """Return next frame, or None at end of file / after stop."""
        if not self.is_stream:
            ok, frame = self.cap.read()
            return frame if ok else None
        while not self._stop:
            with self._lock:
                frame, self._latest = self._latest, None
            if frame is not None:
                return frame
            time.sleep(0.005)
        return None

    def release(self):
        self._stop = True
        self.cap.release()
