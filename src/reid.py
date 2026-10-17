"""OSNet-AIN body re-identification: links a person seen from behind to the same person seen from the front."""
from pathlib import Path

import cv2
import numpy as np
import torch
import torchreid

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BodyEmbedder:
    def __init__(self, weights: str, arch: str = "osnet_ain_x1_0", device: str = "cpu", mean_path: str = None):
        self.device = device
        # Raw OSNet features are all-positive, so any two people look alike under cosine. Subtracting the mean
        # feature (computed by tools/calibrate_reid.py) removes that common component.
        self.mean = np.load(mean_path).astype(np.float32) if mean_path and Path(mean_path).exists() else None
        self.model = torchreid.models.build_model(arch, num_classes=1, pretrained=False)
        # checkpoint has the classifier head stripped -> strict=False only skips classifier.*
        self.model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
        self.model.eval().to(device)

    @torch.no_grad()
    def embed(self, crop_bgr, key=None) -> np.ndarray:
        """L2-normalised 512-d appearance embedding of a person crop."""
        img = cv2.cvtColor(cv2.resize(crop_bgr, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(((img - _MEAN) / _STD).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        v = self.model(x)[0].cpu().numpy().astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        if self.mean is not None:
            v = v - self.mean
            v = v / (np.linalg.norm(v) + 1e-9)
        return v
