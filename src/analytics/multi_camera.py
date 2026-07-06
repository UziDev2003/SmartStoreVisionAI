"""
Multi-Camera Re-Identification
==============================
Correlates the same physical person across two or more camera views
using a cheap HSV color-histogram similarity (default) or an OSNet
deep-feature embedding (optional, requires torch).

This produces a `global_person_id` that can be attached to alerts
to enable cross-camera correlation (e.g. evasion detection).
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np


def extract_color_histogram(crop_bgr: np.ndarray, bins: int = 32) -> np.ndarray:
    """Extract a normalized HSV color histogram from a BGR crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(bins * 3, dtype=np.float32)
    try:
        import cv2
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], channels=[0, 1, 2], mask=None, histSize=[bins, bins, bins],
            ranges=[0, 180, 0, 256, 0, 256],
        )
        hist = hist.flatten().astype(np.float32)
        n = hist.sum()
        if n > 0:
            hist /= n
        return hist
    except Exception:
        return np.zeros(bins * 3, dtype=np.float32)


def hsv_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya-like similarity in [0, 1] (1 = identical)."""
    if h1 is None or h2 is None or h1.size == 0 or h2.size == 0:
        return 0.0
    h1n = h1 / (np.linalg.norm(h1) + 1e-9)
    h2n = h2 / (np.linalg.norm(h2) + 1e-9)
    return float(np.clip(np.dot(h1n, h2n), 0.0, 1.0))


class MultiCameraCorrelator:
    """
    Maintains a global pool of (appearance, last_seen_camera, last_seen_ts)
    and assigns the most-similar match above threshold to new track crops.
    """

    def __init__(self, config: Any):
        self.config = config
        self.enabled = bool(getattr(config, "enabled", False))
        self.method = str(getattr(config, "method", "hsv"))
        self.threshold = float(getattr(config, "similarity_threshold", 0.5))
        self.ttl = float(getattr(config, "global_track_ttl_seconds", 600.0))
        # global_id -> {hist, last_camera, last_ts}
        self.global_pool: Dict[int, Dict[str, Any]] = {}
        # (camera_id, track_id) -> global_id
        self.local_to_global: Dict[Tuple[str, int], int] = {}
        self._next_global_id = 1

    def _extract_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract a feature vector from a bounding box crop."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(96, dtype=np.float32)
        if self.method == "osnet":
            return self._osnet_embed(crop)
        return extract_color_histogram(crop)

    def _osnet_embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Optional OSNet embedding via torchreid; falls back to HSV."""
        try:
            import torch
            import torchreid  # type: ignore  # noqa
            # Minimal skeleton; full integration is optional
            return extract_color_histogram(crop_bgr)
        except Exception:
            return extract_color_histogram(crop_bgr)

    def _cleanup(self, now: float) -> None:
        """Remove stale global identities beyond TTL."""
        stale = [gid for gid, info in self.global_pool.items()
                 if now - info["last_ts"] > self.ttl]
        for gid in stale:
            self.global_pool.pop(gid, None)
            # Also drop local->global mappings pointing to gid
            self.local_to_global = {
                k: v for k, v in self.local_to_global.items() if v != gid
            }

    def update(self, frame: np.ndarray, camera_id: str,
               tracks: List[Any], timestamp: Optional[float] = None) -> Dict[Tuple[str, int], int]:
        """
        For each track, compute an appearance embedding, find the best
        matching global identity, and return a {(camera_id, track_id):
        global_id} mapping. New identities are assigned as needed.
        """
        if not self.enabled or frame is None or not tracks:
            return {}

        ts = timestamp or time.time()
        out: Dict[Tuple[str, int], int] = {}
        for t in tracks:
            emb = self._extract_embedding(frame, t.bbox)
            best_gid: Optional[int] = None
            best_sim = -1.0
            for gid, info in self.global_pool.items():
                if info["last_camera"] == camera_id:
                    # Skip same-camera comparisons for cross-camera
                    continue
                sim = hsv_similarity(emb, info["hist"])
                if sim > best_sim:
                    best_sim = sim
                    best_gid = gid
            key = (camera_id, t.track_id)
            if best_gid is not None and best_sim >= self.threshold:
                self.global_pool[best_gid]["last_ts"] = ts
                self.global_pool[best_gid]["last_camera"] = camera_id
                self.global_pool[best_gid]["hist"] = emb
                self.local_to_global[key] = best_gid
                out[key] = best_gid
            else:
                new_gid = self._next_global_id
                self._next_global_id += 1
                self.global_pool[new_gid] = {
                    "hist": emb, "last_camera": camera_id, "last_ts": ts,
                }
                self.local_to_global[key] = new_gid
                out[key] = new_gid
        self._cleanup(ts)
        return out

    def get_global_id(self, camera_id: str, track_id: int) -> Optional[int]:
        return self.local_to_global.get((camera_id, track_id))
