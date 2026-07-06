"""
Background Subtraction
======================
Wraps OpenCV's MOG2 and KNN background subtractors for motion-only
detection (no YOLO needed). Useful for empty stores, after-hours
monitoring, and motion maps.

Returns a foreground mask plus aggregate statistics (motion area,
largest blob, motion density).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2
import numpy as np


@dataclass
class MotionStats:
    fg_mask: np.ndarray
    motion_area: int          # total foreground pixels
    total_area: int           # total frame pixels
    motion_density: float     # motion_area / total_area
    largest_blob_area: int    # px count of largest connected component
    largest_blob_center: Optional[Tuple[int, int]]
    contours: List


class BackgroundSubtractor:
    """
    Wraps cv2.createBackgroundSubtractorMOG2 or KNN. Each call to
    `apply` returns a `MotionStats` with the foreground mask and
    aggregate metrics.
    """
    def __init__(self, method: str = "MOG2",
                 history: int = 500, var_threshold: float = 16.0,
                 detect_shadows: bool = True, learning_rate: float = -1.0,
                 kernel_size: int = 5, min_blob_area: int = 500):
        self.method = method
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.learning_rate = learning_rate
        self.min_blob_area = min_blob_area
        if method.upper() == "KNN":
            self._sub = cv2.createBackgroundSubtractorKNN(
                history=history, dist2Threshold=var_threshold, detectShadows=detect_shadows)
        else:
            self._sub = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def apply(self, frame: np.ndarray) -> MotionStats:
        if frame is None:
            return MotionStats(np.zeros((0, 0), dtype=np.uint8), 0, 0, 0.0, 0, None, [])
        mask = self._sub.apply(frame, learningRate=self.learning_rate)
        # Remove shadows (gray = 127)
        if self.detect_shadows:
            mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        # Clean up noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        motion_area = int(mask.sum() // 255)
        total_area = int(mask.size)
        density = motion_area / max(1, total_area)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        largest_center: Optional[Tuple[int, int]] = None
        for c in contours:
            a = cv2.contourArea(c)
            if a < self.min_blob_area:
                continue
            if a > largest_area:
                largest_area = int(a)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    largest_center = (int(M["m10"] / M["m00"]),
                                      int(M["m01"] / M["m00"]))
        return MotionStats(
            fg_mask=mask,
            motion_area=motion_area,
            total_area=total_area,
            motion_density=density,
            largest_blob_area=largest_area,
            largest_blob_center=largest_center,
            contours=contours,
        )

    def reset(self) -> None:
        """Re-create the underlying model (e.g. on scene change)."""
        if self.method.upper() == "KNN":
            self._sub = cv2.createBackgroundSubtractorKNN(
                history=self.history, dist2Threshold=self.var_threshold,
                detectShadows=self.detect_shadows)
        else:
            self._sub = cv2.createBackgroundSubtractorMOG2(
                history=self.history, varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows)
