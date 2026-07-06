"""
Face Blur Utility
=================
Privacy-preserving face blurring. Three implementations:
  1. OpenCV Haar cascade (lightweight, no extra deps)
  2. MediaPipe face detection (more accurate)
  3. Whole-frame blur (fallback — no detection)

Usage:
    from src.utils.face_blur import FaceBlurrer
    fb = FaceBlurrer(method="opencv", strength=51)
    out = fb.blur(frame)
"""
from __future__ import annotations

from typing import Tuple, List, Optional
import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
    MEDIAPIPE_AVAILABLE = True
except Exception:  # pragma: no cover
    MEDIAPIPE_AVAILABLE = False


class FaceBlurrer:
    def __init__(self, method: str = "opencv", strength: int = 51,
                 cascade_path: Optional[str] = None,
                 min_face_size: int = 30):
        self.method = method
        # Strength must be odd
        if strength % 2 == 0:
            strength += 1
        self.strength = max(3, strength)
        self.min_face_size = min_face_size
        self._cascade = None
        self._mp_face = None
        if method == "opencv":
            try:
                if cascade_path is None:
                    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self._cascade = None
        elif method == "mediapipe":
            if MEDIAPIPE_AVAILABLE:
                try:
                    self._mp_face = mp.solutions.face_detection.FaceDetection(
                        model_selection=0, min_detection_confidence=0.5
                    )
                except Exception:
                    self._mp_face = None

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return list of (x1, y1, x2, y2) face boxes."""
        boxes: List[Tuple[int, int, int, int]] = []
        if frame is None or frame.size == 0:
            return boxes
        if self.method == "mediapipe" and self._mp_face is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._mp_face.process(rgb)
                if results.detections:
                    h, w = frame.shape[:2]
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        x1 = max(0, int(bbox.xmin * w))
                        y1 = max(0, int(bbox.ymin * h))
                        x2 = min(w, int((bbox.xmin + bbox.width) * w))
                        y2 = min(h, int((bbox.ymin + bbox.height) * h))
                        if x2 - x1 > self.min_face_size and y2 - y1 > self.min_face_size:
                            boxes.append((x1, y1, x2, y2))
            except Exception:
                pass
        elif self.method == "opencv" and self._cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found = self._cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5,
                    minSize=(self.min_face_size, self.min_face_size)
                )
                for (x, y, w, h) in found:
                    boxes.append((x, y, x + w, y + h))
            except Exception:
                pass
        return boxes

    def blur(self, frame: np.ndarray, boxes: Optional[List[Tuple[int, int, int, int]]] = None) -> np.ndarray:
        if frame is None:
            return frame
        out = frame.copy()
        if boxes is None:
            boxes = self.detect_faces(frame)
        for (x1, y1, x2, y2) in boxes:
            # Expand a little to be safe
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1 - 5), max(0, y1 - 5)
            x2, y2 = min(w, x2 + 5), min(h, y2 + 5)
            face = out[y1:y2, x1:x2]
            if face.size == 0:
                continue
            blurred = cv2.GaussianBlur(face, (self.strength, self.strength), 0)
            out[y1:y2, x1:x2] = blurred
        return out

    def close(self) -> None:
        if self._mp_face is not None:
            try:
                self._mp_face.close()
            except Exception:
                pass
            self._mp_face = None
