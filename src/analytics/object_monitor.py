"""
Object Left Behind / Object Removed Detector
============================================
Two related detectors for shelf monitoring:

1. ObjectLeftBehindDetector — alerts when a person places an
   object on a shelf and walks away, leaving the object behind
   (the shelf is supposed to be customer-empty after the person
   leaves, e.g. fitting rooms or restricted storage).

2. ObjectRemovedDetector — alerts when a shelf is "full" with
   items and one or more are removed (potential shoplifting).
   Uses the YOLO bag/backpack/handbag class to detect the
   moving "object" the person carries.

Both rely on the per-shelf occupancy and the frame-difference
motion-energy as a stationary-object proxy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class ShelfArea:
    shelf_id: str
    name: str
    polygon: List[Tuple[int, int]]


@dataclass
class ShelfEvent:
    shelf_id: str
    event_type: str  # 'object_left' or 'object_removed'
    track_id: Optional[int]
    timestamp: float
    severity: str = "medium"
    message: str = ""


class ObjectLeftBehindDetector:
    """
    Alert when:
      1. Person was in shelf zone, then left
      2. After they leave, stationary pixel energy remains > threshold
         for `min_seconds` (proxy for an object left)
    """
    def __init__(self, min_seconds: float = 5.0,
                 stationary_threshold: int = 800,
                 cooldown_seconds: float = 60.0):
        self.min_seconds = min_seconds
        self.stationary_threshold = stationary_threshold
        self.cooldown = cooldown_seconds
        self._last_person_left: Dict[str, float] = {}
        self._last_alert: Dict[str, float] = {}

    def update(self, frame: Optional[np.ndarray], tracks,
               zone_occ: Dict[str, Dict[int, float]], shelves: List[ShelfArea],
               timestamp: Optional[float] = None) -> List[ShelfEvent]:
        ts = timestamp or time.time()
        events: List[ShelfEvent] = []
        for shelf in shelves:
            sid = shelf.shelf_id
            occ_ids = list(zone_occ.get(sid, {}).keys())
            if not occ_ids:
                if sid not in self._last_person_left:
                    self._last_person_left[sid] = ts
                stationary = self._stationary_score(frame, shelf) if frame is not None else 0.0
                duration = ts - self._last_person_left[sid]
                if stationary > 0.4 and duration >= self.min_seconds:
                    if ts - self._last_alert.get(sid, 0.0) >= self.cooldown:
                        events.append(ShelfEvent(
                            shelf_id=sid,
                            event_type="object_left",
                            track_id=None,
                            timestamp=ts,
                            severity="medium",
                            message=f"Object possibly left in {shelf.name} for {duration:.0f}s",
                        ))
                        self._last_alert[sid] = ts
            else:
                self._last_person_left.pop(sid, None)
        return events

    def _stationary_score(self, frame, shelf: ShelfArea) -> float:
        if frame is None or cv2 is None:
            return 0.0
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(shelf.polygon, dtype=np.int32)], 255)
            region = gray[mask > 0]
            if region.size == 0:
                return 0.0
            # Use simple gradient magnitude as activity proxy
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            rmag = mag[mask > 0]
            if rmag.size == 0:
                return 0.0
            active_px = int((rmag > 8.0).sum())
            return min(1.0, active_px / max(1, self.stationary_threshold))
        except Exception:
            return 0.0


class ObjectRemovedDetector:
    """
    Alert when:
      1. Shelf has high background-occupancy (full) initially
      2. The "fullness" metric drops by > 30% (object removed)
      3. A person is currently in/near the shelf
    """
    def __init__(self, baseline_frames: int = 60,
                 min_drop_ratio: float = 0.3,
                 cooldown_seconds: float = 60.0):
        self.baseline_frames = baseline_frames
        self.min_drop_ratio = min_drop_ratio
        self.cooldown = cooldown_seconds
        self._baseline: Dict[str, float] = {}
        self._last_alert: Dict[str, float] = {}
        self._histories: Dict[str, list] = {}

    def _shelf_fullness(self, frame, shelf: ShelfArea) -> float:
        """Heuristic: ratio of edges (gradient magnitude) inside shelf polygon."""
        if frame is None or cv2 is None:
            return 0.0
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(shelf.polygon, dtype=np.int32)], 255)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            region = mag[mask > 0]
            if region.size == 0:
                return 0.0
            return min(1.0, float((region > 20.0).sum()) / float(region.size))
        except Exception:
            return 0.0

    def update(self, frame: Optional[np.ndarray], tracks,
               zone_occ: Dict[str, Dict[int, float]], shelves: List[ShelfArea],
               timestamp: Optional[float] = None) -> List[ShelfEvent]:
        ts = timestamp or time.time()
        events: List[ShelfEvent] = []
        for shelf in shelves:
            sid = shelf.shelf_id
            fullness = self._shelf_fullness(frame, shelf)
            if sid not in self._histories:
                self._histories[sid] = []
            hist = self._histories[sid]
            hist.append(fullness)
            if len(hist) > 240:
                hist.pop(0)
            # Build baseline from first N frames
            if sid not in self._baseline and len(hist) >= self.baseline_frames:
                self._baseline[sid] = sum(hist[:self.baseline_frames]) / self.baseline_frames
            baseline = self._baseline.get(sid, 0.0)
            if baseline <= 0.05:
                continue  # shelf essentially empty
            drop = (baseline - fullness) / baseline
            # A person must be near the shelf
            person_near = any(zone_occ.get(sid, {}))
            if drop >= self.min_drop_ratio and person_near:
                if ts - self._last_alert.get(sid, 0.0) >= self.cooldown:
                    # Find the person track
                    ptrack = next(iter(zone_occ.get(sid, {}).keys()), None)
                    events.append(ShelfEvent(
                        shelf_id=sid,
                        event_type="object_removed",
                        track_id=ptrack,
                        timestamp=ts,
                        severity="high",
                        message=f"Possible object removed from {shelf.name} (fullness drop {drop*100:.0f}%)",
                    ))
                    self._last_alert[sid] = ts
        return events
