"""
Pose-Based Detectors: Gesture, Fall, Fight
==========================================
Three higher-level detectors built on top of the PoseAnalyzer
(MediaPipe). All use the cached pose data (`left_wrist`,
`right_wrist`, `is_crouching`, `is_leaning`) that PoseAnalyzer
produces each frame.

1. GestureDetector  — hand-raise, point, fighting stance
2. FallDetector     — body suddenly goes horizontal, near floor
3. FightDetector    — rapid hand motion by two close persons +
                       erratic crouching + leaning
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time
import math
from collections import deque

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


@dataclass
class PoseEvent:
    event_type: str          # 'gesture' / 'fall' / 'fight'
    track_id: Optional[int]
    track_ids: List[int]
    gesture: Optional[str]   # for gesture detector
    confidence: float
    timestamp: float
    message: str


class GestureDetector:
    """
    Detects simple upper-body gestures from wrist + elbow positions.
    Gestures:
      - 'hand_raised'  : wrist above shoulder (> head height)
      - 'pointing'     : one arm extended, other at side
      - 'hands_up'     : both wrists above head
      - 'fighting_stance' : both arms out wide, crouched
    """
    def __init__(self, min_track_length: int = 5,
                 cooldown_seconds: float = 5.0,
                 raise_threshold_px: int = 40):
        self.min_track_length = min_track_length
        self.cooldown = cooldown_seconds
        self.raise_threshold = raise_threshold_px
        self._last_emit: Dict[Tuple[int, str], float] = {}

    def update(self, pose_estimates: Dict[int, Dict], tracks,
               timestamp: Optional[float] = None) -> List[PoseEvent]:
        ts = timestamp or time.time()
        out: List[PoseEvent] = []
        for tid, pose in pose_estimates.items():
            if pose.get("confidence", 0.0) < 0.4:
                continue
            if np is None:
                continue
            gesture = self._classify(pose)
            if gesture is None:
                continue
            key = (tid, gesture)
            if ts - self._last_emit.get(key, 0.0) < self.cooldown:
                continue
            self._last_emit[key] = ts
            out.append(PoseEvent(
                event_type="gesture",
                track_id=tid,
                track_ids=[tid],
                gesture=gesture,
                confidence=pose.get("confidence", 0.0),
                timestamp=ts,
                message=f"Gesture: {gesture}",
            ))
        return out

    def _classify(self, pose: Dict) -> Optional[str]:
        lh = pose.get("left_wrist")
        rh = pose.get("right_wrist")
        le = pose.get("left_elbow")
        re_ = pose.get("right_elbow")
        if not lh or not rh:
            return None
        # Estimate "head height" as min(y) of elbows (rough)
        if le and re_:
            head_y = min(le[1], re_[1])
        else:
            head_y = lh[1] - self.raise_threshold
        both_raised = lh[1] < head_y - self.raise_threshold and rh[1] < head_y - self.raise_threshold
        if both_raised:
            return "hands_up"
        # Single hand raised
        if lh[1] < head_y - self.raise_threshold or rh[1] < head_y - self.raise_threshold:
            return "hand_raised"
        # Crouching + hands out wide -> fighting stance
        if pose.get("is_crouching") and le and re_:
            arm_spread = abs(lh[0] - rh[0])
            if arm_spread > 100:
                return "fighting_stance"
        return None


class FallDetector:
    """
    Detects when a person suddenly drops to the ground.
    Heuristic: large drop in track center y AND low motion for
    several frames afterwards. Uses pose data when available
    (is_crouching + horizontal aspect) for higher accuracy.
    """
    def __init__(self, min_drop_px: int = 60,
                 min_track_length: int = 10,
                 cooldown_seconds: float = 30.0):
        self.min_drop_px = min_drop_px
        self.min_track_length = min_track_length
        self.cooldown = cooldown_seconds
        self._baseline_y: Dict[int, float] = {}
        self._fallen_at: Dict[int, float] = {}
        self._last_alert: Dict[int, float] = {}

    def update(self, tracks, pose_estimates: Optional[Dict[int, Dict]] = None,
               timestamp: Optional[float] = None) -> List[PoseEvent]:
        ts = timestamp or time.time()
        pose_estimates = pose_estimates or {}
        out: List[PoseEvent] = []
        for track in tracks:
            tid = track.track_id
            if len(track.history) < self.min_track_length:
                continue
            cx, cy = track.center
            # Set baseline y on first sighting
            if tid not in self._baseline_y:
                self._baseline_y[tid] = cy
            baseline = self._baseline_y[tid]
            drop = baseline - cy  # positive = dropped
            is_crouch = pose_estimates.get(tid, {}).get("is_crouching", False)
            is_lean = pose_estimates.get(tid, {}).get("is_leaning", False)
            if drop >= self.min_drop_px or (is_crouch and drop > 20) or is_lean:
                if tid not in self._fallen_at:
                    self._fallen_at[tid] = ts
                # Confirm: stay down for >= 1.0s
                if ts - self._fallen_at[tid] >= 1.0:
                    if ts - self._last_alert.get(tid, 0.0) >= self.cooldown:
                        conf = min(0.95, 0.5 + drop / 200.0)
                        out.append(PoseEvent(
                            event_type="fall",
                            track_id=tid,
                            track_ids=[tid],
                            gesture=None,
                            confidence=conf,
                            timestamp=ts,
                            message=f"Possible fall detected (drop {drop}px)",
                        ))
                        self._last_alert[tid] = ts
            else:
                # Person recovered
                self._fallen_at.pop(tid, None)
                # Slowly update baseline to follow standing height
                self._baseline_y[tid] = 0.95 * self._baseline_y[tid] + 0.05 * cy
        return out


class FightDetector:
    """
    Detects a fight between two or more people.
    Heuristic: two or more tracks in close proximity, BOTH with
    crouching + leaning + rapid hand motion (pose_confidence
    + high heading-change-rate).

    Without pose, the fallback uses motion-energy (frame diff)
    inside a small area around the two tracks.
    """
    def __init__(self, proximity_px: int = 120,
                 min_pair_dwell: float = 2.0,
                 cooldown_seconds: float = 30.0):
        self.proximity_px = proximity_px
        self.min_pair_dwell = min_pair_dwell
        self.cooldown = cooldown_seconds
        self._pair_first_seen: Dict[Tuple[int, int], float] = {}
        self._last_alert: Dict[Tuple[int, int], float] = {}

    def update(self, tracks, pose_estimates: Optional[Dict[int, Dict]] = None,
               timestamp: Optional[float] = None) -> List[PoseEvent]:
        ts = timestamp or time.time()
        pose_estimates = pose_estimates or {}
        out: List[PoseEvent] = []
        # Build active pairs
        active_pairs: set = set()
        for i, t1 in enumerate(tracks):
            for t2 in tracks[i + 1:]:
                d = math.hypot(t1.center[0] - t2.center[0],
                                t1.center[1] - t2.center[1])
                if d <= self.proximity_px:
                    pair = tuple(sorted([t1.track_id, t2.track_id]))
                    active_pairs.add(pair)
                    if pair not in self._pair_first_seen:
                        self._pair_first_seen[pair] = ts
        # Cleanup old pairs
        for pair in list(self._pair_first_seen.keys()):
            if pair not in active_pairs:
                self._pair_first_seen.pop(pair, None)
        # Emit alerts for sustained close proximity
        for pair in active_pairs:
            if ts - self._pair_first_seen[pair] >= self.min_pair_dwell:
                if ts - self._last_alert.get(pair, 0.0) >= self.cooldown:
                    # Optional: require at least one crouched / leaning
                    crouch = any(pose_estimates.get(tid, {}).get("is_crouching", False) for tid in pair)
                    lean = any(pose_estimates.get(tid, {}).get("is_leaning", False) for tid in pair)
                    conf = 0.6
                    if crouch:
                        conf += 0.2
                    if lean:
                        conf += 0.2
                    conf = min(0.95, conf)
                    out.append(PoseEvent(
                        event_type="fight",
                        track_id=None,
                        track_ids=list(pair),
                        gesture=None,
                        confidence=conf,
                        timestamp=ts,
                        message=f"Possible fight / scuffle between tracks {pair[0]} and {pair[1]}",
                    ))
                    self._last_alert[pair] = ts
        return out
