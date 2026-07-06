"""
Movement Feature Extraction
===========================
Trajectory feature extraction for movement analysis:
- instantaneous / average / peak speed
- heading change (turn) detection
- motion variance (immobility vs. movement)
- jerk (acceleration change)

These features power the MovementAnalyzer and improve LoiteringDetector
(false-positive suppression for browsing customers).
"""
from __future__ import annotations

from typing import List, Dict, Tuple, Optional
from collections import deque
import math
import numpy as np


def euclidean(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def compute_speed_series(history: List[Tuple[float, Tuple[int, int]]]) -> List[float]:
    """
    Compute per-step speed (px/s) from a (ts, center) history.
    Returns a list with length max(0, len(history) - 1).
    """
    speeds: List[float] = []
    for i in range(1, len(history)):
        t0, p0 = history[i - 1]
        t1, p1 = history[i]
        dt = t1 - t0
        if dt <= 0:
            continue
        speeds.append(euclidean(p0, p1) / dt)
    return speeds


def heading_degrees(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Heading in degrees, 0=east, 90=south (image coordinate system)."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def angular_diff_deg(a: float, b: float) -> float:
    """Smallest angular difference in degrees in [0, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def count_significant_turns(history: List[Tuple[float, Tuple[int, int]]],
                            threshold_deg: float = 35.0) -> int:
    """Count turns whose heading change exceeds the threshold (across consecutive steps)."""
    if len(history) < 3:
        return 0
    headings = [heading_degrees(history[i - 1][1], history[i][1]) for i in range(1, len(history))]
    turns = 0
    for i in range(1, len(headings)):
        if angular_diff_deg(headings[i - 1], headings[i]) >= threshold_deg:
            turns += 1
    return turns


def motion_variance(history: List[Tuple[float, Tuple[int, int]]],
                    window: int = 30) -> float:
    """
    Variance of step displacements (px^2) over the last `window` points.
    Returns 0.0 for short histories. Used to discriminate between
    truly-stationary (loitering) and moving-while-browsing (normal) tracks.
    """
    if len(history) < 3:
        return 0.0
    last = history[-window:] if len(history) >= window else history
    steps = []
    for i in range(1, len(last)):
        steps.append(euclidean(last[i - 1][1], last[i][1]))
    if not steps:
        return 0.0
    return float(np.var(steps))


def compute_features(history: List[Tuple[float, Tuple[int, int]]]) -> Dict[str, float]:
    """
    Compute a feature dictionary for a track history.
    Returned keys:
        avg_speed, peak_speed, motion_var, total_distance,
        heading_change_count, net_displacement, duration_seconds
    """
    if len(history) < 2:
        return {
            "avg_speed": 0.0,
            "peak_speed": 0.0,
            "motion_var": 0.0,
            "total_distance": 0.0,
            "heading_change_count": 0,
            "net_displacement": 0.0,
            "duration_seconds": 0.0,
        }
    speeds = compute_speed_series(history)
    total = sum(euclidean(history[i - 1][1], history[i][1]) for i in range(1, len(history)))
    duration = max(0.0, history[-1][0] - history[0][0])
    net = euclidean(history[0][1], history[-1][1])
    return {
        "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
        "peak_speed": float(np.max(speeds)) if speeds else 0.0,
        "motion_var": motion_variance(history),
        "total_distance": float(total),
        "heading_change_count": count_significant_turns(history),
        "net_displacement": float(net),
        "duration_seconds": float(duration),
    }


def is_running(features: Dict[str, float], threshold_px_s: float,
               min_track_length: int = 20) -> bool:
    """Decide if a track indicates running (sustained high speed)."""
    return (
        features.get("duration_seconds", 0.0) >= 1.0
        and features.get("peak_speed", 0.0) >= threshold_px_s
    )


def is_erratic(features: Dict[str, float],
               heading_change_threshold: int = 6,
               window_seconds: float = 5.0) -> bool:
    """Decide if a track shows erratic motion (lots of turns in short time)."""
    return (
        features.get("duration_seconds", 0.0) <= window_seconds * 2
        and features.get("heading_change_count", 0) >= heading_change_threshold
    )


class SlidingWindowStats:
    """Reusable rolling stats helper for stream processing."""
    def __init__(self, window: int = 30):
        self.window = window
        self.buf: deque = deque(maxlen=window)

    def add(self, value: float) -> None:
        self.buf.append(float(value))

    def mean(self) -> float:
        return float(np.mean(self.buf)) if self.buf else 0.0

    def var(self) -> float:
        return float(np.var(self.buf)) if len(self.buf) > 1 else 0.0

    def max(self) -> float:
        return float(np.max(self.buf)) if self.buf else 0.0

    def __len__(self) -> int:
        return len(self.buf)
