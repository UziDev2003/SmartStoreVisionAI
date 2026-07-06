"""
Line Crossing + Direction Detection
====================================
Detects when a tracked person crosses a virtual line (entrance,
exit, restricted-boundary). Records direction (N/S/E/W or
"in"/"out") and provides a count per line.

Two complementary mechanisms:
  1. Line crossing via signed-distance sign change between frames
  2. Direction via heading comparison (heading_degrees from
     movement_features)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import math
import time
import numpy as np

from src.analytics.movement_features import heading_degrees


@dataclass
class LineZone:
    """A virtual line defined by two endpoints and a name/direction hint."""
    line_id: str
    name: str
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    # 'in' / 'out' / cardinal direction ('N','S','E','W') label for the side
    # to the LEFT of the line direction (p1 -> p2)
    left_label: str = "in"
    right_label: str = "out"


@dataclass
class LineCrossing:
    line_id: str
    track_id: int
    direction: str  # 'in' or 'out' (or left/right label)
    timestamp: float
    confidence: float = 1.0


def _signed_distance(p: Tuple[int, int], p1: Tuple[int, int],
                     p2: Tuple[int, int]) -> float:
    """Signed distance of point p from line through p1->p2. Positive = left."""
    x, y = p
    x1, y1 = p1
    x2, y2 = p2
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


class LineCrossingDetector:
    """
    Detects when a track crosses any of N virtual lines. Each track
    is tracked by its last position per line, and a sign change
    between consecutive frames triggers a crossing.
    """
    def __init__(self, lines: Optional[List[LineZone]] = None,
                 cooldown_seconds: float = 1.0,
                 cardinal_threshold_deg: float = 45.0):
        self.lines: Dict[str, LineZone] = {}
        if lines:
            for l in lines:
                self.lines[l.line_id] = l
        self.cooldown = cooldown_seconds
        self.cardinal_threshold = cardinal_threshold_deg
        # Per-(line, track) state
        self._last_sign: Dict[Tuple[str, int], float] = {}
        self._last_cross_time: Dict[Tuple[str, int], float] = {}
        # Counters
        self.counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.history: List[LineCrossing] = []

    def add_line(self, line: LineZone) -> None:
        self.lines[line.line_id] = line

    def update(self, tracks, timestamp: Optional[float] = None) -> List[LineCrossing]:
        ts = timestamp or time.time()
        crossings: List[LineCrossing] = []
        for line_id, line in self.lines.items():
            for track in tracks:
                key = (line_id, track.track_id)
                hist = getattr(track, "history", [])
                if len(hist) < 2:
                    continue
                prev_pt = hist[-2][1]
                cur_pt = hist[-1][1]
                d_prev = _signed_distance(prev_pt, line.p1, line.p2)
                d_cur = _signed_distance(cur_pt, line.p1, line.p2)
                # Sign change -> crossing
                if d_prev * d_cur < 0:
                    # Direction = which side we ended up on
                    if d_cur > 0:
                        direction = line.left_label
                    else:
                        direction = line.right_label
                    # Cooldown
                    last_t = self._last_cross_time.get(key, 0.0)
                    if ts - last_t < self.cooldown:
                        continue
                    self._last_cross_time[key] = ts
                    conf = min(1.0, abs(d_cur) / 100.0 + 0.5)
                    cr = LineCrossing(line_id, track.track_id, direction, ts, conf)
                    crossings.append(cr)
                    self.counts[(line_id, direction)] += 1
                    self.history.append(cr)
        return crossings

    def direction_label(self, track) -> str:
        """Cardinal direction of last movement (N/S/E/W)."""
        if len(track.history) < 2:
            return "?"
        h = heading_degrees(track.history[-2][1], track.history[-1][1])
        # image coordinate: 0=east, 90=south
        h = (h + 360) % 360
        if 45 <= h < 135:
            return "S"
        if 135 <= h < 225:
            return "W"
        if 225 <= h < 315:
            return "N"
        return "E"

    def get_counts(self, line_id: Optional[str] = None) -> Dict[Tuple[str, str], int]:
        if line_id is None:
            return dict(self.counts)
        return {k: v for k, v in self.counts.items() if k[0] == line_id}


from collections import defaultdict
