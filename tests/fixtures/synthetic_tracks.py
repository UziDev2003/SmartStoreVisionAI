"""
Synthetic track generator for offline testing.

Produces Track objects with controllable motion patterns:
  - stationary(track_id, t_start, t_end, x, y)         # loitering
  - walking(track_id, t_start, t_end, path)            # normal
  - running(track_id, t_start, t_end, path, dt)        # high speed
  - erratic(track_id, t_start, t_end, area, dt)        # many turns
  - restricted_entry(track_id, t_start, t_end, poly)   # inside restricted zone
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple
from dataclasses import dataclass, field

from src.tracker.bytetrack import Track


def make_stationary(track_id: int, t_start: float, t_end: float,
                    x: int, y: int, fps: int = 30,
                    jitter: int = 0) -> Track:
    """Track that stays in place (with tiny jitter)."""
    n = max(1, int((t_end - t_start) * fps))
    history = []
    for i in range(n):
        ts = t_start + i / fps
        jx = x + random.randint(-jitter, jitter)
        jy = y + random.randint(-jitter, jitter)
        history.append((ts, (jx, jy)))
    x1, y1, x2, y2 = x - 20, y - 40, x + 20, y + 40
    return Track(track_id=track_id, class_id=0, confidence=0.9,
                 bbox=(x1, y1, x2, y2), timestamp=history[-1][0], history=history)


def make_walking(track_id: int, t_start: float, t_end: float,
                 start: Tuple[int, int], end: Tuple[int, int],
                 fps: int = 30) -> Track:
    """Track that moves linearly at walking speed."""
    n = max(1, int((t_end - t_start) * fps))
    history = []
    for i in range(n):
        ts = t_start + i / fps
        f = i / max(1, n - 1)
        x = int(start[0] + (end[0] - start[0]) * f)
        y = int(start[1] + (end[1] - start[1]) * f)
        history.append((ts, (x, y)))
    x, y = history[-1][1]
    return Track(track_id=track_id, class_id=0, confidence=0.9,
                 bbox=(x - 20, y - 40, x + 20, y + 40),
                 timestamp=history[-1][0], history=history)


def make_running(track_id: int, t_start: float, t_end: float,
                 start: Tuple[int, int], end: Tuple[int, int],
                 speed_px_s: float = 350.0, fps: int = 30) -> Track:
    """Track that moves very fast (running)."""
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    n = max(1, int(dist / speed_px_s * fps))
    if n < 5:
        n = 5
    history = []
    for i in range(n):
        ts = t_start + i / fps
        f = i / max(1, n - 1)
        x = int(start[0] + (end[0] - start[0]) * f)
        y = int(start[1] + (end[1] - start[1]) * f)
        history.append((ts, (x, y)))
    x, y = history[-1][1]
    return Track(track_id=track_id, class_id=0, confidence=0.9,
                 bbox=(x - 20, y - 40, x + 20, y + 40),
                 timestamp=history[-1][0], history=history)


def make_erratic(track_id: int, t_start: float, t_end: float,
                 center: Tuple[int, int], radius: int = 60,
                 fps: int = 30, n_turns: int = 8) -> Track:
    """Track that turns rapidly (erratic)."""
    n = max(1, int((t_end - t_start) * fps))
    history = []
    for i in range(n):
        ts = t_start + i / fps
        ang = (i / max(1, n - 1)) * 2 * math.pi * (n_turns / 2)
        x = int(center[0] + radius * math.cos(ang))
        y = int(center[1] + radius * math.sin(ang))
        history.append((ts, (x, y)))
    x, y = history[-1][1]
    return Track(track_id=track_id, class_id=0, confidence=0.9,
                 bbox=(x - 20, y - 40, x + 20, y + 40),
                 timestamp=history[-1][0], history=history)


def make_zone_zone_dwell(track_id: int, t_start: float, t_end: float,
                         center: Tuple[int, int], fps: int = 30) -> Track:
    """Track that dwells inside a small box (e.g., shelf zone)."""
    return make_stationary(track_id, t_start, t_end, center[0], center[1], fps=fps)


def make_track_in_polygon(track_id: int, t_start: float, t_end: float,
                          polygon: List[Tuple[int, int]], fps: int = 30) -> Track:
    """Track that is inside a polygon the whole time."""
    cx = sum(p[0] for p in polygon) // len(polygon)
    cy = sum(p[1] for p in polygon) // len(polygon)
    return make_stationary(track_id, t_start, t_end, cx, cy, fps=fps)


# Pre-made zones for tests
def demo_zones() -> dict:
    return {
        "entrance": {
            "id": "entrance", "name": "Entrance", "type": "entrance",
            "dwell_threshold": 60.0, "polygon": [(100, 400), (300, 400), (300, 600), (100, 600)],
        },
        "checkout": {
            "id": "checkout", "name": "Checkout", "type": "checkout",
            "dwell_threshold": 30.0, "polygon": [(500, 400), (700, 400), (700, 600), (500, 600)],
        },
        "restricted": {
            "id": "restricted", "name": "Backroom", "type": "restricted",
            "dwell_threshold": 0.0, "polygon": [(800, 100), (950, 100), (950, 250), (800, 250)],
        },
        "shelf": {
            "id": "shelf", "name": "Shelf A", "type": "shelf",
            "dwell_threshold": 120.0, "polygon": [(400, 250), (600, 250), (600, 380), (400, 380)],
        },
    }
