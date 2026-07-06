"""
Queue Detection
===============
Detects a queue (line of people waiting) at a checkout/service zone.
Heuristic: a queue exists if there are N+ people inside a queue
"lane" polygon (front-of-zone + trailing space) AND the front
person has dwell time > T.

Provides:
  - queue length (number of people in queue lane)
  - estimated wait time (front-person dwell)
  - average wait time (rolling)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import time
from collections import deque


@dataclass
class QueueZone:
    """Polygon in front of a checkout/service counter where the queue forms."""
    queue_id: str
    name: str
    polygon: List[Tuple[int, int]]
    capacity: int = 10
    min_queue_length: int = 2
    wait_threshold_seconds: float = 60.0


@dataclass
class QueueStatus:
    queue_id: str
    name: str
    length: int
    estimated_wait_seconds: float
    avg_wait_seconds: float
    timestamp: float
    is_queue: bool  # True if length >= min_queue_length


def _point_in_polygon(pt: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class QueueDetector:
    def __init__(self, queues: Optional[List[QueueZone]] = None,
                 history_window: int = 20):
        self.queues: Dict[str, QueueZone] = {}
        if queues:
            for q in queues:
                self.queues[q.queue_id] = q
        self._dwell_history: Dict[str, deque] = {}
        self._first_seen: Dict[Tuple[str, int], float] = {}

    def add_queue(self, q: QueueZone) -> None:
        self.queues[q.queue_id] = q

    def update(self, tracks, zone_occ: Dict[str, Dict[int, float]],
               timestamp: Optional[float] = None) -> List[QueueStatus]:
        ts = timestamp or time.time()
        out: List[QueueStatus] = []
        for qid, q in self.queues.items():
            in_queue = []
            for track in tracks:
                if _point_in_polygon(track.center, q.polygon):
                    in_queue.append(track)
            length = len(in_queue)
            # Compute estimated wait = max dwell in queue
            estimated = 0.0
            for t in in_queue:
                d = self._get_dwell(qid, t.track_id, ts)
                if d > estimated:
                    estimated = d
            # Update rolling history
            if qid not in self._dwell_history:
                self._dwell_history[qid] = deque(maxlen=20)
            if length >= q.min_queue_length and estimated > 0:
                self._dwell_history[qid].append(estimated)
            avg = (sum(self._dwell_history[qid]) / len(self._dwell_history[qid])
                   if self._dwell_history[qid] else 0.0)
            # Cleanup
            for tid in list(self._first_seen.keys()):
                if tid[0] == qid and tid[1] not in [t.track_id for t in in_queue]:
                    del self._first_seen[tid]
            is_queue = length >= q.min_queue_length and estimated >= q.wait_threshold_seconds
            out.append(QueueStatus(
                queue_id=qid,
                name=q.name,
                length=length,
                estimated_wait_seconds=float(estimated),
                avg_wait_seconds=float(avg),
                timestamp=ts,
                is_queue=is_queue,
            ))
        return out

    def _get_dwell(self, qid: str, track_id: int, ts: float) -> float:
        key = (qid, track_id)
        if key not in self._first_seen:
            self._first_seen[key] = ts
        return ts - self._first_seen[key]
