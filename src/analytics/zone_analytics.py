"""Zone-Based Analytics Module"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time, numpy as np, cv2

@dataclass
class Zone:
    zone_id: str; name: str; zone_type: str; polygon: List[Tuple[int, int]]
    capacity: Optional[int] = None; dwell_threshold: float = 120.0
    
    def contains_point(self, pt: Tuple[int, int]) -> bool:
        x, y = pt; n = len(self.polygon); inside = False; j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]; xj, yj = self.polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi): inside = not inside
            j = i
        return inside
    
    def get_center(self) -> Tuple[int, int]:
        return (sum(p[0] for p in self.polygon) // len(self.polygon), sum(p[1] for p in self.polygon) // len(self.polygon))

@dataclass
class ZoneEvent:
    track_id: int; zone_id: str; event_type: str; timestamp: float; dwell_time: Optional[float] = None

@dataclass
class ZoneStats:
    zone_id: str; current_occupancy: int = 0; total_entries: int = 0; total_exits: int = 0
    avg_dwell_time: float = 0.0; max_dwell_time: float = 0.0; congestion_level: float = 0.0

class ZoneAnalytics:
    def __init__(self, zones: List[Zone], fw: int = 1920, fh: int = 1080):
        self.zones = {z.zone_id: z for z in zones}
        self.zone_occ: Dict[str, Dict[int, float]] = defaultdict(dict)
        self.dwell_times: Dict[str, List[float]] = defaultdict(list)
        self.entries: Dict[str, int] = defaultdict(int)
        self.exits: Dict[str, int] = defaultdict(int)
        self.stats: Dict[str, ZoneStats] = {z.zone_id: ZoneStats(z.zone_id) for z in zones}
        self.heatmap = np.zeros((fh // 10, fw // 10), dtype=np.float32)
        self.events: List[ZoneEvent] = []
    
    def update(self, tracks: List, ts: Optional[float] = None) -> List[ZoneEvent]:
        ts = ts or time.time(); evts = []
        for t in tracks:
            cx, cy = t.center; tid = t.track_id
            for zid, z in self.zones.items():
                in_z = z.contains_point((cx, cy)); was_in = tid in self.zone_occ[zid]
                if in_z and not was_in:
                    self.zone_occ[zid][tid] = ts; self.entries[zid] += 1
                    e = ZoneEvent(tid, zid, "enter", ts); evts.append(e); self.events.append(e)
                elif not in_z and was_in:
                    dwell = ts - self.zone_occ[zid].pop(tid, ts)
                    self.dwell_times[zid].append(dwell); self.exits[zid] += 1
                    e = ZoneEvent(tid, zid, "exit", ts, dwell); evts.append(e); self.events.append(e)
                elif in_z:
                    gx, gy = int(cx // 10), int(cy // 10)
                    if 0 <= gy < self.heatmap.shape[0] and 0 <= gx < self.heatmap.shape[1]: self.heatmap[gy, gx] += 1
        self._update_stats()
        if len(self.events) > 1000: self.events = self.events[-1000:]
        return evts
    
    def _update_stats(self):
        for zid, z in self.zones.items():
            s = self.stats[zid]; s.current_occupancy = len(self.zone_occ[zid])
            s.total_entries = self.entries[zid]; s.total_exits = self.exits[zid]
            if self.dwell_times[zid]:
                s.avg_dwell_time = sum(self.dwell_times[zid]) / len(self.dwell_times[zid])
                s.max_dwell_time = max(self.dwell_times[zid])
            if z.capacity: s.congestion_level = min(1.0, s.current_occupancy / z.capacity)
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        colors = {"entrance": (0,255,255), "checkout": (255,0,255), "restricted": (0,0,255), "general": (0,255,0)}
        for zid, z in self.zones.items():
            c = colors.get(z.zone_type, (255,255,255)); pts = np.array(z.polygon, dtype=np.int32)
            ov = out.copy(); cv2.fillPoly(ov, [pts], (*c, 50)); cv2.addWeighted(ov, 0.3, out, 0.7, 0, out)
            cv2.polylines(out, [pts], True, c, 2)
            cx, cy = z.get_center(); cv2.putText(out, f"{z.name}: {self.stats[zid].current_occupancy}", (cx-40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        return out
