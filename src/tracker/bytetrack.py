"""ByteTrack Multi-Object Tracker - Simple IoU-based"""
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import cv2

@dataclass
class Track:
    track_id: int; class_id: int; confidence: float
    bbox: Tuple[int, int, int, int]; timestamp: float
    history: List = field(default_factory=list)
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox; return ((x1+x2)//2, (y1+y2)//2)

@dataclass
class TrackingResult:
    tracks: List[Track]; frame_id: int; timestamp: float
    num_detections: int; num_tracks: int

class ByteTracker:
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30):
        self.thresh = track_thresh
        self.buffer = track_buffer
        self.frame_count = 0
        self.tracks: Dict[int, Track] = {}
        self.histories: Dict[int, List] = defaultdict(list)
        self.next_id = 1
        self.times: List[float] = []
    
    def update(self, frame, detections: List, ts: Optional[float] = None) -> TrackingResult:
        ts = ts or time.time()
        start = time.perf_counter()
        self.frame_count += 1
        result_tracks = []
        
        for d in detections:
            # Extract bbox and conf from Detection object or list
            if hasattr(d, 'bbox'):
                x1, y1, x2, y2 = d.bbox; conf = float(d.confidence)
            elif isinstance(d, (list, np.ndarray)) and len(d) >= 5:
                vals = list(map(float, d[:5])); x1, y1, x2, y2 = map(int, vals[:4]); conf = vals[4]
            else: continue
            
            center = ((x1+x2)//2, (y1+y2)//2)
            matched = False
            
            # Match with existing tracks using IoU
            for tid, t in list(self.tracks.items()):
                if self._iou(t.bbox, (x1, y1, x2, y2)) > 0.3:
                    t.bbox = (x1, y1, x2, y2); t.confidence = conf; t.timestamp = ts
                    t.history.append((ts, center)); self.histories[tid].append((ts, center))
                    if len(self.histories[tid]) > 30: self.histories[tid].pop(0)
                    result_tracks.append(t); matched = True; break
            
            # Create new track
            if not matched and conf > self.thresh:
                tid = self.next_id; self.next_id += 1
                t = Track(tid, 0, conf, (x1, y1, x2, y2), ts, [(ts, center)])
                self.tracks[tid] = t; self.histories[tid] = [(ts, center)]
                result_tracks.append(t)
        
        # Remove stale tracks
        for tid in list(self.tracks.keys()):
            if ts - self.tracks[tid].timestamp > self.buffer / 30.0:
                del self.tracks[tid]
                if tid in self.histories: del self.histories[tid]
        
        self.times.append(time.perf_counter() - start)
        return TrackingResult(result_tracks, self.frame_count, ts, len(detections), len(result_tracks))
    
    def _iou(self, b1: Tuple, b2: Tuple) -> float:
        x1, y1, x2, y2 = b1; x1b, y1b, x2b, y2b = b2
        xi1, yi1 = max(x1, x1b), max(y1, y1b); xi2, yi2 = min(x2, x2b), min(y2, y2b)
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        union = (x2-x1)*(y2-y1) + (x2b-x1b)*(y2b-y1b) - inter
        return inter/union if union > 0 else 0
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Track], trail: int = 15) -> np.ndarray:
        out = frame.copy()
        colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            c = colors[t.track_id % len(colors)]
            cv2.rectangle(out, (x1,y1), (x2,y2), c, 2)
            if len(t.history) > 1:
                pts = [p for _,p in t.history[-trail:]]
                for j in range(1, len(pts)): cv2.line(out, pts[j-1], pts[j], c, max(1, j//4))
            cv2.putText(out, f"ID:{t.track_id}", (x1, max(y1-5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        return out
    
    def get_stats(self) -> Dict:
        if not self.times: return {"fps": 0}
        t = sum(self.times) / len(self.times)
        return {"fps": 1/t if t > 0 else 0}
    
    def reset(self):
        self.frame_count = 0; self.tracks.clear(); self.histories.clear(); self.times.clear()
