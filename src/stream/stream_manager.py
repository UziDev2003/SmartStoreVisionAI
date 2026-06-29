"""Stream Manager Module"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import time, threading, queue
import numpy as np, cv2
from pathlib import Path

@dataclass
class StreamConfig:
    stream_id: str; source: str; name: str; enabled: bool = True; fps: int = 30

@dataclass
class StreamFrame:
    stream_id: str; frame: np.ndarray; timestamp: float; frame_number: int; width: int; height: int

class StreamManager:
    def __init__(self, streams: List[StreamConfig], buffer_size: int = 10):
        self.streams = {s.stream_id: s for s in streams}
        self.buffers: Dict[str, queue.Queue] = {}
        self.captures: Dict[str, cv2.VideoCapture] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.running: Dict[str, bool] = {}
        self.stats = {s.stream_id: {"frames": 0, "fps": 0.0, "dropped": 0, "errors": 0} for s in streams}
        self.last_time: Dict[str, float] = {}
        self.callbacks: List[Callable] = []
    
    def start(self, sid: Optional[str] = None):
        if sid: self._start(sid)
        else:
            for s in self.streams: self._start(s)
    
    def _start(self, sid: str):
        if self.running.get(sid, False): return
        cfg = self.streams.get(sid); 
        if not cfg or not cfg.enabled: return
        cap = cv2.VideoCapture(cfg.source)
        if not cap.isOpened(): self.stats[sid]["errors"] += 1; return
        self.captures[sid] = cap; self.buffers[sid] = queue.Queue(maxsize=10)
        self.running[sid] = True
        t = threading.Thread(target=self._loop, args=(sid,), daemon=True); t.start(); self.threads[sid] = t
    
    def _loop(self, sid: str):
        cap = self.captures[sid]; buf = self.buffers[sid]
        fn = 0; last_t = time.time(); frames = 0
        while self.running.get(sid, False):
            ret, frame = cap.read()
            if not ret:
                self.stats[sid]["errors"] += 1
                if cfg.source.startswith('rtsp'):
                    time.sleep(2); cap.release(); cap = cv2.VideoCapture(cfg.source); self.captures[sid] = cap
                else: break
                continue
            fn += 1; ts = time.time(); frames += 1
            if ts - last_t >= 1.0: self.stats[sid]["fps"] = frames / (ts - last_t); frames = 0; last_t = ts
            try: buf.put_nowait(StreamFrame(sid, frame, ts, fn, frame.shape[1], frame.shape[0])); self.stats[sid]["frames"] += 1
            except queue.Full: self.stats[sid]["dropped"] += 1
            self.last_time[sid] = ts
        cap.release()
    
    def get_frame(self, sid: str, timeout: float = 1.0) -> Optional[StreamFrame]:
        buf = self.buffers.get(sid)
        if not buf: return None
        try: return buf.get(timeout=timeout)
        except queue.Empty: return None
    
    def stop(self, sid: Optional[str] = None):
        if sid: self._stop(sid)
        else:
            for s in self.streams: self._stop(s)
    
    def _stop(self, sid: str):
        self.running[sid] = False
        if sid in self.threads: self.threads[sid].join(timeout=2)
        if sid in self.captures: self.captures[sid].release()
    
    def is_healthy(self, sid: str, max_age: float = 5.0) -> bool:
        return sid in self.last_time and time.time() - self.last_time[sid] < max_age
