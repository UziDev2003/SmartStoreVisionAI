"""Alert Manager Module"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import time, json, cv2
from pathlib import Path

@dataclass
class Alert:
    alert_id: str; alert_type: str; severity: str; camera_id: str; track_id: int
    zone_id: Optional[str]; message: str; timestamp: float; snapshot_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict: return self.__dict__

class AlertManager:
    def __init__(self, snapshot_dir: str = "data/snapshots", max_alerts: int = 1000, rate_limit: float = 30.0):
        self.snapshot_dir = Path(snapshot_dir); self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.max_alerts = max_alerts; self.rate_limit = rate_limit
        self.alerts: deque = deque(maxlen=max_alerts)
        self.last_alert: Dict[str, float] = {}
        self.callbacks: List[Callable] = []
        self.stats = {"total": 0, "by_type": {}, "by_severity": {}}
    
    def create(self, alert_type: str, camera_id: str, track_id: int, message: str,
               severity: str = "medium", zone_id: Optional[str] = None,
               frame: Any = None, bbox: Optional[tuple] = None, metadata: Optional[Dict] = None) -> Optional[Alert]:
        ts = time.time(); key = f"{camera_id}:{alert_type}"
        if key in self.last_alert and ts - self.last_alert[key] < self.rate_limit: return None
        self.last_alert[key] = ts
        alert_id = f"{camera_id}_{alert_type}_{int(ts * 1000)}"
        snap = self._save_snap(alert_id, frame, bbox) if frame is not None else None
        alert = Alert(alert_id, alert_type, severity, camera_id, track_id, zone_id, message, ts, snap, metadata or {})
        self.alerts.append(alert)
        self.stats["total"] += 1
        self.stats["by_type"][alert_type] = self.stats["by_type"].get(alert_type, 0) + 1
        self.stats["by_severity"][severity] = self.stats["by_severity"].get(severity, 0) + 1
        for cb in self.callbacks:
            try: cb(alert)
            except: pass
        return alert
    
    def _save_snap(self, aid: str, frame: Any, bbox: Optional[tuple], pad: int = 50) -> Optional[str]:
        try:
            if bbox:
                x1, y1, x2, y2 = bbox; h, w = frame.shape[:2]
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                snap = frame[y1:y2, x1:x2]
            else: snap = frame
            p = self.snapshot_dir / f"{aid}.jpg"; cv2.imwrite(str(p), snap); return str(p)
        except: return None
    
    def register(self, cb: Callable): self.callbacks.append(cb)
    
    def get_alerts(self, camera_id: Optional[str] = None, alert_type: Optional[str] = None,
                   since: Optional[float] = None, limit: int = 100) -> List[Alert]:
        r = list(self.alerts)
        if camera_id: r = [a for a in r if a.camera_id == camera_id]
        if alert_type: r = [a for a in r if a.alert_type == alert_type]
        if since: r = [a for a in r if a.timestamp >= since]
        return sorted(r, key=lambda a: a.timestamp, reverse=True)[:limit]
    
    def get_recent(self, secs: float = 300) -> List[Alert]:
        return [a for a in self.alerts if a.timestamp >= time.time() - secs]
