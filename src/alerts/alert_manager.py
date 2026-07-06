"""Alert Manager Module (extended with confidence, evidence, severity_score)."""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import time, json, cv2
from pathlib import Path


@dataclass
class Alert:
    alert_id: str
    alert_type: str
    severity: str
    camera_id: str
    track_id: int
    zone_id: Optional[str]
    message: str
    timestamp: float
    snapshot_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    # New fields for suspicious activity integration
    confidence: float = 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    severity_score: int = 0
    global_person_id: Optional[int] = None
    rule_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


# Severity ordering for sort/filter
SEVERITY_ORDER = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


class AlertManager:
    def __init__(self, snapshot_dir: str = "data/snapshots", max_alerts: int = 1000,
                 rate_limit: float = 30.0):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.max_alerts = max_alerts
        self.rate_limit = rate_limit
        self.alerts: deque = deque(maxlen=max_alerts)
        self.last_alert: Dict[str, float] = {}
        self.callbacks: List[Callable] = []
        self.acknowledged: Dict[str, float] = {}
        self.stats = {"total": 0, "by_type": {}, "by_severity": {}}

    def create(self, alert_type: str, camera_id: str, track_id: int, message: str,
               severity: str = "medium", zone_id: Optional[str] = None,
               frame: Any = None, bbox: Optional[tuple] = None,
               metadata: Optional[Dict] = None,
               confidence: float = 1.0,
               evidence: Optional[Dict] = None,
               global_person_id: Optional[int] = None,
               rule_violations: Optional[List[str]] = None) -> Optional[Alert]:
        ts = time.time()
        key = f"{camera_id}:{alert_type}:{zone_id or '_'}:{track_id}"
        if key in self.last_alert and ts - self.last_alert[key] < self.rate_limit:
            return None
        self.last_alert[key] = ts
        alert_id = f"{camera_id}_{alert_type}_{int(ts * 1000)}"
        snap = self._save_snap(alert_id, frame, bbox) if frame is not None else None
        sev_score = SEVERITY_ORDER.get(severity.upper(), 1) if isinstance(severity, str) else 1
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            camera_id=camera_id,
            track_id=track_id,
            zone_id=zone_id,
            message=message,
            timestamp=ts,
            snapshot_path=snap,
            metadata=metadata or {},
            confidence=float(confidence),
            evidence=evidence or {},
            severity_score=sev_score,
            global_person_id=global_person_id,
            rule_violations=rule_violations or [],
        )
        self.alerts.append(alert)
        self.stats["total"] += 1
        self.stats["by_type"][alert_type] = self.stats["by_type"].get(alert_type, 0) + 1
        self.stats["by_severity"][severity] = self.stats["by_severity"].get(severity, 0) + 1
        for cb in self.callbacks:
            try:
                cb(alert)
            except Exception:
                pass
        return alert

    def create_from_activity(self, activity, camera_id: str, frame: Any = None,
                             bbox: Optional[tuple] = None) -> Optional[Alert]:
        """Convenience: create an alert from a SuspiciousActivity."""
        return self.create(
            alert_type=activity.activity_type.value,
            camera_id=camera_id,
            track_id=activity.track_id if activity.track_id is not None else -1,
            message=activity.message,
            severity=activity.severity.value,
            zone_id=activity.zone_id,
            frame=frame,
            bbox=bbox,
            metadata={
                "activity_id": activity.activity_id,
                "zone_name": activity.zone_name,
            },
            confidence=activity.confidence,
            evidence=activity.evidence,
            global_person_id=activity.global_person_id,
            rule_violations=activity.rule_violations,
        )

    def _save_snap(self, aid: str, frame: Any, bbox: Optional[tuple], pad: int = 50) -> Optional[str]:
        try:
            if bbox:
                x1, y1, x2, y2 = bbox
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                snap = frame[y1:y2, x1:x2]
            else:
                snap = frame
            p = self.snapshot_dir / f"{aid}.jpg"
            cv2.imwrite(str(p), snap)
            return str(p)
        except Exception:
            return None

    def register(self, cb: Callable) -> None:
        self.callbacks.append(cb)

    def acknowledge(self, alert_id: str) -> bool:
        if alert_id in [a.alert_id for a in self.alerts]:
            self.acknowledged[alert_id] = time.time()
            return True
        return False

    def get_alerts(self, camera_id: Optional[str] = None, alert_type: Optional[str] = None,
                   since: Optional[float] = None, limit: int = 100,
                   min_severity: Optional[str] = None,
                   min_confidence: float = 0.0,
                   include_acknowledged: bool = True) -> List[Alert]:
        r = list(self.alerts)
        if camera_id:
            r = [a for a in r if a.camera_id == camera_id]
        if alert_type:
            r = [a for a in r if a.alert_type == alert_type]
        if since:
            r = [a for a in r if a.timestamp >= since]
        if min_severity:
            min_score = SEVERITY_ORDER.get(min_severity.upper(), 0)
            r = [a for a in r if a.severity_score >= min_score]
        if min_confidence > 0:
            r = [a for a in r if a.confidence >= min_confidence]
        if not include_acknowledged:
            r = [a for a in r if a.alert_id not in self.acknowledged]
        return sorted(r, key=lambda a: a.timestamp, reverse=True)[:limit]

    def get_recent(self, secs: float = 300) -> List[Alert]:
        return [a for a in self.alerts if a.timestamp >= time.time() - secs]
