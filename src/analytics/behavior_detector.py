"""
Behavior Detection Module (backwards-compatible thin wrapper)
============================================================
Retains the original BehaviorAlert/BehaviorDetector public API for
backwards compatibility, while internally forwarding to the new
SuspiciousActivityDetector when available.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class AlertType(Enum):
    LOITERING = "loitering"
    INTRUSION = "intrusion"
    CONGESTION = "congestion"


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class BehaviorAlert:
    alert_type: AlertType
    severity: AlertSeverity
    track_id: int
    zone_id: Optional[str]
    message: str
    timestamp: float
    metadata: Dict = None


class BehaviorDetector:
    """
    Backwards-compatible behavior detector. Optionally wraps a
    SuspiciousActivityDetector for richer features.
    """
    def __init__(self, loiter_threshold: float = 120.0, intrusion_zones: List = None,
                 congestion_threshold: float = 0.8,
                 suspicious_detector: Any = None):
        self.loiter_threshold = loiter_threshold
        self.intrusion_zones = intrusion_zones or []
        self.congestion_threshold = congestion_threshold
        self.alert_cooldowns: Dict[str, float] = {}
        self.cooldown_period = 30.0
        self.alert_history: List[BehaviorAlert] = []
        self.suspicious_detector = suspicious_detector  # optional

    def detect(self, tracks: List, zones: Dict, zone_occupancy: Dict,
               timestamp: Optional[float] = None,
               frame: Any = None,
               camera_id: str = "cam_01",
               detections: Any = None) -> List[BehaviorAlert]:
        if timestamp is None:
            timestamp = time.time()
        alerts: List[BehaviorAlert] = []

        # If a SuspiciousActivityDetector is wired in, optionally consume its output
        if self.suspicious_detector is not None:
            try:
                activities = self.suspicious_detector.detect(
                    tracks=tracks,
                    zones=zones,
                    zone_occ=zone_occupancy,
                    timestamp=timestamp,
                    frame=frame,
                    detections=detections,
                    camera_id=camera_id,
                )
                for a in activities:
                    sev = AlertSeverity.MEDIUM
                    sev_v = a.severity.value.upper() if hasattr(a.severity, "value") else str(a.severity).upper()
                    if sev_v in ("HIGH", "CRITICAL"):
                        sev = AlertSeverity.HIGH
                    elif sev_v == "LOW" or sev_v == "INFO":
                        sev = AlertSeverity.LOW
                    try:
                        atype = AlertType(a.activity_type.value)
                    except ValueError:
                        continue
                    alerts.append(BehaviorAlert(
                        alert_type=atype,
                        severity=sev,
                        track_id=a.track_id if a.track_id is not None else -1,
                        zone_id=a.zone_id,
                        message=a.message,
                        timestamp=timestamp,
                        metadata={"confidence": a.confidence, "evidence": a.evidence},
                    ))
                self.alert_history.extend(alerts)
                if len(self.alert_history) > 1000:
                    self.alert_history = self.alert_history[-1000:]
                return alerts
            except Exception:
                pass  # fall back to legacy detection

        # Legacy detection
        alerts.extend(self._check_loitering(tracks, zones, zone_occupancy, timestamp))
        alerts.extend(self._check_intrusion(tracks, timestamp))
        alerts.extend(self._check_congestion(zones, zone_occupancy, timestamp))
        self.alert_history.extend(alerts)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        return alerts

    def _check_loitering(self, tracks: List, zones: Dict, zone_occ: Dict, ts: float) -> List:
        alerts = []
        for zid, zone in zones.items():
            threshold = zone.get("dwell_threshold", self.loiter_threshold)
            for track in tracks:
                if track.track_id in zone_occ.get(zid, {}):
                    dwell = ts - zone_occ[zid][track.track_id]
                    if dwell > threshold and self._can_alert(f"loit_{zid}_{track.track_id}", ts):
                        alerts.append(BehaviorAlert(AlertType.LOITERING, AlertSeverity.MEDIUM,
                            track.track_id, zid, f"Loitering in {zone.get('name', zid)}: {dwell:.0f}s", ts))
        return alerts

    def _check_intrusion(self, tracks: List, ts: float) -> List:
        alerts = []
        if cv2 is None:
            return alerts
        for zid, poly in self.intrusion_zones:
            pts = np.array(poly, dtype=np.int32)
            for track in tracks:
                cx, cy = track.center
                if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
                    if self._can_alert(f"intr_{zid}_{track.track_id}", ts):
                        alerts.append(BehaviorAlert(AlertType.INTRUSION, AlertSeverity.HIGH,
                            track.track_id, zid, f"Entered restricted zone {zid}", ts))
        return alerts

    def _check_congestion(self, zones: Dict, zone_occ: Dict, ts: float) -> List:
        alerts = []
        for zid, zone in zones.items():
            cap = zone.get("capacity")
            if not cap:
                continue
            occ = len(zone_occ.get(zid, {}))
            if occ / cap > self.congestion_threshold and self._can_alert(f"cong_{zid}", ts):
                alerts.append(BehaviorAlert(AlertType.CONGESTION, AlertSeverity.MEDIUM, -1, zid,
                    f"Congestion: {occ}/{cap}", ts))
        return alerts

    def _can_alert(self, key: str, ts: float) -> bool:
        if key not in self.alert_cooldowns or ts - self.alert_cooldowns[key] >= self.cooldown_period:
            self.alert_cooldowns[key] = ts
            return True
        return False

    def get_recent_alerts(self, seconds: float = 300) -> List:
        return [a for a in self.alert_history if a.timestamp >= time.time() - seconds]
