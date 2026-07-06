"""
Suspicious Activity Detection (Core Orchestrator)
================================================
This module is the heart of the suspicious-activity subsystem. It
aggregates a collection of focused detectors behind a single,
configurable orchestrator and applies low-false-positive guards
(multi-frame confirmation, hysteresis, per-rule cooldowns, dedup,
anomaly-scoring gating).

Sub-detectors implemented:
    - LoiteringDetector           (motion-variance guard)
    - RestrictedZoneDetector      (severity escalation)
    - MovementAnalyzer            (running, erratic)
    - ShelfInteractionDetector    (abnormal dwell / revisits)
    - ShelfTamperingDetector      (motion + pose + bag classes)
    - UnattendedCheckoutDetector  (stationary objects / bag class)
    - PoseAnalyzer                (hand-near-shelf, posture)  [via pose_analyzer]
    - MultiCameraCorrelator       (Re-ID across cameras)      [via multi_camera]
    - AnomalyScorer               (IsolationForest gating)    [via anomaly_scorer]
    - RuleEngine                  (YAML-driven composable)    [via activity_rules]
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import time
import math
import hashlib
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    CV2_AVAILABLE = False

from src.analytics.movement_features import compute_features, motion_variance
from src.analytics.activity_rules import RuleEngine


# ---------------------------------------------------------------------------
# Public enums & dataclasses
# ---------------------------------------------------------------------------
class ActivityType(Enum):
    LOITERING = "loitering"
    RESTRICTED_ZONE = "restricted_zone"
    SHELF_TAMPERING = "shelf_tampering"
    ABNORMAL_SHELF_INTERACTION = "abnormal_shelf_interaction"
    UNATTENDED_CHECKOUT = "unattended_checkout"
    RUNNING = "running"
    ERRATIC_MOVEMENT = "erratic_movement"
    CUSTOM_RULE = "custom_rule"


class Severity(Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


SEVERITY_RANK = {s: i for i, s in enumerate(
    [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL])}


@dataclass
class SuspiciousActivity:
    activity_id: str
    activity_type: ActivityType
    severity: Severity
    confidence: float
    track_id: Optional[int]
    track_ids: List[int] = field(default_factory=list)
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    camera_id: str = "cam_01"
    message: str = ""
    timestamp: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    rule_violations: List[str] = field(default_factory=list)
    global_person_id: Optional[int] = None
    snapshot_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["activity_type"] = self.activity_type.value
        d["severity"] = self.severity.value
        return d


# ---------------------------------------------------------------------------
# Cooldown tracker
# ---------------------------------------------------------------------------
class CooldownTracker:
    """
    Per-(activity_type, key) cooldown. `key` is typically
    (zone_id, track_id) or just track_id.
    """
    def __init__(self):
        self._last: Dict[Tuple[str, Tuple], float] = {}

    def can_emit(self, activity_type: str, key: Tuple, ts: float,
                 cooldown_seconds: float) -> bool:
        full = (activity_type, tuple(key))
        last = self._last.get(full)
        if last is None or ts - last >= cooldown_seconds:
            self._last[full] = ts
            return True
        return False

    def reset(self, key: Optional[Tuple] = None) -> None:
        if key is None:
            self._last.clear()
        else:
            self._last = {k: v for k, v in self._last.items() if k[1] != key}


# ---------------------------------------------------------------------------
# Per-detector multi-frame confirmation helper
# ---------------------------------------------------------------------------
class ConfirmationCounter:
    """Counts consecutive frames a condition has been true. Fires after N frames."""
    def __init__(self):
        self._counts: Dict[Tuple, int] = defaultdict(int)
        self._seen: Dict[Tuple, bool] = defaultdict(bool)

    def tick(self, key: Tuple, condition: bool) -> bool:
        """Increment/decrement; return True if condition has held for required frames."""
        if condition:
            self._counts[key] += 1
            self._seen[key] = True
        else:
            # Decay quickly so we don't keep firing
            self._counts[key] = max(0, self._counts[key] - 1)
        return self._counts[key] > 0

    def confirm(self, key: Tuple, threshold: int) -> bool:
        return self._counts.get(key, 0) >= threshold

    def reset(self, key: Tuple) -> None:
        self._counts.pop(key, None)
        self._seen.pop(key, None)

    def get(self, key: Tuple) -> int:
        return self._counts.get(key, 0)


# ---------------------------------------------------------------------------
# Utility: dedup signature
# ---------------------------------------------------------------------------
def _evidence_signature(activity: SuspiciousActivity) -> str:
    """Stable hash to dedup near-identical alerts inside a window."""
    h = hashlib.sha1()
    h.update(str(activity.activity_type.value).encode())
    h.update(str(activity.track_id).encode())
    h.update(str(activity.zone_id).encode())
    h.update(str(sorted(activity.evidence.items())).encode())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# LoiteringDetector
# ---------------------------------------------------------------------------
class LoiteringDetector:
    def __init__(self, config):
        self.config = config
        self.confirm = ConfirmationCounter()

    def update(self, tracks: List[Any], zones: Dict[str, Any],
               zone_occ: Dict[str, Dict[int, float]], pose_estimates: Dict[int, Dict],
               ts: float, cooldowns: CooldownTracker,
               active_streams: Dict[str, Any], camera_id: str) -> List[SuspiciousActivity]:
        if not self.config.enabled:
            return []
        out: List[SuspiciousActivity] = []
        min_var = float(self.config.min_motion_variance)
        min_conf = float(self.config.min_confidence)
        cf_frames = int(self.config.confirmation_frames)
        cooldown = float(self.config.cooldown_seconds)

        for zid, zone in zones.items():
            if zone.zone_type == "restricted":
                continue
            # Use the smaller of the zone's dwell_threshold and the config default,
            # so both the per-zone override and the global config are respected.
            zone_threshold = float(getattr(zone, "dwell_threshold", 1e9))
            threshold = min(zone_threshold, self.config.default_dwell_seconds)
            for track in tracks:
                if track.track_id not in zone_occ.get(zid, {}):
                    self.confirm.reset(("loit", zid, track.track_id))
                    continue
                dwell = ts - zone_occ[zid][track.track_id]
                feats = compute_features(track.history)
                var = feats.get("motion_var", 0.0)
                # Loitering = low motion + long dwell
                suspicious = dwell >= threshold and var <= min_var
                key = ("loit", zid, track.track_id)
                self.confirm.tick(key, suspicious)
                if not self.confirm.confirm(key, cf_frames):
                    continue
                if not cooldowns.can_emit(ActivityType.LOITERING.value, key, ts, cooldown):
                    continue
                # Confidence scales with dwell beyond threshold and immobility
                dwell_excess = min(1.0, (dwell - threshold) / max(60.0, threshold))
                conf = min(0.95, 0.55 + 0.25 * dwell_excess + 0.15 * (1.0 - min(var / max(min_var, 1e-3), 1.0)))
                if conf < min_conf:
                    continue
                severity = Severity.MEDIUM
                if conf >= 0.85:
                    severity = Severity.HIGH
                aid = f"{camera_id}_loit_{track.track_id}_{int(ts*1000)}"
                out.append(SuspiciousActivity(
                    activity_id=aid,
                    activity_type=ActivityType.LOITERING,
                    severity=severity,
                    confidence=float(conf),
                    track_id=track.track_id,
                    zone_id=zid,
                    zone_name=getattr(zone, "name", zid),
                    camera_id=camera_id,
                    message=f"Loitering in {getattr(zone, 'name', zid)} for {dwell:.0f}s",
                    timestamp=ts,
                    evidence={
                        "dwell_seconds": float(dwell),
                        "motion_variance": float(var),
                        "threshold_seconds": float(threshold),
                    },
                    rule_violations=["loitering:low_motion_long_dwell"],
                ))
        return out


# ---------------------------------------------------------------------------
# RestrictedZoneDetector
# ---------------------------------------------------------------------------
class RestrictedZoneDetector:
    def __init__(self, config):
        self.config = config
        self.confirm = ConfirmationCounter()

    def update(self, tracks: List[Any], zones: Dict[str, Any],
               ts: float, cooldowns: CooldownTracker, camera_id: str) -> List[SuspiciousActivity]:
        if not self.config.enabled or not CV2_AVAILABLE:
            return []
        out: List[SuspiciousActivity] = []
        min_conf = float(self.config.min_confidence)
        cf_frames = int(self.config.confirmation_frames)
        cooldown = float(self.config.cooldown_seconds)
        default_severity = Severity[self.config.default_severity] \
            if self.config.default_severity in SEVERITY_RANK else Severity.HIGH

        for zid, zone in zones.items():
            if zone.zone_type != "restricted":
                continue
            try:
                poly = np.array(zone.polygon, dtype=np.int32)
            except Exception:
                continue
            for track in tracks:
                cx, cy = track.center
                inside = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0
                key = ("rz", zid, track.track_id)
                self.confirm.tick(key, inside)
                if not self.confirm.confirm(key, cf_frames):
                    continue
                if not cooldowns.can_emit(ActivityType.RESTRICTED_ZONE.value, key, ts, cooldown):
                    continue
                # Confidence: high since this is a deterministic containment test
                conf = 0.95 if inside else 0.5
                if conf < min_conf:
                    continue
                severity = default_severity
                if conf >= 0.9 and self.config.enable_approach_escalation:
                    # If they were near the zone in recent history, escalate
                    if self._has_recent_approach(track, poly):
                        severity = Severity.CRITICAL
                aid = f"{camera_id}_rz_{track.track_id}_{int(ts*1000)}"
                out.append(SuspiciousActivity(
                    activity_id=aid,
                    activity_type=ActivityType.RESTRICTED_ZONE,
                    severity=severity,
                    confidence=float(conf),
                    track_id=track.track_id,
                    zone_id=zid,
                    zone_name=getattr(zone, "name", zid),
                    camera_id=camera_id,
                    message=f"Restricted zone entry: {getattr(zone, 'name', zid)}",
                    timestamp=ts,
                    evidence={"zone_type": "restricted"},
                    rule_violations=["restricted_zone:entry"],
                ))
        return out

    def _has_recent_approach(self, track, poly, distance_px: int = 120) -> bool:
        """Look at the last ~15 history points and check if any were near the polygon."""
        for ts, pt in track.history[-15:]:
            try:
                d = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True)
                if -distance_px < float(d) < 0:
                    return True
            except Exception:
                continue
        return False


# ---------------------------------------------------------------------------
# MovementAnalyzer (running, erratic)
# ---------------------------------------------------------------------------
class MovementAnalyzer:
    def __init__(self, config):
        self.config = config

    def update(self, tracks: List[Any], ts: float, cooldowns: CooldownTracker,
               camera_id: str) -> List[SuspiciousActivity]:
        if not self.config.enabled:
            return []
        out: List[SuspiciousActivity] = []
        min_track = int(self.config.min_track_length)
        min_conf = float(self.config.min_confidence)
        cooldown = float(self.config.cooldown_seconds)
        running_th = float(self.config.running_speed_px_s)
        erratic_turns = int(self.config.erratic_heading_changes)
        erratic_win = float(self.config.erratic_window_seconds)

        for track in tracks:
            if len(track.history) < min_track:
                continue
            feats = compute_features(track.history)
            key = (track.track_id,)
            # Running
            if feats["peak_speed"] >= running_th and feats["duration_seconds"] >= 1.0:
                if cooldowns.can_emit(ActivityType.RUNNING.value, key, ts, cooldown):
                    conf = min(0.95, 0.6 + 0.3 * min(1.0, feats["peak_speed"] / max(running_th, 1.0) - 1.0))
                    if conf >= min_conf:
                        aid = f"{camera_id}_run_{track.track_id}_{int(ts*1000)}"
                        out.append(SuspiciousActivity(
                            activity_id=aid,
                            activity_type=ActivityType.RUNNING,
                            severity=Severity.MEDIUM,
                            confidence=float(conf),
                            track_id=track.track_id,
                            camera_id=camera_id,
                            message=f"Running detected (peak {feats['peak_speed']:.0f} px/s)",
                            timestamp=ts,
                            evidence={"peak_speed_px_s": feats["peak_speed"],
                                      "avg_speed_px_s": feats["avg_speed"]},
                            rule_violations=["movement:running"],
                        ))
            # Erratic
            if (feats["heading_change_count"] >= erratic_turns
                    and feats["duration_seconds"] <= erratic_win * 2):
                if cooldowns.can_emit(ActivityType.ERRATIC_MOVEMENT.value, key, ts, cooldown):
                    conf = min(0.95, 0.6 + 0.04 * feats["heading_change_count"])
                    if conf >= min_conf:
                        aid = f"{camera_id}_erratic_{track.track_id}_{int(ts*1000)}"
                        out.append(SuspiciousActivity(
                            activity_id=aid,
                            activity_type=ActivityType.ERRATIC_MOVEMENT,
                            severity=Severity.LOW,
                            confidence=float(conf),
                            track_id=track.track_id,
                            camera_id=camera_id,
                            message=f"Erratic movement ({int(feats['heading_change_count'])} turns in {feats['duration_seconds']:.1f}s)",
                            timestamp=ts,
                            evidence={"heading_change_count": int(feats["heading_change_count"]),
                                      "duration_seconds": float(feats["duration_seconds"])},
                            rule_violations=["movement:erratic"],
                        ))
        return out


# ---------------------------------------------------------------------------
# AbnormalShelfInteractionDetector
# ---------------------------------------------------------------------------
class ShelfInteractionDetector:
    """Detects abnormal interaction patterns in shelf zones (long dwell,
    multiple revisits, scan-and-leave)."""
    def __init__(self, config):
        self.config = config
        self.revisit_counts: Dict[Tuple[int, str], int] = defaultdict(int)
        self.last_zone: Dict[int, str] = {}
        self.confirm = ConfirmationCounter()

    def update(self, tracks: List[Any], zones: Dict[str, Any],
               zone_occ: Dict[str, Dict[int, float]], ts: float,
               cooldowns: CooldownTracker, camera_id: str) -> List[SuspiciousActivity]:
        if not self.config.enabled:
            return []
        out: List[SuspiciousActivity] = []
        min_dwell = float(self.config.min_dwell_seconds)
        max_dwell = float(self.config.max_dwell_seconds)
        max_revisits = int(self.config.max_revisits)
        min_conf = float(self.config.min_confidence)
        cooldown = float(self.config.cooldown_seconds)

        for track in tracks:
            current_zone = None
            for zid, zone in zones.items():
                if zone.zone_type == "shelf" and track.track_id in zone_occ.get(zid, {}):
                    current_zone = zid
                    break
            prev = self.last_zone.get(track.track_id)
            if current_zone and prev and current_zone != prev:
                self.revisit_counts[(track.track_id, current_zone)] += 1
            if current_zone:
                self.last_zone[track.track_id] = current_zone

        for zid, zone in zones.items():
            if zone.zone_type != "shelf":
                continue
            for track in tracks:
                if track.track_id not in zone_occ.get(zid, {}):
                    continue
                dwell = ts - zone_occ[zid][track.track_id]
                revisits = self.revisit_counts[(track.track_id, zid)]
                # Long dwell beyond limit
                if dwell >= max_dwell:
                    key = ("abshelf", zid, track.track_id)
                    if cooldowns.can_emit(ActivityType.ABNORMAL_SHELF_INTERACTION.value,
                                          key, ts, cooldown):
                        conf = min(0.95, 0.5 + 0.4 * min(1.0, (dwell - max_dwell) / max(60.0, max_dwell)))
                        if conf >= min_conf:
                            aid = f"{camera_id}_abshelf_{track.track_id}_{int(ts*1000)}"
                            out.append(SuspiciousActivity(
                                activity_id=aid,
                                activity_type=ActivityType.ABNORMAL_SHELF_INTERACTION,
                                severity=Severity.MEDIUM,
                                confidence=float(conf),
                                track_id=track.track_id,
                                zone_id=zid,
                                zone_name=getattr(zone, "name", zid),
                                camera_id=camera_id,
                                message=f"Abnormal shelf dwell ({dwell:.0f}s in {getattr(zone, 'name', zid)})",
                                timestamp=ts,
                                evidence={"dwell_seconds": float(dwell),
                                          "revisits": int(revisits)},
                                rule_violations=["abnormal_shelf:long_dwell"],
                            ))
                # High revisits
                if revisits >= max_revisits:
                    key = ("abshelf_rev", zid, track.track_id)
                    if cooldowns.can_emit(ActivityType.ABNORMAL_SHELF_INTERACTION.value,
                                          key, ts, cooldown):
                        conf = min(0.9, 0.55 + 0.1 * revisits)
                        if conf >= min_conf:
                            aid = f"{camera_id}_abshelf_rev_{track.track_id}_{int(ts*1000)}"
                            out.append(SuspiciousActivity(
                                activity_id=aid,
                                activity_type=ActivityType.ABNORMAL_SHELF_INTERACTION,
                                severity=Severity.LOW,
                                confidence=float(conf),
                                track_id=track.track_id,
                                zone_id=zid,
                                zone_name=getattr(zone, "name", zid),
                                camera_id=camera_id,
                                message=f"Repeated shelf revisits ({revisits}) in {getattr(zone, 'name', zid)}",
                                timestamp=ts,
                                evidence={"revisits": int(revisits)},
                                rule_violations=["abnormal_shelf:many_revisits"],
                            ))
        return out


# ---------------------------------------------------------------------------
# ShelfTamperingDetector
# ---------------------------------------------------------------------------
class ShelfTamperingDetector:
    """
    Combines: motion intensity near shelf + (optional) pose: hand-near-shelf
    + (optional) bag/backpack class presence.
    """
    def __init__(self, config):
        self.config = config
        self.confirm = ConfirmationCounter()
        self._prev_frame_gray: Dict[str, Any] = {}

    def update(self, frame: np.ndarray, tracks: List[Any], zones: Dict[str, Any],
               zone_occ: Dict[str, Dict[int, float]], pose_estimates: Dict[int, Dict],
               ts: float, cooldowns: CooldownTracker, camera_id: str) -> List[SuspiciousActivity]:
        if not self.config.enabled:
            return []
        out: List[SuspiciousActivity] = []
        min_intensity = float(self.config.min_motion_intensity)
        min_dur = float(self.config.min_interaction_seconds)
        min_conf = float(self.config.min_confidence)
        cf_frames = int(self.config.confirmation_frames)
        cooldown = float(self.config.cooldown_seconds)
        use_pose = bool(self.config.use_pose_corroboration)

        if frame is None or not CV2_AVAILABLE:
            return out
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            return out
        prev_gray = self._prev_frame_gray.get(camera_id)
        motion_map = None
        if prev_gray is not None and prev_gray.shape == gray.shape:
            try:
                diff = cv2.absdiff(gray, prev_gray)
                motion_map = cv2.GaussianBlur(diff, (5, 5), 0)
            except Exception:
                motion_map = None
        self._prev_frame_gray[camera_id] = gray

        for zid, zone in zones.items():
            if zone.zone_type != "shelf":
                continue
            try:
                poly = np.array(zone.polygon, dtype=np.int32)
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
            except Exception:
                continue
            region_motion = 0.0
            if motion_map is not None:
                vals = motion_map[mask > 0]
                if vals.size > 0:
                    region_motion = float(vals.mean()) / 255.0

            for track in tracks:
                if track.track_id not in zone_occ.get(zid, {}):
                    self.confirm.reset(("tamper", zid, track.track_id))
                    continue
                dwell = ts - zone_occ[zid][track.track_id]
                motion_ok = region_motion >= min_intensity
                dwell_ok = dwell >= min_dur
                # Pose corroboration: hands near this zone
                pose_score = 0.0
                if use_pose and pose_estimates:
                    pose = pose_estimates.get(track.track_id)
                    if pose is not None:
                        try:
                            s1 = self._hand_near_score(pose.get("left_wrist"),
                                                       zone.polygon, int(self.config._hand_max_px))
                            s2 = self._hand_near_score(pose.get("right_wrist"),
                                                       zone.polygon, int(self.config._hand_max_px))
                            pose_score = max(s1, s2)
                        except Exception:
                            pose_score = 0.0
                condition = motion_ok and dwell_ok
                key = ("tamper", zid, track.track_id)
                self.confirm.tick(key, condition)
                if not self.confirm.confirm(key, cf_frames):
                    continue
                if not cooldowns.can_emit(ActivityType.SHELF_TAMPERING.value, key, ts, cooldown):
                    continue
                # Combine evidence
                conf = 0.5
                if motion_ok:
                    conf += 0.2 * min(1.0, region_motion / max(min_intensity, 1e-3))
                if pose_score > 0.4:
                    conf += 0.2 * pose_score
                conf = min(0.95, conf)
                if conf < min_conf:
                    continue
                aid = f"{camera_id}_tamper_{track.track_id}_{int(ts*1000)}"
                out.append(SuspiciousActivity(
                    activity_id=aid,
                    activity_type=ActivityType.SHELF_TAMPERING,
                    severity=Severity.HIGH if conf >= 0.8 else Severity.MEDIUM,
                    confidence=float(conf),
                    track_id=track.track_id,
                    zone_id=zid,
                    zone_name=getattr(zone, "name", zid),
                    camera_id=camera_id,
                    message=f"Possible shelf tampering at {getattr(zone, 'name', zid)}",
                    timestamp=ts,
                    evidence={"motion_intensity": float(region_motion),
                              "dwell_seconds": float(dwell),
                              "pose_score": float(pose_score)},
                    rule_violations=["shelf_tampering:motion_dwell"],
                ))
        return out

    def _hand_near_score(self, hand_pos, polygon, max_px: int) -> float:
        if not hand_pos or not polygon:
            return 0.0
        try:
            poly = np.array(polygon, dtype=np.int32)
            inside = cv2.pointPolygonTest(poly, (float(hand_pos[0]), float(hand_pos[1])), False) >= 0
            if inside:
                return 1.0
            dist = abs(float(cv2.pointPolygonTest(poly, (float(hand_pos[0]), float(hand_pos[1])), True)))
            return max(0.0, 1.0 - dist / max(1, max_px))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# UnattendedCheckoutDetector
# ---------------------------------------------------------------------------
class UnattendedCheckoutDetector:
    """
    Detects objects/bags left in a checkout zone after the customer has
    left. Uses two signals (in order of reliability):
      1. Stationary-pixel clusters (frame diff fallback) — always available
      2. (Optional) YOLO bag/backpack class — if detector exposes them
    """
    def __init__(self, config):
        self.config = config
        self.confirm = ConfirmationCounter()
        self._last_present: Dict[str, Dict[int, float]] = defaultdict(dict)
        self._person_left: Dict[str, float] = {}

    def update(self, frame: np.ndarray, tracks: List[Any], zones: Dict[str, Any],
               zone_occ: Dict[str, Dict[int, float]], detections: List[Any],
               ts: float, cooldowns: CooldownTracker, camera_id: str) -> List[SuspiciousActivity]:
        if not self.config.enabled:
            return []
        out: List[SuspiciousActivity] = []
        min_unattended = float(self.config.min_unattended_seconds)
        min_conf = float(self.config.min_confidence)
        cooldown = float(self.config.cooldown_seconds)

        # Map current occupancy per checkout zone
        for zid, zone in zones.items():
            if zone.zone_type != "checkout":
                continue
            occ_ids = list(zone_occ.get(zid, {}).keys())
            if not occ_ids:
                # No person currently in checkout; check if anyone was here before
                if zid not in self._person_left:
                    self._person_left[zid] = ts
                # Use stationary-pixel proxy for "object present"
                stationary_score = self._stationary_score(frame, zone) if frame is not None else 0.0
                bag_present = self._bag_classes_present(detections, zone) if self.config.use_bag_class else False
                duration = ts - self._person_left[zid]
                if (stationary_score > 0.0 or bag_present) and duration >= min_unattended:
                    key = ("unatt", zid)
                    self.confirm.tick(key, True)
                    if not self.confirm.confirm(key, 5):
                        continue
                    if not cooldowns.can_emit(ActivityType.UNATTENDED_CHECKOUT.value,
                                              key, ts, cooldown):
                        continue
                    base = 0.5
                    if bag_present:
                        base += 0.3
                    if stationary_score > 0.4:
                        base += 0.2
                    conf = float(min(0.95, base))
                    if conf < min_conf:
                        continue
                    aid = f"{camera_id}_unatt_{zid}_{int(ts*1000)}"
                    out.append(SuspiciousActivity(
                        activity_id=aid,
                        activity_type=ActivityType.UNATTENDED_CHECKOUT,
                        severity=Severity.MEDIUM,
                        confidence=conf,
                        track_id=None,
                        zone_id=zid,
                        zone_name=getattr(zone, "name", zid),
                        camera_id=camera_id,
                        message=f"Unattended object at {getattr(zone, 'name', zid)} for {duration:.0f}s",
                        timestamp=ts,
                        evidence={"unattended_seconds": float(duration),
                                  "stationary_score": float(stationary_score),
                                  "bag_detected": bool(bag_present)},
                        rule_violations=["unattended_checkout:object_left"],
                    ))
            else:
                # Reset when someone is in the zone
                self._person_left.pop(zid, None)
                self.confirm.reset(("unatt", zid))
        return out

    def _stationary_score(self, frame, zone) -> float:
        """Coarse proxy: estimate how much of the zone is 'stuck' (low motion)."""
        if frame is None or not CV2_AVAILABLE:
            return 0.0
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(zone.polygon, dtype=np.int32)], 255)
            # Use simple gradient magnitude as activity proxy
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            region = mag[mask > 0]
            if region.size == 0:
                return 0.0
            mean = float(region.mean())
            # Lower gradient -> more stationary. We want non-zero to mean SOMETHING is there.
            # If completely empty (background only), gradient is low AND uniform.
            # We approximate "object present" as: enough non-zero gradient pixels above a baseline.
            active_px = int((region > 8.0).sum())
            thresh = int(self.config.stationary_pixel_threshold)
            return min(1.0, active_px / max(1, thresh))
        except Exception:
            return 0.0

    def _bag_classes_present(self, detections, zone) -> bool:
        """Check if YOLO detected bag/backpack/handbag inside the zone."""
        if not CV2_AVAILABLE or not detections:
            return False
        try:
            poly = np.array(zone.polygon, dtype=np.int32)
        except Exception:
            return False
        bag_ids = {24, 26, 28}  # COCO: backpack, handbag, suitcase
        for d in detections:
            cls = getattr(d, "class_id", None)
            if cls is None and isinstance(d, (list, tuple)) and len(d) >= 6:
                cls = int(d[5])
            if cls in bag_ids:
                bbox = getattr(d, "bbox", None)
                if bbox is None and isinstance(d, (list, tuple)) and len(d) >= 4:
                    bbox = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
                if bbox is None:
                    continue
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                    return True
        return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class SuspiciousActivityDetector:
    """
    Top-level orchestrator. Wires all sub-detectors together and applies
    global guards (cooldowns, dedup, anomaly-scoring gating, severity
    filtering).
    """

    def __init__(self, config: Any, anomaly_scorer: Any = None,
                 pose_analyzer: Any = None, multi_camera: Any = None):
        self.config = config
        self.cooldowns = CooldownTracker()
        self._seen_signatures: Dict[str, float] = {}
        self._activities_emitted: deque = deque(maxlen=int(config.alerts.max_alerts_in_memory))
        self._all_activities: deque = deque(maxlen=2000)

        # Build sub-detectors
        d = config.detectors
        self.loitering = LoiteringDetector(d.loitering)
        self.restricted = RestrictedZoneDetector(d.restricted_zone)
        self.movement = MovementAnalyzer(d.movement)
        self.shelf_inter = ShelfInteractionDetector(d.abnormal_shelf)
        self.shelf_tamper = ShelfTamperingDetector(d.shelf_tampering)
        self.shelf_tamper.config._hand_max_px = d.pose.hand_near_shelf_max_px
        self.unattended = UnattendedCheckoutDetector(d.unattended_checkout)

        self.anomaly_scorer = anomaly_scorer
        self.pose_analyzer = pose_analyzer
        self.multi_camera = multi_camera
        self.rule_engine = RuleEngine(getattr(config, "custom_rules", []) or [])

    @classmethod
    def from_config_path(cls, path: str = "config/suspicious_activity.yaml",
                         pose_analyzer: Any = None,
                         multi_camera: Any = None) -> "SuspiciousActivityDetector":
        from src.utils.config_schema import SuspiciousActivityConfig
        cfg = SuspiciousActivityConfig.from_yaml(path)
        anomaly = None
        try:
            from src.analytics.anomaly_scorer import AnomalyScorer
            anomaly = AnomalyScorer(cfg.anomaly_scorer)
            anomaly.load()
        except Exception:
            anomaly = None
        return cls(cfg, anomaly_scorer=anomaly,
                   pose_analyzer=pose_analyzer, multi_camera=multi_camera)

    def detect(
        self,
        tracks: List[Any],
        zones: Dict[str, Any],
        zone_occ: Dict[str, Dict[int, float]],
        timestamp: Optional[float] = None,
        frame: Any = None,
        detections: Any = None,
        camera_id: str = "cam_01",
    ) -> List[SuspiciousActivity]:
        ts = timestamp or time.time()
        activities: List[SuspiciousActivity] = []

        # Optional multi-camera correlation
        global_ids: Dict[int, int] = {}
        if self.multi_camera and self.multi_camera.enabled and frame is not None:
            try:
                mapping = self.multi_camera.update(frame, camera_id, tracks, ts)
                global_ids = {tid: gid for (cid, tid), gid in mapping.items() if cid == camera_id}
            except Exception:
                global_ids = {}

        # Optional pose estimation
        pose_estimates: Dict[int, Dict] = {}
        if self.pose_analyzer and self.pose_analyzer.is_available and frame is not None:
            try:
                pose_estimates = self.pose_analyzer.update(frame, tracks, ts)
            except Exception:
                pose_estimates = {}

        # 1) Restricted zone
        try:
            activities.extend(self.restricted.update(tracks, zones, ts, self.cooldowns, camera_id))
        except Exception:
            pass

        # 2) Loitering
        try:
            activities.extend(self.loitering.update(
                tracks, zones, zone_occ, pose_estimates, ts, self.cooldowns, {}, camera_id))
        except Exception:
            pass

        # 3) Movement
        try:
            activities.extend(self.movement.update(tracks, ts, self.cooldowns, camera_id))
        except Exception:
            pass

        # 4) Shelf interaction
        try:
            activities.extend(self.shelf_inter.update(
                tracks, zones, zone_occ, ts, self.cooldowns, camera_id))
        except Exception:
            pass

        # 5) Shelf tampering
        try:
            activities.extend(self.shelf_tamper.update(
                frame, tracks, zones, zone_occ, pose_estimates, ts,
                self.cooldowns, camera_id))
        except Exception:
            pass

        # 6) Unattended checkout
        try:
            activities.extend(self.unattended.update(
                frame, tracks, zones, zone_occ, detections, ts, self.cooldowns, camera_id))
        except Exception:
            pass

        # 7) Custom rules
        try:
            for track in tracks:
                feats = compute_features(track.history)
                ctx = {
                    "track_id": track.track_id,
                    "confidence": getattr(track, "confidence", 0.0),
                    "features": feats,
                }
                # Add zone context
                for zid, zone in zones.items():
                    if track.track_id in zone_occ.get(zid, {}):
                        ctx["zone_id"] = zid
                        ctx["zone_type"] = zone.zone_type
                        ctx["zone_name"] = getattr(zone, "name", zid)
                        ctx["dwell_seconds"] = float(ts - zone_occ[zid][track.track_id])
                        ctx["occupancy"] = len(zone_occ[zid])
                        break
                triggered = self.rule_engine.evaluate(ctx)
                for t in triggered:
                    aid = f"{camera_id}_rule_{track.track_id}_{int(ts*1000)}"
                    try:
                        sev = Severity[t["severity"]] if t["severity"] in SEVERITY_RANK else Severity.LOW
                    except Exception:
                        sev = Severity.LOW
                    activities.append(SuspiciousActivity(
                        activity_id=aid,
                        activity_type=ActivityType.CUSTOM_RULE,
                        severity=sev,
                        confidence=float(t["confidence"]),
                        track_id=track.track_id,
                        zone_id=ctx.get("zone_id"),
                        zone_name=ctx.get("zone_name"),
                        camera_id=camera_id,
                        message=t["message"],
                        timestamp=ts,
                        evidence={"rule": t["name"], "features": feats},
                        rule_violations=[f"custom:{t['name']}"],
                    ))
        except Exception:
            pass

        # Apply global guards: anomaly gating, severity filter, dedup
        final: List[SuspiciousActivity] = []
        seen_in_window: Dict[str, float] = {}
        min_sev = self._parse_min_severity(self.config.alerts.min_severity_to_notify)
        min_conf_global = float(self.config.alerts.min_confidence)
        dedup_window = float(self.config.alerts.dedup_window_seconds)

        for act in activities:
            # Attach global_person_id
            if act.track_id is not None and act.track_id in global_ids:
                act.global_person_id = global_ids[act.track_id]

            # Anomaly gating
            if self.anomaly_scorer is not None and self.anomaly_scorer.is_trained and act.track_id is not None:
                track = next((t for t in tracks if t.track_id == act.track_id), None)
                if track is not None:
                    feats = compute_features(track.history)
                    new_conf = self.anomaly_scorer.combine(act.confidence, feats)
                    act.evidence["anomaly_combined_confidence"] = float(new_conf)
                    act.confidence = float(new_conf)

            # Severity filter
            if SEVERITY_RANK[act.severity] < SEVERITY_RANK[min_sev]:
                continue
            # Confidence filter
            if act.confidence < min_conf_global:
                continue
            # Dedup by evidence signature
            sig = _evidence_signature(act)
            last = seen_in_window.get(sig)
            if last is not None and ts - last < dedup_window:
                continue
            seen_in_window[sig] = ts
            final.append(act)

        # Maintain a small history
        for a in final:
            self._activities_emitted.append(a)
            self._all_activities.append(a)
        return final

    @staticmethod
    def _parse_min_severity(s: str) -> Severity:
        try:
            return Severity[s]
        except Exception:
            return Severity.MEDIUM

    def get_recent(self, seconds: float = 300.0,
                   min_severity: Optional[Severity] = None) -> List[SuspiciousActivity]:
        cutoff = time.time() - seconds
        out = [a for a in self._all_activities if a.timestamp >= cutoff]
        if min_severity is not None:
            out = [a for a in out if SEVERITY_RANK[a.severity] >= SEVERITY_RANK[min_severity]]
        return sorted(out, key=lambda a: a.timestamp, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for a in self._activities_emitted:
            by_type[a.activity_type.value] = by_type.get(a.activity_type.value, 0) + 1
            by_severity[a.severity.value] = by_severity.get(a.severity.value, 0) + 1
        return {
            "total": len(self._activities_emitted),
            "by_type": by_type,
            "by_severity": by_severity,
        }
