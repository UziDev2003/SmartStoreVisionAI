"""
Pydantic configuration schema for Smart Store Vision AI
======================================================
Provides typed, validated configuration models for the suspicious
activity detection system. Backed by pydantic v2.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Zone model
# ---------------------------------------------------------------------------
class ZoneConfig(BaseModel):
    """Polygon zone configuration."""
    id: str
    name: str
    type: str = Field(default="general", description="entrance|checkout|restricted|shelf|general")
    capacity: Optional[int] = None
    dwell_threshold: float = 120.0
    polygon: List[Tuple[int, int]]

    @field_validator("polygon")
    @classmethod
    def _at_least_3_points(cls, v: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(v) < 3:
            raise ValueError("polygon must contain at least 3 points")
        return [(int(x), int(y)) for x, y in v]

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        allowed = {"entrance", "checkout", "restricted", "shelf", "general"}
        if v not in allowed:
            raise ValueError(f"zone type must be one of {allowed}")
        return v


# ---------------------------------------------------------------------------
# Per-detector configuration models
# ---------------------------------------------------------------------------
class LoiteringConfig(BaseModel):
    enabled: bool = True
    default_dwell_seconds: float = 120.0
    min_motion_variance: float = 0.05
    min_confidence: float = 0.6
    confirmation_frames: int = 10
    cooldown_seconds: float = 60.0


class RestrictedZoneConfig(BaseModel):
    enabled: bool = True
    default_severity: str = "HIGH"
    min_confidence: float = 0.85
    confirmation_frames: int = 3
    cooldown_seconds: float = 30.0
    enable_approach_escalation: bool = True


class ShelfTamperingConfig(BaseModel):
    enabled: bool = True
    min_motion_intensity: float = 0.3
    min_interaction_seconds: float = 5.0
    min_confidence: float = 0.7
    confirmation_frames: int = 8
    cooldown_seconds: float = 45.0
    use_pose_corroboration: bool = True
    use_bag_class: bool = True


class AbnormalShelfConfig(BaseModel):
    enabled: bool = True
    min_dwell_seconds: float = 15.0
    max_revisits: int = 3
    max_dwell_seconds: float = 240.0
    min_confidence: float = 0.55
    cooldown_seconds: float = 60.0


class UnattendedCheckoutConfig(BaseModel):
    enabled: bool = True
    min_unattended_seconds: float = 30.0
    use_bag_class: bool = True
    stationary_pixel_threshold: int = 800
    min_confidence: float = 0.65
    cooldown_seconds: float = 90.0


class MovementConfig(BaseModel):
    enabled: bool = True
    running_speed_px_s: float = 220.0
    erratic_heading_changes: int = 6
    erratic_window_seconds: float = 5.0
    min_track_length: int = 20
    min_confidence: float = 0.7
    cooldown_seconds: float = 30.0


class PoseConfig(BaseModel):
    enabled: bool = True
    hand_near_shelf_max_px: int = 80
    posture_analysis: bool = True
    min_pose_confidence: float = 0.5


class AnomalyScorerConfig(BaseModel):
    enabled: bool = True
    model_path: str = "models/anomaly_iforest.joblib"
    contamination: float = 0.05
    n_estimators: int = 200
    confidence_weight: float = 0.5
    min_samples_to_train: int = 100
    auto_train_on_start: bool = False


class MultiCameraConfig(BaseModel):
    enabled: bool = False
    method: str = Field(default="hsv", description="hsv|osnet")
    similarity_threshold: float = 0.5
    global_track_ttl_seconds: float = 600.0


# ---------------------------------------------------------------------------
# Top-level models
# ---------------------------------------------------------------------------
class DetectorConfigs(BaseModel):
    loitering: LoiteringConfig = LoiteringConfig()
    restricted_zone: RestrictedZoneConfig = RestrictedZoneConfig()
    shelf_tampering: ShelfTamperingConfig = ShelfTamperingConfig()
    abnormal_shelf: AbnormalShelfConfig = AbnormalShelfConfig()
    unattended_checkout: UnattendedCheckoutConfig = UnattendedCheckoutConfig()
    movement: MovementConfig = MovementConfig()
    pose: PoseConfig = PoseConfig()


class GlobalAlertConfig(BaseModel):
    min_severity_to_notify: str = "MEDIUM"
    global_cooldown_seconds: float = 30.0
    min_confidence: float = 0.5
    max_alerts_in_memory: int = 1000
    dedup_window_seconds: float = 15.0


class SuspiciousActivityConfig(BaseModel):
    """Top-level configuration for the suspicious activity detector."""
    detectors: DetectorConfigs = DetectorConfigs()
    alerts: GlobalAlertConfig = GlobalAlertConfig()
    anomaly_scorer: AnomalyScorerConfig = AnomalyScorerConfig()
    multi_camera: MultiCameraConfig = MultiCameraConfig()
    custom_rules: List[Dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SuspiciousActivityConfig":
        import yaml
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        import yaml
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
