"""
Streamlit Threshold Tuner
==========================
Live sliders to tune detection thresholds. Writes an override file that
the runtime SuspiciousActivityDetector can hot-reload.
"""
from __future__ import annotations

import streamlit as st
from pathlib import Path
import yaml
from typing import Dict, Any

from src.utils.config_schema import SuspiciousActivityConfig


OVERRIDE_PATH = Path("config/suspicious_activity_override.yaml")


def _load_base_config() -> Dict[str, Any]:
    cfg = SuspiciousActivityConfig.from_yaml("config/suspicious_activity.yaml")
    return cfg.model_dump(mode="json")


def _save_override(values: Dict[str, Any]) -> Path:
    OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OVERRIDE_PATH, "w") as f:
        yaml.safe_dump(values, f, default_flow_style=False, sort_keys=False)
    return OVERRIDE_PATH


def render() -> Dict[str, Any]:
    st.markdown("### 🎛️ Threshold Tuner")
    st.caption("Tune detection thresholds live. Saves an override file that "
               "can be reloaded at runtime.")

    cfg = _load_base_config()
    detectors = cfg.get("detectors", {})

    out: Dict[str, Any] = {}

    # Loitering
    with st.expander("🕐 Loitering", expanded=True):
        c = detectors.get("loitering", {})
        c["enabled"] = st.checkbox("Enabled", value=c.get("enabled", True), key="t_loit_en")
        c["default_dwell_seconds"] = st.slider("Dwell Threshold (s)", 30, 600,
                                                int(c.get("default_dwell_seconds", 120)), 5,
                                                key="t_loit_dwell")
        c["min_motion_variance"] = st.slider("Min Motion Variance (≤)",
                                              0.0, 1.0, float(c.get("min_motion_variance", 0.05)),
                                              0.01, key="t_loit_var")
        c["min_confidence"] = st.slider("Min Confidence", 0.0, 1.0,
                                         float(c.get("min_confidence", 0.6)), 0.05, key="t_loit_conf")
        c["confirmation_frames"] = st.slider("Confirmation Frames", 1, 60,
                                              int(c.get("confirmation_frames", 10)), 1, key="t_loit_cf")
        c["cooldown_seconds"] = st.slider("Cooldown (s)", 5, 600,
                                           int(c.get("cooldown_seconds", 60)), 5, key="t_loit_cd")
        detectors["loitering"] = c

    # Restricted zone
    with st.expander("🚧 Restricted Zone"):
        c = detectors.get("restricted_zone", {})
        c["enabled"] = st.checkbox("Enabled", value=c.get("enabled", True), key="t_rz_en")
        c["default_severity"] = st.selectbox("Severity",
                                              ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                                              index=["LOW","MEDIUM","HIGH","CRITICAL"].index(
                                                  c.get("default_severity", "HIGH")),
                                              key="t_rz_sev")
        c["min_confidence"] = st.slider("Min Confidence", 0.0, 1.0,
                                         float(c.get("min_confidence", 0.85)), 0.05, key="t_rz_conf")
        c["confirmation_frames"] = st.slider("Confirmation Frames", 1, 30,
                                              int(c.get("confirmation_frames", 3)), 1, key="t_rz_cf")
        c["enable_approach_escalation"] = st.checkbox(
            "Escalate on approach history",
            value=c.get("enable_approach_escalation", True), key="t_rz_esc")
        detectors["restricted_zone"] = c

    # Movement
    with st.expander("🏃 Movement"):
        c = detectors.get("movement", {})
        c["enabled"] = st.checkbox("Enabled", value=c.get("enabled", True), key="t_mv_en")
        c["running_speed_px_s"] = st.slider("Running Speed (px/s)", 50, 1000,
                                              int(c.get("running_speed_px_s", 220)), 10, key="t_mv_run")
        c["erratic_heading_changes"] = st.slider("Erratic Turn Count", 2, 30,
                                                    int(c.get("erratic_heading_changes", 6)), 1, key="t_mv_turns")
        c["erratic_window_seconds"] = st.slider("Erratic Window (s)", 1.0, 30.0,
                                                  float(c.get("erratic_window_seconds", 5.0)), 0.5, key="t_mv_win")
        c["min_track_length"] = st.slider("Min Track Length", 5, 100,
                                            int(c.get("min_track_length", 20)), 1, key="t_mv_len")
        detectors["movement"] = c

    # Shelf interaction
    with st.expander("🛍️ Shelf Interaction"):
        c = detectors.get("abnormal_shelf", {})
        c["enabled"] = st.checkbox("Enabled", value=c.get("enabled", True), key="t_sh_en")
        c["min_dwell_seconds"] = st.slider("Min Dwell (s)", 5, 300,
                                             int(c.get("min_dwell_seconds", 15)), 5, key="t_sh_min")
        c["max_dwell_seconds"] = st.slider("Max Dwell (s)", 30, 1200,
                                             int(c.get("max_dwell_seconds", 240)), 10, key="t_sh_max")
        c["max_revisits"] = st.slider("Max Revisits", 1, 20,
                                        int(c.get("max_revisits", 3)), 1, key="t_sh_rev")
        detectors["abnormal_shelf"] = c

    # Shelf tampering
    with st.expander("🛒 Shelf Tampering"):
        c = detectors.get("shelf_tampering", {})
        c["enabled"] = st.checkbox("Enabled", value=c.get("enabled", True), key="t_tm_en")
        c["min_motion_intensity"] = st.slider("Min Motion Intensity", 0.0, 1.0,
                                                float(c.get("min_motion_intensity", 0.3)), 0.05, key="t_tm_mi")
        c["min_interaction_seconds"] = st.slider("Min Interaction (s)", 1.0, 60.0,
                                                    float(c.get("min_interaction_seconds", 5.0)), 0.5, key="t_tm_dur")
        c["use_pose_corroboration"] = st.checkbox(
            "Use Pose Corroboration (MediaPipe)",
            value=c.get("use_pose_corroboration", True), key="t_tm_pose")
        detectors["shelf_tampering"] = c

    # Unattended checkout
    with st.expander("🛒 Unattended Checkout"):
        c = detectors.get("unattended_checkout", {})
        c["enabled"] = st.checkbox("Enabled", value=c.get("enabled", True), key="t_uc_en")
        c["min_unattended_seconds"] = st.slider("Min Unattended (s)", 5, 600,
                                                  int(c.get("min_unattended_seconds", 30)), 5, key="t_uc_min")
        c["use_bag_class"] = st.checkbox("Use YOLO Bag Class",
                                          value=c.get("use_bag_class", True), key="t_uc_bag")
        detectors["unattended_checkout"] = c

    # Anomaly scorer
    with st.expander("🤖 Anomaly Scorer"):
        c = cfg.get("anomaly_scorer", {})
        c["enabled"] = st.checkbox("Enabled", value=c.get("enabled", True), key="t_as_en")
        c["contamination"] = st.slider("Contamination", 0.001, 0.2,
                                         float(c.get("contamination", 0.05)), 0.005, key="t_as_co")
        c["confidence_weight"] = st.slider("Confidence Weight", 0.0, 1.0,
                                            float(c.get("confidence_weight", 0.5)), 0.05, key="t_as_cw")
        cfg["anomaly_scorer"] = c

    # Alerts
    with st.expander("🔔 Alerts"):
        a = cfg.get("alerts", {})
        a["min_severity_to_notify"] = st.selectbox(
            "Min Severity to Notify",
            ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
            index=["INFO","LOW","MEDIUM","HIGH","CRITICAL"].index(
                a.get("min_severity_to_notify", "MEDIUM")),
            key="t_al_sev",
        )
        a["min_confidence"] = st.slider("Global Min Confidence", 0.0, 1.0,
                                          float(a.get("min_confidence", 0.5)), 0.05, key="t_al_conf")
        a["dedup_window_seconds"] = st.slider("Dedup Window (s)", 1.0, 120.0,
                                                float(a.get("dedup_window_seconds", 15.0)), 1.0, key="t_al_dd")
        cfg["alerts"] = a

    cfg["detectors"] = detectors
    out = cfg

    if st.button("💾 Save Override", type="primary"):
        path = _save_override(out)
        st.success(f"Saved override to {path}")
    return out
