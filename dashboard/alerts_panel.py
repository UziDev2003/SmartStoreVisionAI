"""
Streamlit Alerts Panel
=======================
Displays live suspicious-activity alerts with filters, severity badges,
evidence drill-down, and acknowledgement.
"""
from __future__ import annotations

import streamlit as st
from typing import List, Optional, Any
from datetime import datetime
import os

from src.analytics.suspicious_activity import (
    SuspiciousActivity, ActivityType, Severity, SEVERITY_RANK
)


SEVERITY_COLORS = {
    "INFO": "#9ca3af",
    "LOW": "#3b82f6",
    "MEDIUM": "#f59e0b",
    "HIGH": "#ef4444",
    "CRITICAL": "#7c2d12",
}

ACTIVITY_LABELS = {
    ActivityType.LOITERING.value: "🕐 Loitering",
    ActivityType.RESTRICTED_ZONE.value: "🚧 Restricted Zone",
    ActivityType.SHELF_TAMPERING.value: "🛒 Shelf Tampering",
    ActivityType.ABNORMAL_SHELF_INTERACTION.value: "🛍️ Abnormal Shelf",
    ActivityType.UNATTENDED_CHECKOUT.value: "🛒 Unattended Checkout",
    ActivityType.RUNNING.value: "🏃 Running",
    ActivityType.ERRATIC_MOVEMENT.value: "🌀 Erratic Movement",
    ActivityType.CUSTOM_RULE.value: "⚙️ Custom Rule",
}


def _activity_badge(act: SuspiciousActivity) -> str:
    color = SEVERITY_COLORS.get(act.severity.value, "#9ca3af")
    label = ACTIVITY_LABELS.get(act.activity_type.value, act.activity_type.value)
    return (
        f"<span style='background-color:{color}; color:white; padding:2px 8px; "
        f"border-radius:6px; font-size:0.75em; font-weight:bold;'>{act.severity.value}</span> "
        f"<span style='font-weight:600;'>{label}</span>"
    )


def render(activities: List[SuspiciousActivity],
           max_display: int = 100,
           show_evidence: bool = True,
           allow_ack: bool = True) -> None:
    """Render the alerts panel."""
    st.markdown("### 🚨 Suspicious Activity Feed")

    if not activities:
        st.info("No suspicious activities in the selected window. "
                "Run the pipeline to populate alerts.")
        return

    # Sort newest first
    activities = sorted(activities, key=lambda a: a.timestamp, reverse=True)[:max_display]

    # Filters
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        type_filter = st.selectbox(
            "Type",
            ["ALL"] + [a.value for a in ActivityType],
            index=0,
        )
    with c2:
        sev_filter = st.selectbox(
            "Min Severity",
            ["ALL", "INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
            index=2,
        )
    with c3:
        cam_filter = st.text_input("Camera ID (contains)", value="")
    with c4:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

    # Apply filters
    filtered = activities
    if type_filter != "ALL":
        filtered = [a for a in filtered if a.activity_type.value == type_filter]
    if sev_filter != "ALL":
        # Handle both string and Severity enum for a.severity
        def _get_severity_rank(sev):
            if isinstance(sev, str):
                try:
                    sev = Severity(sev)
                except (ValueError, KeyError):
                    return 0
            return SEVERITY_RANK.get(sev, 0)
        min_rank = _get_severity_rank(sev_filter)
        filtered = [a for a in filtered if _get_severity_rank(a.severity) >= min_rank]
    if cam_filter:
        filtered = [a for a in filtered if cam_filter.lower() in (a.camera_id or "").lower()]
    filtered = [a for a in filtered if a.confidence >= min_conf]

    st.caption(f"Showing {len(filtered)} of {len(activities)} activities")

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total", len(filtered))
    with m2:
        st.metric("Critical/High",
                  sum(1 for a in filtered if a.severity in (Severity.HIGH, Severity.CRITICAL)))
    with m3:
        st.metric("Avg Confidence",
                  f"{(sum(a.confidence for a in filtered) / len(filtered)):.2f}" if filtered else "0.00")
    with m4:
        st.metric("Unique Persons",
                  len({a.global_person_id for a in filtered if a.global_person_id is not None}))

    st.markdown("---")

    # List
    for i, act in enumerate(filtered):
        ts_str = datetime.fromtimestamp(act.timestamp).strftime("%H:%M:%S")
        with st.container():
            st.markdown(_activity_badge(act), unsafe_allow_html=True)
            st.markdown(
                f"**{ts_str}** &nbsp; 📷 `{act.camera_id}` &nbsp; "
                f"🎯 `track {act.track_id if act.track_id is not None else '-'}`"
                + (f" &nbsp; 🌐 `global #{act.global_person_id}`" if act.global_person_id else "")
            )
            st.markdown(f"> {act.message}")
            st.progress(min(1.0, float(act.confidence)),
                        text=f"Confidence: {act.confidence:.2f}")
            if act.zone_name:
                st.caption(f"Zone: {act.zone_name} ({act.zone_id})")
            if act.rule_violations:
                st.caption("Rules: " + ", ".join(act.rule_violations))
            if show_evidence and act.evidence:
                with st.expander("Evidence", expanded=False):
                    st.json(act.evidence)
            if act.snapshot_path and os.path.exists(act.snapshot_path):
                with st.expander("Snapshot", expanded=False):
                    st.image(act.snapshot_path, width=320)
            if allow_ack:
                if st.button("✓ Acknowledge", key=f"ack_{i}_{act.activity_id}"):
                    st.success(f"Acknowledged {act.activity_id}")
            st.markdown("---")
