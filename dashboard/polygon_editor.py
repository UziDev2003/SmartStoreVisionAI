"""
Streamlit Polygon Editor
=========================
Interactive zone editor that lets operators draw, edit, and save
polygons directly on top of a video frame using
`streamlit-drawable-canvas` (the de-facto industry-standard
Streamlit canvas component).

Features:
  - Load first frame from a video file or URL
  - Pre-populate canvas with existing zones
  - Polygon / Rect drawing tools
  - Color-coded zone-type selector (restricted/checkout/shelf/entrance/general)
  - Capacity & dwell threshold inputs
  - Save to per-camera runtime YAML or to main zones.yaml
  - Delete & reset
"""
from __future__ import annotations

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.analytics.zone_analytics import Zone
from src.utils.zone_io import save_zones, load_zones, RUNTIME_ZONES_DIR


ZONE_COLORS = {
    "restricted": "#ff0000",
    "checkout": "#00ff00",
    "shelf": "#ffff00",
    "entrance": "#00ffff",
    "general": "#ffffff",
}


def _zones_to_canvas_objects(zones):
    """Convert existing zones to a fabric.js initial drawing."""
    objects = []
    for z in zones:
        color = ZONE_COLORS.get(z.zone_type, "#ffffff")
        objects.append({
            "type": "polygon",
            "version": "5.3.0",
            "originX": "left", "originY": "top",
            "left": 0, "top": 0,
            "fill": "rgba(0,0,0,0)",
            "stroke": color,
            "strokeWidth": 3,
            "strokeDashArray": None,
            "label": z.name,
            "points": [{"x": float(p[0]), "y": float(p[1])} for p in z.polygon],
            "selectable": True,
        })
    return {"version": "5.3.0", "objects": objects}


def _canvas_to_zones(canvas_data, zone_type, default_dwell):
    """Convert canvas objects to Zone list (preserving assigned types if present)."""
    from src.utils.zone_io import zones_from_canvas
    color_to_type = {v: k for k, v in ZONE_COLORS.items()}
    objects = canvas_data.get("objects", []) if canvas_data else []
    zones = []
    for obj in objects:
        stroke = (obj.get("stroke") or "").lower()
        inferred_type = color_to_type.get(stroke, zone_type)
        zlist = zones_from_canvas([obj])
        for z in zlist:
            z.zone_type = inferred_type
            z.dwell_threshold = default_dwell
            zones.append(z)
    return zones


def _extract_first_frame(video_path):
    """Read the first frame of a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def render(camera_id="cam_01", video_path=None, existing_zones=None, on_save=None):
    """
    Render the polygon editor. Returns a dict with keys:
        - 'zones': List[Zone] currently configured
        - 'saved_to': path of last save (or None)
    """
    st.markdown("### 🗺️ Interactive Zone Editor")
    st.caption("Draw polygons directly on a video frame. Use color codes to "
               "classify zone type. Click **Save** to persist.")

    with st.sidebar:
        st.subheader("Zone Editor")
        cam = st.text_input("Camera ID", value=camera_id, key="zone_cam_id")
        tool = st.selectbox("Drawing Tool", ["polygon", "rect"], index=0)
        default_type = st.selectbox(
            "Default Zone Type (for new shapes)",
            list(ZONE_COLORS.keys()),
            index=4,
        )
        default_color = ZONE_COLORS[default_type]
        default_dwell = st.slider("Default Dwell Threshold (s)", 10, 600, 120, 10)
        stroke_width = st.slider("Stroke Width", 1, 8, 3)
        save_to = st.radio("Save Target", ["Runtime (per camera)", "Main (zones.yaml)"], index=0)
        enable_smoothing = st.checkbox("Smooth polygons (Douglas-Peucker)", value=True)

    frame = None
    if video_path and Path(video_path).exists():
        frame = _extract_first_frame(video_path)
    if frame is None:
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        st.info("No video loaded - drawing on a blank canvas. Upload a video "
                "in the Live tab to see the first frame here.")

    h, w = frame.shape[:2]
    target_w = 960
    if w != target_w:
        scale = target_w / w
        frame = cv2.resize(frame, (target_w, int(h * scale)))
    rgb = frame[:, :, ::-1]
    bg_image = Image.fromarray(rgb)

    initial_drawing = None
    if existing_zones:
        scaled = []
        scale = target_w / w
        for z in existing_zones:
            z2 = Zone(z.zone_id, z.name, z.zone_type,
                      [(int(p[0] * scale), int(p[1] * scale)) for p in z.polygon],
                      z.capacity, z.dwell_threshold)
            scaled.append(z2)
        initial_drawing = _zones_to_canvas_objects(scaled)

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=default_color,
        background_image=bg_image,
        update_streamlit=True,
        height=frame.shape[0],
        width=frame.shape[1],
        drawing_mode=tool,
        initial_drawing=initial_drawing,
        key="zone_canvas",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        save_clicked = st.button("💾 Save Zones", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("🗑️ Clear Canvas", use_container_width=True)
    with col3:
        load_runtime_clicked = st.button("📂 Load Saved", use_container_width=True)

    saved_to = None
    zones_final = []

    if save_clicked and canvas_result and canvas_result.json_data:
        zones_final = _canvas_to_zones(canvas_result.json_data, default_type, default_dwell)
        if enable_smoothing:
            zones_final = [_smooth_zone(z) for z in zones_final]
        if save_to.startswith("Runtime"):
            saved_to = save_zones(zones_final, camera_id=cam)
        else:
            saved_to = save_zones(zones_final, path=Path("config/zones.yaml"))
        st.success(f"Saved {len(zones_final)} zones to {saved_to}")
        if on_save:
            on_save(zones_final, str(saved_to))

    if clear_clicked:
        st.rerun()

    if load_runtime_clicked:
        loaded = load_zones(camera_id=cam)
        st.session_state["_loaded_zones"] = loaded
        st.info(f"Loaded {len(loaded)} zones from runtime file for {cam}")

    if zones_final:
        st.markdown("#### Saved Zones")
        for z in zones_final:
            c = ZONE_COLORS.get(z.zone_type, "#888")
            st.markdown(
                f"<span style='color:{c}; font-weight:bold;'>●</span> "
                f"**{z.name}** ({z.zone_type}) - {len(z.polygon)} vertices",
                unsafe_allow_html=True,
            )

    return {"zones": zones_final, "saved_to": str(saved_to) if saved_to else None}


def _smooth_zone(zone, epsilon=4.0):
    """Apply Douglas-Peucker polygon simplification."""
    if not zone.polygon or len(zone.polygon) < 3:
        return zone
    try:
        arr = np.array(zone.polygon, dtype=np.int32)
        peri = cv2.arcLength(arr, True)
        approx_eps = max(0.1, float(epsilon) / max(peri, 1.0) * 100.0)
        approx = cv2.approxPolyDP(arr, approx_eps, True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(pts) < 3:
            return zone
        return Zone(zone.zone_id, zone.name, zone.zone_type, pts,
                    zone.capacity, zone.dwell_threshold)
    except Exception:
        return zone
