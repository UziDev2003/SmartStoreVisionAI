"""
Streamlit Polygon Editor (resilient)
====================================
Interactive zone editor that lets operators define, edit, and save
zones. Two modes are supported:

  1. **Visual canvas** (preferred) — uses `streamlit-drawable-canvas`
     when it works on the installed Streamlit version.
  2. **Text fallback** — pure-Streamlit editor that lets the user
     define polygons via JSON coordinates and number inputs. Always
     available, even on Streamlit >= 1.39 where the drawable-canvas
     library is broken.

The editor automatically selects the right mode at runtime.

Features:
  - Load first frame from a video file
  - Pre-populate editor with existing zones
  - Polygon / Rect drawing tools (visual mode)
  - Color-coded zone-type selector (restricted/checkout/shelf/entrance/general)
  - Capacity & dwell threshold inputs
  - Save to per-camera runtime YAML or to main zones.yaml
  - Delete & reset
"""
from __future__ import annotations

import json
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.analytics.zone_analytics import Zone
from src.utils.zone_io import (
    save_zones, load_zones, zones_from_canvas, RUNTIME_ZONES_DIR,
)


ZONE_COLORS = {
    "restricted": "#ff0000",
    "checkout": "#00ff00",
    "shelf": "#ffff00",
    "entrance": "#00ffff",
    "general": "#ffffff",
}


# ---------------------------------------------------------------------------
# canvas helper
# ---------------------------------------------------------------------------
def _has_working_canvas() -> bool:
    """Check whether `streamlit_drawable_canvas` works with the installed
    version of Streamlit (newer versions removed `image.image_to_url`)."""
    try:
        import streamlit.elements.image as _img
        if not hasattr(_img, "image_to_url") and hasattr(_img, "_image_to_url"):
            _img.image_to_url = _img._image_to_url
        from streamlit_drawable_canvas import st_canvas  # noqa: F401
        return hasattr(_img, "image_to_url")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Conversion helpers (used by visual mode)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Fallback: text-based editor
# ---------------------------------------------------------------------------
def _render_text_editor(existing_zones, default_type, default_dwell, cam):
    """Pure-Streamlit fallback editor. Lets the user edit a list of zones
    via JSON coordinates, name, type, capacity, and dwell inputs."""
    if "_text_zones" not in st.session_state:
        st.session_state["_text_zones"] = [
            {
                "zone_id": z.zone_id,
                "name": z.name,
                "zone_type": z.zone_type,
                "capacity": z.capacity,
                "dwell_threshold": z.dwell_threshold,
                "polygon": list(z.polygon),
            }
            for z in (existing_zones or [])
        ]

    zones_state = st.session_state["_text_zones"]
    new_zones = []
    st.caption("Edit zones in the table. Polygon = list of (x, y) tuples, "
               "e.g. `[(100,200),(300,200),(300,400)]`.")

    for i, z in enumerate(zones_state):
        with st.expander(
            f"📍 {z['name'] or 'Unnamed'} ({z['zone_type']})",
            expanded=False,
        ):
            c1, c2 = st.columns(2)
            with c1:
                z["name"] = st.text_input("Name", value=z["name"], key=f"z_name_{i}")
                z["zone_id"] = st.text_input("ID", value=z["zone_id"], key=f"z_id_{i}")
                z["zone_type"] = st.selectbox(
                    "Type", list(ZONE_COLORS.keys()),
                    index=list(ZONE_COLORS.keys()).index(z["zone_type"])
                    if z["zone_type"] in ZONE_COLORS else 4,
                    key=f"z_type_{i}",
                )
            with c2:
                z["capacity"] = st.number_input(
                    "Capacity", 0, 1000, int(z["capacity"]), key=f"z_cap_{i}")
                z["dwell_threshold"] = st.number_input(
                    "Dwell (s)", 0, 600, int(z["dwell_threshold"]), key=f"z_dw_{i}")
            poly_str = st.text_area(
                "Polygon (JSON list of [x,y])",
                value=json.dumps(z["polygon"]),
                key=f"z_poly_{i}",
                height=80,
            )
            try:
                parsed = json.loads(poly_str)
                if isinstance(parsed, list) and all(
                    isinstance(p, (list, tuple)) and len(p) == 2 for p in parsed
                ):
                    z["polygon"] = [(int(p[0]), int(p[1])) for p in parsed]
                else:
                    st.warning("Polygon must be a list of [x, y] pairs.")
            except Exception:
                st.warning("Invalid JSON — keeping previous polygon.")
            if st.button("🗑️ Delete", key=f"z_del_{i}"):
                st.session_state["_text_zones"].pop(i)
                st.rerun()
            new_zones.append(z)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ Add Zone", use_container_width=True):
            st.session_state["_text_zones"].append({
                "zone_id": f"zone_{len(st.session_state['_text_zones']) + 1}",
                "name": f"Zone {len(st.session_state['_text_zones']) + 1}",
                "zone_type": default_type,
                "capacity": 5,
                "dwell_threshold": default_dwell,
                "polygon": [(100, 100), (300, 100), (300, 300), (100, 300)],
            })
            st.rerun()
    with c2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state["_text_zones"] = []
            st.rerun()

    # Convert dicts to Zone objects
    return [
        Zone(
            z["zone_id"], z["name"], z["zone_type"],
            z["polygon"], int(z["capacity"]), int(z["dwell_threshold"]),
        )
        for z in st.session_state["_text_zones"]
    ]


# ---------------------------------------------------------------------------
# Visual canvas editor
# ---------------------------------------------------------------------------
def _render_canvas_editor(frame, existing_zones, tool, default_color,
                          default_dwell, stroke_width, cam):
    """The original canvas-based editor. Falls back to text editor on failure."""
    from streamlit_drawable_canvas import st_canvas

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
            scaled.append(Zone(
                z.zone_id, z.name, z.zone_type,
                [(int(p[0] * scale), int(p[1] * scale)) for p in z.polygon],
                z.capacity, z.dwell_threshold,
            ))
        initial_drawing = _zones_to_canvas_objects(scaled)

    try:
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
    except AttributeError as e:
        # Newer Streamlit broke streamlit_drawable_canvas
        st.warning(
            f"⚠️ Visual canvas unavailable in this Streamlit version: `{e}`. "
            "Switching to **text-based zone editor**."
        )
        return _render_text_editor(existing_zones, "general", default_dwell, cam)

    if canvas_result and canvas_result.json_data:
        return _canvas_to_zones(canvas_result.json_data, "general", default_dwell)
    return []


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render(camera_id="cam_01", video_path=None, existing_zones=None, on_save=None):
    """Render the polygon editor. Returns a dict with keys:
        - 'zones': List[Zone] currently configured
        - 'saved_to': path of last save (or None)
    """
    st.markdown("### 🗺️ Interactive Zone Editor")
    st.caption(
        "Draw polygons directly on a video frame, or use the text editor. "
        "Click **💾 Save Zones** to persist."
    )

    # --- Sidebar controls ---
    with st.sidebar:
        st.subheader("Zone Editor")
        cam = st.text_input("Camera ID", value=camera_id, key="zone_cam_id")
        has_canvas = _has_working_canvas()
        mode_options = ["Visual canvas"] if has_canvas else []
        mode_options.append("Text editor")
        mode = st.radio(
            "Editor mode",
            mode_options,
            index=0,
            help="Visual canvas requires `streamlit-drawable-canvas` to be "
                 "compatible with the installed Streamlit version.",
        )
        tool = st.selectbox("Drawing Tool", ["polygon", "rect"], index=0)
        default_type = st.selectbox(
            "Default Zone Type",
            list(ZONE_COLORS.keys()),
            index=4,
        )
        default_color = ZONE_COLORS[default_type]
        default_dwell = st.slider("Default Dwell Threshold (s)", 10, 600, 120, 10)
        stroke_width = st.slider("Stroke Width", 1, 8, 3)
        save_to = st.radio(
            "Save Target",
            ["Runtime (per camera)", "Main (zones.yaml)"],
            index=0,
        )
        enable_smoothing = st.checkbox(
            "Smooth polygons (Douglas-Peucker)", value=True,
        )

    # --- Load first frame ---
    frame = None
    if video_path and Path(video_path).exists():
        frame = _extract_first_frame(video_path)
    if frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                 channels="RGB", use_container_width=True)
    else:
        st.info(
            "📹 No video loaded — drawing on a blank canvas. Upload a video "
            "in the Live tab to see the first frame here."
        )

    # --- Choose editor mode ---
    if mode == "Visual canvas" and frame is not None:
        zones_final = _render_canvas_editor(
            frame, existing_zones, tool, default_color,
            default_dwell, stroke_width, cam,
        )
    else:
        zones_final = _render_text_editor(existing_zones, default_type, default_dwell, cam)

    # --- Save / Clear / Load buttons ---
    col1, col2, col3 = st.columns(3)
    with col1:
        save_clicked = st.button("💾 Save Zones", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("🗑️ Clear Canvas", use_container_width=True)
    with col3:
        load_runtime_clicked = st.button("📂 Load Saved", use_container_width=True)

    saved_to = None
    if save_clicked and zones_final:
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
        if "_text_zones" in st.session_state:
            st.session_state["_text_zones"] = []
        st.rerun()

    if load_runtime_clicked:
        loaded = load_zones(camera_id=cam)
        st.session_state["_loaded_zones"] = loaded
        st.session_state["_text_zones"] = [
            {
                "zone_id": z.zone_id,
                "name": z.name,
                "zone_type": z.zone_type,
                "capacity": z.capacity,
                "dwell_threshold": z.dwell_threshold,
                "polygon": list(z.polygon),
            }
            for z in loaded
        ]
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
