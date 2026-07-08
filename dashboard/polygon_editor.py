"""
Streamlit Polygon Editor (image-based interactive)
====================================================
Interactive zone editor that lets operators define, edit, and save
zones directly on the video frame image.

Two modes:
  1. **Image Click Editor** (default) — click on the image to place
     polygon vertices; uses Streamlit-native click coordinates.
  2. **OpenCV Interactive** — opens a dedicated OpenCV window with
     full mouse-drawing support (separate process, best experience).

Features:
  - Load first frame from a video file or use a blank canvas
  - Pre-populate editor with existing zones
  - Polygon drawing via mouse clicks on the image
  - Color-coded zone-type selector
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
from typing import List, Optional, Dict, Any, Tuple
import tempfile
import os
import sys
import subprocess

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

ZONE_COLORS_BGR = {
    "restricted": (0, 0, 255),
    "checkout": (0, 255, 0),
    "shelf": (0, 255, 255),
    "entrance": (255, 255, 0),
    "general": (255, 255, 255),
}


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
# Draw zones on an image for display
# ---------------------------------------------------------------------------
def _draw_zones_on_image(img: np.ndarray, zones: List[Zone],
                          highlight_idx: Optional[int] = None,
                          draw_labels: bool = True) -> np.ndarray:
    """Draw all zones on a copy of the image with color coding."""
    out = img.copy()
    for i, z in enumerate(zones):
        color = ZONE_COLORS_BGR.get(z.zone_type, (255, 255, 255))
        if highlight_idx is not None and i == highlight_idx:
            # Highlight selected zone with thicker lines
            thickness = 4
        else:
            thickness = 2

        pts = np.array(z.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], True, color, thickness)
        if draw_labels and z.name:
            cx = int(np.mean([p[0] for p in z.polygon]))
            cy = int(np.mean([p[1] for p in z.polygon]))
            cv2.putText(out, z.name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
            # Draw vertex markers
            for p in z.polygon:
                cv2.circle(out, (int(p[0]), int(p[1])), 4, color, -1)
    return out


# ---------------------------------------------------------------------------
# Image Click Editor (replaces the old text editor)
# ---------------------------------------------------------------------------
# Streamlit 1.36+ compatibility hack for streamlit-drawable-canvas 0.9.3
try:
    import streamlit.elements.image as st_image
    def _mock_image_to_url(image, width, height, image_format, *args, **kwargs):
        import io, base64
        from PIL import Image
        import numpy as np
        buffered = io.BytesIO()
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        fmt = image_format.upper() if isinstance(image_format, str) else "PNG"
        if fmt not in ["PNG", "JPEG", "GIF", "BMP", "WEBP"]:
            fmt = "PNG"
        image.save(buffered, format=fmt)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{fmt.lower()};base64,{img_str}"
    
    st_image.image_to_url = _mock_image_to_url
except Exception:
    pass

# Try to import the drawable canvas; fall back to a coordinate-based editor
# if it's not installed (this is what was causing the image to not show)
try:
    from streamlit_drawable_canvas import st_canvas
    _CANVAS_AVAILABLE = True
    _CANVAS_ERROR = None
except Exception as _e:
    _CANVAS_AVAILABLE = False
    _CANVAS_ERROR = str(_e)
    st_canvas = None


def _render_image_editor(frame: np.ndarray, existing_zones: List[Zone],
                          default_type: str, default_dwell: int,
                          cam: str) -> List[Zone]:
    """
    Image-based polygon editor.
    - If streamlit-drawable-canvas is available, uses interactive canvas.
    - Otherwise falls back to a coordinate-based editor with the image shown
      via st.image() so the user can at least see the frame.
    """
    h, w = frame.shape[:2]
    # Cap canvas size for performance & Cloudflare-tunnel friendliness
    MAX_W = 800
    if w > MAX_W:
        scale = MAX_W / w
        frame = cv2.resize(frame, (MAX_W, int(h * scale)))
        h, w = frame.shape[:2]

    # Draw existing zones on the background image
    display_img = _draw_zones_on_image(frame, existing_zones or [], draw_labels=True)
    display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(display_img_rgb)

    st.write(
        "Draw new zones directly on the image below. Click 'Polygon' to start "
        "drawing. Double click or click the start point to close."
    )

    # Select drawing tool and color
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        drawing_mode = st.selectbox("Drawing tool:",
                                     ("polygon", "rect", "freedraw", "transform"))
    with col2:
        stroke_width = st.slider("Stroke width: ", 1, 10, 3)
    with col3:
        new_zone_type = st.selectbox(
            "New Zone Type:", list(ZONE_COLORS.keys()),
            index=list(ZONE_COLORS.keys()).index(default_type)
            if default_type in ZONE_COLORS else 4,
        )

    stroke_color = ZONE_COLORS.get(new_zone_type, "#00ff00")

    zones_final = list(existing_zones) if existing_zones else []

    if _CANVAS_AVAILABLE:
        try:
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=pil_image,
                update_streamlit=True,
                height=h,
                width=w,
                drawing_mode=drawing_mode,
                key="canvas",
            )
            if canvas_result is not None and getattr(canvas_result, "json_data", None):
                objects = canvas_result.json_data["objects"]
                for obj in objects:
                    obj["stroke"] = stroke_color
                new_zones = zones_from_canvas(objects)
                for z in new_zones:
                    z.dwell_threshold = default_dwell
                zones_final.extend(new_zones)
            elif canvas_result is None:
                # Canvas returned nothing - likely a render issue. Fall back.
                st.warning(
                    "⚠️ Interactive canvas didn't render. Showing the frame "
                    "as a static image — use the coordinate editor below to "
                    "add zones manually."
                )
                st.image(pil_image, caption="Video frame (no interactive canvas)",
                         use_column_width=True)
                zones_final = _render_coordinate_editor(
                    frame, existing_zones or [], new_zone_type, default_dwell,
                )
        except Exception as _canvas_exc:
            st.error(
                f"❌ Interactive canvas error: {_canvas_exc}\n\n"
                f"Falling back to coordinate-based editor. "
                f"You can also try installing/upgrading the library: "
                f"`pip install -U streamlit-drawable-canvas`"
            )
            st.image(pil_image, caption="Video frame", use_column_width=True)
            zones_final = _render_coordinate_editor(
                frame, existing_zones or [], new_zone_type, default_dwell,
            )
    else:
        # Canvas library not installed - fall back to coordinate editor
        st.info(
            f"ℹ️ `streamlit-drawable-canvas` is not available ({_CANVAS_ERROR}). "
            f"Using the coordinate-based editor below instead."
        )
        st.image(pil_image, caption="Video frame", use_column_width=True)
        zones_final = _render_coordinate_editor(
            frame, existing_zones or [], new_zone_type, default_dwell,
        )

    return zones_final


def _render_coordinate_editor(frame: np.ndarray, existing_zones: List[Zone],
                               zone_type: str, default_dwell: int) -> List[Zone]:
    """
    Fallback coordinate-based zone editor. The user enters polygon vertex
    coordinates manually. This is used when the interactive canvas
    component fails to render.
    """
    zones_final = list(existing_zones) if existing_zones else []

    st.markdown("#### ✏️ Coordinate-Based Zone Editor")
    st.caption(
        "Add zones by entering polygon vertices as `(x, y)` coordinates. "
        "Reference the image above to pick points."
    )

    # Add a new zone
    with st.expander("➕ Add a new zone", expanded=False):
        new_name = st.text_input("Zone name", value=f"Zone {len(zones_final) + 1}",
                                  key="coord_zone_name")
        new_type = st.selectbox("Type", list(ZONE_COLORS.keys()),
                                 index=list(ZONE_COLORS.keys()).index(zone_type)
                                 if zone_type in ZONE_COLORS else 4,
                                 key="coord_zone_type")
        new_dwell = st.number_input("Dwell threshold (s)", 0, 600,
                                     default_dwell, 10, key="coord_zone_dwell")
        new_cap = st.number_input("Capacity", 0, 100, 5, 1, key="coord_zone_cap")

        # Vertex input
        n_points = st.number_input("Number of vertices", 3, 12, 4, 1,
                                    key="coord_n_points")
        pts = []
        cols = st.columns(2)
        for i in range(int(n_points)):
            with cols[i % 2]:
                c1, c2 = st.columns(2)
                with c1:
                    x = st.number_input(f"x{i+1}", 0, 2000, 100, 10,
                                         key=f"coord_x_{i}")
                with c2:
                    y = st.number_input(f"y{i+1}", 0, 2000, 100, 10,
                                         key=f"coord_y_{i}")
            pts.append((int(x), int(y)))

        if st.button("➕ Add Zone", key="coord_add_zone") and len(pts) >= 3:
            from src.analytics.zone_analytics import Zone as _Zone
            zone = _Zone(
                zone_id=new_name.lower().replace(" ", "_"),
                name=new_name,
                zone_type=new_type,
                polygon=pts,
                capacity=int(new_cap),
                dwell_threshold=int(new_dwell),
            )
            zones_final.append(zone)
            st.success(f"Added zone '{new_name}' with {len(pts)} vertices")
            st.rerun()

    return zones_final


# ---------------------------------------------------------------------------
# OpenCV Interactive Editor (dedicated window with mouse callbacks)
# ---------------------------------------------------------------------------
def _run_opencv_editor(frame: np.ndarray, existing_zones: List[Zone],
                        default_type: str, output_file: str) -> str:
    """
    Launch an OpenCV window for interactive polygon drawing.
    Saves zones to output_file as JSON. Returns the file path.
    """
    script = f"""
import cv2, json, sys, numpy as np
from pathlib import Path

frame = np.array({frame.tolist()})
existing = {json.dumps([
    {{
        "zone_id": z.zone_id, "name": z.name, "zone_type": z.zone_type,
        "polygon": list(z.polygon), "capacity": z.capacity,
        "dwell_threshold": z.dwell_threshold
    }} for z in existing_zones
])}
default_type = "{default_type}"
output_file = "{output_file}"

# Colors
ZONE_COLORS_BGR = {{
    "restricted": (0, 0, 255),
    "checkout": (0, 255, 0),
    "shelf": (0, 255, 255),
    "entrance": (255, 255, 0),
    "general": (255, 255, 255),
}}

zones = existing.copy()
current_points = []
drawing = False
zone_name = ""
zone_type = default_type
current_idx = -1

def draw_zones(img, zones_list, highlight=-1):
    out = img.copy()
    for i, z in enumerate(zones_list):
        color = ZONE_COLORS_BGR.get(z["zone_type"], (255, 255, 255))
        thick = 4 if i == highlight else 2
        pts = np.array(z["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], True, color, thick)
        cx = int(np.mean([p[0] for p in z["polygon"]]))
        cy = int(np.mean([p[1] for p in z["polygon"]]))
        cv2.putText(out, z["name"], (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out

def mouse_callback(event, x, y, flags, param):
    global current_points, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if near an existing point (remove)
        for i, (px, py) in enumerate(current_points):
            if abs(px - x) < 10 and abs(py - y) < 10:
                current_points.pop(i)
                return
        # Check if close to first point (close polygon)
        if len(current_points) >= 3:
            px, py = current_points[0]
            if abs(px - x) < 15 and abs(py - y) < 15:
                drawing = False
                return
        current_points.append((x, y))

cv2.namedWindow("Zone Editor")
cv2.setMouseCallback("Zone Editor", mouse_callback)

print("=== OpenCV Zone Editor ===")
print("Left-click: add vertex")
print("Click near first vertex: close polygon")
print("Click near existing vertex: remove")
print("Keys:")
print("  's' - Save zones")
print("  'n' - New zone")
print("  'd' - Delete last zone")
print("  'ESC'/'q' - Quit without saving")
print("  '1-5' - Set zone type (1=restricted,2=checkout,3=shelf,4=entrance,5=general)")

while True:
    img = draw_zones(frame, zones)
    # Draw current drawing
    if current_points:
        for i in range(len(current_points) - 1):
            cv2.line(img, current_points[i], current_points[i + 1], (0, 255, 255), 2)
        for pt in current_points:
            cv2.circle(img, pt, 4, (0, 255, 255), -1)
        if len(current_points) >= 3:
            cv2.line(img, current_points[-1], current_points[0], (0, 200, 200), 1)

    # Instructions
    cv2.putText(img, f"Type: {{zone_type}} | S:save N:new D:del Q:quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if current_points:
        cv2.putText(img, f"Points: {{len(current_points)}} (click near start to close)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Zone Editor", img)
    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord('q'):
        break
    elif key == ord('s') and current_points and len(current_points) >= 3:
        name = input(f"Name for zone ({{len(zones)+1}}): ") or f"Zone {{len(zones)+1}}"
        zones.append({{
            "zone_id": name.lower().replace(" ", "_"),
            "name": name,
            "zone_type": zone_type,
            "polygon": list(current_points),
            "capacity": 5,
            "dwell_threshold": 120,
        }})
        current_points = []
        print(f"Zone saved: {{name}}")
    elif key == ord('n'):
        current_points = []
        zone_type = default_type
        print("New zone started")
    elif key == ord('d'):
        if zones:
            removed = zones.pop()
            print(f"Deleted: {{removed['name']}}")
    elif key == ord('1'):
        zone_type = "restricted"
    elif key == ord('2'):
        zone_type = "checkout"
    elif key == ord('3'):
        zone_type = "shelf"
    elif key == ord('4'):
        zone_type = "entrance"
    elif key == ord('5'):
        zone_type = "general"

cv2.destroyAllWindows()

# Save
with open(output_file, 'w') as f:
    json.dump(zones, f)
print(f"Saved {{len(zones)}} zones to {{output_file}}")
"""
    # Create temp script file
    script_path = tempfile.mktemp(suffix=".py", prefix="zone_editor_")
    with open(script_path, "w") as f:
        f.write(script)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=300
        )
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        st.error("Editor timed out after 5 minutes")
        return None
    except Exception as e:
        st.error(f"Editor error: {e}")
        return None
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass

    return output_file


def _render_opencv_editor_button(frame: np.ndarray, existing_zones: List[Zone],
                                   default_type: str, default_dwell: int,
                                   cam: str) -> Optional[List[Zone]]:
    """Render a button to launch the OpenCV interactive editor."""
    if st.button("🖱️ Open Interactive Zone Editor (OpenCV window)",
                 use_container_width=True, type="primary"):
        out_file = tempfile.mktemp(suffix=".json", prefix="zones_")
        result_file = _run_opencv_editor(frame, existing_zones, default_type, out_file)
        if result_file and Path(result_file).exists():
            with open(result_file, "r") as f:
                zones_data = json.load(f)
            try:
                os.unlink(result_file)
            except Exception:
                pass
            if zones_data:
                zones = []
                for z in zones_data:
                    zones.append(Zone(
                        z["zone_id"], z["name"], z["zone_type"],
                        z["polygon"], z.get("capacity", 5),
                        z.get("dwell_threshold", default_dwell),
                    ))
                st.success(f"Loaded {len(zones)} zones from editor")
                return zones
            else:
                st.warning("No zones were created")
        else:
            st.warning("Editor was closed or no zones saved")
    return None


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
        "Draw polygons directly by clicking on the video frame image. "
        "Click **Add Point** to place vertices, **Close Polygon** to finish, "
        "or use the OpenCV interactive editor for mouse drawing."
    )

    # --- Sidebar controls ---
    with st.sidebar:
        st.subheader("Zone Editor")
        cam = st.text_input("Camera ID", value=camera_id, key="zone_cam_id")
        tool = st.radio("Editor Mode", ["Image Click Editor", "OpenCV Interactive"],
                        index=0,
                        help="Image Click: place vertices via coordinate inputs. "
                             "OpenCV: dedicated window with mouse drawing.")
        default_type = st.selectbox(
            "Default Zone Type",
            list(ZONE_COLORS.keys()),
            index=4,
        )
        default_color = ZONE_COLORS[default_type]
        default_dwell = st.slider("Default Dwell Threshold (s)", 10, 600, 120, 10)
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
    if frame is None:
        # Create a blank canvas
        frame = np.ones((540, 960, 3), dtype=np.uint8) * 30
        cv2.putText(frame, "No video loaded - blank canvas",
                    (240, 270), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (100, 100, 100), 2)
        st.info(
            "📹 No video loaded — drawing on a blank canvas. Upload a video "
            "in the Live tab to see the first frame here."
        )
    else:
        h, w = frame.shape[:2]
        target_w = 960
        if w != target_w:
            scale = target_w / w
            frame = cv2.resize(frame, (target_w, int(h * scale)))

    # --- Editor modes ---
    zones_final = []

    if tool == "OpenCV Interactive":
        # Show existing zones on frame preview
        if existing_zones:
            preview = _draw_zones_on_image(frame, existing_zones)
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                     use_column_width=True,
                     caption="Current zones (preview)")
        else:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     use_column_width=True,
                     caption="Video frame")
        result = _render_opencv_editor_button(frame, existing_zones or [],
                                               default_type, default_dwell, cam)
        if result is not None:
            zones_final = result
        else:
            # Fall back to existing zones
            zones_final = existing_zones or []
    else:
        # Image Click Editor
        zones_final = _render_image_editor(
            frame, existing_zones or [], default_type, default_dwell, cam,
        )

    # --- Save / Clear / Load buttons ---
    col1, col2, col3 = st.columns(3)
    with col1:
        save_clicked = st.button("💾 Save Zones", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("🗑️ Clear All Zones", use_container_width=True)
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
        if "_image_zones" in st.session_state:
            st.session_state["_image_zones"] = []
        if "_temp_polygon" in st.session_state:
            st.session_state["_temp_polygon"] = {"points": [], "closed": False, "editing_zone_idx": None}
        st.rerun()

    if load_runtime_clicked:
        loaded = load_zones(camera_id=cam)
        st.session_state["_image_zones"] = [
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
                f"**{z.name}** ({z.zone_type}) — {len(z.polygon)} vertices",
                unsafe_allow_html=True,
            )

    return {"zones": zones_final, "saved_to": str(saved_to) if saved_to else None}