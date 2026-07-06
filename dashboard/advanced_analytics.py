"""
Advanced Analytics Tab
======================
Streamlit panel for configuring and visualizing the new
feature modules (frame preprocessor, background subtraction,
line crossing, queue detection, object monitor, face blur,
video clipper, pose detectors).

Each section has:
  - Toggle to enable/disable
  - Parameter sliders
  - Live status display
"""
from __future__ import annotations
import streamlit as st
import cv2
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.frame_preprocessor import FramePreprocessor, PreprocessConfig
from src.analytics.background_subtractor import BackgroundSubtractor
from src.analytics.line_detector import LineCrossingDetector, LineZone
from src.analytics.queue_detector import QueueDetector, QueueZone
from src.analytics.object_monitor import (
    ObjectLeftBehindDetector, ObjectRemovedDetector, ShelfArea,
)
from src.analytics.video_clipper import VideoClipper
from src.analytics.pose_detectors import (
    GestureDetector, FallDetector, FightDetector,
)
from src.utils.face_blur import FaceBlurrer


# ---------------------------------------------------------------------------
# Factory helpers (cached per session)
# ---------------------------------------------------------------------------
def _get_preprocessor() -> FramePreprocessor:
    cfg = PreprocessConfig(enabled=True)
    # Read the toggle values
    cfg.target_width = st.session_state.get("pp_w", 0)
    cfg.target_height = st.session_state.get("pp_h", 0)
    cfg.letterbox = st.session_state.get("pp_letterbox", False)
    cfg.clahe = st.session_state.get("pp_clahe", False)
    cfg.clahe_clip_limit = st.session_state.get("pp_clahe_clip", 2.0)
    cfg.white_balance = st.session_state.get("pp_wb", False)
    cfg.gamma = st.session_state.get("pp_gamma", 1.0)
    cfg.blur_ksize = st.session_state.get("pp_blur", 0)
    cfg.sharpen = st.session_state.get("pp_sharpen", False)
    cfg.denoise = st.session_state.get("pp_denoise", False)
    cfg.convert_rgb = st.session_state.get("pp_rgb", False)
    return FramePreprocessor(cfg)


def _get_bg_subtractor() -> Optional[BackgroundSubtractor]:
    if not st.session_state.get("bg_enabled", False):
        return None
    method = st.session_state.get("bg_method", "MOG2")
    history = st.session_state.get("bg_history", 500)
    return BackgroundSubtractor(method=method, history=history)


def _get_line_detector() -> Optional[LineCrossingDetector]:
    if not st.session_state.get("line_enabled", False):
        return None
    if "line_detector" not in st.session_state:
        st.session_state["line_detector"] = LineCrossingDetector()
        # Seed default line
        st.session_state["line_detector"].add_line(LineZone(
            line_id="entrance",
            name="Front Door",
            p1=(100, 400), p2=(860, 400),
            left_label="in", right_label="out",
        ))
    return st.session_state["line_detector"]


def _get_queue_detector() -> Optional[QueueDetector]:
    if not st.session_state.get("queue_enabled", False):
        return None
    if "queue_detector" not in st.session_state:
        st.session_state["queue_detector"] = QueueDetector(queues=[
            QueueZone("checkout_q", "Checkout Queue",
                      polygon=[(600, 200), (900, 200), (900, 500), (600, 500)],
                      min_queue_length=2, wait_threshold_seconds=30.0),
        ])
    return st.session_state["queue_detector"]


def _get_object_detector():
    """Returns (ObjectLeftBehindDetector, ObjectRemovedDetector, [ShelfArea])"""
    if not st.session_state.get("obj_enabled", False):
        return None, None, []
    if "obj_shelves" not in st.session_state:
        st.session_state["obj_shelves"] = [
            ShelfArea("shelf_a", "Shelf A",
                      polygon=[(200, 100), (450, 100), (450, 350), (200, 350)]),
            ShelfArea("shelf_b", "Shelf B",
                      polygon=[(500, 100), (750, 100), (750, 350), (500, 350)]),
        ]
        st.session_state["olb"] = ObjectLeftBehindDetector(min_seconds=8.0)
        st.session_state["orm"] = ObjectRemovedDetector(min_drop_ratio=0.3)
    return st.session_state["olb"], st.session_state["orm"], st.session_state["obj_shelves"]


def _get_face_blurrer() -> Optional[FaceBlurrer]:
    if not st.session_state.get("face_blur_enabled", False):
        return None
    if "face_blurrer" not in st.session_state:
        method = st.session_state.get("face_blur_method", "opencv")
        st.session_state["face_blurrer"] = FaceBlurrer(method=method)
    return st.session_state["face_blurrer"]


def _get_video_clipper() -> Optional[VideoClipper]:
    if not st.session_state.get("clip_enabled", False):
        return None
    if "video_clipper" not in st.session_state:
        pre = st.session_state.get("clip_pre", 10)
        post = st.session_state.get("clip_post", 10)
        st.session_state["video_clipper"] = VideoClipper(
            pre_seconds=pre, post_seconds=post,
            output_dir="data/clips", fps=25)
    return st.session_state["video_clipper"]


def _get_pose_detectors():
    """Returns (gesture, fall, fight) detectors."""
    g = GestureDetector() if st.session_state.get("pose_gesture", True) else None
    f = FallDetector() if st.session_state.get("pose_fall", True) else None
    ft = FightDetector() if st.session_state.get("pose_fight", True) else None
    return g, f, ft


# ---------------------------------------------------------------------------
# The main render function
# ---------------------------------------------------------------------------
def render() -> Dict[str, Any]:
    """Render the Advanced Analytics configuration tab. Returns the
    live state for use in the Live tab."""
    st.markdown("### 🛠️ Advanced Analytics Configuration")
    st.caption("Configure all 10 additional modules: frame preprocessor, "
               "background subtraction, line crossing, queue detection, "
               "object monitor, face blur, video clipper, pose detectors.")

    state: Dict[str, Any] = {}

    with st.expander("🖼️ Frame Preprocessor", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.checkbox("Enable preprocessing", True, key="pp_enabled")
            st.checkbox("Letterbox (square)", False, key="pp_letterbox")
            st.checkbox("CLAHE", False, key="pp_clahe")
            st.slider("CLAHE clip limit", 0.5, 5.0, 2.0, 0.1, key="pp_clahe_clip")
            st.checkbox("Gray-world white balance", False, key="pp_wb")
            st.slider("Gamma", 0.3, 2.5, 1.0, 0.05, key="pp_gamma")
        with c2:
            st.number_input("Target width (0=keep)", 0, 4096, 0, key="pp_w")
            st.number_input("Target height (0=keep)", 0, 4096, 0, key="pp_h")
            st.slider("Blur kernel", 0, 21, 0, 1, key="pp_blur")
            st.checkbox("Sharpen", False, key="pp_sharpen")
            st.checkbox("Denoise (slow)", False, key="pp_denoise")
            st.checkbox("BGR->RGB", False, key="pp_rgb")
        state["preprocessor"] = _get_preprocessor if st.session_state.get("pp_enabled", True) else None

    with st.expander("🎬 Background Subtraction", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.checkbox("Enable BG subtraction", False, key="bg_enabled")
        with c2:
            st.selectbox("Method", ["MOG2", "KNN"], key="bg_method")
            st.slider("History", 50, 2000, 500, 50, key="bg_history")
        if st.session_state.get("bg_enabled", False):
            st.info("💡 Background mask will be displayed as a small overlay in the Live tab.")

    with st.expander("➡️ Line Crossing", expanded=False):
        st.checkbox("Enable line crossing", False, key="line_enabled")
        if st.session_state.get("line_enabled", False):
            det = _get_line_detector()
            st.caption(f"Active lines: {len(det.lines)}")
            with st.form("add_line"):
                st.markdown("**Add a new line:**")
                c1, c2, c3, c4 = st.columns(4)
                p1x = c1.number_input("p1.x", 0, 4096, 100)
                p1y = c2.number_input("p1.y", 0, 4096, 400)
                p2x = c3.number_input("p2.x", 0, 4096, 860)
                p2y = c4.number_input("p2.y", 0, 4096, 400)
                name = st.text_input("Line name", "New Line")
                c5, c6 = st.columns(2)
                left_label = c5.text_input("Left label", "in")
                right_label = c6.text_input("Right label", "out")
                if st.form_submit_button("Add Line"):
                    det.add_line(LineZone(
                        line_id=f"L{len(det.lines) + 1}",
                        name=name,
                        p1=(p1x, p1y), p2=(p2x, p2y),
                        left_label=left_label, right_label=right_label,
                    ))
                    st.success(f"Added line: {name}")
            # Show counts
            counts = det.get_counts()
            if counts:
                st.markdown("**Crossing counts:**")
                for (lid, d), c in counts.items():
                    st.markdown(f"- {lid} → {d}: **{c}**")
            else:
                st.caption("No crossings yet.")

    with st.expander("🧍 Queue Detection", expanded=False):
        st.checkbox("Enable queue detection", False, key="queue_enabled")
        if st.session_state.get("queue_enabled", False):
            det = _get_queue_detector()
            for q in det.queues.values():
                st.markdown(f"**{q.name}** — min {q.min_queue_length} ppl, wait ≥ {q.wait_threshold_seconds}s")
            st.caption("Queue status will be displayed in the Live tab.")

    with st.expander("📦 Object Monitor (left-behind / removed)", expanded=False):
        st.checkbox("Enable object monitor", False, key="obj_enabled")
        if st.session_state.get("obj_enabled", False):
            _, _, shelves = _get_object_detector()
            st.markdown("**Active shelves:**")
            for s in shelves:
                st.markdown(f"- {s.name} (id: {s.shelf_id})")
            st.caption("Object events will appear in the Suspicious Activity feed.")

    with st.expander("😶 Face Blur (Privacy)", expanded=False):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.checkbox("Enable face blur", False, key="face_blur_enabled")
        with c2:
            st.radio("Method", ["opencv", "mediapipe"],
                     horizontal=True, key="face_blur_method")
        st.caption("Faces are Gaussian-blurred before display / saving.")

    with st.expander("🎥 Video Clipper", expanded=False):
        st.checkbox("Enable video clipper", False, key="clip_enabled")
        if st.session_state.get("clip_enabled", False):
            c1, c2 = st.columns(2)
            with c1:
                st.slider("Pre-event seconds", 1, 60, 10, key="clip_pre")
            with c2:
                st.slider("Post-event seconds", 1, 60, 10, key="clip_post")
            st.info("Clips are saved to `data/clips/<alert_id>.mp4` on each alert.")

    with st.expander("🧍‍♀️ Pose Detectors (Gesture / Fall / Fight)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.checkbox("Gesture", True, key="pose_gesture")
        with c2:
            st.checkbox("Fall", True, key="pose_fall")
        with c3:
            st.checkbox("Fight", True, key="pose_fight")
        st.caption("Requires MediaPipe (Enable Pose in the sidebar).")

    # Save state for use elsewhere
    st.session_state["_advanced_state"] = state
    return state
