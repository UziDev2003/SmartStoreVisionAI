"""
Smart Store Vision AI - Streamlit App (v4 with batch video processing)
======================================================================
- **Video Processing** mode: upload a video → process entire file → download
  results. Much faster than frame-by-frame streaming.
- **Live Stream** mode: webcam / RTSP with frame dropping for smooth
  real-time playback.
- Clean modern UI with processing stats.
"""
import torch
# Patch torch.classes.__path__ to prevent Streamlit's file watcher from raising warnings/errors
try:
    torch.classes.__path__ = []
except AttributeError:
    pass

import streamlit as st
import cv2, time, tempfile, os
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from collections import deque
import threading
import logging
import warnings

# Initialize Streamlit page config FIRST - before any other Streamlit calls
st.set_page_config(page_title="Smart Store Vision AI", page_icon="🏪", layout="wide")

# NOW set up logging and warnings AFTER page config
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("streamlit.runtime.memory_media_file_storage").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.media_file_manager").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Initialize session state IMMEDIATELY after set_page_config
# ---------------------------------------------------------------------------
def _init_session():
    defaults = {
        "run": False,
        "stats": [],
        "am": None,
        "hm": None,
        "zA": None,
        "zones": None,
        "sus_detector": None,
        "pose_analyzer": None,
        "uploaded_video_path": None,
        "activities_log": [],
        "frame_skip": 2,
        "last_processed_time": 0.0,
        "frame_count": 0,
        "processed_video_path": None,
        # Advanced analytics module keys used by _get_* functions
        "pp_enabled": True,
        "pp_w": 0,
        "pp_h": 0,
        "pp_letterbox": False,
        "pp_clahe": False,
        "pp_clahe_clip": 2.0,
        "pp_wb": False,
        "pp_gamma": 1.0,
        "pp_blur": 0,
        "pp_sharpen": False,
        "pp_denoise": False,
        "pp_rgb": False,
        "bg_enabled": False,
        "bg_method": "MOG2",
        "bg_history": 500,
        "line_enabled": False,
        "queue_enabled": False,
        "obj_enabled": False,
        "face_blur_enabled": False,
        "face_blur_method": "opencv",
        "clip_enabled": False,
        "clip_pre": 10,
        "clip_post": 10,
        "pose_gesture": True,
        "pose_fall": True,
        "pose_fight": True,
        # Queue/live processing state
        "last_queue_status": [],
        "last_object_events": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()

# Now apply the CSS after session init
st.markdown(
    """<style>
/* Modern dark theme */
.stApp { background: #0a0a1a; color: #ffffff; }
[data-testid="stSidebar"] { background-color: #1a1a2e !important; }
h1, h2, h3, h4, h5, h6, p, label { color: #ffffff !important; }
.stMarkdown p { color: #ffffff !important; }

/* Selectbox dropdown fix */
div[data-baseweb="select"] > div { background-color: #1a1a2e; }
div[data-baseweb="popover"] ul { background-color: #1a1a2e; }
div[data-baseweb="popover"] li { color: #ffffff; }
div[data-baseweb="popover"] li:hover { background-color: #00d9ff; color: #000000; }
div[data-baseweb="popover"] li[aria-selected="true"] { background-color: #0f3460; color: #ffffff; }

/* Modal / Exception dialog fix */
div[data-baseweb="modal"] section { background-color: #1a1a2e !important; color: #ffffff !important; }
div[data-baseweb="modal"] div { color: #ffffff; }

.stMetric>div:first-child{color:#00d9ff!important;font-size:1.5rem!important}
.stMetric>div:nth-child(2){color:#ffffff!important}
.zone{background:#0f3460;color:#ffffff;padding:10px;border-radius:8px;margin:5px 0}
.severity-LOW{color:#3b82f6;font-weight:bold}
.severity-MEDIUM{color:#f59e0b;font-weight:bold}
.severity-HIGH{color:#ef4444;font-weight:bold}
.severity-CRITICAL{color:#7c2d12;font-weight:bold}
/* Card style for results */
.result-card{background:#1a1a2e;border-radius:12px;padding:20px;margin:10px 0;border:1px solid #2a2a4a}
.stButton>button{background:#0f3460;color:white;border-radius:8px;border:none;padding:8px 20px}
.stButton>button:hover{background:#1a4a8a}
/* Tabs and expanders */
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { color: #ffffff !important; }
.streamlit-expanderHeader { color: #ffffff !important; }
[data-testid="stExpander"] details summary p { color: #ffffff !important; }
</style>""",
    unsafe_allow_html=True,
)

# Add path for local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import all required modules AFTER session init
from src.detector.yolo_detector import YOLODetector
from src.tracker.bytetrack import ByteTracker
from src.analytics.zone_analytics import ZoneAnalytics, Zone
from src.analytics.heatmap import HeatmapGenerator
from src.analytics.behavior_detector import BehaviorDetector
from src.alerts.alert_manager import AlertManager
from src.analytics.suspicious_activity import (
    SuspiciousActivityDetector, SuspiciousActivity, ActivityType, Severity, SEVERITY_RANK,
)
from src.utils.zone_io import load_zones, save_zones, list_runtime_zone_files
from dashboard.polygon_editor import render as render_polygon_editor
from dashboard.alerts_panel import render as render_alerts_panel
from dashboard.threshold_tuner import render as render_threshold_tuner
from dashboard.advanced_analytics import (
    render as render_advanced_analytics,
    _get_preprocessor, _get_bg_subtractor, _get_line_detector,
    _get_queue_detector, _get_object_detector, _get_face_blurrer,
    _get_video_clipper, _get_pose_detectors,
)
from src.analytics.pose_detectors import PoseEvent


# ---------------------------------------------------------------------------
# Resource singletons
# ---------------------------------------------------------------------------
@st.cache_resource
def get_det(ms="s", cf=0.5, dv="cpu"):
    return YOLODetector(model_size=ms, confidence_threshold=cf, device=dv)


def _build_suspicious_detector(enable_pose: bool) -> Optional[SuspiciousActivityDetector]:
    pose = None
    if enable_pose:
        try:
            from src.utils.config_schema import PoseConfig
            from src.analytics.pose_analyzer import PoseAnalyzer
            pose = PoseAnalyzer(PoseConfig(enabled=True))
        except Exception:
            pose = None
    try:
        return SuspiciousActivityDetector.from_config_path(
            "config/suspicious_activity.yaml", pose_analyzer=pose,
        )
    except Exception as e:
        st.error(f"Failed to build suspicious detector: {e}")
        return None


def _default_zones(width: int = 960, height: int = 540) -> List[Zone]:
    sx, sy = width / 1920, height / 1080
    def s(p): return (int(p[0] * sx), int(p[1] * sy))
    return [
        Zone("entrance", "Store Entrance", "entrance",
             [s((100, 800)), s((400, 800)), s((400, 1080)), s((100, 1080))], 10, 120),
        Zone("checkout", "Checkout", "checkout",
             [s((700, 900)), s((1200, 900)), s((1200, 1080)), s((700, 1080))], 8, 180),
        Zone("restricted", "Restricted", "restricted",
             [s((800, 100)), s((950, 100)), s((950, 250)), s((800, 250))], 0, 0),
        Zone("shelf_a", "Shelf A", "shelf",
             [s((450, 200)), s((800, 200)), s((800, 650)), s((450, 650))], 5, 240),
    ]


# ---------------------------------------------------------------------------
# Batch Video Processing (for uploaded files)
# ---------------------------------------------------------------------------
def process_video_batch(
    video_path: str,
    output_path: str,
    det,
    trk,
    zones: List[Zone],
    zA: ZoneAnalytics,
    hm: HeatmapGenerator,
    am: AlertManager,
    sus_detector,
    bhv: BehaviorDetector,
    show_hm: bool,
    enable_pose: bool,
    preproc,
    bg_sub,
    line_det,
    queue_det,
    olb_det, orm_det, obj_shelves,
    face_blur,
    video_clip,
    gest_d, fall_d, fight_d,
    progress_callback=None,
    frame_callback=None,
) -> dict:
    """
    Process an entire video file in batch mode.
    Returns stats dict.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    # Use H.264 (avc1) for browser-streamable fragmented MP4.
    # This fixes Cloudflare tunnel "stream canceled" errors because the file
    # can be played progressively instead of requiring the full file first.
    # Fall back to mp4v if avc1 is not available (rare on Colab/Linux).
    out_w, out_h = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        # Fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    fc = 0
    start_time = time.time()
    last_frame_time = 0.0
    activities_log = []
    last_queue_status = []
    last_object_events = []

    while True:
        ret, fr = cap.read()
        if not ret:
            break

        fc += 1
        fr = cv2.resize(fr, (out_w, out_h))
        ts = time.time()

        # --- Preprocessing ---
        if preproc is not None:
            try: fr = preproc(fr)
            except Exception: pass
        if face_blur is not None:
            try: fr = face_blur.blur(fr)
            except Exception: pass
        if video_clip is not None:
            try: video_clip.add_frame(fr, ts)
            except Exception: pass
        bg_stats = None
        if bg_sub is not None:
            try: bg_stats = bg_sub.apply(fr)
            except Exception: pass

        # --- Core detection & tracking ---
        dets = det.detect(fr)
        res = trk.update(fr, dets, ts)
        zA.update(res.tracks, ts)
        hm.update(res.tracks, ts)

        cfg = {z.zone_id: {"name": z.name, "dwell_threshold": z.dwell_threshold}
               for z in zones}

        # Line crossing
        if line_det is not None:
            try: line_det.update(res.tracks, ts)
            except Exception: pass
        # Queue detection
        if queue_det is not None:
            try: last_queue_status = queue_det.update(res.tracks, zA.zone_occ, ts)
            except Exception: pass
        # Object monitor
        obj_events = []
        if olb_det is not None and obj_shelves:
            try: obj_events.extend(olb_det.update(fr, res.tracks, zA.zone_occ, obj_shelves, ts))
            except Exception: pass
        if orm_det is not None and obj_shelves:
            try: obj_events.extend(orm_det.update(fr, res.tracks, zA.zone_occ, obj_shelves, ts))
            except Exception: pass
        if obj_events:
            last_object_events.extend(obj_events)
            last_object_events = last_object_events[-200:]

        # Pose
        pose_ests = {}
        if enable_pose and st.session_state.get("pose_analyzer"):
            try: pose_ests = st.session_state["pose_analyzer"].estimate(fr, res.tracks) or {}
            except Exception: pass
        for ev in (gest_d.update(pose_ests, res.tracks, ts) if gest_d else []):
            activities_log.append(ev)
        for ev in (fall_d.update(res.tracks, pose_ests, ts) if fall_d else []):
            activities_log.append(ev)
        for ev in (fight_d.update(res.tracks, pose_ests, ts) if fight_d else []):
            activities_log.append(ev)

        # Suspicious activity
        if sus_detector is not None:
            try:
                activities = sus_detector.detect(
                    tracks=res.tracks,
                    zones={z.zone_id: z for z in zones},
                    zone_occ=zA.zone_occ,
                    timestamp=ts, frame=fr, detections=dets,
                    camera_id="v1",
                )
                for a in activities:
                    track = next((t for t in res.tracks if t.track_id == a.track_id), None)
                    bbox = track.bbox if track else None
                    am.create_from_activity(a, "v1", fr, bbox)
                    if video_clip is not None:
                        try: video_clip.save_clip(a.alert_id, fr)
                        except Exception: pass
                activities_log.extend(activities)
                if len(activities_log) > 500:
                    activities_log = activities_log[-500:]
            except Exception:
                pass

        # Behavior detector
        for alert in bhv.detect(res.tracks, cfg, zA.zone_occ, ts, fr, "v1", dets):
            t = next((tr for tr in res.tracks if tr.track_id == alert.track_id), None)
            am.create(alert.alert_type.value, "v1", alert.track_id,
                      alert.message, alert.severity.value,
                      alert.zone_id, fr, t.bbox if t else None)

        # --- Drawing ---
        out = det.draw_detections(fr.copy(), dets)
        out = trk.draw_tracks(out, res.tracks)
        out = zA.draw_zones(out)
        if show_hm:
            out = hm.get_overlay(out, 0.3)

        # BG mask
        if bg_stats is not None and bg_stats.fg_mask is not None and bg_stats.fg_mask.size > 0:
            try:
                small = cv2.resize(bg_stats.fg_mask, (160, 90))
                rgb_mask = cv2.applyColorMap(small, cv2.COLORMAP_JET)
                out[0:90, out.shape[1]-160:out.shape[1]] = rgb_mask
                cv2.putText(out, f"Motion: {bg_stats.motion_density*100:.0f}%",
                            (out.shape[1]-160, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            except Exception:
                pass
        # Line zones
        if line_det is not None:
            for line in line_det.lines.values():
                try:
                    cv2.line(out, line.p1, line.p2, (0, 255, 255), 2)
                    cv2.putText(out, line.name, line.p1,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                except Exception:
                    pass
        # Queue polygons
        if queue_det is not None:
            for q in queue_det.queues.values():
                try:
                    pts = np.array(q.polygon, dtype=np.int32)
                    cv2.polylines(out, [pts], True, (255, 200, 0), 2)
                except Exception:
                    pass
        # Shelf polygons
        for s in obj_shelves:
            try:
                pts = np.array(s.polygon, dtype=np.int32)
                cv2.polylines(out, [pts], True, (200, 100, 200), 2)
            except Exception:
                pass

        s_stats = det.get_performance_stats()
        cv2.putText(out,
                    f"FPS:{s_stats['fps']:.0f} T:{len(res.tracks)} F:{hm.total_footfall}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(out,
                    f"Suspicious: {len(activities_log)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        cv2.putText(out,
                    f"Frame: {fc}/{total_frames}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(out)

        # ONLY call frame_callback if provided AND throttled.
        # For batch processing, the user can disable preview entirely for max speed.
        if frame_callback:
            now = time.time()
            if now - last_frame_time >= 0.066:  # ~15 FPS preview cap
                try:
                    frame_callback(out)
                except Exception:
                    pass
                last_frame_time = now

        # Progress
        if progress_callback and fc % 30 == 0:
            elapsed = time.time() - start_time
            progress_callback(fc / total_frames, {
                "frame": fc, "total": total_frames,
                "fps": fc / max(elapsed, 0.001),
                "tracks": len(res.tracks),
                "footfall": hm.total_footfall,
                "suspicious": len(activities_log),
            })

    cap.release()
    writer.release()
    elapsed = time.time() - start_time

    return {
        "total_frames": total_frames,
        "processed_frames": fc,
        "elapsed_seconds": elapsed,
        "avg_fps": fc / max(elapsed, 0.001),
        "total_footfall": hm.total_footfall,
        "suspicious_count": len(activities_log),
        "output_path": output_path,
        "activities_log": activities_log,
        "last_queue_status": last_queue_status,
        "last_object_events": last_object_events,
    }


# ---------------------------------------------------------------------------
# Adaptive frame processing (for live streams)
# ---------------------------------------------------------------------------
def _calculate_target_fps(processing_time, min_fps=5, max_fps=30):
    if processing_time <= 0:
        return max_fps
    target = 1.0 / processing_time
    return max(min_fps, min(max_fps, target))


def _should_process_frame(frame_idx, frame_skip):
    return frame_idx % (frame_skip + 1) == 0


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
st.markdown("## 🏪 Smart Store Vision AI")
st.caption("YOLOv8 + ByteTrack — Batch Video Processing & Live Stream")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    ms = st.selectbox("Model", ["n", "s", "m"], 1, key="cfg_model")
    cf = st.slider("Confidence", 0.2, 0.9, 0.5, 0.05, key="cfg_conf")
    device_options = ["cpu"]
    if torch.cuda.is_available():
        device_options.append("cuda")
    default_device_idx = device_options.index("cuda") if "cuda" in device_options else 0
    dv = st.selectbox("Device", device_options, default_device_idx, key="cfg_device")
    
    if "cuda" not in device_options:
        st.warning("⚠️ CUDA GPU not detected. Running on CPU (slow).")
        
    loiter = st.slider("Loiter (s)", 30, 300, 120, 10, key="cfg_loiter")
    show_hm = st.checkbox("Show Heatmap", False, key="cfg_showhm")
    enable_pose = st.checkbox("Enable Pose (MediaPipe)", False, key="cfg_pose")
    enable_multi = st.checkbox("Multi-Camera Re-ID", False, key="cfg_mcam")

    st.markdown("---")
    st.subheader("⚡ Performance")
    frame_skip = st.slider("Frame Skip (live only)", 0, 10, 2,
                           key="cfg_frame_skip",
                           help="For live streams only. Higher = smoother display.")
    st.session_state["frame_skip"] = frame_skip

# Tabs
tab_live, tab_zones, tab_adv, tab_sus, tab_tune, tab_dash = st.tabs(
    ["📹 Live", "🗺️ Zone Editor", "🛠️ Advanced",
     "🚨 Activity", "🎛️ Tuner", "📊 Dashboard"]
)

# --- Tab: Live ---
with tab_live:
    # Mode selector
    mode = st.radio(
        "Processing Mode",
        ["🎬 Batch Video Processing", "📡 Live Stream (Webcam/RTSP)"],
        horizontal=True,
        key="live_mode",
        help="Batch: upload a video file and get a processed output video. "
             "Live: real-time webcam or RTSP stream."
    )

    if mode == "🎬 Batch Video Processing":
        # ===== BATCH MODE =====
        st.markdown("""
        <div class="result-card">
        <h3>🎬 Batch Video Processing</h3>
        <p>Upload a video file. The system will process the entire video and
        produce an output video with detections, tracking, zones, and analytics
        drawn on it. Much faster than frame-by-frame streaming.</p>
        </div>
        """, unsafe_allow_html=True)

        up = st.file_uploader("Choose video file", type=["mp4", "avi", "mov"],
                              key="batch_uploader")
        if up:
            # Save uploaded file
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tf.write(up.read())
            tf.close()
            input_path = tf.name
            st.session_state["uploaded_video_path"] = input_path

            # Output path
            output_path = tempfile.mktemp(suffix=".mp4", prefix="processed_")

            # Show input info
            cap_check = cv2.VideoCapture(input_path)
            if cap_check.isOpened():
                total = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
                vid_fps = cap_check.get(cv2.CAP_PROP_FPS) or 25.0
                dur = total / vid_fps if vid_fps > 0 else 0
                st.info(f"📹 **{up.name}** — {total} frames, {vid_fps:.1f} FPS, "
                        f"{dur:.1f}s duration")
            cap_check.release()

            # Zone loading
            cam_id = st.text_input("Camera ID for zone loading", value="v1",
                                   key="batch_cam")
            existing_zones = []
            try:
                existing_zones = load_zones(camera_id=cam_id)
            except Exception:
                pass
            if not existing_zones:
                try:
                    existing_zones = load_zones(path=Path("config/zones.yaml"))
                except Exception:
                    pass
            if not existing_zones:
                existing_zones = _default_zones(960, 540)
                st.caption("Using default zones (no saved zones found)")

            # Performance options for batch processing
            st.markdown("##### ⚙️ Batch Options")
            show_preview = st.checkbox(
                "📺 Show Live Preview While Processing",
                value=False,
                key="batch_show_preview",
                help="Disable for MAXIMUM processing speed. The output video will "
                     "still have all the detections drawn on it. Re-enable to watch "
                     "progress live (slower)."
            )
            preview_every = st.slider(
                "Preview every N frames (if enabled)",
                1, 60, 10,
                key="batch_preview_every",
                help="Higher = faster but jumpier preview. Only used when preview is ON."
            )

            # Process button
            if st.button("▶️ Process Video", type="primary", use_container_width=True):
                try:
                    det = get_det(ms, cf, dv)
                except Exception as e:
                    st.error(f"Model error: {e}")
                    det = None

                if det:
                    # Initialize components
                    trk = ByteTracker()
                    zones = existing_zones
                    zA = ZoneAnalytics(zones, 960, 540)
                    hm = HeatmapGenerator(960, 540)
                    am = AlertManager()
                    sus_det = _build_suspicious_detector(enable_pose)
                    bhv = BehaviorDetector(loiter_threshold=loiter,
                                           suspicious_detector=sus_det)

                    preproc = _get_preprocessor() if st.session_state.get("pp_enabled", True) else None
                    bg_sub = _get_bg_subtractor()
                    line_det = _get_line_detector()
                    queue_det = _get_queue_detector()
                    olb_det, orm_det, obj_shelves = _get_object_detector()
                    face_blur = _get_face_blurrer()
                    video_clip = _get_video_clipper()
                    gest_d, fall_d, fight_d = _get_pose_detectors()

                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    stats_cols = st.columns(4)
                    fps_m = stats_cols[0].empty()
                    frame_m = stats_cols[1].empty()
                    foot_m = stats_cols[2].empty()
                    sus_m = stats_cols[3].empty()

                    def on_progress(pct, info):
                        progress_bar.progress(pct)
                        status_text.markdown(
                            f"Processing frame **{info['frame']}/{info['total']}** "
                            f"| FPS: {info['fps']:.1f} | "
                            f"Tracks: {info['tracks']} | "
                            f"Footfall: {info['footfall']}"
                        )
                        fps_m.metric("Processing FPS", f"{info['fps']:.1f}")
                        frame_m.metric("Frame", f"{info['frame']}/{info['total']}")
                        foot_m.metric("Footfall", info['footfall'])
                        sus_m.metric("Suspicious", info['suspicious'])
                        
                    # Only create video_placeholder if preview is enabled
                    video_placeholder = st.empty() if show_preview else None
                    preview_counter = {"n": 0}
                    
                    def on_frame(frame):
                        if not show_preview:
                            return
                        # Only display every N frames to avoid saturating Streamlit
                        preview_counter["n"] += 1
                        if preview_counter["n"] % max(1, preview_every) != 0:
                            return
                        try:
                            # Resize for faster display
                            small = cv2.resize(frame, (480, 270))
                            video_placeholder.image(small, channels="BGR", use_column_width=True)
                        except Exception:
                            pass

                    # Run batch processing
                    result = process_video_batch(
                        video_path=input_path,
                        output_path=output_path,
                        det=det, trk=trk,
                        zones=zones, zA=zA, hm=hm, am=am,
                        sus_detector=sus_det, bhv=bhv,
                        show_hm=show_hm, enable_pose=enable_pose,
                        preproc=preproc, bg_sub=bg_sub,
                        line_det=line_det, queue_det=queue_det,
                        olb_det=olb_det, orm_det=orm_det, obj_shelves=obj_shelves,
                        face_blur=face_blur, video_clip=video_clip,
                        gest_d=gest_d, fall_d=fall_d, fight_d=fight_d,
                        progress_callback=on_progress,
                        frame_callback=on_frame,
                    )

                    progress_bar.progress(1.0)
                    status_text.success("✅ Processing complete!")

                    # Store results
                    st.session_state["processed_video_path"] = output_path
                    st.session_state["stats"] = [{
                        "fps": result["avg_fps"],
                        "tr": 0,
                        "fh": result["total_footfall"],
                        "proc_ms": (result["elapsed_seconds"] / max(result["processed_frames"], 1)) * 1000,
                    }]
                    st.session_state["hm"] = hm
                    st.session_state["am"] = am
                    st.session_state["activities_log"] = result["activities_log"]
                    st.session_state["zones"] = zones
                    st.session_state["zA"] = zA

                    # Show results
                    st.markdown("---")
                    st.markdown("### 📊 Processing Results")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Total Frames", result["processed_frames"])
                    r2.metric("Avg FPS", f"{result['avg_fps']:.1f}")
                    r3.metric("Duration", f"{result['elapsed_seconds']:.1f}s")
                    r4.metric("Footfall", result["total_footfall"])

                    # Show processed video
                    # File-size aware: only stream inline if < 30MB to avoid
                    # Cloudflare tunnel "stream canceled" errors on large media.
                    st.markdown("### 🎬 Processed Output")
                    try:
                        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    except OSError:
                        file_size_mb = 0
                    if file_size_mb < 30:
                        st.success(
                            f"📦 Output size: {file_size_mb:.1f} MB — streaming inline"
                        )
                        st.video(output_path)
                    else:
                        st.warning(
                            f"📦 Output size: {file_size_mb:.1f} MB — too large to "
                            f"stream through the tunnel. Use the download button "
                            f"below to save it locally."
                        )

                    # Download button (always available - works for any file size)
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Processed Video",
                            f,
                            file_name=f"processed_{up.name}",
                            mime="video/mp4",
                            use_container_width=True,
                        )

                    # Clean up input temp file
                    try:
                        os.unlink(input_path)
                    except Exception:
                        pass

        else:
            st.info("👆 Upload a video file to begin batch processing")

    else:
        # ===== LIVE STREAM MODE =====
        st.markdown("""
        <div class="result-card">
        <h3>📡 Live Stream</h3>
        <p>Connect to a webcam or RTSP camera for real-time analysis.
        Use <strong>Frame Skip</strong> in the sidebar to adjust performance.</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c2:
            st.subheader("📊 Stats")
            fps_metric = st.empty()
            tracks_metric = st.empty()
            footfall_metric = st.empty()
            alerts_metric = st.empty()
            sus_metric = st.empty()
            proc_time_metric = st.empty()
            fps_metric.metric("Display FPS", "0.0")
            tracks_metric.metric("Tracks", "0")
            footfall_metric.metric("Footfall", "0")
            alerts_metric.metric("Alerts", "0")
            sus_metric.metric("Suspicious", "0")
            proc_time_metric.metric("Proc Time", "0ms")

            st.subheader("📍 Zones")
            zones_container = st.empty()
            st.subheader("⚠️ Alerts")
            alerts_container = st.empty()
            st.subheader("🗺️ Heatmap")
            heatmap_container = st.empty()

        with c1:
            src_type = st.radio("Source Type", ["Webcam", "RTSP URL"],
                                horizontal=True, key="live_src_type")

            video_source = None
            if src_type == "Webcam":
                camera_idx = st.number_input("Camera index", 0, 10, 0, 1, key="cam_idx")
                video_source = camera_idx
                st.caption(f"Webcam #{camera_idx}")
            else:
                rtsp_url = st.text_input("RTSP URL",
                                          "rtsp://admin:password@192.168.1.100:554/stream",
                                          key="rtsp_url")
                if rtsp_url and rtsp_url.startswith("rtsp://"):
                    video_source = rtsp_url
                else:
                    st.info("Enter an RTSP URL to stream from an IP camera")

            b1, b2 = st.columns(2)
            with b1:
                start_enabled = video_source is not None
                if st.button("▶️ Start", type="primary", use_container_width=True,
                             key="start_live", disabled=not start_enabled):
                    st.session_state["run"] = True
            with b2:
                if st.button("⏹️ Stop", use_container_width=True, key="stop_live"):
                    st.session_state["run"] = False

            fp = st.empty()

            if st.session_state["run"] and video_source is not None:
                try:
                    det = get_det(ms, cf, dv)
                except Exception as e:
                    st.error(f"Model error: {e}")
                    det = None

                if det:
                    st.session_state["stats"] = []
                    st.session_state["zones"] = _default_zones(960, 540)
                    st.session_state["zA"] = ZoneAnalytics(st.session_state["zones"], 960, 540)
                    st.session_state["hm"] = HeatmapGenerator(960, 540)
                    st.session_state["am"] = AlertManager()
                    st.session_state["sus_detector"] = _build_suspicious_detector(enable_pose)
                    st.session_state["activities_log"] = []
                    st.session_state["last_queue_status"] = []
                    st.session_state["last_object_events"] = []

                    preproc = _get_preprocessor() if st.session_state.get("pp_enabled", True) else None
                    bg_sub = _get_bg_subtractor()
                    line_det = _get_line_detector()
                    queue_det = _get_queue_detector()
                    olb_det, orm_det, obj_shelves = _get_object_detector()
                    face_blur = _get_face_blurrer()
                    video_clip = _get_video_clipper()
                    gest_d, fall_d, fight_d = _get_pose_detectors()

                    trk = ByteTracker()
                    bhv = BehaviorDetector(loiter_threshold=loiter,
                                           suspicious_detector=st.session_state["sus_detector"])

                    cap = cv2.VideoCapture(video_source)
                    if not cap.isOpened():
                        st.error("Failed to open video source")
                        st.session_state["run"] = False
                    else:
                        fc = 0
                        last_display_time = time.time()
                        frame_times = deque(maxlen=30)
                        proc_times = deque(maxlen=30)
                        last_frame = None

                        while st.session_state["run"]:
                            loop_start = time.time()
                            ret, fr = cap.read()
                            if not ret:
                                st.warning("Lost connection, reconnecting...")
                                time.sleep(1)
                                cap.release()
                                cap = cv2.VideoCapture(video_source)
                                if not cap.isOpened():
                                    break
                                continue

                            fc += 1
                            fr = cv2.resize(fr, (960, 540))
                            ts = time.time()

                            avg_proc = np.mean(proc_times) if proc_times else 0.033
                            target_fps = _calculate_target_fps(avg_proc)
                            should_process = _should_process_frame(fc, st.session_state["frame_skip"])

                            if should_process:
                                proc_start = time.time()

                                if preproc is not None:
                                    try: fr = preproc(fr)
                                    except Exception: pass
                                if face_blur is not None:
                                    try: fr = face_blur.blur(fr)
                                    except Exception: pass
                                if video_clip is not None:
                                    try: video_clip.add_frame(fr, ts)
                                    except Exception: pass
                                bg_stats = None
                                if bg_sub is not None:
                                    try: bg_stats = bg_sub.apply(fr)
                                    except Exception: pass

                                dets = det.detect(fr)
                                res = trk.update(fr, dets, ts)
                                st.session_state["zA"].update(res.tracks, ts)
                                st.session_state["hm"].update(res.tracks, ts)

                                cfg = {z.zone_id: {"name": z.name, "dwell_threshold": z.dwell_threshold}
                                       for z in st.session_state["zones"]}

                                if line_det is not None:
                                    try: line_det.update(res.tracks, ts)
                                    except Exception: pass
                                if queue_det is not None:
                                    try: st.session_state["last_queue_status"] = queue_det.update(
                                        res.tracks, st.session_state["zA"].zone_occ, ts)
                                    except Exception: pass
                                obj_events = []
                                if olb_det is not None and obj_shelves:
                                    try: obj_events.extend(olb_det.update(
                                        fr, res.tracks, st.session_state["zA"].zone_occ, obj_shelves, ts))
                                    except Exception: pass
                                if orm_det is not None and obj_shelves:
                                    try: obj_events.extend(orm_det.update(
                                        fr, res.tracks, st.session_state["zA"].zone_occ, obj_shelves, ts))
                                    except Exception: pass
                                if obj_events:
                                    st.session_state["last_object_events"].extend(obj_events)
                                    st.session_state["last_object_events"] = st.session_state["last_object_events"][-200:]

                                pose_ests = {}
                                if enable_pose and st.session_state.get("pose_analyzer"):
                                    try: pose_ests = st.session_state["pose_analyzer"].estimate(fr, res.tracks) or {}
                                    except Exception: pass
                                for ev in (gest_d.update(pose_ests, res.tracks, ts) if gest_d else []):
                                    st.session_state["activities_log"].append(ev)
                                for ev in (fall_d.update(res.tracks, pose_ests, ts) if fall_d else []):
                                    st.session_state["activities_log"].append(ev)
                                for ev in (fight_d.update(res.tracks, pose_ests, ts) if fight_d else []):
                                    st.session_state["activities_log"].append(ev)

                                sus_d = st.session_state["sus_detector"]
                                if sus_d is not None:
                                    try:
                                        activities = sus_d.detect(
                                            tracks=res.tracks,
                                            zones={z.zone_id: z for z in st.session_state["zones"]},
                                            zone_occ=st.session_state["zA"].zone_occ,
                                            timestamp=ts, frame=fr, detections=dets,
                                            camera_id="v1",
                                        )
                                        for a in activities:
                                            track = next((t for t in res.tracks if t.track_id == a.track_id), None)
                                            bbox = track.bbox if track else None
                                            st.session_state["am"].create_from_activity(a, "v1", fr, bbox)
                                            if video_clip is not None:
                                                try: video_clip.save_clip(a.alert_id, fr)
                                                except Exception: pass
                                        st.session_state["activities_log"].extend(activities)
                                        if len(st.session_state["activities_log"]) > 500:
                                            st.session_state["activities_log"] = st.session_state["activities_log"][-500:]
                                    except Exception:
                                        pass

                                for alert in bhv.detect(res.tracks, cfg, st.session_state["zA"].zone_occ, ts,
                                                         fr, "v1", dets):
                                    t = next((tr for tr in res.tracks if tr.track_id == alert.track_id), None)
                                    st.session_state["am"].create(alert.alert_type.value, "v1", alert.track_id,
                                                                  alert.message, alert.severity.value,
                                                                  alert.zone_id, fr, t.bbox if t else None)

                                out = det.draw_detections(fr.copy(), dets)
                                out = trk.draw_tracks(out, res.tracks)
                                out = st.session_state["zA"].draw_zones(out)
                                if show_hm:
                                    out = st.session_state["hm"].get_overlay(out, 0.3)

                                if bg_stats is not None and bg_stats.fg_mask is not None and bg_stats.fg_mask.size > 0:
                                    try:
                                        small = cv2.resize(bg_stats.fg_mask, (160, 90))
                                        rgb_mask = cv2.applyColorMap(small, cv2.COLORMAP_JET)
                                        out[0:90, out.shape[1]-160:out.shape[1]] = rgb_mask
                                        cv2.putText(out, f"Motion: {bg_stats.motion_density*100:.0f}%",
                                                    (out.shape[1]-160, 105),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                                    except Exception:
                                        pass
                                if line_det is not None:
                                    for line in line_det.lines.values():
                                        try:
                                            cv2.line(out, line.p1, line.p2, (0, 255, 255), 2)
                                            cv2.putText(out, line.name, line.p1,
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                        except Exception:
                                            pass
                                if queue_det is not None:
                                    for q in queue_det.queues.values():
                                        try:
                                            pts = np.array(q.polygon, dtype=np.int32)
                                            cv2.polylines(out, [pts], True, (255, 200, 0), 2)
                                        except Exception:
                                            pass
                                for s in obj_shelves:
                                    try:
                                        pts = np.array(s.polygon, dtype=np.int32)
                                        cv2.polylines(out, [pts], True, (200, 100, 200), 2)
                                    except Exception:
                                        pass

                                s_stats = det.get_performance_stats()
                                proc_time = (time.time() - proc_start) * 1000
                                proc_times.append(proc_time / 1000.0)

                                cv2.putText(out,
                                            f"FPS:{s_stats['fps']:.0f} T:{len(res.tracks)} F:{st.session_state['hm'].total_footfall}",
                                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.putText(out,
                                            f"Suspicious: {len(st.session_state['activities_log'])}",
                                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                                cv2.putText(out,
                                            f"Proc: {proc_time:.0f}ms | Skip: {st.session_state['frame_skip']}",
                                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                                last_frame = out
                                st.session_state["last_processed_time"] = time.time()

                            display_frame = last_frame if last_frame is not None else fr
                            fp.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                                     channels="RGB", use_column_width=True)

                            frame_times.append(time.time() - last_display_time)
                            last_display_time = time.time()

                            if should_process:
                                st.session_state["stats"].append({
                                    "fps": s_stats["fps"], "tr": len(res.tracks),
                                    "fh": st.session_state["hm"].total_footfall,
                                    "proc_ms": proc_time,
                                })

                            if fc % 5 == 0 and should_process:
                                avg_display_fps = len(frame_times) / max(sum(frame_times), 0.001) if frame_times else 0
                                avg_proc_ms = np.mean(proc_times) * 1000 if proc_times else 0

                                fps_metric.metric("Display FPS", f"{avg_display_fps:.1f}")
                                tracks_metric.metric("Tracks", len(res.tracks))
                                footfall_metric.metric("Footfall", st.session_state["hm"].total_footfall)
                                alerts_metric.metric("Alerts", len(st.session_state["am"].get_recent(60)))
                                sus_metric.metric("Suspicious", len(st.session_state["activities_log"]))
                                proc_time_metric.metric("Proc Time", f"{avg_proc_ms:.0f}ms")

                                with zones_container.container():
                                    for zid, st_ in st.session_state["zA"].stats.items():
                                        z = next((zz for zz in st.session_state["zones"]
                                                  if zz.zone_id == zid), None)
                                        if z:
                                            c = "🔴" if z.zone_type == "restricted" else "🟡"
                                            st.markdown(f'<div class="zone">{c} {z.name}: {st_.current_occupancy}</div>',
                                                        unsafe_allow_html=True)
                                cur = st.session_state["am"].get_recent(60)
                                if cur:
                                    with alerts_container.container():
                                        for a in cur[:4]:
                                            sev_cls = f"severity-{a.severity.upper()}"
                                            st.markdown(
                                                f"**{datetime.fromtimestamp(a.timestamp).strftime('%H:%M:%S')}** "
                                                f"<span class='{sev_cls}'>[{a.severity}]</span> {a.message}",
                                                unsafe_allow_html=True)
                                else:
                                    alerts_container.info("No recent alerts")
                                if st.session_state["hm"]:
                                    heatmap_container.image(
                                        cv2.cvtColor(st.session_state["hm"].get_image(), cv2.COLOR_BGR2RGB),
                                        use_column_width=True)

                            elapsed = time.time() - loop_start
                            target_interval = 1.0 / target_fps
                            if elapsed < target_interval:
                                time.sleep(max(0.001, target_interval - elapsed))

                        cap.release()
                        st.session_state["run"] = False

# --- Tab: Zone Editor ---
with tab_zones:
    cam_id = st.text_input("Camera ID", value="v1", key="zone_cam")
    video_for_frame = st.session_state.get("uploaded_video_path")
    existing = []
    try:
        existing = load_zones(camera_id=cam_id)
    except Exception:
        pass
    if not existing:
        try:
            existing = load_zones(path=Path("config/zones.yaml"))
        except Exception:
            pass
    result = render_polygon_editor(camera_id=cam_id, video_path=video_for_frame,
                                   existing_zones=existing)
    if result.get("saved_to"):
        st.success(f"Saved to {result['saved_to']}")
    st.markdown("---")
    st.subheader("📁 Saved Runtime Zone Files")
    files = list_runtime_zone_files()
    if files:
        for p in files:
            st.markdown(f"- `{p}`")
    else:
        st.caption("No runtime zone files yet.")

# --- Tab: Advanced Analytics ---
with tab_adv:
    render_advanced_analytics()

# --- Tab: Suspicious Activity ---
with tab_sus:
    st.markdown("### 🚨 Suspicious Activity Feed")
    activities: List[SuspiciousActivity] = st.session_state.get("activities_log", [])
    window = st.slider("Time window (s)", 30, 1800, 300, 30)
    cutoff = time.time() - window
    activities = [a for a in activities if a.timestamp >= cutoff]
    render_alerts_panel(activities)

    line_det = _get_line_detector()
    if line_det is not None:
        st.markdown("---")
        st.markdown("### ➡️ Line Crossing Counts")
        counts = line_det.get_counts()
        if counts:
            for (lid, d), c in counts.items():
                st.markdown(f"- **{lid}** ({d}): `{c}`")
        else:
            st.caption("No crossings yet.")

    queue_det = _get_queue_detector()
    if queue_det is not None and st.session_state.get("last_queue_status"):
        st.markdown("---")
        st.markdown("### 🧍 Queue Status")
        for s in st.session_state["last_queue_status"]:
            color = "🟢" if not s.is_queue else "🟠" if s.length < 4 else "🔴"
            st.markdown(
                f"{color} **{s.name}** — {s.length} ppl, "
                f"est wait **{s.estimated_wait_seconds:.0f}s**, "
                f"avg **{s.avg_wait_seconds:.0f}s**"
            )

    if st.session_state.get("last_object_events"):
        st.markdown("---")
        st.markdown("### 📦 Object Monitor Events")
        for e in st.session_state["last_object_events"][-10:]:
            icon = "🟡" if e.event_type == "object_left" else "🔴"
            st.markdown(f"{icon} {e.message}")

# --- Tab: Tuner ---
with tab_tune:
    render_threshold_tuner()

# --- Tab: Dashboard ---
with tab_dash:
    st.markdown("### 📊 Analytics Dashboard")
    st.caption("Quick view of recent activity.")
    if st.session_state.get("stats"):
        df_stats = st.session_state["stats"][-200:]
        fps_vals = [s["fps"] for s in df_stats]
        proc_vals = [s.get("proc_ms", 0) for s in df_stats]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg FPS", f"{sum(fps_vals) / max(1, len(fps_vals)):.1f}")
        c2.metric("Peak FPS", f"{max(fps_vals) if fps_vals else 0:.1f}")
        c3.metric("Total Footfall", st.session_state["hm"].total_footfall if st.session_state["hm"] else 0)
        c4.metric("Suspicious (recent)",
                  sum(1 for a in st.session_state.get("activities_log", [])
                      if a.timestamp >= time.time() - 300))
        if proc_vals:
            c5, c6 = st.columns(2)
            c5.metric("Avg Proc Time", f"{sum(proc_vals) / max(1, len(proc_vals)):.0f}ms")
            c6.metric("Frame Skip", st.session_state.get("frame_skip", 2))
    else:
        st.info("Run the live analysis to populate dashboard stats.")

    st.markdown("---")
    st.markdown("### 📦 Module Status")
    cm1, cm2, cm3 = st.columns(3)
    cm1.markdown("**Detection / Tracking**")
    cm1.markdown("- ✅ YOLOv8 person detection")
    cm1.markdown("- ✅ ByteTrack tracking")
    cm1.markdown("- ✅ Re-ID (optional)")
    cm1.markdown("- ✅ Heatmap / footfall")
    cm2.markdown("**Advanced Analytics**")
    cm2.markdown(f"- {'✅' if st.session_state.get('pp_enabled', True) else '⬜'} Frame preprocessor")
    cm2.markdown(f"- {'✅' if st.session_state.get('bg_enabled') else '⬜'} Background subtraction")
    cm2.markdown(f"- {'✅' if st.session_state.get('line_enabled') else '⬜'} Line crossing")
    cm2.markdown(f"- {'✅' if st.session_state.get('queue_enabled') else '⬜'} Queue detection")
    cm2.markdown(f"- {'✅' if st.session_state.get('obj_enabled') else '⬜'} Object monitor")
    cm3.markdown("**Privacy / Output**")
    cm3.markdown(f"- {'✅' if st.session_state.get('face_blur_enabled') else '⬜'} Face blur")
    cm3.markdown(f"- {'✅' if st.session_state.get('clip_enabled') else '⬜'} Video clipper")
    cm3.markdown(f"- {'✅' if st.session_state.get('pose_gesture') else '⬜'} Gesture detection")
    cm3.markdown(f"- {'✅' if st.session_state.get('pose_fall') else '⬜'} Fall detection")
    cm3.markdown(f"- {'✅' if st.session_state.get('pose_fight') else '⬜'} Fight detection")

    clips_dir = Path("data/clips")
    if clips_dir.exists():
        clips = sorted(clips_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if clips:
            st.markdown("---")
            st.markdown("### 🎥 Recent Clips")
            for c in clips[:5]:
                st.markdown(f"- `{c.name}` ({c.stat().st_size // 1024} KB)")