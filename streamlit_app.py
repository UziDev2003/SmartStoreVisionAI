"""
Smart Store Vision AI - Streamlit App (v2 with tabs and polygon editor)
======================================================================
Adds tabs:
  - 📹 Live Analysis
  - 🗺️ Zone Editor (interactive polygon drawing)
  - 🚨 Suspicious Activity
  - 🎛️ Threshold Tuner
  - 📊 Analytics Dashboard
"""
import streamlit as st
import cv2, time, tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional

st.set_page_config(page_title="Smart Store Vision AI", page_icon="🏪", layout="wide")
st.markdown(
    """<style>
.stMetric>div:first-child{color:#00d9ff!important;font-size:1.5rem!important}
.stMetric>div:nth-child(2){color:#ffffff!important}
.zone{background:#0f3460;color:#ffffff;padding:10px;border-radius:8px;margin:5px 0}
.severity-LOW{color:#3b82f6;font-weight:bold}
.severity-MEDIUM{color:#f59e0b;font-weight:bold}
.severity-HIGH{color:#ef4444;font-weight:bold}
.severity-CRITICAL{color:#7c2d12;font-weight:bold}
</style>""",
    unsafe_allow_html=True,
)

import sys
sys.path.insert(0, str(Path(__file__).parent))

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


def _init_session():
    defaults = {
        "run": False,
        "stats": [],
        "am": AlertManager(),
        "hm": None,
        "zA": None,
        "zones": None,
        "sus_detector": None,
        "pose_analyzer": None,
        "uploaded_video_path": None,
        "activities_log": [],   # list of SuspiciousActivity
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
    # Scale default zones to match frame size
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
# Main app
# ---------------------------------------------------------------------------
_init_session()
st.markdown("## 🏪 Smart Store Vision AI")
st.caption("YOLOv8 + ByteTrack + Suspicious Activity Detection")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    ms = st.selectbox("Model", ["n", "s", "m"], 1, key="cfg_model")
    cf = st.slider("Confidence", 0.2, 0.9, 0.5, 0.05, key="cfg_conf")
    dv = st.selectbox("Device", ["cpu", "cuda"], 0, key="cfg_device")
    loiter = st.slider("Loiter (s)", 30, 300, 120, 10, key="cfg_loiter")
    show_hm = st.checkbox("Show Heatmap", False, key="cfg_showhm")
    enable_pose = st.checkbox("Enable Pose (MediaPipe)", False, key="cfg_pose")
    enable_multi = st.checkbox("Multi-Camera Re-ID", False, key="cfg_mcam")
    st.markdown("---")
    st.caption("Tip: use the Zone Editor tab to draw zones on a video frame.")

# Tabs
tab_live, tab_zones, tab_adv, tab_sus, tab_tune, tab_dash = st.tabs(
    ["📹 Live", "🗺️ Zone Editor", "🛠️ Advanced",
     "🚨 Activity", "🎛️ Tuner", "📊 Dashboard"]
)

# --- Tab: Live ---
with tab_live:
    c1, c2 = st.columns([3, 1])
    with c2:
        st.subheader("📊 Stats")
        fps_metric = st.empty()
        tracks_metric = st.empty()
        footfall_metric = st.empty()
        alerts_metric = st.empty()
        sus_metric = st.empty()
        fps_metric.metric("FPS", "0.0")
        tracks_metric.metric("Tracks", "0")
        footfall_metric.metric("Footfall", "0")
        alerts_metric.metric("Alerts", "0")
        sus_metric.metric("Suspicious", "0")

        st.subheader("📍 Zones")
        zones_container = st.empty()
        st.subheader("⚠️ Alerts")
        alerts_container = st.empty()
        st.subheader("🗺️ Heatmap")
        heatmap_container = st.empty()

    with c1:
        st.subheader("📹 Upload Video")
        up = st.file_uploader("Choose video", type=["mp4", "avi", "mov"],
                              label_visibility="collapsed", key="live_uploader")
        if up:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tf.write(up.read())
            tf.close()
            st.session_state["uploaded_video_path"] = tf.name
            b1, b2 = st.columns(2)
            with b1:
                if st.button("▶️ Start", type="primary", use_container_width=True, key="start_live"):
                    st.session_state["run"] = True
            with b2:
                if st.button("⏹️ Stop", use_container_width=True, key="stop_live"):
                    st.session_state["run"] = False
            fp = st.empty()
            bar = st.progress(0)
            if st.session_state["run"]:
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

                    # Build advanced detectors
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
                    cap = cv2.VideoCapture(tf.name)
                    fc = 0
                    while st.session_state["run"]:
                        ret, fr = cap.read()
                        if not ret:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        fc += 1
                        fr = cv2.resize(fr, (960, 540))
                        ts = time.time()
                        # 1) Optional frame preprocessing
                        if preproc is not None:
                            try:
                                fr = preproc(fr)
                            except Exception:
                                pass
                        # 2) Optional face blur (privacy)
                        if face_blur is not None:
                            try:
                                fr = face_blur.blur(fr)
                            except Exception:
                                pass
                        # 3) Optional video clipper buffer
                        if video_clip is not None:
                            try:
                                video_clip.add_frame(fr, ts)
                            except Exception:
                                pass
                        # 4) Optional background subtraction
                        bg_stats = None
                        if bg_sub is not None:
                            try:
                                bg_stats = bg_sub.apply(fr)
                            except Exception:
                                bg_stats = None
                        dets = det.detect(fr)
                        res = trk.update(fr, dets, ts)
                        st.session_state["zA"].update(res.tracks, ts)
                        st.session_state["hm"].update(res.tracks, ts)
                        cfg = {z.zone_id: {"name": z.name, "dwell_threshold": z.dwell_threshold}
                               for z in st.session_state["zones"]}
                        # Line crossing
                        if line_det is not None:
                            try:
                                line_det.update(res.tracks, ts)
                            except Exception:
                                pass
                        # Queue detection
                        if queue_det is not None:
                            try:
                                st.session_state["last_queue_status"] = queue_det.update(
                                    res.tracks, st.session_state["zA"].zone_occ, ts)
                            except Exception:
                                pass
                        # Object monitor
                        obj_events = []
                        if olb_det is not None and obj_shelves:
                            try:
                                obj_events.extend(olb_det.update(
                                    fr, res.tracks,
                                    st.session_state["zA"].zone_occ, obj_shelves, ts))
                            except Exception:
                                pass
                        if orm_det is not None and obj_shelves:
                            try:
                                obj_events.extend(orm_det.update(
                                    fr, res.tracks,
                                    st.session_state["zA"].zone_occ, obj_shelves, ts))
                            except Exception:
                                pass
                        if obj_events:
                            st.session_state["last_object_events"].extend(obj_events)
                            st.session_state["last_object_events"] = st.session_state["last_object_events"][-200:]
                        # Pose detectors (gesture/fall/fight)
                        pose_ests = {}
                        if enable_pose and st.session_state.get("pose_analyzer"):
                            try:
                                pose_ests = st.session_state["pose_analyzer"].estimate(
                                    fr, res.tracks) or {}
                            except Exception:
                                pose_ests = {}
                        for ev in (gest_d.update(pose_ests, res.tracks, ts) if gest_d else []):
                            st.session_state["activities_log"].append(ev)
                        for ev in (fall_d.update(res.tracks, pose_ests, ts) if fall_d else []):
                            st.session_state["activities_log"].append(ev)
                        for ev in (fight_d.update(res.tracks, pose_ests, ts) if fight_d else []):
                            st.session_state["activities_log"].append(ev)
                        # Suspicious activity
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
                                    # Save video clip on alert
                                    if video_clip is not None:
                                        try:
                                            video_clip.save_clip(a.alert_id, fr)
                                        except Exception:
                                            pass
                                st.session_state["activities_log"].extend(activities)
                                if len(st.session_state["activities_log"]) > 500:
                                    st.session_state["activities_log"] = st.session_state["activities_log"][-500:]
                            except Exception as e:
                                pass
                        # Behavior detector (legacy / fallback)
                        for alert in bhv.detect(res.tracks, cfg, st.session_state["zA"].zone_occ, ts,
                                                 fr, "v1", dets):
                            t = next((tr for tr in res.tracks if tr.track_id == alert.track_id), None)
                            st.session_state["am"].create(alert.alert_type.value, "v1", alert.track_id,
                                                          alert.message, alert.severity.value,
                                                          alert.zone_id, fr, t.bbox if t else None)
                        # Draw
                        out = det.draw_detections(fr.copy(), dets)
                        out = trk.draw_tracks(out, res.tracks)
                        out = st.session_state["zA"].draw_zones(out)
                        if show_hm:
                            out = st.session_state["hm"].get_overlay(out, 0.3)
                        # Optional: draw BG mask in corner
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
                        # Draw line zones
                        if line_det is not None:
                            for line in line_det.lines.values():
                                try:
                                    cv2.line(out, line.p1, line.p2, (0, 255, 255), 2)
                                    cv2.putText(out, line.name, line.p1,
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                except Exception:
                                    pass
                        # Draw queue polygons
                        if queue_det is not None:
                            for q in queue_det.queues.values():
                                try:
                                    pts = np.array(q.polygon, dtype=np.int32)
                                    cv2.polylines(out, [pts], True, (255, 200, 0), 2)
                                except Exception:
                                    pass
                        # Draw shelf polygons (object monitor)
                        for s in obj_shelves:
                            try:
                                pts = np.array(s.polygon, dtype=np.int32)
                                cv2.polylines(out, [pts], True, (200, 100, 200), 2)
                            except Exception:
                                pass
                        s_stats = det.get_performance_stats()
                        cv2.putText(out,
                                    f"FPS:{s_stats['fps']:.0f} T:{len(res.tracks)} F:{st.session_state['hm'].total_footfall}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(out,
                                    f"Suspicious: {len(st.session_state['activities_log'])}",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                        fp.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                                 channels="RGB", use_container_width=True)
                        bar.progress(min(fc / 100, 1.0))
                        st.session_state["stats"].append({
                            "fps": s_stats["fps"], "tr": len(res.tracks),
                            "fh": st.session_state["hm"].total_footfall,
                        })
                        # Sidebar updates
                        fps_metric.metric("FPS", f"{s_stats['fps']:.1f}")
                        tracks_metric.metric("Tracks", len(res.tracks))
                        footfall_metric.metric("Footfall", st.session_state["hm"].total_footfall)
                        alerts_metric.metric("Alerts", len(st.session_state["am"].get_recent(60)))
                        sus_metric.metric("Suspicious", len(st.session_state["activities_log"]))
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
                                use_container_width=True)
                        time.sleep(0.03)
                    cap.release()
                    bar.empty()
                    st.session_state["run"] = False
        else:
            st.info("👆 Upload a video to start")

# --- Tab: Zone Editor ---
with tab_zones:
    cam_id = st.text_input("Camera ID", value="v1", key="zone_cam")
    video_for_frame = st.session_state.get("uploaded_video_path")
    # Load existing
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

    # Line crossing counts
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

    # Queue status
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

    # Object monitor events
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
    st.caption("Quick view of recent activity. Live charts render in the Live tab.")
    if st.session_state.get("stats"):
        df_stats = st.session_state["stats"][-200:]
        fps_vals = [s["fps"] for s in df_stats]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg FPS", f"{sum(fps_vals) / max(1, len(fps_vals)):.1f}")
        c2.metric("Peak FPS", f"{max(fps_vals) if fps_vals else 0:.1f}")
        c3.metric("Total Footfall", st.session_state["hm"].total_footfall if st.session_state["hm"] else 0)
        c4.metric("Suspicious (recent)",
                  sum(1 for a in st.session_state.get("activities_log", [])
                      if a.timestamp >= time.time() - 300))
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

    # Saved clips
    clips_dir = Path("data/clips")
    if clips_dir.exists():
        clips = sorted(clips_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if clips:
            st.markdown("---")
            st.markdown("### 🎥 Recent Clips")
            for c in clips[:5]:
                st.markdown(f"- `{c.name}` ({c.stat().st_size // 1024} KB)")
