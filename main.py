"""Smart Store Vision AI - Main Entry Point (extended with suspicious activity)"""
import argparse, sys, time, cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.detector.yolo_detector import YOLODetector
from src.tracker.bytetrack import ByteTracker
from src.analytics.zone_analytics import ZoneAnalytics, Zone
from src.analytics.heatmap import HeatmapGenerator
from src.analytics.behavior_detector import BehaviorDetector
from src.alerts.alert_manager import AlertManager
from src.utils.config import load_config
from src.utils.zone_io import load_zones
from src.analytics.suspicious_activity import (
    SuspiciousActivityDetector, SuspiciousActivity, Severity, ActivityType,
)
from src.analytics.pose_analyzer import PoseAnalyzer
from src.analytics.multi_camera import MultiCameraCorrelator


def parse_args():
    p = argparse.ArgumentParser(description="Smart Store Vision AI")
    p.add_argument("--config", type=str, default="config/settings.yaml")
    p.add_argument("--zones", type=str, default=None,
                   help="Path to zones.yaml (defaults to config/zones.yaml)")
    p.add_argument("--camera-id", type=str, default="cam_01")
    p.add_argument("--sus-config", type=str, default="config/suspicious_activity.yaml")
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--rtsp", type=str, default=None)
    p.add_argument("--model-size", type=str, default="s", choices=["n","s","m","l","x"])
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu","mps"])
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--detect-classes", type=str, default="0,24,26,28",
                   help="Comma-separated YOLO class IDs (default: person+bag+handbag+suitcase)")
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-alerts", action="store_true")
    p.add_argument("--no-suspicious", action="store_true",
                   help="Disable the new suspicious-activity detector")
    p.add_argument("--no-pose", action="store_true", help="Disable MediaPipe pose")
    p.add_argument("--multi-cam", action="store_true", help="Enable multi-camera Re-ID")
    return p.parse_args()


def create_demo_zones():
    return [
        Zone("entrance","Store Entrance","entrance",[(100,800),(400,800),(400,1080),(100,1080)],capacity=10),
        Zone("checkout","Checkout","checkout",[(700,900),(1200,900),(1200,1080),(700,1080)],capacity=15),
        Zone("restricted","Backroom","restricted",[(1600,100),(1900,100),(1900,400),(1600,400)],capacity=0),
        Zone("shelf_a","Shelf A","shelf",[(450,200),(800,200),(800,650),(450,650)],capacity=5),
        Zone("shelf_b","High-Value Display","shelf",[(850,250),(1200,250),(1200,600),(850,600)],capacity=3),
    ]


def setup_pipeline(args):
    """Build the full pipeline (detector, tracker, analytics, suspicious)."""
    # Detector
    detector = YOLODetector(
        model_size=args.model_size,
        confidence_threshold=args.conf,
        device=args.device,
    )
    # Tracker
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30)
    # Zones
    if args.zones:
        try:
            zones = load_zones(path=Path(args.zones))
        except Exception:
            zones = create_demo_zones()
    else:
        zones = create_demo_zones()
    # Analytics
    zone_analytics = ZoneAnalytics(zones)
    heatmap = HeatmapGenerator()
    # Alert manager
    alert_manager = AlertManager() if not args.no_alerts else None
    # Optional: Pose analyzer
    pose_analyzer = None
    if not args.no_pose:
        try:
            from src.utils.config_schema import PoseConfig
            pose_analyzer = PoseAnalyzer(PoseConfig(enabled=True))
        except Exception:
            pose_analyzer = None
    # Optional: Multi-camera correlator
    multi_cam = None
    if args.multi_cam:
        try:
            from src.utils.config_schema import MultiCameraConfig
            multi_cam = MultiCameraCorrelator(MultiCameraConfig(enabled=True))
        except Exception:
            multi_cam = None
    # Suspicious-activity detector
    sus_detector = None
    if not args.no_suspicious:
        try:
            sus_detector = SuspiciousActivityDetector.from_config_path(
                args.sus_config, pose_analyzer=pose_analyzer, multi_camera=multi_cam,
            )
        except Exception as e:
            print(f"[warn] Could not init suspicious-activity detector: {e}")
            sus_detector = None
    return {
        "detector": detector,
        "tracker": tracker,
        "zones": zones,
        "zone_analytics": zone_analytics,
        "heatmap": heatmap,
        "alert_manager": alert_manager,
        "pose_analyzer": pose_analyzer,
        "multi_cam": multi_cam,
        "sus_detector": sus_detector,
    }


def run_stream(source: str, args):
    print(f"Running on: {source}")
    p = setup_pipeline(args)
    detector = p["detector"]
    tracker = p["tracker"]
    zones = p["zones"]
    zone_analytics = p["zone_analytics"]
    heatmap = p["heatmap"]
    alert_manager = p["alert_manager"]
    sus_detector = p["sus_detector"]
    pose_analyzer = p["pose_analyzer"]
    multi_cam = p["multi_cam"]

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open: {source}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # YOLO classes
    class_ids = [int(c) for c in args.detect_classes.split(",") if c.strip()]

    frame_count = 0
    start_time = time.time()
    sus_alert_count = 0
    fps_smooth = fps

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_count += 1
        timestamp = time.time()

        # Detect
        detections = detector.detect(frame, classes=class_ids)
        # Track
        tracking_result = tracker.update(frame, detections, timestamp)
        # Zone analytics
        zone_analytics.update(tracking_result.tracks, timestamp)
        # Heatmap
        heatmap.update(tracking_result.tracks, timestamp)

        # Suspicious-activity detection
        activities: list = []
        if sus_detector is not None:
            try:
                activities = sus_detector.detect(
                    tracks=tracking_result.tracks,
                    zones={z.zone_id: z for z in zones},
                    zone_occ=zone_analytics.zone_occ,
                    timestamp=timestamp,
                    frame=frame,
                    detections=detections,
                    camera_id=args.camera_id,
                )
            except Exception as e:
                activities = []

        # Emit alerts
        if alert_manager is not None:
            for act in activities:
                track = next((t for t in tracking_result.tracks if t.track_id == act.track_id), None)
                bbox = track.bbox if track else None
                a = alert_manager.create_from_activity(act, args.camera_id, frame, bbox)
                if a is not None:
                    sus_alert_count += 1
            # Also legacy alerts (backwards compat)
            try:
                behavior = BehaviorDetector(suspicious_detector=sus_detector)
                _ = behavior.detect(
                    tracking_result.tracks,
                    {z.zone_id: {"name": z.name, "dwell_threshold": z.dwell_threshold} for z in zones},
                    zone_analytics.zone_occ,
                    timestamp, frame, args.camera_id, detections,
                )
            except Exception:
                pass

        # Render
        if args.show:
            out = detector.draw_detections(frame, detections)
            out = tracker.draw_tracks(out, tracking_result.tracks)
            out = zone_analytics.draw_zones(out)
            out = heatmap.get_overlay(out, alpha=0.25)

            # Draw pose wrists if available
            if pose_analyzer is not None and pose_analyzer.is_available:
                try:
                    pose_map = pose_analyzer.update(frame, tracking_result.tracks, timestamp)
                    for tid, p in pose_map.items():
                        cv2.circle(out, p["left_wrist"], 5, (0, 255, 255), -1)
                        cv2.circle(out, p["right_wrist"], 5, (255, 255, 0), -1)
                except Exception:
                    pass

            # Overlay text
            stats = detector.get_performance_stats()
            cur_fps = stats.get("fps", 0.0)
            fps_smooth = 0.9 * fps_smooth + 0.1 * cur_fps
            sus_stats = sus_detector.get_stats() if sus_detector else {"total": 0}
            cv2.putText(out, f"FPS: {fps_smooth:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(out, f"Tracks: {len(tracking_result.tracks)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(out, f"Footfall: {heatmap.total_footfall}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(out, f"Suspicious: {sus_stats['total']}  (last: {len(activities)})",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            y = 150
            for zid, s in zone_analytics.stats.items():
                zname = next((z.name for z in zones if z.zone_id == zid), zid)
                cv2.putText(out, f"{zname}: {s.current_occupancy}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 25

            # Show recent activity messages in upper-right
            x0 = width - 480
            y0 = 30
            cv2.rectangle(out, (x0 - 10, y0 - 10), (width - 10, y0 + 30 * min(5, len(activities)) + 10),
                          (0, 0, 0), -1)
            for i, a in enumerate(activities[:5]):
                cv2.putText(out, f"[{a.severity.value}] {a.message[:60]}",
                            (x0, y0 + 25 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

            cv2.imshow("Smart Store Vision AI", out)
            if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                break

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(
                f"Frames: {frame_count}/{total_frames}, "
                f"FPS: {frame_count / max(elapsed, 1e-3):.1f}, "
                f"Tracks: {len(tracking_result.tracks)}, "
                f"Sus alerts: {sus_alert_count}"
            )

    cap.release()
    cv2.destroyAllWindows()
    print("\n=== Final Stats ===")
    print(f"Detector: {detector.get_performance_stats()}")
    print(f"Tracker: {tracker.get_stats()}")
    print(f"Heatmap: {heatmap.get_stats()}")
    if sus_detector:
        print(f"Suspicious: {sus_detector.get_stats()}")
    if alert_manager:
        print(f"Alerts: {alert_manager.stats}")


def main():
    args = parse_args()
    try:
        config = load_config(args.config)
    except Exception:
        config = {}
    if args.video:
        run_stream(args.video, args)
    elif args.rtsp:
        run_stream(args.rtsp, args)
    else:
        print("Usage: python main.py --video <path> --show")
        print("       python main.py --rtsp <url> --show")
        for path in ["data/demo.mp4", "demo.mp4", "test.mp4"]:
            if Path(path).exists():
                print(f"Found: {path}")
                args.video = path
                run_stream(path, args)
                break


if __name__ == "__main__":
    main()
