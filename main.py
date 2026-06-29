"""Smart Store Vision AI - Main Entry Point"""
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

def parse_args():
    p = argparse.ArgumentParser(description="Smart Store Vision AI")
    p.add_argument("--config", type=str, default="config/settings.yaml")
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--rtsp", type=str, default=None)
    p.add_argument("--camera-id", type=str, default="cam_01")
    p.add_argument("--model-size", type=str, default="s", choices=["n","s","m","l","x"])
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu","mps"])
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-alerts", action="store_true")
    return p.parse_args()

def create_demo_zones():
    return [
        Zone("entrance","Store Entrance","entrance",[(100,800),(400,800),(400,1080),(100,1080)],capacity=10),
        Zone("checkout","Checkout","checkout",[(700,900),(1200,900),(1200,1080),(700,1080)],capacity=15),
        Zone("restricted","Backroom","restricted",[(1600,100),(1900,100),(1900,400),(1600,400)],capacity=0),
    ]

def run_demo_video(video_path: str, args):
    print(f"Running on: {video_path}")
    detector = YOLODetector(model_size=args.model_size, confidence_threshold=args.conf, device=args.device)
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30)
    zones = create_demo_zones()
    zone_analytics = ZoneAnalytics(zones)
    heatmap = HeatmapGenerator()
    behavior_detector = BehaviorDetector()
    alert_manager = AlertManager() if not args.no_alerts else None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        timestamp = time.time()
        
        detections = detector.detect(frame)
        tracking_result = tracker.update(frame, detections, timestamp)
        zone_analytics.update(tracking_result.tracks, timestamp)
        heatmap.update(tracking_result.tracks, timestamp)
        
        if alert_manager:
            alerts = behavior_detector.detect(tracking_result.tracks,
                {z.zone_id:{"name":z.name,"dwell_threshold":z.dwell_threshold} for z in zones},
                zone_analytics.zone_occ, timestamp)
            for alert in alerts:
                t = tracking_result.get_track_by_id(alert.track_id)
                bbox = t.bbox if t else None
                alert_manager.create(alert.alert_type.value, args.camera_id, alert.track_id,
                    alert.message, alert.severity.value, alert.zone_id, frame, bbox)
        
        if args.show:
            output = detector.draw_detections(frame, detections)
            output = tracker.draw_tracks(output, tracking_result.tracks)
            output = zone_analytics.draw_zones(output)
            output = heatmap.get_overlay(output, alpha=0.3)
            
            stats = detector.get_performance_stats()
            cv2.putText(output, f"FPS: {stats['fps']:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(output, f"Tracks: {len(tracking_result.tracks)}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            y = 110
            for zid, s in zone_analytics.stats.items():
                zname = next((z.name for z in zones if z.zone_id == zid), zid)
                cv2.putText(output, f"{zname}: {s.current_occupancy}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                y += 30
            
            cv2.imshow("Smart Store Vision AI", output)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: break
        
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Frames: {frame_count}/{total_frames}, FPS: {frame_count/elapsed:.1f}, Tracks: {len(tracking_result.tracks)}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n=== Stats ===\nDetector: {detector.get_performance_stats()}\nTracker: {tracker.get_performance_stats()}\nHeatmap: {heatmap.get_stats()}")
    if alert_manager: print(f"Alerts: {alert_manager.get_stats()}")

def main():
    args = parse_args()
    try: config = load_config(args.config)
    except: config = {}
    
    if args.video:
        run_demo_video(args.video, args)
    elif args.rtsp:
        run_demo_video(args.rtsp, args)
    else:
        print("Usage: python main.py --video <path> --show")
        print("       python main.py --rtsp <url> --show")
        for path in ["data/demo.mp4","demo.mp4","test.mp4"]:
            if Path(path).exists():
                print(f"Found: {path}")
                args.video = path
                run_demo_video(path, args)
                break

if __name__ == "__main__": main()
