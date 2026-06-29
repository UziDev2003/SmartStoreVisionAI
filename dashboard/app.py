"""Smart Store Vision AI - Dashboard Server"""
import os, sys, time, base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.detector.yolo_detector import YOLODetector
from src.tracker.bytetrack import ByteTracker
from src.analytics.zone_analytics import ZoneAnalytics, Zone
from src.analytics.heatmap import HeatmapGenerator
from src.alerts.alert_manager import AlertManager

class VisionPipeline:
    def __init__(self, config=None):
        self.config = config or {}
        self.detector = YOLODetector(model_size="s", confidence_threshold=0.5, device="cuda")
        self.tracker = ByteTracker(track_thresh=0.5)
        self.zones = [Zone("entrance","Store Entrance","entrance",[(100,800),(400,800),(400,1080),(100,1080)],capacity=10),
                      Zone("checkout","Checkout","checkout",[(700,900),(1200,900),(1200,1080),(700,1080)],capacity=15),
                      Zone("restricted","Backroom","restricted",[(1600,100),(1900,100),(1900,400),(1600,400)],capacity=0)]
        self.zone_analytics = ZoneAnalytics(self.zones)
        self.heatmap = HeatmapGenerator()
        self.alert_manager = AlertManager()
        self.current_frame = None
    
    def process(self, frame):
        ts = time.time()
        detections = self.detector.detect(frame)
        result = self.tracker.update(frame, detections, ts)
        self.zone_analytics.update(result.tracks, ts)
        self.heatmap.update(result.tracks, ts)
        self.current_frame = frame.copy()
        output = frame.copy()
        output = self.detector.draw_detections(output, detections)
        output = self.tracker.draw_tracks(output, result.tracks)
        output = self.zone_analytics.draw_zones(output)
        return output
    
    def get_stats(self):
        return {"detector": self.detector.get_performance_stats(), "tracker": self.tracker.get_performance_stats(),
                "heatmap": self.heatmap.get_stats(), "alerts": self.alert_manager.get_stats(),
                "occupancy": {zid: s.current_occupancy for zid, s in self.zone_analytics.stats.items()}}
    
    def get_alerts(self, limit=50):
        return [a.to_dict() for a in self.alert_manager.get_alerts(limit=limit)]

app = Flask(__name__)
CORS(app)
pipeline = VisionPipeline()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/stats')
def get_stats(): return jsonify(pipeline.get_stats())

@app.route('/api/alerts')
def get_alerts(): return jsonify(pipeline.get_alerts(limit=request.args.get('limit',50,type=int)))

@app.route('/api/zones')
def get_zones():
    return jsonify([{"id":z.zone_id,"name":z.name,"type":z.zone_type,"capacity":z.capacity,
                     "occupancy":pipeline.zone_analytics.stats.get(z.zone_id,None).current_occupancy if pipeline.zone_analytics.stats.get(z.zone_id) else 0}
                    for z in pipeline.zones])

@app.route('/api/heatmap')
def get_heatmap():
    img = pipeline.heatmap.get_image()
    _, buf = cv2.imencode('.png', img)
    return jsonify({"image": f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"})

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            if pipeline.current_frame is not None:
                _, buf = cv2.imencode('.jpg', pipeline.current_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.033)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
