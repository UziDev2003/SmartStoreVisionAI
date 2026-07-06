"""Unit tests for Smart Store Vision AI components."""
import pytest, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.detector.yolo_detector import Detection
from src.analytics.zone_analytics import Zone, ZoneAnalytics
from src.analytics.heatmap import HeatmapGenerator
from src.analytics.behavior_detector import BehaviorDetector
from src.alerts.alert_manager import AlertManager

class TestDetection:
    def test_bbox_properties(self):
        det = Detection(class_id=0, class_name="person", confidence=0.95, bbox=(100,100,200,300))
        cx, cy, w, h = det.bbox_xywh
        assert cx == 150 and cy == 200 and w == 100 and h == 200
        assert det.center == (150, 200)
        assert det.area == 20000

class TestZone:
    def test_point_inside(self):
        z = Zone("z1","Test","general",[(0,0),(100,0),(100,100),(0,100)])
        assert z.contains_point((50,50)) is True
        assert z.contains_point((150,50)) is False
    def test_center(self):
        z = Zone("z1","Test","general",[(0,0),(100,0),(100,100),(0,100)])
        assert z.get_center() == (50,50)

class TestZoneAnalytics:
    def test_init(self):
        z = Zone("z1","Test","general",[(0,0),(100,0),(100,100),(0,100)])
        a = ZoneAnalytics([z], fw=400, fh=200)
        assert len(a.zones) == 1
        assert "z1" in a.zones
    def test_occupancy(self):
        z = Zone("z1","Test","general",[(50,50),(150,50),(150,150),(50,150)])
        a = ZoneAnalytics([z])
        class T: track_id = 1; center = (100,100)
        evts = a.update([T()])
        assert len(evts) == 1 and evts[0].event_type == "enter"

class TestHeatmap:
    def test_init(self):
        hm = HeatmapGenerator(640, 480, 10)
        assert hm.grid_w == 64 and hm.grid_h == 48
    def test_update(self):
        hm = HeatmapGenerator(640, 480, 10)
        class T: track_id = 1; center = (100,100)
        hm.update([T()])
        assert hm.total_footfall == 1
        assert hm.heatmap[10,10] == 1

class TestBehavior:
    def test_rate_limit(self):
        d = BehaviorDetector()
        zones = {"z1":{"name":"Test","dwell_threshold":10.0}}
        occ = {"z1":{1:0}}
        alerts = d._check_loitering([], zones, occ, 15.0)
        assert len(alerts) == 0

class TestAlerts:
    def test_creation(self):
        m = AlertManager(rate_limit=0)
        a = m.create("loitering","cam_01",1,"Test","medium","z1")
        assert a is not None and a.alert_type == "loitering"
    def test_rate_limit(self):
        # New rate-limit key includes track_id, so two different tracks
        # are rate-limited independently. Same track_id + alert_type
        # within the cooldown window should be suppressed.
        m = AlertManager(rate_limit=60)
        a1 = m.create("test", "cam_01", 1, "A1")
        a2 = m.create("test", "cam_01", 1, "A1-dup")  # same track
        a3 = m.create("test", "cam_01", 2, "A2")      # different track
        assert a1 is not None
        assert a2 is None       # rate-limited (same track+type)
        assert a3 is not None   # different track -> allowed
    def test_filter(self):
        m = AlertManager(rate_limit=0)
        m.create("loitering","cam_01",1,"A1","medium")
        m.create("intrusion","cam_01",2,"A2","high")
        assert len(m.get_alerts(alert_type="loitering")) == 1
        assert len(m.get_alerts(camera_id="cam_01")) == 2

if __name__ == "__main__": pytest.main([__file__,"-v"])
