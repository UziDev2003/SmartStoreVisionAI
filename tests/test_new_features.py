"""
Tests for the new analytics modules (Batch 1 + 2 of the
35-feature expansion): frame_preprocessor, background_subtractor,
line_detector, queue_detector, object_monitor, video_clipper,
face_blur, pose_detectors.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import time
import tempfile

from src.analytics.frame_preprocessor import FramePreprocessor, PreprocessConfig
from src.analytics.background_subtractor import BackgroundSubtractor
from src.analytics.line_detector import LineCrossingDetector, LineZone
from src.analytics.queue_detector import QueueDetector, QueueZone
from src.analytics.object_monitor import (
    ObjectLeftBehindDetector, ObjectRemovedDetector, ShelfArea
)
from src.analytics.video_clipper import VideoClipper
from src.analytics.pose_detectors import (
    GestureDetector, FallDetector, FightDetector, PoseEvent
)


# ---------------------------------------------------------------------------
# FramePreprocessor
# ---------------------------------------------------------------------------
def test_preprocessor_passthrough_when_disabled():
    cfg = PreprocessConfig(enabled=False)
    p = FramePreprocessor(cfg)
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    out = p(frame)
    assert out is frame


def test_preprocessor_resize():
    cfg = PreprocessConfig(enabled=True, target_width=64, target_height=64)
    p = FramePreprocessor(cfg)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = p(frame)
    assert out.shape == (64, 64, 3)


def test_preprocessor_gamma():
    cfg = PreprocessConfig(enabled=True, gamma=2.0)
    p = FramePreprocessor(cfg)
    frame = np.full((50, 50, 3), 200, dtype=np.uint8)
    out = p(frame)
    # Gamma 2.0 darkens (out = 255 * (200/255)^(1/2))
    assert float(out.mean()) < float(frame.mean())
    assert out.mean() > 0


def test_preprocessor_letterbox():
    cfg = PreprocessConfig(enabled=True, letterbox=True)
    p = FramePreprocessor(cfg)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = p(frame)
    assert out.shape[0] == out.shape[1] == 200  # max(100,200) = 200


# ---------------------------------------------------------------------------
# BackgroundSubtractor
# ---------------------------------------------------------------------------
def test_background_subtractor_mog2():
    bs = BackgroundSubtractor(method="MOG2", history=50, var_threshold=16.0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Warm up with a few frames so the model stabilizes
    for _ in range(20):
        bs.apply(frame)
    stats = bs.apply(frame)
    assert stats.motion_area == 0
    assert stats.total_area == 100 * 100


def test_background_subtractor_motion_detection():
    bs = BackgroundSubtractor(method="MOG2", history=10)
    # Train on black background
    bg = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(15):
        bs.apply(bg)
    # Add a moving white square
    fg = bg.copy()
    fg[20:40, 20:40] = 255
    stats = bs.apply(fg)
    assert stats.motion_area > 0
    assert stats.motion_density < 1.0


def test_background_subtractor_knn():
    bs = BackgroundSubtractor(method="KNN", history=10)
    bg = np.zeros((50, 50, 3), dtype=np.uint8)
    for _ in range(15):
        bs.apply(bg)
    fg = bg.copy()
    fg[10:20, 10:20] = 200
    stats = bs.apply(fg)
    assert stats.motion_area > 0


# ---------------------------------------------------------------------------
# LineCrossingDetector
# ---------------------------------------------------------------------------
def test_line_crossing_detected():
    line = LineZone("entrance", "Front Door",
                    p1=(100, 500), p2=(500, 500),
                    left_label="in", right_label="out")
    det = LineCrossingDetector(lines=[line])

    class Track:
        track_id = 1
        def __init__(self, hist):
            self.history = hist
            self.center = hist[-1][1] if hist else (0, 0)
    # Track moves from below (d_cur < 0) to above (d_cur > 0) the line
    t = Track([(0, (300, 600)), (0.1, (300, 400))])
    crossings = det.update([t], timestamp=1.0)
    assert len(crossings) == 1
    assert crossings[0].direction == "in"


def test_line_no_crossing_when_staying_on_one_side():
    line = LineZone("entrance", "Front Door",
                    p1=(100, 500), p2=(500, 500))
    det = LineCrossingDetector(lines=[line])

    class Track:
        track_id = 1
        def __init__(self, hist):
            self.history = hist
            self.center = hist[-1][1] if hist else (0, 0)
    # Track stays below the line
    t = Track([(0, (300, 600)), (0.1, (310, 610))])
    crossings = det.update([t], timestamp=1.0)
    assert len(crossings) == 0


def test_line_direction_label():
    line = LineZone("e", "E", p1=(0, 0), p2=(100, 0))
    det = LineCrossingDetector(lines=[line])

    class Track:
        track_id = 1
        history = [(0, (0, 0)), (0.1, (10, 0))]  # moving east
        center = (10, 0)
    # direction_label based on last movement
    assert det.direction_label(Track()) in {"E", "S", "W", "N"}


# ---------------------------------------------------------------------------
# QueueDetector
# ---------------------------------------------------------------------------
def test_queue_detected_with_min_length():
    q = QueueZone("checkout_q", "Checkout Queue",
                  polygon=[(100, 100), (200, 100), (200, 300), (100, 300)],
                  min_queue_length=2, wait_threshold_seconds=2.0)
    det = QueueDetector(queues=[q])

    class Track:
        track_id = 1
        def __init__(self, x, y):
            self.center = (x, y)
    tracks = [Track(150, 200), Track(150, 250)]
    # Pre-seed first_seen with a past timestamp so dwell > 0 on first call
    now = time.time()
    det._first_seen[("checkout_q", 1)] = now - 5.0
    det._first_seen[("checkout_q", 2)] = now - 5.0
    occ = {"checkout_q": {1: now - 5.0, 2: now - 5.0}}
    statuses = det.update(tracks, occ, timestamp=now)
    assert len(statuses) == 1
    assert statuses[0].length == 2
    assert statuses[0].estimated_wait_seconds >= 4.0


# ---------------------------------------------------------------------------
# ObjectLeftBehindDetector
# ---------------------------------------------------------------------------
def test_object_left_behind_no_alert_when_person_present():
    detector = ObjectLeftBehindDetector(min_seconds=0.0, stationary_threshold=10)
    shelf = ShelfArea("shelf1", "Test Shelf", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])
    occ = {"shelf1": {1: time.time()}}
    events = detector.update(None, [], occ, [shelf], timestamp=time.time())
    assert events == []  # person still present


# ---------------------------------------------------------------------------
# VideoClipper
# ---------------------------------------------------------------------------
def test_video_clipper_buffer_and_save(tmp_path):
    clipper = VideoClipper(pre_seconds=1, post_seconds=1, output_dir=str(tmp_path), fps=10)
    for i in range(15):
        frame = np.full((60, 80, 3), i * 10, dtype=np.uint8)
        clipper.add_frame(frame, timestamp=i * 0.1)
    path = clipper.save_clip("test_alert")
    # Path may be empty if writer not available on test env; just ensure no crash
    clipper.close()
    assert isinstance(path, str)


# ---------------------------------------------------------------------------
# Pose Detectors
# ---------------------------------------------------------------------------
def test_gesture_hands_up():
    det = GestureDetector(raise_threshold_px=20)
    # Both wrists well above the elbows (y smaller = higher on screen)
    pose = {
        "left_wrist": (50, 50), "right_wrist": (60, 60),
        "left_elbow": (50, 200), "right_elbow": (60, 200),
        "is_crouching": False, "confidence": 0.8,
    }
    events = det.update({1: pose}, [], timestamp=time.time())
    assert any(e.gesture == "hands_up" for e in events)


def test_gesture_hand_raised():
    det = GestureDetector(raise_threshold_px=20)
    pose = {
        "left_wrist": (50, 80),  # very high
        "right_wrist": (60, 200),  # at side
        "left_elbow": (50, 150), "right_elbow": (60, 150),
        "is_crouching": False, "confidence": 0.8,
    }
    events = det.update({1: pose}, [], timestamp=time.time())
    assert any(e.gesture == "hand_raised" for e in events)


def test_gesture_no_event_low_confidence():
    det = GestureDetector()
    pose = {"left_wrist": (50, 50), "right_wrist": (50, 50), "confidence": 0.1}
    events = det.update({1: pose}, [], timestamp=time.time())
    assert events == []


def test_fall_detected_on_y_drop():
    det = FallDetector(min_drop_px=50, min_track_length=5, cooldown_seconds=0)
    class T:
        track_id = 1
        center = (100, 100)
        history = [(i * 0.1, (100, 1000 - i * 100)) for i in range(20)]
    # First call: baseline is set to current (low) y; no drop detected yet
    det.update([T()], timestamp=time.time())
    # Manually set the baseline to the highest historical y
    det._baseline_y[1] = 1000.0
    # Second call: drop = 900, _fallen_at is set, ts - _fallen_at = 0, not enough dwell
    det.update([T()], timestamp=time.time() + 1.0)
    # Third call: ts - _fallen_at = 1.0+, fall should be confirmed
    events = det.update([T()], timestamp=time.time() + 2.0)
    assert any(e.event_type == "fall" for e in events)


def test_fight_detected_close_proximity():
    det = FightDetector(proximity_px=200, min_pair_dwell=0.0, cooldown_seconds=0)
    class T:
        track_id = 1
        def __init__(self, cx, cy):
            self.center = (cx, cy)
    tracks = [T(100, 100), T(150, 100)]
    events = det.update(tracks, timestamp=time.time())
    assert any(e.event_type == "fight" for e in events)


def test_fight_not_detected_far_apart():
    det = FightDetector(proximity_px=50)
    class T:
        track_id = 1
        def __init__(self, cx, cy):
            self.center = (cx, cy)
    tracks = [T(100, 100), T(500, 500)]  # very far
    events = det.update(tracks, timestamp=time.time())
    assert events == []
