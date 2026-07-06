"""Tests for movement feature extraction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import math
import pytest
from src.analytics.movement_features import (
    compute_features, motion_variance, count_significant_turns,
    is_running, is_erratic,
)


def _make_history(points):
    """Convert list of (x, y) to history with timestamps."""
    return [(i * 0.1, p) for i, p in enumerate(points)]


def test_compute_features_short():
    feats = compute_features([(0.0, (0, 0))])
    assert feats["avg_speed"] == 0.0
    assert feats["heading_change_count"] == 0


def test_motion_variance_zero_for_stationary():
    h = _make_history([(100, 100)] * 5)
    assert motion_variance(h) == 0.0


def test_motion_variance_positive_for_moving():
    pts = [(0, 0), (5, 0), (10, 0), (15, 0), (20, 0), (5, 0), (10, 0), (20, 0)]
    h = _make_history(pts)
    v = motion_variance(h)
    assert v > 0.0


def test_count_significant_turns():
    # Straight line: 0 turns
    pts = [(0, 0), (10, 0), (20, 0), (30, 0)]
    assert count_significant_turns(_make_history(pts)) == 0
    # Sharp turn at end
    pts = [(0, 0), (10, 0), (20, 0), (20, 10), (20, 20)]
    h = _make_history(pts)
    assert count_significant_turns(h) >= 1


def test_is_running_threshold():
    feats = {"peak_speed": 250.0, "duration_seconds": 1.0}
    assert is_running(feats, 200.0) is True
    feats = {"peak_speed": 50.0, "duration_seconds": 1.0}
    assert is_running(feats, 200.0) is False


def test_is_erratic():
    feats = {"heading_change_count": 10, "duration_seconds": 4.0}
    assert is_erratic(feats) is True
    feats = {"heading_change_count": 1, "duration_seconds": 1.0}
    assert is_erratic(feats) is False
