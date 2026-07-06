"""Tests for zone_io utilities (YAML save/load, canvas conversion)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tempfile
import pytest
from src.analytics.zone_analytics import Zone
from src.utils.zone_io import save_zones, load_zones, zones_from_canvas


def test_save_load_roundtrip():
    zones = [
        Zone("e", "Entrance", "entrance", [(10, 20), (30, 20), (30, 50), (10, 50)],
             capacity=10, dwell_threshold=120.0),
        Zone("r", "Restricted", "restricted", [(100, 100), (200, 100), (200, 200), (100, 200)],
             capacity=0, dwell_threshold=0.0),
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "z.yaml"
        save_zones(zones, path=path)
        loaded = load_zones(path=path)
        assert len(loaded) == 2
        assert loaded[0].zone_id == "e"
        assert loaded[0].capacity == 10
        assert loaded[1].zone_type == "restricted"


def test_zones_from_canvas_polygon():
    obj = {
        "type": "polygon",
        "label": "Test Zone",
        "stroke": "#ff0000",
        "points": [{"x": 10, "y": 10}, {"x": 50, "y": 10}, {"x": 30, "y": 50}],
    }
    zs = zones_from_canvas([obj])
    assert len(zs) == 1
    assert zs[0].zone_type == "restricted"  # color #ff0000 -> restricted
    assert len(zs[0].polygon) >= 3


def test_zones_from_canvas_rect():
    obj = {
        "type": "rect",
        "label": "Counter",
        "stroke": "#00ff00",
        "left": 100, "top": 200, "width": 50, "height": 80,
    }
    zs = zones_from_canvas([obj])
    assert len(zs) == 1
    assert zs[0].zone_type == "checkout"  # color #00ff00 -> checkout


def test_zones_from_canvas_ignores_too_few_points():
    obj = {"type": "polygon", "label": "tiny", "stroke": "#fff",
           "points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]}
    assert zones_from_canvas([obj]) == []
