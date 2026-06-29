"""
Smart Store Vision AI - Main Package
=====================================
An intelligent retail analytics system using YOLOv8 for person detection
and ByteTrack for multi-object tracking, with real-time analytics and alerts.
"""

__version__ = "1.0.0"
__author__ = "Smart Store Vision AI Team"

from src.detector.yolo_detector import YOLODetector
from src.tracker.bytetrack import ByteTracker
from src.analytics.zone_analytics import ZoneAnalytics
from src.analytics.heatmap import HeatmapGenerator
from src.analytics.behavior_detector import BehaviorDetector
from src.alerts.alert_manager import AlertManager
from src.stream.stream_manager import StreamManager

__all__ = [
    "YOLODetector",
    "ByteTracker", 
    "ZoneAnalytics",
    "HeatmapGenerator",
    "BehaviorDetector",
    "AlertManager",
    "StreamManager"
]
