#!/usr/bin/env python
"""
Quick test script to verify the Smart Store Vision AI installation.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO")
    except ImportError as e:
        print(f"✗ Ultralytics: {e}")
        return False
    
    try:
        import flask
        print(f"✓ Flask: {flask.__version__}")
    except ImportError as e:
        print(f"✗ Flask: {e}")
    
    # Test project modules
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from src.detector.yolo_detector import YOLODetector
        print("✓ YOLODetector")
    except ImportError as e:
        print(f"✗ YOLODetector: {e}")
        return False
    
    try:
        from src.tracker.bytetrack import ByteTracker
        print("✓ ByteTracker")
    except ImportError as e:
        print(f"✗ ByteTracker: {e}")
        return False
    
    try:
        from src.analytics.zone_analytics import ZoneAnalytics
        print("✓ ZoneAnalytics")
    except ImportError as e:
        print(f"✗ ZoneAnalytics: {e}")
        return False
    
    try:
        from src.analytics.heatmap import HeatmapGenerator
        print("✓ HeatmapGenerator")
    except ImportError as e:
        print(f"✗ HeatmapGenerator: {e}")
        return False
    
    try:
        from src.alerts.alert_manager import AlertManager
        print("✓ AlertManager")
    except ImportError as e:
        print(f"✗ AlertManager: {e}")
        return False
    
    return True


def test_yolo_model():
    """Test YOLOv8 model loading."""
    print("\nTesting YOLOv8 model loading...")
    
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # Use smallest model for test
        print("✓ YOLOv8n model loaded")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def test_inference():
    """Test basic inference."""
    print("\nTesting inference...")
    
    try:
        import numpy as np
        from ultralytics import YOLO
        
        model = YOLO("yolov8n.pt")
        
        # Create dummy image
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(img, verbose=False)
        print(f"✓ Inference successful, detected {len(results[0].boxes)} objects")
        return True
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False


def main():
    print("=" * 50)
    print("Smart Store Vision AI - Installation Test")
    print("=" * 50)
    
    success = True
    
    if not test_imports():
        success = False
    
    if success:
        test_yolo_model()
    
    if success:
        test_inference()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed!")
        print("\nTo run the demo:")
        print("  python main.py --video data/demo.mp4 --show")
        print("\nTo start the dashboard:")
        print("  python dashboard/app.py")
        print("  Then open http://localhost:5000")
    else:
        print("✗ Some tests failed. Please check the output above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
