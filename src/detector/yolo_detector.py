"""
YOLOv8 Human Detector Module
============================
Provides fast and accurate person detection using Ultralytics YOLOv8.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import cv2
import numpy as np

from ultralytics import YOLO


@dataclass
class Detection:
    """Represents a single detection with bounding box and confidence."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    
    @property
    def bbox_xywh(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1)
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class YOLODetector:
    """YOLOv8-based person detector for retail analytics."""
    
    PERSON_CLASS_ID = 0
    
    def __init__(
        self,
        model_size: str = "s",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        model_path: Optional[str] = None
    ):
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.inference_times: List[float] = []
        self.frame_count = 0
        
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(f"yolov8{model_size}.pt")
        
        self.model.to(device)
        self.class_names = self.model.names
    
    def detect(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> List[Detection]:
        if classes is None:
            classes = [self.PERSON_CLASS_ID]
        
        start_time = time.perf_counter()
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False,
            device=self.device
        )
        self.inference_times.append(time.perf_counter() - start_time)
        self.frame_count += 1
        
        return self._parse_results(results)
    
    def _parse_results(self, results) -> List[Detection]:
        detections = []
        if len(results) == 0:
            return detections
        
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        for i in range(len(result.boxes)):
            box = result.boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            detections.append(Detection(
                class_id=class_id,
                class_name=self.class_names.get(class_id, f"class_{class_id}"),
                confidence=confidence,
                bbox=(int(x1), int(y1), int(x2), int(y2))
            ))
        return detections
    
    def get_performance_stats(self) -> Dict[str, float]:
        if not self.inference_times:
            return {"avg_inference_ms": 0.0, "fps": 0.0, "frames_processed": 0}
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        return {
            "avg_inference_ms": avg_time * 1000,
            "fps": 1.0 / avg_time if avg_time > 0 else 0.0,
            "frames_processed": self.frame_count
        }
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection],
                        track_ids: Optional[Dict[int, int]] = None) -> np.ndarray:
        output = frame.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"ID:{track_ids.get(i, '?')} {det.class_name.upper()} {det.confidence:.2f}" \
                    if track_ids else f"{det.class_name.upper()} {det.confidence:.2f}"
            
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - lh - 10), (x1 + lw, y1), (0, 255, 0), -1)
            cv2.putText(output, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return output


def create_detector(config: Optional[Dict[str, Any]] = None) -> YOLODetector:
    if config is None:
        config = {}
    return YOLODetector(
        model_size=config.get("model_size", "s"),
        confidence_threshold=config.get("confidence_threshold", 0.5),
        iou_threshold=config.get("iou_threshold", 0.45),
        device=config.get("device", "cuda"),
        model_path=config.get("model_path")
    )
