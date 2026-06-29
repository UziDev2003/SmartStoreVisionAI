"""
Heatmap Generator Module
Generates foot traffic heatmaps from tracked person positions.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import time

class HeatmapGenerator:
    def __init__(self, width: int = 1920, height: int = 1080, grid_size: int = 10, 
                 decay: float = 0.995, blur: int = 15):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.decay = decay
        self.blur = blur
        self.grid_w = width // grid_size
        self.grid_h = height // grid_size
        self.heatmap = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.seen_track_ids = set()
        self.total_footfall = 0
        self.peak_density = 0.0
    
    def update(self, tracks: List, timestamp: Any = None) -> np.ndarray:
        self.heatmap *= self.decay
        for t in tracks:
            cx, cy = t.center
            gx, gy = int(cx // self.grid_size), int(cy // self.grid_size)
            if 0 <= gy < self.grid_h and 0 <= gx < self.grid_w:
                self.heatmap[gy, gx] += 1
            if hasattr(t, 'track_id') and t.track_id not in self.seen_track_ids:
                self.seen_track_ids.add(t.track_id)
                self.total_footfall += 1
        self.peak_density = max(self.peak_density, self.heatmap.max())
        return self.heatmap.copy()
    
    def get_image(self, normalize: bool = True) -> np.ndarray:
        h = self.heatmap.copy()
        if normalize and h.max() > 0:
            h = (h / h.max() * 255).astype(np.uint8)
        else:
            h = np.clip(h, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(h, cv2.COLORMAP_JET)
    
    def get_overlay(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        img = self.get_image()
        resized = cv2.resize(img, (frame.shape[1], frame.shape[0]))
        return cv2.addWeighted(frame, 1-alpha, resized, alpha, 0)
    
    def get_hotspots(self, threshold: float = 0.7, min_area: int = 25) -> List[Dict]:
        if self.heatmap.max() == 0:
            return []
        norm = self.heatmap / self.heatmap.max()
        binary = (norm > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hotspots = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(c)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) * self.grid_size
                    cy = int(M["m01"] / M["m00"]) * self.grid_size
                else:
                    cx, cy = x * self.grid_size, y * self.grid_size
                hotspots.append({"center": (cx, cy), "bounds": (x, y, x+w, y+h), 
                              "density": norm[y:y+h, x:x+w].mean()})
        return sorted(hotspots, key=lambda h: h["density"], reverse=True)
    
    def draw_hotspots(self, frame: np.ndarray, hotspots: List[Dict]) -> np.ndarray:
        out = frame.copy()
        for i, hs in enumerate(hotspots[:5]):
            x1, y1, x2, y2 = hs["bounds"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"Hotspot {i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(out, hs["center"], 5, (0, 255, 255), -1)
        return out
    
    def get_stats(self) -> Dict[str, Any]:
        return {"footfall": self.total_footfall, "peak_density": float(self.peak_density),
                "coverage": float((self.heatmap > 0).sum() / self.heatmap.size)}
    
    def reset(self):
        self.heatmap.fill(0)
        self.seen_track_ids.clear()
        self.total_footfall = 0
        self.peak_density = 0.0
