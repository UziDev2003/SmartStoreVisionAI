"""
Supervision library wrapper
============================
Lightweight adapter around the `supervision` package (Roboflow). Provides
zone containment checks, annotators, and heatmap utilities using an
industry-standard API while remaining a soft dependency.

If `supervision` is not installed, falls back to OpenCV-only operations.
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Any
import numpy as np

try:
    import supervision as sv  # type: ignore
    SUPERVISION_AVAILABLE = True
except Exception:  # pragma: no cover
    sv = None  # type: ignore
    SUPERVISION_AVAILABLE = False


def is_available() -> bool:
    return SUPERVISION_AVAILABLE


def make_polygon_zone(polygon: List[Tuple[int, int]], frame_wh: Tuple[int, int]):
    """
    Build a supervision PolygonZone for the given polygon and frame.
    Returns None if supervision is not available.
    """
    if not SUPERVISION_AVAILABLE:
        return None
    h, w = frame_wh[1], frame_wh[0]
    poly = np.array(polygon, dtype=np.int32)
    return sv.PolygonZone(polygon=poly, frame_resolution_wh=(w, h))


def annotate_zones(
    frame: np.ndarray,
    zones_with_names: List[Tuple[List[Tuple[int, int]], str, str]],
    occupancy: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Annotate a frame with labeled polygons.

    zones_with_names: list of (polygon, name, color_hex)
    occupancy: optional list of current occupancy values to display
    """
    if SUPERVISION_AVAILABLE:
        try:
            annotator = sv.PolygonZoneAnnotator()
            for i, (poly, name, color) in enumerate(zones_with_names):
                occ = occupancy[i] if occupancy and i < len(occupancy) else 0
                label = f"{name}: {occ}"
                # Convert hex color to BGR
                if isinstance(color, str) and color.startswith("#"):
                    r = int(color[1:3], 16); g = int(color[3:5], 16); b = int(color[5:7], 16)
                    color_bgr = (b, g, r)
                else:
                    color_bgr = color
                pts = np.array(poly, dtype=np.int32)
                cv_frame = frame.copy()
                cv2_polys = [pts]
                overlay = cv_frame.copy()
                import cv2
                cv2.fillPoly(overlay, cv2_polys, color_bgr)
                cv2.addWeighted(overlay, 0.25, cv_frame, 0.75, 0, cv_frame)
                cv2.polylines(cv_frame, cv2_polys, True, color_bgr, 2)
                cx = sum(p[0] for p in poly) // len(poly)
                cy = sum(p[1] for p in poly) // len(poly)
                cv2.putText(cv_frame, label, (cx - 60, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
                frame = cv_frame
            return frame
        except Exception:
            pass
    # Fallback: plain OpenCV
    import cv2
    for i, (poly, name, color) in enumerate(zones_with_names):
        occ = occupancy[i] if occupancy and i < len(occupancy) else 0
        pts = np.array(poly, dtype=np.int32)
        if isinstance(color, str) and color.startswith("#"):
            r = int(color[1:3], 16); g = int(color[3:5], 16); b = int(color[5:7], 16)
            color_bgr = (b, g, r)
        else:
            color_bgr = color or (255, 255, 255)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color_bgr)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [pts], True, color_bgr, 2)
        cx = sum(p[0] for p in poly) // len(poly)
        cy = sum(p[1] for p in poly) // len(poly)
        cv2.putText(frame, f"{name}: {occ}", (cx - 60, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
    return frame


def make_box_annotator(color: Tuple[int, int, int] = (0, 255, 0)):
    """Build a supervision BoxAnnotator; returns None if unavailable."""
    if not SUPERVISION_AVAILABLE:
        return None
    try:
        return sv.BoxAnnotator(color=sv.Color.from_bgr_tuple(color) if hasattr(sv, "Color") else None)
    except Exception:
        try:
            return sv.BoxAnnotator()
        except Exception:
            return None
