"""
Pose Analyzer (MediaPipe)
=========================
Uses MediaPipe Pose to extract per-track hand positions and posture,
enabling hand-near-shelf detection, leaning/crouching detection, and
two-person interaction heuristics.

MediaPipe is an optional dependency; if unavailable, the analyzer
returns empty results and other detectors continue to function.
"""
from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import math

try:
    import mediapipe as mp  # type: ignore
    import numpy as np
    MEDIAPIPE_AVAILABLE = True
except Exception:  # pragma: no cover
    MEDIAPIPE_AVAILABLE = False
    mp = None  # type: ignore
    np = None  # type: ignore


class PoseAnalyzer:
    """
    Wraps MediaPipe Pose. Performs person-agnostic pose detection
    (since the ByteTracker already gave us bounding boxes).
    """

    def __init__(self, config: Any):
        self.config = config
        self.pose = None
        self.enabled = bool(getattr(config, "enabled", True)) and MEDIAPIPE_AVAILABLE
        if self.enabled:
            try:
                self.pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=float(getattr(config, "min_pose_confidence", 0.5)),
                    min_tracking_confidence=0.5,
                )
            except Exception:
                self.pose = None
                self.enabled = False
        # Per-track pose cache: track_id -> {hand_positions, posture, timestamp}
        self.pose_cache: Dict[int, Dict[str, Any]] = {}

    @property
    def is_available(self) -> bool:
        return self.enabled and self.pose is not None

    def update(self, frame, tracks: List[Any], timestamp: float) -> Dict[int, Dict[str, Any]]:
        """
        Update pose estimates for all tracks. Returns a dict keyed by
        track_id with: { 'left_hand': (x,y), 'right_hand': (x,y),
        'left_wrist': (x,y), 'right_wrist': (x,y),
        'is_crouching': bool, 'is_leaning': bool,
        'confidence': float, 'bbox': tuple }
        """
        if not self.is_available or frame is None or not tracks:
            return {}

        out: Dict[int, Dict[str, Any]] = {}
        h, w = frame.shape[:2]
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            # Crop with padding
            pad = 10
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            try:
                rgb = crop[:, :, ::-1]
                result = self.pose.process(rgb)
            except Exception:
                continue
            if not result or not result.pose_landmarks:
                continue

            lm = result.pose_landmarks.landmark
            ch, cw = cy2 - cy1, cx2 - cx1
            # 15 = left wrist, 16 = right wrist
            try:
                lh = lm[15]
                rh = lm[16]
            except IndexError:
                continue

            def to_px(landmark):
                return (int(cx1 + landmark.x * cw), int(cy1 + landmark.y * ch))

            left_wrist = to_px(lh)
            right_wrist = to_px(rh)
            left_elbow = to_px(lm[13]) if len(lm) > 13 else left_wrist
            right_elbow = to_px(lm[14]) if len(lm) > 14 else right_wrist

            # Crouch: hip (23,24) below knee (25,26) by large margin
            is_crouching = False
            if len(lm) > 26:
                lhip = lm[23]; rhip = lm[24]
                lknee = lm[25]; rknee = lm[26]
                hip_y = (lhip.y + rhip.y) / 2.0
                knee_y = (lknee.y + rknee.y) / 2.0
                is_crouching = (knee_y - hip_y) < 0.05  # hip close to knee

            # Lean: shoulder (11,12) significantly offset from hip
            is_leaning = False
            if len(lm) > 12:
                lsh = lm[11]; rsh = lm[12]
                lhip = lm[23]; rhip = lm[24]
                sh_cx = (lsh.x + rsh.x) / 2.0
                hip_cx = (lhip.x + rhip.x) / 2.0
                is_leaning = abs(sh_cx - hip_cx) > 0.12

            confidence = float(min(lh.visibility, rh.visibility)) if (lh and rh) else 0.0

            entry = {
                "left_wrist": left_wrist,
                "right_wrist": right_wrist,
                "left_elbow": left_elbow,
                "right_elbow": right_elbow,
                "is_crouching": is_crouching,
                "is_leaning": is_leaning,
                "confidence": confidence,
                "bbox": t.bbox,
            }
            out[t.track_id] = entry
            self.pose_cache[t.track_id] = {**entry, "timestamp": timestamp}

        return out

    def hand_near_polygon(self, hand_pos: Tuple[int, int], polygon: List[Tuple[int, int]],
                          max_distance_px: int = 80) -> float:
        """
        Return a score in [0, 1] indicating how close a hand is to a
        polygon. 0 = far, 1 = inside. Uses distance to nearest edge.
        """
        if not self.is_available or hand_pos is None or not polygon:
            return 0.0
        try:
            import cv2
            hx, hy = hand_pos
            poly = np.array(polygon, dtype=np.int32)
            inside = cv2.pointPolygonTest(poly, (float(hx), float(hy)), False) >= 0
            if inside:
                return 1.0
            dist_outside = cv2.pointPolygonTest(poly, (float(hx), float(hy)), True)
            dist = abs(float(dist_outside))
            if dist <= 0:
                return 1.0
            score = max(0.0, 1.0 - dist / float(max_distance_px))
            return score
        except Exception:
            return 0.0

    def get_cached(self, track_id: int) -> Optional[Dict[str, Any]]:
        return self.pose_cache.get(track_id)

    def cleanup_stale(self, active_track_ids: List[int], ttl_seconds: float = 5.0) -> None:
        """Remove stale pose cache entries to bound memory."""
        keep = set(active_track_ids)
        for tid in list(self.pose_cache.keys()):
            if tid not in keep:
                self.pose_cache.pop(tid, None)

    def close(self) -> None:
        if self.pose is not None:
            try:
                self.pose.close()
            except Exception:
                pass
            self.pose = None
