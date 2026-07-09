"""
Video Clip Extractor
====================
Saves a clip (e.g. 10 sec before + 10 sec after) when an alert
fires. Uses a rolling buffer of recent frames + the recorded
file output. Works with cv2.VideoWriter (mp4v codec).

Usage:
    clipper = VideoClipper(pre_seconds=10, post_seconds=10)
    clipper.start_recording(cap, fps, frame_size)
    for frame, ts in stream:
        clipper.add_frame(frame, ts)
        if should_save_alert:
            path = clipper.save_clip(alert_id="...")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
from collections import deque
import time
import cv2
import numpy as np


class VideoClipper:
    """
    Maintains a rolling buffer of recent frames and their
    timestamps. When `save_clip` is called, writes the buffered
    frames plus a configurable number of post-event seconds to
    an MP4 file.
    """
    def __init__(self, pre_seconds: float = 10.0, post_seconds: float = 10.0,
                 output_dir: str = "data/clips", fps: int = 25,
                 codec: str = "mp4v"):
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.codec = codec
        self._buffer: deque = deque()  # list of (ts, frame)
        self._max_buffer_size = int(pre_seconds * fps) + 5
        self._writer: Optional[cv2.VideoWriter] = None
        self._current_clip_path: Optional[str] = None
        self._current_clip_id: Optional[str] = None
        self._post_frames_remaining: int = 0
        self._frame_size: Optional[Tuple[int, int]] = None

    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        if frame is None:
            return
        ts = timestamp if timestamp is not None else time.time()
        self._buffer.append((ts, frame.copy()))
        while len(self._buffer) > self._max_buffer_size:
            self._buffer.popleft()
        # If we are recording a post-event clip, write the frame
        if self._writer is not None and self._post_frames_remaining > 0:
            self._writer.write(frame)
            self._post_frames_remaining -= 1
            if self._post_frames_remaining == 0:
                self._finalize_clip()

    def save_clip(self, alert_id: str, frame: Optional[np.ndarray] = None) -> str:
        """
        Begin recording a clip: flush the pre-event buffer to a file,
        then keep writing the post-event frames for `post_seconds`.
        """
        if not self._buffer:
            return ""
        # If a clip is already in progress, finalize it first
        if self._writer is not None:
            self._finalize_clip()
        # Use the latest frame's size
        ts, latest = self._buffer[-1]
        h, w = latest.shape[:2]
        self._frame_size = (w, h)
        out_path = self.output_dir / f"{alert_id}.mp4"
        
        # Try codecs in order of compatibility: mp4v -> avc1 -> XVID -> MJPG
        # mp4v is universally supported across all OpenCV builds
        # avc1/H264 requires specific backends that may not be available
        codec_priority = ["mp4v", "avc1", "XVID", "MJPG"]
        if self.codec in codec_priority:
            # Move current codec to front of priority list
            codecs_to_try = [self.codec] + [c for c in codec_priority if c != self.codec]
        else:
            codecs_to_try = codec_priority
        
        self._writer = None
        for codec in codecs_to_try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self._writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))
            if self._writer.isOpened():
                print(f"[INFO] VideoClipper initialized with codec: {codec}")
                break
            else:
                self._writer.release()
                self._writer = None
        
        if self._writer is None:
            print("[WARN] VideoClipper failed to initialize with any codec")
            return ""
        # Flush pre-event buffer
        for _, f in self._buffer:
            self._writer.write(f)
        # Add the current frame if provided (synchronously)
        if frame is not None:
            self._writer.write(frame)
        self._current_clip_path = str(out_path)
        self._current_clip_id = alert_id
        # Schedule post-event frames
        self._post_frames_remaining = int(self.post_seconds * self.fps)
        return str(out_path)

    def _finalize_clip(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None
        self._post_frames_remaining = 0
        self._current_clip_path = None
        self._current_clip_id = None

    def is_recording(self) -> bool:
        return self._writer is not None

    def get_current_clip(self) -> Optional[str]:
        return self._current_clip_path

    def close(self) -> None:
        self._finalize_clip()
