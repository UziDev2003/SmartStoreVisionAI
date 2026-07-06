"""
Frame Preprocessor
==================
Optional image-quality pipeline that runs on each frame before
inference. All operations are gated by config flags so a user
can pick the exact preprocessing chain they want.

Pipeline order (configurable):
  1. Resize (preserving aspect ratio)
  2. Letterbox (square padding for YOLO)
  3. Color conversion (BGR -> RGB or grayscale)
  4. White balance (Gray World)
  5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
  6. Gamma correction
  7. Gaussian blur / sharpening
  8. Denoising (fastNlMeans)
  9. Normalize (subtract mean, divide std) - for ONNX/Torch models
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional
import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    enabled: bool = True
    target_width: int = 0  # 0 = no resize
    target_height: int = 0
    letterbox: bool = False
    letterbox_color: Tuple[int, int, int] = (114, 114, 114)
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    white_balance: bool = False
    gamma: float = 1.0  # 1.0 = no change
    blur_ksize: int = 0  # 0/1 = no blur
    sharpen: bool = False
    denoise: bool = False
    denoise_strength: int = 7
    convert_rgb: bool = False
    normalize: bool = False  # mean/std normalization (for ONNX/Torch)


class FramePreprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        if self.config.clahe:
            try:
                self._clahe = cv2.createCLAHE(
                    clipLimit=self.config.clahe_clip_limit,
                    tileGridSize=(self.config.clahe_grid_size,
                                  self.config.clahe_grid_size),
                )
            except Exception:
                self._clahe = None
        else:
            self._clahe = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.enabled or frame is None:
            return frame
        out = frame

        # 1) Resize
        if self.config.target_width > 0 and self.config.target_height > 0:
            out = cv2.resize(out, (self.config.target_width, self.config.target_height),
                             interpolation=cv2.INTER_AREA)
        elif self.config.target_width > 0:
            h, w = out.shape[:2]
            new_h = int(h * self.config.target_width / w)
            out = cv2.resize(out, (self.config.target_width, new_h),
                             interpolation=cv2.INTER_AREA)

        # 2) Letterbox to square
        if self.config.letterbox:
            out = self._letterbox(out)

        # 3) Color convert
        if self.config.convert_rgb and len(out.shape) == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        # 4) White balance (Gray World)
        if self.config.white_balance and len(out.shape) == 3:
            out = self._gray_world(out)

        # 5) CLAHE (on L channel)
        if self.config.clahe and self._clahe is not None and len(out.shape) == 3:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self._clahe.apply(l)
            out = cv2.merge((l2, a, b))
            out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
        elif self.config.clahe and self._clahe is not None and len(out.shape) == 2:
            out = self._clahe.apply(out)

        # 6) Gamma
        if abs(self.config.gamma - 1.0) > 1e-3:
            g = float(self.config.gamma)
            table = np.array([((i / 255.0) ** g) * 255
                              for i in range(256)]).astype("uint8")
            out = cv2.LUT(out, table)

        # 7) Blur
        if self.config.blur_ksize and self.config.blur_ksize > 1:
            k = self.config.blur_ksize
            if k % 2 == 0:
                k += 1  # Gaussian kernel must be odd
            out = cv2.GaussianBlur(out, (k, k), 0)

        # 8) Sharpen
        if self.config.sharpen:
            out = self._sharpen(out)

        # 9) Denoise
        if self.config.denoise and len(out.shape) == 3:
            out = cv2.fastNlMeansDenoisingColored(
                out, None, self.config.denoise_strength,
                self.config.denoise_strength, 7, 21)

        # 10) Normalize (model input)
        if self.config.normalize:
            out = out.astype(np.float32) / 255.0
            # mean and std for ImageNet
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            if len(out.shape) == 3 and out.shape[2] == 3:
                out = (out - mean) / std

        return out

    def _letterbox(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        size = max(h, w)
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left
        return cv2.copyMakeBorder(frame, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=self.config.letterbox_color)

    def _gray_world(self, frame: np.ndarray) -> np.ndarray:
        avg_b = frame[:, :, 0].mean()
        avg_g = frame[:, :, 1].mean()
        avg_r = frame[:, :, 2].mean()
        avg = (avg_b + avg_g + avg_r) / 3.0 + 1e-6
        out = frame.astype(np.float32)
        out[:, :, 0] = np.clip(out[:, :, 0] * (avg / avg_b), 0, 255)
        out[:, :, 1] = np.clip(out[:, :, 1] * (avg / avg_g), 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] * (avg / avg_r), 0, 255)
        return out.astype(np.uint8)

    def _sharpen(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
