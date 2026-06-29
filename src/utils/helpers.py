"""
Helper Utilities
================
Common utility functions for visualization and data processing.
"""

import cv2
import numpy as np
from typing import Tuple, List


def draw_text_with_background(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.6,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    padding: int = 5
) -> np.ndarray:
    """Draw text with a colored background rectangle."""
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    
    # Draw background
    cv2.rectangle(
        img,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
    
    return img


def create_grid_layout(
    num_images: int,
    grid_cols: int = 4,
    cell_width: int = 320,
    cell_height: int = 240
) -> Tuple[int, int]:
    """Calculate grid dimensions for image layout."""
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    width = grid_cols * cell_width
    height = grid_rows * cell_height
    return width, height


def stack_images_horizontal(images: List[np.ndarray], target_height: int = 480) -> np.ndarray:
    """Stack images horizontally with consistent height."""
    if not images:
        return np.zeros((target_height, 10, 3), dtype=np.uint8)
    
    # Resize all to same height
    resized = []
    for img in images:
        if img is None or img.size == 0:
            continue
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, target_height)))
    
    if not resized:
        return np.zeros((target_height, 10, 3), dtype=np.uint8)
    
    return np.hstack(resized)


def stack_images_vertical(images: List[np.ndarray], target_width: int = 640) -> np.ndarray:
    """Stack images vertically with consistent width."""
    if not images:
        return np.zeros((10, target_width, 3), dtype=np.uint8)
    
    resized = []
    for img in images:
        if img is None or img.size == 0:
            continue
        h, w = img.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        resized.append(cv2.resize(img, (target_width, new_h)))
    
    if not resized:
        return np.zeros((10, target_width, 3), dtype=np.uint8)
    
    return np.vstack(resized)


def create_grid_view(images: List[np.ndarray], cols: int = 2, cell_size: Tuple[int, int] = (320, 240)) -> np.ndarray:
    """Create a grid view of images."""
    if not images:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)
    
    rows = (len(images) + cols - 1) // cols
    cw, ch = cell_size
    
    # Create output grid
    grid = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        if img is None or img.size == 0:
            continue
        row = i // cols
        col = i % cols
        resized = cv2.resize(img, (cw, ch))
        grid[row*ch:(row+1)*ch, col*cw:(col+1)*cw] = resized
    
    return grid


def draw_progress_bar(
    img: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    progress: float,
    bg_color: Tuple[int, int, int] = (50, 50, 50),
    fg_color: Tuple[int, int, int] = (0, 255, 0),
    border_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Draw a progress bar."""
    # Background
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    
    # Progress fill
    fill_width = int(width * min(1.0, max(0.0, progress)))
    if fill_width > 0:
        cv2.rectangle(img, (x, y), (x + fill_width, y + height), fg_color, -1)
    
    # Border
    cv2.rectangle(img, (x, y), (x + width, y + height), border_color, 1)
    
    return img


def timestamp_to_str(timestamp: float) -> str:
    """Convert Unix timestamp to formatted string."""
    import datetime
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"
