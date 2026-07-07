"""
Zone I/O utilities — save/load zones to YAML with per-camera overrides.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

from src.analytics.zone_analytics import Zone


RUNTIME_ZONES_DIR = Path("config/zones_runtime")
MAIN_ZONES_FILE = Path("config/zones.yaml")


def save_zones(
    zones: List[Zone],
    path: Optional[Path] = None,
    camera_id: Optional[str] = None,
) -> Path:
    """Save zones to YAML. If `camera_id` is given, writes to runtime dir."""
    if path is None:
        if camera_id is None:
            path = MAIN_ZONES_FILE
        else:
            RUNTIME_ZONES_DIR.mkdir(parents=True, exist_ok=True)
            path = RUNTIME_ZONES_DIR / f"{camera_id}.yaml"
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    data = {
        "zones": [
            {
                "id": z.zone_id,
                "name": z.name,
                "type": z.zone_type,
                "capacity": z.capacity,
                "dwell_threshold": z.dwell_threshold,
                "polygon": [list(p) for p in z.polygon],
            }
            for z in zones
        ]
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    return Path(path)


def load_zones(
    path: Optional[Path] = None,
    camera_id: Optional[str] = None,
) -> List[Zone]:
    """Load zones from YAML. Falls back to main zones file if runtime file missing."""
    if path is None:
        if camera_id is not None:
            runtime = RUNTIME_ZONES_DIR / f"{camera_id}.yaml"
            if runtime.exists():
                path = runtime
            else:
                path = MAIN_ZONES_FILE
        else:
            path = MAIN_ZONES_FILE

    if not Path(path).exists():
        return []

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    zones: List[Zone] = []
    for z in data.get("zones", []):
        zones.append(
            Zone(
                zone_id=z["id"],
                name=z["name"],
                zone_type=z.get("type", "general"),
                polygon=[(p[0], p[1]) for p in z["polygon"]],
                capacity=z.get("capacity"),
                dwell_threshold=z.get("dwell_threshold", 120.0),
            )
        )
    return zones


def list_runtime_zone_files() -> List[Path]:
    """List all saved per-camera zone files."""
    if not RUNTIME_ZONES_DIR.exists():
        return []
    return sorted(RUNTIME_ZONES_DIR.glob("*.yaml"))


def delete_runtime_zones(camera_id: str) -> bool:
    """Delete a per-camera runtime zone file."""
    p = RUNTIME_ZONES_DIR / f"{camera_id}.yaml"
    if p.exists():
        p.unlink()
        return True
    return False


def zones_from_canvas(canvas_objects: List[Dict[str, Any]]) -> List[Zone]:
    """
    Convert streamlit-drawable-canvas JSON objects to Zone list.
    Supports 'polygon' and 'rect' shape types.
    """
    zones: List[Zone] = []
    for i, obj in enumerate(canvas_objects):
        t = obj.get("type")
        # Default label if not provided
        name = obj.get("label", f"Zone {i+1}")
        zone_id = name.lower().replace(" ", "_")[:32] or f"zone_{i+1}"
        # Default zone type from stroke color or "general"
        # streamlit-drawable-canvas stores color in obj['stroke'] (string like '#ff0000')
        ztype = "general"
        try:
            stroke = (obj.get("stroke") or "").lower()
            if "#ff0000" in stroke or "#f00" in stroke:
                ztype = "restricted"
            elif "#00ff00" in stroke or "#0f0" in stroke:
                ztype = "checkout"
            elif "#ffff00" in stroke or "#ff0" in stroke:
                ztype = "shelf"
            elif "#00ffff" in stroke or "#0ff" in stroke:
                ztype = "entrance"
        except Exception:
            ztype = "general"

        polygon: List[tuple] = []
        if t == "polygon" and "points" in obj:
            for pt in obj["points"]:
                if isinstance(pt, dict):
                    polygon.append((int(pt.get("x", 0)), int(pt.get("y", 0))))
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    polygon.append((int(pt[0]), int(pt[1])))
        elif t == "rect":
            left = float(obj.get("left", 0))
            top = float(obj.get("top", 0))
            width = float(obj.get("width", 0))
            height = float(obj.get("height", 0))
            polygon = [
                (int(left), int(top)),
                (int(left + width), int(top)),
                (int(left + width), int(top + height)),
                (int(left), int(top + height)),
            ]
        elif t == "path" and "path" in obj:
            # Path is a list of commands like [['M', x, y], ['L', x, y], ...]
            try:
                raw = obj["path"]
                
                # Fabric.js path coordinates are relative to the center of the bounding box if shifted
                # But when drawn by the user and not moved, they are absolute, or left/top are the bounding box top-left
                # We can just extract the x, y which are the last two elements of the command array
                
                # For `st_canvas`, when left and top are provided, they offset the path points
                # Actually, fabric.js path commands are relative to center of width/height?
                # It's safer to use the points if they are absolute. Let's just grab the x, y values and add left/top
                
                left = float(obj.get("left", 0))
                top = float(obj.get("top", 0))
                
                for p in raw:
                    if isinstance(p, (list, tuple)) and len(p) >= 3:
                        try:
                            # Usually ['M', x, y] or ['L', x, y]
                            # Path is usually relative to center if left/top are set, or absolute if not
                            # For streamlit_drawable_canvas, if it's drawn, it's usually absolute + offset
                            # Actually, `st_canvas` paths have `path` as relative to `left`, `top`
                            # Wait, actually let's just grab the numbers and see
                            x = float(p[-2]) + left
                            y = float(p[-1]) + top
                            polygon.append((int(x), int(y)))
                        except (ValueError, TypeError):
                            continue
            except Exception:
                polygon = []

        if len(polygon) >= 3:
            zones.append(
                Zone(
                    zone_id=zone_id,
                    name=name,
                    zone_type=ztype,
                    polygon=polygon,
                    capacity=None,
                    dwell_threshold=120.0,
                )
            )
    return zones
