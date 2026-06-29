"""
Configuration Utilities
======================
Handles loading and saving configuration files.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    p = Path(path)
    
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    if p.suffix in ['.yaml', '.yml']:
        with open(p, 'r') as f:
            return yaml.safe_load(f) or {}
    elif p.suffix == '.json':
        with open(p, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")


def save_config(config: Dict[str, Any], path: str, format: str = 'yaml'):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        path: Output file path
        format: 'yaml' or 'json'
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'yaml':
        with open(p, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif format == 'json':
        with open(p, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries (override takes precedence)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
