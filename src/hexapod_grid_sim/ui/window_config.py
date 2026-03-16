"""Window configuration manager for multi-window layout persistence."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

_lock = threading.Lock()

_DEFAULT_CONFIG_DIR = Path.home() / ".hexapod_grid_sim"
_CONFIG_FILENAME = "window_layout.json"


def _config_path() -> Path:
    """Return the path to the window layout JSON file."""
    env_override = os.environ.get("HEXAPOD_WINDOW_CONFIG")
    if env_override:
        return Path(env_override)
    return _DEFAULT_CONFIG_DIR / _CONFIG_FILENAME


def _load() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        with _lock:
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_window_position(name: str) -> Optional[tuple[int, int]]:
    """Return (x, y) for a named window, or None if not stored."""
    entry = _load().get(name)
    if entry and "x" in entry and "y" in entry:
        return (int(entry["x"]), int(entry["y"]))
    return None


def get_window_size(name: str) -> Optional[tuple[int, int]]:
    """Return (width, height) for a named window, or None if not stored."""
    entry = _load().get(name)
    if entry and "w" in entry and "h" in entry:
        return (int(entry["w"]), int(entry["h"]))
    return None


def save_window_geometry(name: str, x: int, y: int, w: int, h: int) -> None:
    """Persist the position and size of a named window."""
    data = _load()
    data[name] = {"x": x, "y": y, "w": w, "h": h}
    _save(data)


def get_all_window_config() -> dict[str, dict[str, int]]:
    """Return the full window configuration dictionary."""
    return _load()
