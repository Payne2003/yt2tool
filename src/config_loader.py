"""
src/config_loader.py — Load YAML config + env-var overrides
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | None = None) -> dict:
    """Load config.yaml relative to this file's parent (project root)."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # ── Env-var overrides ─────────────────────────────────────────────────────
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        cfg.setdefault("openai", {})["api_key"] = openai_key

    return cfg


def get_nested(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested config dict."""
    node = cfg
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
    return node
