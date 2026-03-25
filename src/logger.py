"""
src/logger.py — Colored rotating logger
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGERS: dict[str, logging.Logger] = {}
_ROOT_LOGGER: logging.Logger | None = None


def setup_logger(
    level: str = "INFO",
    log_file: str = "logs/yt2dataset.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
) -> None:
    global _ROOT_LOGGER

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("yt2dataset")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    _ROOT_LOGGER = root


def get_logger(name: str) -> logging.Logger:
    full_name = f"yt2dataset.{name}"
    if full_name not in _LOGGERS:
        _LOGGERS[full_name] = logging.getLogger(full_name)
    return _LOGGERS[full_name]
