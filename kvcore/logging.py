"""Logging helpers for KVCore."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure a concise package-local logger once."""
    logger = logging.getLogger("kvcore")
    if logger.handlers:
        logger.setLevel(level.upper())
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under kvcore."""
    return logging.getLogger(f"kvcore.{name}")
