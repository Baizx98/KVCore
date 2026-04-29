from __future__ import annotations

import logging
import os

DEFAULT_LOG_LEVEL = "WARNING"
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOGGER_NAME = "kvcore"
_CONFIGURED = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a KVCore logger with lazy default configuration."""
    configure_logging()
    if name is None or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    if name.startswith(_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def configure_logging(
    level: str | int | None = None,
    *,
    force: bool = False,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> None:
    """Configure KVCore logging.

    The default level is intentionally WARNING to keep library imports quiet.
    Users can set KVCORE_LOG_LEVEL or call this function from scripts/tests.
    """
    global _CONFIGURED
    if level is None and _CONFIGURED and not force:
        return
    resolved_level = _coerce_level(
        level or os.environ.get("KVCORE_LOG_LEVEL") or DEFAULT_LOG_LEVEL
    )
    root_logger = logging.getLogger()
    if force or not root_logger.handlers:
        logging.basicConfig(level=resolved_level, format=log_format, force=force)
    logging.getLogger(_LOGGER_NAME).setLevel(resolved_level)
    _CONFIGURED = True


def set_log_level(level: str | int) -> None:
    logging.getLogger(_LOGGER_NAME).setLevel(_coerce_level(level))


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    normalized = level.upper()
    if normalized.isdigit():
        return int(normalized)
    value = getattr(logging, normalized, None)
    if not isinstance(value, int):
        raise ValueError(f"Unknown log level: {level!r}")
    return value


__all__ = [
    "configure_logging",
    "get_logger",
    "set_log_level",
]
