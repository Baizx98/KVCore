from __future__ import annotations

import logging

import pytest

from kvcore.utils.log import configure_logging, get_logger, set_log_level


def test_configure_logging_sets_kvcore_level() -> None:
    configure_logging("DEBUG", force=True)

    assert get_logger().getEffectiveLevel() == logging.DEBUG
    assert get_logger("kvcore.test").getEffectiveLevel() == logging.DEBUG


def test_set_log_level_accepts_names_and_rejects_unknown() -> None:
    configure_logging("WARNING", force=True)

    set_log_level("ERROR")
    assert get_logger("kvcore.test").getEffectiveLevel() == logging.ERROR

    with pytest.raises(ValueError, match="Unknown log level"):
        set_log_level("LOUD")
