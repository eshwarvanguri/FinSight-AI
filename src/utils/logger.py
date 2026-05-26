"""Centralized logging using loguru."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.utils.config import settings


def setup_logger(name: str = "finsight", log_file: str | None = None) -> "logger":
    """Configure loguru logger with sensible defaults."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=settings.LOG_LEVEL,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

    return logger


log = setup_logger()
