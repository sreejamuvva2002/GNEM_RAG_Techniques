"""Centralised logging configuration for the Simple RAG pipeline.

Usage::

    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root ``simple_rag`` logger once.

    Args:
        level: Logging level name (e.g. ``"DEBUG"``, ``"INFO"``).
               Parsed via ``logging.getLevelName``.

    This function is idempotent — repeated calls are harmless.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    root_logger = logging.getLogger("simple_rag")
    root_logger.setLevel(numeric_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``simple_rag`` namespace.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance scoped under ``simple_rag``.
    """
    return logging.getLogger(f"simple_rag.{name}")
