"""YAML configuration loader with validation.

The loader resolves all relative paths against the project root
(``simple_rag/``) so that ``main.py`` and tests can run from any cwd.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.exceptions import ConfigError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Project root is two levels up from this file:
#   simple_rag/src/utils/config_loader.py  →  simple_rag/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate ``config.yaml``.

    Args:
        config_path: Absolute or project-relative path to the YAML file.
                     Defaults to ``config/config.yaml`` inside the project
                     root.

    Returns:
        Parsed configuration dictionary with path values resolved to
        absolute ``Path`` objects.

    Raises:
        ConfigError: If the file is missing, unparseable, or lacks
                     required top-level keys.
    """
    if config_path is None:
        config_path = _PROJECT_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = _PROJECT_ROOT / config_path

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg: dict[str, Any] = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML: {exc}") from exc

    _validate(cfg)
    _resolve_paths(cfg)

    logger.info("Configuration loaded from %s", config_path)
    return cfg


def get_project_root() -> Path:
    """Return the resolved project root (``simple_rag/``)."""
    return _PROJECT_ROOT


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_REQUIRED_SECTIONS = ("paths", "data", "chunking", "embedding", "retrieval")


def _validate(cfg: dict[str, Any]) -> None:
    """Check that all mandatory top-level sections exist."""
    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            raise ConfigError(
                f"Missing required config section: '{section}'"
            )


def _resolve_paths(cfg: dict[str, Any]) -> None:
    """Convert path values under ``paths`` to absolute ``Path`` objects."""
    paths_section = cfg.get("paths", {})
    for key, value in paths_section.items():
        p = Path(value)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        paths_section[key] = p

    # Ensure output parent directory exists.
    output_path: Path = paths_section.get("output", _PROJECT_ROOT / "output" / "contexts.xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure index/cache directory exists.
    index_dir: Path = paths_section.get("index_dir", _PROJECT_ROOT / ".cache")
    index_dir.mkdir(parents=True, exist_ok=True)
