from __future__ import annotations
"""Custom exception hierarchy for the Simple RAG pipeline.

All domain-specific exceptions inherit from ``SimpleRAGError`` so callers
can catch the entire family with a single handler when needed.
"""


class SimpleRAGError(Exception):
    """Base exception for all Simple RAG errors."""


class ConfigError(SimpleRAGError):
    """Raised when configuration loading or validation fails.

    Examples:
        Missing required keys, malformed YAML, invalid value ranges.
    """


class EmbeddingError(SimpleRAGError):
    """Raised when embedding generation fails.

    Examples:
        Timeout, unexpected response shape, model not loaded.
    """


class OllamaUnavailableError(EmbeddingError):
    """Raised when the Ollama server cannot be reached.

    This is a specialisation of ``EmbeddingError`` because the root
    cause is the embedding provider being offline.
    """


class RetrievalError(SimpleRAGError):
    """Raised when the retrieval pipeline encounters an error.

    Examples:
        Empty index, score fusion failure, candidate pool exhaustion.
    """


class DataLoaderError(SimpleRAGError):
    """Raised when data loading or parsing fails.

    Examples:
        File not found, unexpected schema, corrupt workbook.
    """


"""Centralised logging configuration for the Simple RAG pipeline.

Usage::

            logger.info("Pipeline started")
"""


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



logger = get_logger(__name__)

"""YAML configuration loader with validation.

The loader resolves all relative paths against the project root
(``simple_rag/``) so that ``main.py`` and tests can run from any cwd.
"""


import os
from pathlib import Path
from typing import Any

import yaml



# Project root is two levels up from this file:
#   simple_rag/src/utils.py  →  simple_rag/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

_REQUIRED_SECTIONS = ("paths", "data", "chunking", "embedding", "vector_store", "retrieval")


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

    # Ensure output parent directories exist for all output paths.
    for key in ("output", "chunks_output", "embeddings_output"):
        out_path: Path | None = paths_section.get(key)
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure index/cache directory exists.
    index_dir: Path = paths_section.get("index_dir", _PROJECT_ROOT / ".cache")
    index_dir.mkdir(parents=True, exist_ok=True)


"""Excel and data I/O utilities.

Centralises all Excel reading / writing so that pipeline scripts
stay focused on orchestration logic.  Uses ``openpyxl`` as the
engine and ``pandas`` for DataFrame ↔ Excel conversions.
"""


from pathlib import Path
from typing import Any

import pandas as pd




# ------------------------------------------------------------------
# Question loading
# ------------------------------------------------------------------

def load_questions(path: str | Path) -> pd.DataFrame:
    """Load questions from CSV or XLSX.

    Auto-detects format by file extension.  Returns a DataFrame
    with columns ``question_id`` and ``question``.

    If the source lacks a ``question_id`` column, one is generated
    as ``Q001``, ``Q002``, …

    Args:
        path: Path to the questions file.

    Returns:
        DataFrame with ``question_id`` and ``question`` columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the ``question`` column cannot be found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Normalise column names to lowercase for matching.
    col_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_map)

    # Find the question column (case-insensitive).
    q_col = None
    for c in df.columns:
        if c in ("question", "questions"):
            q_col = c
            break
    if q_col is None:
        raise ValueError(
            f"Cannot find 'question' column in {path}. "
            f"Found columns: {list(df.columns)}"
        )

    # Generate question_id if missing.
    if "question_id" not in df.columns:
        df["question_id"] = [f"Q{i + 1:03d}" for i in range(len(df))]

    result = df[["question_id", q_col]].copy()
    result.columns = ["question_id", "question"]

    logger.info("Loaded %d questions from %s", len(result), path.name)
    return result


# ------------------------------------------------------------------
# Multi-sheet Excel writing
# ------------------------------------------------------------------

def write_multi_sheet_excel(
    path: str | Path,
    sheets: dict[str, pd.DataFrame],
) -> None:
    """Write multiple DataFrames to an Excel file, one per sheet.

    Args:
        path: Destination ``.xlsx`` path.
        sheets: Dict mapping sheet name → DataFrame.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    sheet_summary = ", ".join(
        f"{name}({len(df)} rows)" for name, df in sheets.items()
    )
    logger.info("Wrote Excel to %s — sheets: %s", path, sheet_summary)


# ------------------------------------------------------------------
# Single-sheet Excel reading
# ------------------------------------------------------------------

def read_excel_sheet(
    path: str | Path,
    sheet_name: str,
) -> pd.DataFrame:
    """Read a specific sheet from an Excel file.

    Args:
        path: Path to the ``.xlsx`` file.
        sheet_name: Name of the sheet to read.

    Returns:
        DataFrame with the sheet contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the sheet is not found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except ValueError as exc:
        raise ValueError(
            f"Sheet '{sheet_name}' not found in {path}. Error: {exc}"
        ) from exc
