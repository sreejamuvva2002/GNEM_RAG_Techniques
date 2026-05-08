"""Excel data loader adapter.

Handles the GNEM schema including trailing empty columns and
missing / NaN cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.exceptions import DataLoaderError
from src.interfaces.data_loader_interface import DataLoaderInterface
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExcelDataLoaderAdapter(DataLoaderInterface):
    """Load tabular data from ``.xlsx`` files using pandas + openpyxl."""

    # ------------------------------------------------------------------ #
    # DataLoaderInterface implementation                                  #
    # ------------------------------------------------------------------ #

    def load(
        self,
        path: Path,
        sheet_name: str,
        columns: list[str],
    ) -> list[dict[str, Any]]:
        """Load rows from an Excel workbook.

        Trailing empty columns in the file are ignored automatically
        because we only keep the columns listed in ``columns``.

        Missing or NaN values are converted to ``None`` so downstream
        code can handle them uniformly.

        Args:
            path: Path to the ``.xlsx`` file.
            sheet_name: Worksheet name to read.
            columns: Ordered column names to retain.

        Returns:
            List of row dicts.

        Raises:
            DataLoaderError: On I/O errors, missing sheet, or missing
                             expected columns.
        """
        path = Path(path)
        if not path.exists():
            raise DataLoaderError(f"Data file not found: {path}")

        try:
            df = pd.read_excel(
                path,
                sheet_name=sheet_name,
                engine="openpyxl",
            )
        except Exception as exc:
            raise DataLoaderError(
                f"Failed to read Excel file '{path}', sheet '{sheet_name}': {exc}"
            ) from exc

        # Validate expected columns are present.
        missing = set(columns) - set(df.columns)
        if missing:
            raise DataLoaderError(
                f"Missing expected columns in '{path}': {sorted(missing)}. "
                f"Available columns: {list(df.columns)}"
            )

        # Keep only the requested columns (ignores trailing empties).
        df = df[columns]

        # Replace NaN with None for uniform downstream handling.
        df = df.where(pd.notna(df), None)

        rows: list[dict[str, Any]] = df.to_dict(orient="records")
        logger.info(
            "Loaded %d rows from '%s' (sheet '%s')",
            len(rows),
            path.name,
            sheet_name,
        )
        return rows
