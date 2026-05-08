"""Interface for data-source loaders.

Implementations may read Excel, CSV, databases, APIs, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DataLoaderInterface(ABC):
    """Contract for loading tabular datasets."""

    @abstractmethod
    def load(
        self,
        path: Path,
        sheet_name: str,
        columns: list[str],
    ) -> list[dict[str, Any]]:
        """Load rows from a data source.

        Args:
            path: File path to the data source.
            sheet_name: Sheet / table name within the source.
            columns: Ordered list of column names to retain.

        Returns:
            A list of row dicts.  Keys are column names from
            ``columns``; values are the cell values (may be ``None``
            for missing data).  Rows preserve the original order.

        Raises:
            DataLoaderError: On any I/O or schema error.
        """
