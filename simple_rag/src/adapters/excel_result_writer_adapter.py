"""Excel result writer adapter.

Writes RAG pipeline output to ``.xlsx`` files using pandas + openpyxl.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.interfaces.result_writer_interface import ResultWriterInterface
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExcelResultWriterAdapter(ResultWriterInterface):
    """Write pipeline results to an Excel workbook."""

    # ------------------------------------------------------------------ #
    # ResultWriterInterface implementation                                #
    # ------------------------------------------------------------------ #

    def write(self, results: list[dict[str, Any]], path: Path) -> None:
        """Write result records to an ``.xlsx`` file.

        Parent directories are created automatically if absent.

        Args:
            results: List of result dicts.  Expected keys:
                     ``question``, ``retrieved_context``,
                     ``parent_ids``, ``companies``,
                     ``similarity_scores``, ``keyword_scores``,
                     ``combined_scores``.
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(results)

        # Enforce column order.
        ordered_columns = [
            "question",
            "retrieved_context",
            "parent_ids",
            "companies",
            "similarity_scores",
            "keyword_scores",
            "combined_scores",
        ]
        # Only keep columns that exist in the data.
        ordered_columns = [c for c in ordered_columns if c in df.columns]
        df = df[ordered_columns]

        df.to_excel(path, index=False, engine="openpyxl")
        logger.info("Results written to %s (%d rows)", path, len(df))
