"""Interface for result writers.

Implementations may write Excel, CSV, JSON, databases, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ResultWriterInterface(ABC):
    """Contract for writing RAG pipeline output."""

    @abstractmethod
    def write(self, results: list[dict[str, Any]], path: Path) -> None:
        """Write a list of result records to the target destination.

        Args:
            results: Each dict represents one query result.  Expected
                     keys depend on the pipeline, but typically include
                     ``question``, ``retrieved_context``,
                     ``parent_ids``, ``companies``, and score columns.
            path: Destination file path.  Parent directories are
                  created automatically if absent.

        Raises:
            SimpleRAGError: On any I/O error.
        """
