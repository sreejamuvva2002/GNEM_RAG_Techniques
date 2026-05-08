"""Parent–child chunker for tabular (row-level) data.

Each spreadsheet row becomes one *parent* chunk (all fields) plus
several *child* chunks (field-grouped slices).  The groupings are
read from ``config.yaml`` so they can be adjusted without touching
code.

This module is **pure domain logic** — no I/O, no external services.
"""

from __future__ import annotations

import math
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------

class Chunk:
    """Lightweight container for a text chunk and its metadata.

    Attributes:
        chunk_id: Globally unique identifier for the chunk.
        parent_id: Identifier of the parent record this chunk
                   derives from.
        text: The formatted text content.
        metadata: Arbitrary metadata dict (e.g. company name, group).
        chunk_type: ``"parent"`` or ``"child"``.
        group_name: Name of the child group (``None`` for parents).
    """

    __slots__ = (
        "chunk_id",
        "parent_id",
        "text",
        "metadata",
        "chunk_type",
        "group_name",
    )

    def __init__(
        self,
        chunk_id: str,
        parent_id: str,
        text: str,
        metadata: dict[str, Any],
        chunk_type: str = "parent",
        group_name: str | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.parent_id = parent_id
        self.text = text
        self.metadata = metadata
        self.chunk_type = chunk_type
        self.group_name = group_name

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for adapter consumption."""
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "text": self.text,
            "metadata": self.metadata,
            "chunk_type": self.chunk_type,
            "group_name": self.group_name,
        }


# ------------------------------------------------------------------
# Chunker
# ------------------------------------------------------------------

class ParentChildChunker:
    """Converts tabular rows into parent and child chunks.

    Args:
        text_columns: Ordered list of column names to include in
                      parent chunks.
        child_groups: Mapping of group name → list of column names.
                      Loaded from ``config.yaml``'s
                      ``chunking.child_groups`` section.
    """

    def __init__(
        self,
        text_columns: list[str],
        child_groups: dict[str, list[str]],
    ) -> None:
        self._text_columns = text_columns
        self._child_groups = child_groups

    # -------------------------------------------------------------- #
    # Public API                                                      #
    # -------------------------------------------------------------- #

    def chunk_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[list[Chunk], list[Chunk]]:
        """Convert every row into parent + child chunks.

        Args:
            rows: List of row dicts as returned by the data loader.

        Returns:
            A 2-tuple ``(parents, children)`` where each element is a
            list of ``Chunk`` objects.
        """
        parents: list[Chunk] = []
        children: list[Chunk] = []

        for idx, row in enumerate(rows):
            parent_id = f"parent_{idx}"

            parent_text = self._format_parent(row)
            company_name = self._safe_str(row.get("Company"))

            parent = Chunk(
                chunk_id=parent_id,
                parent_id=parent_id,
                text=parent_text,
                metadata={"company": company_name, "row_index": idx},
                chunk_type="parent",
            )
            parents.append(parent)

            for group_name, columns in self._child_groups.items():
                child_text = self._format_child(row, columns, group_name)
                if not child_text:
                    # All fields in this group were empty — skip.
                    continue

                child_id = f"child_{idx}_{group_name}"
                child = Chunk(
                    chunk_id=child_id,
                    parent_id=parent_id,
                    text=child_text,
                    metadata={
                        "company": company_name,
                        "row_index": idx,
                        "group": group_name,
                    },
                    chunk_type="child",
                    group_name=group_name,
                )
                children.append(child)

        logger.info(
            "Chunked %d rows → %d parents, %d children",
            len(rows),
            len(parents),
            len(children),
        )
        return parents, children

    # -------------------------------------------------------------- #
    # Internal helpers                                                #
    # -------------------------------------------------------------- #

    def _format_parent(self, row: dict[str, Any]) -> str:
        """Format a full row as a labeled-field text block.

        Empty / NaN values are skipped so the text stays clean.
        """
        lines: list[str] = []
        for col in self._text_columns:
            value = row.get(col)
            if self._is_empty(value):
                continue
            lines.append(f"{col}: {self._safe_str(value)}")
        return "\n".join(lines)

    def _format_child(
        self,
        row: dict[str, Any],
        columns: list[str],
        group_name: str,
    ) -> str:
        """Format a field-grouped slice of a row."""
        lines: list[str] = []
        for col in columns:
            value = row.get(col)
            if self._is_empty(value):
                continue
            lines.append(f"{col}: {self._safe_str(value)}")
        return "\n".join(lines)

    @staticmethod
    def _is_empty(value: Any) -> bool:
        """Return ``True`` for None, NaN, or whitespace-only strings."""
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    @staticmethod
    def _safe_str(value: Any) -> str:
        """Convert a cell value to a clean string."""
        if value is None:
            return ""
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            # Avoid trailing ".0" for integer-valued floats.
            if value == int(value):
                return str(int(value))
        return str(value).strip()
