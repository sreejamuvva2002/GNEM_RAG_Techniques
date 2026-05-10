from __future__ import annotations
"""Parent–child chunker for tabular (row-level) data.

Each spreadsheet row becomes one chunk with:
  - Structured metadata (Record_ID, Company_Clean, County, Tier_Level, etc.)
  - A formatted ``Embedding_Text`` built from a configurable template.
  - ``Char_Count`` and ``Token_Estimate`` for the embedding text.

The metadata schema and embedding text template are driven entirely
by ``config.yaml`` — nothing is hardcoded.

This module is **pure domain logic** — no I/O, no external services.
"""

from src.utils import get_logger

import hashlib
import math
import re
from typing import Any


# Field groups for true parent-child chunking.
# Each company record is split into these 3 child chunks.
# Templates use the same placeholder names as the normal embedding_text_template
# plus EV_Relevant and Primary_OEMs from metadata.
CHILD_FIELD_GROUPS = [
    {
        "name": "identity",
        "template": (
            "Company: {Company}\n"
            "Location: {County}, {State}\n"
            "Supply Chain Tier: {Tier_Level} ({Tier_Confidence})\n"
            "Status: {Status}"
        ),
    },
    {
        "name": "industry",
        "template": (
            "Industry: {Industry_Name}\n"
            "Product/Service: {Product_Service}"
        ),
    },
    {
        "name": "profile",
        "template": (
            "Employment: {Employment} employees\n"
            "OEM Status: {OEM_Status}\n"
            "EV Relevant: {EV_Relevant}\n"
            "Primary OEMs: {Primary_OEMs}"
        ),
    },
]


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Data container
# ------------------------------------------------------------------

class Chunk:
    """Container for a processed chunk with metadata.

    Attributes:
        chunk_id: Unique chunk identifier (e.g. ``GA_AUTO_0000``).
        record_id: MD5 hash identifying the source company record.
        metadata: Full metadata dict for the chunk.
        embedding_text: Formatted text to be embedded.
        char_count: Length of ``embedding_text``.
        token_estimate: Rough token estimate (``char_count // 4``).
    """

    __slots__ = (
        "chunk_id",
        "record_id",
        "metadata",
        "embedding_text",
        "char_count",
        "token_estimate",
    )

    def __init__(
        self,
        chunk_id: str,
        record_id: str,
        metadata: dict[str, Any],
        embedding_text: str,
    ) -> None:
        self.chunk_id = chunk_id
        self.record_id = record_id
        self.metadata = metadata
        self.embedding_text = embedding_text
        self.char_count = len(embedding_text)
        # Rough token estimate: ~4 characters per token for English.
        self.token_estimate = max(1, self.char_count // 4)

    def to_flat_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict for Excel export / adapter consumption."""
        base = {
            "Chunk_ID": self.chunk_id,
            "Record_ID": self.record_id,
        }
        base.update(self.metadata)
        base["Embedding_Text"] = self.embedding_text
        base["Char_Count"] = self.char_count
        base["Token_Estimate"] = self.token_estimate
        return base


# ------------------------------------------------------------------
# Chunker
# ------------------------------------------------------------------

class ParentChildChunker:
    """Converts tabular rows into chunks with rich metadata.

    Args:
        chunking_config: The ``chunking`` section from ``config.yaml``.
    """

    def __init__(self, chunking_config: dict[str, Any]) -> None:
        self._cfg = chunking_config
        self._prefix = chunking_config["chunk_id_prefix"]
        self._state = chunking_config["state"]
        self._field_map = chunking_config["field_mappings"]
        self._announcement_marker = chunking_config["announcement_marker"]
        self._oem_categories = [
            c.lower() for c in chunking_config["oem_categories"]
        ]
        self._template_lines: list[str] = chunking_config["embedding_text_template"]

    # -------------------------------------------------------------- #
    # Public API                                                      #
    # -------------------------------------------------------------- #

    def chunk_rows(self, rows: list[dict[str, Any]]) -> list[Chunk]:
        """Convert every row into a chunk with metadata.

        Args:
            rows: List of row dicts as returned by the data loader.

        Returns:
            A list of ``Chunk`` objects, one per input row.
        """
        chunks: list[Chunk] = []

        for idx, row in enumerate(rows):
            chunk = self._build_chunk(row, idx)
            chunks.append(chunk)

        logger.info("Chunked %d rows → %d chunks", len(rows), len(chunks))
        return chunks

    # -------------------------------------------------------------- #
    # Internal helpers                                                #
    # -------------------------------------------------------------- #

    def _build_chunk(self, row: dict[str, Any], idx: int) -> Chunk:
        """Build a single chunk from a data row."""

        # --- Raw field extraction (config-driven) ---
        company_raw = self._safe_str(row.get(self._field_map["company"]))
        category = self._safe_str(row.get(self._field_map["category"]))
        industry_group = self._safe_str(row.get(self._field_map["industry_group"]))
        location = self._safe_str(row.get(self._field_map["location"]))
        employment_raw = row.get(self._field_map["employment"])
        product_service = self._safe_str(row.get(self._field_map["product_service"]))
        ev_relevant = self._safe_str(row.get(self._field_map["ev_relevant"]))
        classification = self._safe_str(row.get(self._field_map["classification_method"]))

        # --- Derived metadata ---
        is_announcement = self._announcement_marker in company_raw if company_raw else False
        company_clean = company_raw.replace(self._announcement_marker, "").strip() if company_raw else ""
        county = self._extract_county(location)
        employment = self._format_employment(employment_raw)
        tier_level, tier_confidence = self._parse_tier(category, classification)
        is_oem = category.strip().lower() in self._oem_categories if category else False
        industry_code, industry_name = self._parse_industry_group(industry_group)

        # --- IDs ---
        record_id = self._make_record_id(company_raw, idx)
        chunk_id = f"{self._prefix}_{idx:04d}"

        # --- OEM status and operational status ---
        oem_status = "OEM" if is_oem else "Supplier/Service Provider"
        status = "Announced" if is_announcement else "Operational"

        # --- Build Embedding_Text from template ---
        template_vars = {
            "Company": company_clean or company_raw,
            "County": county,
            "State": self._state,
            "Employment": employment,
            "Industry_Name": industry_name,
            "Tier_Level": tier_level,
            "Tier_Confidence": tier_confidence,
            "Product_Service": product_service,
            "OEM_Status": oem_status,
            "Status": status,
        }
        embedding_text = self._render_template(template_vars)

        # --- Additional raw field extractions for metadata ---
        ev_supply_chain_role = self._safe_str(
            row.get(self._field_map["ev_supply_chain_role"])
        )
        primary_oems = self._safe_str(row.get(self._field_map["primary_oems"]))
        facility_type = self._safe_str(row.get(self._field_map["facility_type"]))
        supplier_type = self._safe_str(row.get(self._field_map["supplier_type"]))

        # --- Metadata dict ---
        metadata = {
            "Company": company_raw,
            "Company_Clean": company_clean or company_raw,
            "County": county,
            "Employment": employment_raw if not self._is_empty(employment_raw) else None,
            "Employment_Formatted": employment,
            "Industry_Code": industry_code,
            "Industry_Name": industry_name,
            "Tier_Level": tier_level,
            "Tier_Confidence": tier_confidence,
            "Is_OEM": is_oem,
            "Is_Announcement": is_announcement,
            "EV_Relevant": ev_relevant,
            "EV_Supply_Chain_Role": ev_supply_chain_role,
            "Primary_OEMs": primary_oems,
            "Facility_Type": facility_type,
            "Supplier_Type": supplier_type,
            "Product_Service": product_service,
            "Classification_Method": classification,
        }

        return Chunk(
            chunk_id=chunk_id,
            record_id=record_id,
            metadata=metadata,
            embedding_text=embedding_text,
        )

    def _render_template(self, variables: dict[str, Any]) -> str:
        """Render embedding text from the configurable template.

        Lines whose placeholder values are empty are skipped.
        """
        lines: list[str] = []
        for template_line in self._template_lines:
            # Extract placeholder names from the template line.
            placeholders = re.findall(r"\{(\w+)\}", template_line)

            # Check if any placeholder resolved to a non-empty value.
            has_content = any(
                str(variables.get(p, "")).strip()
                for p in placeholders
                if p != "State"  # State is always present.
            )
            if not has_content:
                continue

            try:
                rendered = template_line.format(**variables)
            except KeyError:
                continue

            lines.append(rendered)
        return "\n".join(lines)

    @staticmethod
    def _extract_county(location: str) -> str:
        """Extract county name from location string.

        Expected format: ``"City, County County"`` or ``"City, County"``.
        """
        if not location:
            return ""
        parts = location.split(",")
        if len(parts) >= 2:
            county_part = parts[-1].strip()
            # Remove trailing "County" if present to get just the name.
            return county_part
        return location.strip()

    @staticmethod
    def _parse_tier(category: str, classification: str) -> tuple[str, str]:
        """Extract tier level and confidence from category.

        Examples:
            ``"Tier 2/3"``  → ``("2/3", "likely")``
            ``"Tier 1"``    → ``("1", "likely")``
            ``"Tier 1/2"``  → ``("1/2", "likely")``
            ``"OEM"``       → ``("OEM", "confirmed")``
        """
        if not category:
            return ("", "")

        cat_lower = category.strip().lower()

        if "oem" in cat_lower:
            return ("OEM", "confirmed")

        # Extract tier number pattern (e.g., "1", "2/3", "1/2").
        match = re.search(r"tier\s+([\d/]+)", cat_lower)
        if match:
            return (match.group(1), "likely")

        return (category.strip(), "likely")

    @staticmethod
    def _parse_industry_group(industry_group: str) -> tuple[int | None, str]:
        """Parse industry group into code and name.

        Handles formats like ``"37: Transportation Equipment"`` or
        plain ``"Transportation Equipment"``.
        """
        if not industry_group:
            return (None, "")

        ig = industry_group.strip()

        # Try to extract leading numeric code.
        match = re.match(r"^(\d+)[:\s]+(.+)$", ig)
        if match:
            return (int(match.group(1)), match.group(2).strip())

        # No numeric code — return name only.
        return (None, ig)

    @staticmethod
    def _format_employment(value: Any) -> str:
        """Format employment as a clean string."""
        if value is None:
            return ""
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            if value == int(value):
                return str(int(value))
        return str(value).strip()

    @staticmethod
    def _make_record_id(company: str, idx: int) -> str:
        """Generate a stable MD5 hash for the record.

        Uses company name + row index to ensure uniqueness even if
        two rows share the same company name.
        """
        raw = f"{company}_{idx}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _is_empty(value: Any) -> bool:
        """Return True for None, NaN, or whitespace-only strings."""
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
            if value == int(value):
                return str(int(value))
        return str(value).strip()


# ------------------------------------------------------------------
# True parent-child chunker
# ------------------------------------------------------------------

class TrueParentChildChunker:
    """Splits each company record into 3 field-group child chunks.

    Child chunks are indexed in ChromaDB and BM25 for retrieval.
    On retrieval, every child's ``record_id`` maps back to the full
    parent company record via the returned ``parent_lookup`` dict.
    The LLM always receives the full parent text — never just the
    matched child snippet.

    Child grouping:
      - ``identity``: Company + Location + Tier + Status
      - ``industry``:  Industry + Product/Service
      - ``profile``:   Employment + OEM Status + EV Relevance + Primary OEMs

    Args:
        chunking_config: The ``chunking`` section from ``config.yaml``.
    """

    _CHILD_ID_PREFIX = "GA_PC"

    def __init__(self, chunking_config: dict[str, Any]) -> None:
        # Delegate all parent-level chunking to the existing chunker.
        self._row_chunker = ParentChildChunker(chunking_config)
        self._state = chunking_config["state"]

    def chunk_rows(
        self, rows: list[dict[str, Any]]
    ) -> tuple[list[Chunk], dict[str, dict[str, Any]]]:
        """Split each row into child chunks and build the parent lookup.

        Args:
            rows: List of row dicts as returned by the data loader.

        Returns:
            A tuple of:
            - ``child_chunks``: All child ``Chunk`` objects (up to 3 per row).
              Each child's ``record_id`` is the parent record's ``record_id``
              so ``HybridRetriever`` deduplicates by parent correctly.
            - ``parent_lookup``: ``{record_id: {"text": full_text, "company": ...}}``
              for passing to ``HybridRetriever``.
        """
        parent_chunks = self._row_chunker.chunk_rows(rows)

        parent_lookup: dict[str, dict[str, Any]] = {
            pc.record_id: {
                "text": pc.embedding_text,
                "company": pc.metadata.get("Company_Clean", ""),
            }
            for pc in parent_chunks
        }

        child_chunks: list[Chunk] = []
        for parent_chunk in parent_chunks:
            children = self._split_into_children(parent_chunk)
            child_chunks.extend(children)

        logger.info(
            "True parent-child: %d parent records → %d child chunks",
            len(parent_chunks),
            len(child_chunks),
        )
        return child_chunks, parent_lookup

    # -------------------------------------------------------------- #
    # Internal helpers                                                #
    # -------------------------------------------------------------- #

    def _split_into_children(self, parent: Chunk) -> list[Chunk]:
        """Produce up to 3 child chunks from a single parent chunk."""
        meta = parent.metadata

        # Extract numeric row index from parent chunk_id (e.g. "GA_AUTO_0042" → "0042")
        idx_match = re.search(r"(\d+)$", parent.chunk_id)
        idx_str = idx_match.group(1) if idx_match else parent.chunk_id

        extended_vars = {
            "Company": meta.get("Company_Clean", ""),
            "County": meta.get("County", ""),
            "State": self._state,
            "Tier_Level": meta.get("Tier_Level", ""),
            "Tier_Confidence": meta.get("Tier_Confidence", ""),
            "Status": "Announced" if meta.get("Is_Announcement", False) else "Operational",
            "Industry_Name": meta.get("Industry_Name", ""),
            "Product_Service": meta.get("Product_Service", ""),
            "Employment": meta.get("Employment_Formatted", ""),
            "OEM_Status": "OEM" if meta.get("Is_OEM", False) else "Supplier/Service Provider",
            "EV_Relevant": meta.get("EV_Relevant", ""),
            "Primary_OEMs": meta.get("Primary_OEMs", ""),
        }

        children: list[Chunk] = []
        for group in CHILD_FIELD_GROUPS:
            child_text = self._render_child(group["template"], extended_vars)
            if not child_text.strip():
                continue
            child_chunk = Chunk(
                chunk_id=f"{self._CHILD_ID_PREFIX}_{idx_str}_{group['name']}",
                record_id=parent.record_id,  # shared with parent → enables dedup
                metadata={**meta, "child_group": group["name"]},
                embedding_text=child_text,
            )
            children.append(child_chunk)

        return children

    @staticmethod
    def _render_child(template: str, variables: dict[str, Any]) -> str:
        """Render a child template, skipping lines whose values are all empty."""
        lines: list[str] = []
        for line in template.split("\n"):
            placeholders = re.findall(r"\{(\w+)\}", line)
            has_content = any(
                str(variables.get(p, "")).strip()
                for p in placeholders
                if p != "State"
            )
            if not has_content:
                continue
            try:
                lines.append(line.format(**variables))
            except KeyError:
                continue
        return "\n".join(lines)
