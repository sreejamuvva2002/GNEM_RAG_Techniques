"""Tests for the parent-child chunker.

Covers:
- Parent chunk created per row.
- Children inherit parent_id.
- Empty / NaN fields are skipped cleanly.
- Configurable child groups are respected.
"""

from __future__ import annotations

import math

import pytest

from src.core.parent_child_chunker import ParentChildChunker, Chunk


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def text_columns() -> list[str]:
    """Ordered list of columns matching the GNEM schema."""
    return [
        "Company",
        "Category",
        "Industry Group",
        "Updated Location",
        "Address",
        "Latitude",
        "Longitude",
        "Primary Facility Type",
        "EV Supply Chain Role",
        "Primary OEMs",
        "Supplier or Affiliation Type",
        "Employment",
        "Product / Service",
        "EV / Battery Relevant",
        "Classification Method",
    ]


@pytest.fixture
def child_groups() -> dict[str, list[str]]:
    """Child group config matching config.yaml."""
    return {
        "identity": ["Company", "Category", "Industry Group", "Supplier or Affiliation Type"],
        "location": ["Updated Location", "Address", "Latitude", "Longitude", "Primary Facility Type"],
        "supply_chain": ["EV Supply Chain Role", "Primary OEMs", "EV / Battery Relevant", "Classification Method"],
        "business": ["Employment", "Product / Service"],
    }


@pytest.fixture
def sample_row() -> dict:
    """A complete GNEM-like row."""
    return {
        "Company": "ACM Georgia LLC",
        "Category": "Tier 2/3",
        "Industry Group": "Textile Products",
        "Updated Location": "Warrenton, Warren County",
        "Address": "975 Thomson Hwy, Warrenton, GA 30828",
        "Latitude": 33.4123,
        "Longitude": -82.6406,
        "Primary Facility Type": "Manufacturing Plant",
        "EV Supply Chain Role": "General Automotive",
        "Primary OEMs": "Multiple OEMs",
        "Supplier or Affiliation Type": "Automotive supply chain participant",
        "Employment": 400.0,
        "Product / Service": "Automotive floor mats made of purchased wire",
        "EV / Battery Relevant": "Indirect",
        "Classification Method": "Supplier",
    }


@pytest.fixture
def chunker(text_columns, child_groups) -> ParentChildChunker:
    return ParentChildChunker(text_columns=text_columns, child_groups=child_groups)


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestParentCreation:
    """Parent chunk is created for every row."""

    def test_one_parent_per_row(self, chunker, sample_row):
        parents, _ = chunker.chunk_rows([sample_row])
        assert len(parents) == 1

    def test_parent_id_format(self, chunker, sample_row):
        parents, _ = chunker.chunk_rows([sample_row])
        assert parents[0].parent_id == "parent_0"

    def test_parent_contains_all_fields(self, chunker, sample_row):
        parents, _ = chunker.chunk_rows([sample_row])
        text = parents[0].text
        assert "Company: ACM Georgia LLC" in text
        assert "Category: Tier 2/3" in text
        assert "Employment: 400" in text

    def test_multiple_rows_produce_multiple_parents(self, chunker, sample_row):
        rows = [sample_row.copy() for _ in range(5)]
        parents, _ = chunker.chunk_rows(rows)
        assert len(parents) == 5
        ids = [p.parent_id for p in parents]
        assert ids == [f"parent_{i}" for i in range(5)]


class TestChildCreation:
    """Children inherit parent_id and are grouped correctly."""

    def test_children_inherit_parent_id(self, chunker, sample_row):
        parents, children = chunker.chunk_rows([sample_row])
        for child in children:
            assert child.parent_id == "parent_0"

    def test_four_child_groups(self, chunker, sample_row):
        _, children = chunker.chunk_rows([sample_row])
        group_names = {c.group_name for c in children}
        assert group_names == {"identity", "location", "supply_chain", "business"}

    def test_child_text_contains_group_fields(self, chunker, sample_row):
        _, children = chunker.chunk_rows([sample_row])
        identity = [c for c in children if c.group_name == "identity"][0]
        assert "Company: ACM Georgia LLC" in identity.text
        assert "Category: Tier 2/3" in identity.text
        # Location fields should NOT be in the identity child.
        assert "Address:" not in identity.text

    def test_child_type_is_child(self, chunker, sample_row):
        _, children = chunker.chunk_rows([sample_row])
        for child in children:
            assert child.chunk_type == "child"


class TestEmptyFieldHandling:
    """Empty and NaN fields are skipped cleanly."""

    def test_none_fields_skipped(self, chunker, text_columns, child_groups):
        row = {col: None for col in text_columns}
        row["Company"] = "Test Corp"
        parents, children = chunker.chunk_rows([row])
        # Parent should only have the Company field.
        assert parents[0].text == "Company: Test Corp"

    def test_nan_fields_skipped(self, chunker, text_columns, child_groups):
        row = {col: float("nan") for col in text_columns}
        row["Company"] = "NaN Test Corp"
        parents, _ = chunker.chunk_rows([row])
        assert "Company: NaN Test Corp" in parents[0].text
        # No other fields should appear.
        assert parents[0].text.count("\n") == 0

    def test_empty_string_fields_skipped(self, chunker, text_columns):
        row = {col: "  " for col in text_columns}
        row["Company"] = "Blank Corp"
        parents, _ = chunker.chunk_rows([row])
        assert parents[0].text == "Company: Blank Corp"

    def test_child_group_skipped_when_all_empty(self, chunker, text_columns):
        """If every field in a child group is empty, no child is created."""
        row = {col: None for col in text_columns}
        row["Company"] = "Partial Corp"
        row["Category"] = "Tier 1"
        _, children = chunker.chunk_rows([row])
        group_names = {c.group_name for c in children}
        # Only identity group has data (Company, Category).
        assert "identity" in group_names
        # Business group (Employment, Product/Service) is all None → skipped.
        assert "business" not in group_names


class TestIntegerFloatFormatting:
    """Integer-valued floats like 400.0 render as '400', not '400.0'."""

    def test_employment_renders_as_int(self, chunker, sample_row):
        parents, _ = chunker.chunk_rows([sample_row])
        assert "Employment: 400" in parents[0].text
        assert "400.0" not in parents[0].text


class TestChunkSerialization:
    """Chunk.to_dict() produces a usable dict."""

    def test_to_dict_keys(self, chunker, sample_row):
        parents, _ = chunker.chunk_rows([sample_row])
        d = parents[0].to_dict()
        assert set(d.keys()) == {
            "chunk_id", "parent_id", "text", "metadata",
            "chunk_type", "group_name",
        }
