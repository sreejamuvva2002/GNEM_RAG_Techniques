"""Tests for score normalisation and fusion math.

Covers:
- BM25 min-max normalisation.
- Vector distance → similarity inversion + normalisation.
- Weighted combination.
- Edge cases: all-equal scores, single document, empty arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.score_fusion import (
    normalize_bm25_scores,
    normalize_vector_scores,
    combine_scores,
)


# ------------------------------------------------------------------ #
# BM25 normalisation
# ------------------------------------------------------------------ #

class TestNormalizeBM25:
    """Min-max normalisation for BM25 scores (higher = better)."""

    def test_basic_range(self):
        scores = np.array([2.0, 5.0, 8.0])
        result = normalize_bm25_scores(scores)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_already_01(self):
        scores = np.array([0.0, 0.5, 1.0])
        result = normalize_bm25_scores(scores)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_all_equal_returns_ones(self):
        scores = np.array([4.0, 4.0, 4.0])
        result = normalize_bm25_scores(scores)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])

    def test_single_document_returns_one(self):
        scores = np.array([7.0])
        result = normalize_bm25_scores(scores)
        np.testing.assert_allclose(result, [1.0])

    def test_empty_array(self):
        scores = np.array([])
        result = normalize_bm25_scores(scores)
        assert result.size == 0

    def test_preserves_order(self):
        """Highest raw score should remain highest after normalisation."""
        scores = np.array([1.0, 3.0, 2.0])
        result = normalize_bm25_scores(scores)
        assert result[1] > result[2] > result[0]


# ------------------------------------------------------------------ #
# Vector normalisation (distance → similarity → min-max)
# ------------------------------------------------------------------ #

class TestNormalizeVector:
    """Invert distances and min-max normalise."""

    def test_basic_inversion(self):
        distances = np.array([0.0, 1.0, 3.0])
        result = normalize_vector_scores(distances)
        # Distance 0 → sim 1.0 (highest), distance 3 → sim 0.25 (lowest).
        assert result[0] > result[1] > result[2]
        # Min-max: best should be 1.0, worst should be 0.0.
        np.testing.assert_allclose(result[0], 1.0)
        np.testing.assert_allclose(result[2], 0.0)

    def test_all_equal_distances_returns_ones(self):
        distances = np.array([2.0, 2.0, 2.0])
        result = normalize_vector_scores(distances)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])

    def test_single_document(self):
        distances = np.array([0.5])
        result = normalize_vector_scores(distances)
        np.testing.assert_allclose(result, [1.0])

    def test_empty_array(self):
        distances = np.array([])
        result = normalize_vector_scores(distances)
        assert result.size == 0

    def test_already_similarity_skips_inversion(self):
        """When already_similarity=True, no inversion is applied."""
        similarities = np.array([0.2, 0.5, 0.9])
        result = normalize_vector_scores(similarities, already_similarity=True)
        # Min-max of [0.2, 0.5, 0.9] → [0.0, ~0.4286, 1.0]
        np.testing.assert_allclose(result[0], 0.0)
        np.testing.assert_allclose(result[2], 1.0)
        assert 0.0 < result[1] < 1.0

    def test_zero_distance_is_best(self):
        """Zero distance should map to the highest normalised score."""
        distances = np.array([0.0, 10.0, 100.0])
        result = normalize_vector_scores(distances)
        np.testing.assert_allclose(result[0], 1.0)


# ------------------------------------------------------------------ #
# Weighted combination
# ------------------------------------------------------------------ #

class TestCombineScores:
    """Weighted sum: combined = alpha * vector + (1-alpha) * bm25."""

    def test_basic_combination(self):
        vec = np.array([1.0, 0.0, 0.5])
        bm25 = np.array([0.0, 1.0, 0.5])
        alpha = 0.6

        result = combine_scores(vec, bm25, alpha)
        expected = 0.6 * vec + 0.4 * bm25
        np.testing.assert_allclose(result, expected)

    def test_alpha_zero_uses_only_bm25(self):
        vec = np.array([0.9, 0.1])
        bm25 = np.array([0.3, 0.7])
        result = combine_scores(vec, bm25, alpha=0.0)
        np.testing.assert_allclose(result, bm25)

    def test_alpha_one_uses_only_vector(self):
        vec = np.array([0.9, 0.1])
        bm25 = np.array([0.3, 0.7])
        result = combine_scores(vec, bm25, alpha=1.0)
        np.testing.assert_allclose(result, vec)

    def test_equal_weight(self):
        vec = np.array([0.8, 0.2])
        bm25 = np.array([0.4, 0.6])
        result = combine_scores(vec, bm25, alpha=0.5)
        np.testing.assert_allclose(result, [0.6, 0.4])

    def test_combined_in_01_range(self):
        """Combined scores should stay in [0, 1] when inputs are in [0, 1]."""
        rng = np.random.default_rng(42)
        vec = rng.random(100)
        bm25 = rng.random(100)
        result = combine_scores(vec, bm25, alpha=0.6)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
