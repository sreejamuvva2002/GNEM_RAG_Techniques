"""Score normalisation and fusion for hybrid retrieval.

This module implements the exact normalisation and combination logic
described in the architecture specification:

1. **BM25 normalisation** — min-max scaling to [0, 1].  Higher is better.
2. **Vector normalisation** — convert distances (lower = better) to
   similarity via ``sim = 1 / (1 + distance)``, then min-max to [0, 1].
   If the store already returns cosine similarity (higher = better),
   the inversion step is a no-op but min-max still runs.
3. **Weighted combination** —
   ``combined = alpha * vector_norm + (1 - alpha) * bm25_norm``

This module is **pure domain logic** — stdlib + numpy only, no I/O.
"""

from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_bm25_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise BM25 scores to [0, 1].

    BM25 scores are already "higher = better".

    Args:
        scores: 1-D array of raw BM25 scores.

    Returns:
        Normalised scores as a 1-D numpy array.  If all scores are
        equal, returns an array of ones (all candidates are equivalent).
    """
    if scores.size == 0:
        return scores.copy()

    min_s = scores.min()
    max_s = scores.max()
    span = max_s - min_s

    if span == 0.0:
        # All scores identical — treat every candidate as equally good.
        return np.ones_like(scores, dtype=np.float64)

    return (scores - min_s) / span


def normalize_vector_scores(
    distances: np.ndarray,
    already_similarity: bool = False,
) -> np.ndarray:
    """Invert distances to similarities, then min-max normalise to [0, 1].

    The inversion formula is ``sim = 1 / (1 + distance)`` which maps
    distance 0 → similarity 1 and large distance → near 0.

    If the vector store already returns cosine *similarity* (higher is
    better), set ``already_similarity=True`` to skip the inversion —
    the min-max normalisation still runs so the output is in [0, 1].

    Args:
        distances: 1-D array of raw distance (or similarity) values.
        already_similarity: If ``True``, treat values as similarities
                           (higher = better) and skip inversion.

    Returns:
        Normalised scores in [0, 1], higher = better.  Returns ones
        when all values are equal.
    """
    if distances.size == 0:
        return distances.copy()

    if already_similarity:
        # No inversion needed — higher is already better.
        # Still min-max normalise for consistency.
        similarities = distances.astype(np.float64)
    else:
        # Invert: distance → similarity.
        # sim = 1 / (1 + distance).  Lower distance → higher similarity.
        similarities = 1.0 / (1.0 + distances.astype(np.float64))

    min_s = similarities.min()
    max_s = similarities.max()
    span = max_s - min_s

    if span == 0.0:
        return np.ones_like(similarities, dtype=np.float64)

    return (similarities - min_s) / span


def combine_scores(
    vector_norm: np.ndarray,
    bm25_norm: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute the weighted combination of normalised scores.

    Formula::

        combined = alpha * vector_norm + (1 - alpha) * bm25_norm

    Args:
        vector_norm: Min-max normalised vector similarity scores.
        bm25_norm: Min-max normalised BM25 scores.
        alpha: Semantic (vector) weight in ``[0, 1]``.

    Returns:
        Combined scores as a 1-D numpy array.
    """
    return alpha * vector_norm + (1.0 - alpha) * bm25_norm
