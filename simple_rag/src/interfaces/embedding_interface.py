"""Interface for text embedding providers.

Any embedding adapter (Ollama, OpenAI, HuggingFace, …) must implement
this protocol so that core/services never depend on a concrete provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingInterface(ABC):
    """Contract for generating dense vector embeddings from text."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings.

        Args:
            texts: Non-empty list of plain-text strings.

        Returns:
            A list of embedding vectors (each a list of floats), one
            per input text, in the same order.

        Raises:
            EmbeddingError: On any failure during embedding generation.
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        This is separated from ``embed_texts`` because some providers
        use a different prompt prefix for queries vs. documents.

        Args:
            query: The query text to embed.

        Returns:
            A single embedding vector as a list of floats.

        Raises:
            EmbeddingError: On any failure during embedding generation.
        """
