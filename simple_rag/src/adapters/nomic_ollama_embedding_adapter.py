"""Nomic Embed Text adapter via Ollama's REST API.

This is the **only** embedding adapter in the project.  It hits
Ollama's ``/api/embeddings`` endpoint with batched requests and
surfaces clear errors when Ollama is unreachable or the model is
not pulled.
"""

from __future__ import annotations

import requests

from src.exceptions import EmbeddingError, OllamaUnavailableError
from src.interfaces.embedding_interface import EmbeddingInterface
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NomicOllamaEmbeddingAdapter(EmbeddingInterface):
    """Generate embeddings using ``nomic-embed-text`` through Ollama.

    Args:
        model: Ollama model name (default ``nomic-embed-text``).
        ollama_host: Base URL of the Ollama server.
        batch_size: Maximum number of texts per batch request.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434",
        batch_size: int = 32,
        timeout: int = 60,
    ) -> None:
        self._model = model
        self._host = ollama_host.rstrip("/")
        self._batch_size = batch_size
        self._timeout = timeout
        self._endpoint = f"{self._host}/api/embeddings"

    # ------------------------------------------------------------------ #
    # EmbeddingInterface implementation                                   #
    # ------------------------------------------------------------------ #

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts.

        Texts are sent in sub-batches of ``batch_size`` to avoid
        overwhelming the server.

        Args:
            texts: Non-empty list of plain-text strings.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            OllamaUnavailableError: If the Ollama server is unreachable.
            EmbeddingError: On any other embedding failure.
        """
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

            if start % (self._batch_size * 10) == 0 and start > 0:
                logger.info(
                    "Embedded %d / %d texts", start, len(texts)
                )

        logger.info("Embedded %d texts total", len(all_embeddings))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: The query text.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            OllamaUnavailableError: If the Ollama server is unreachable.
            EmbeddingError: On any other embedding failure.
        """
        results = self._embed_batch([query])
        return results[0]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send each text individually to Ollama's /api/embeddings.

        Ollama's embeddings endpoint processes one prompt at a time,
        so we loop over the batch sequentially.
        """
        embeddings: list[list[float]] = []

        for text in texts:
            payload = {
                "model": self._model,
                "prompt": text,
            }
            try:
                response = requests.post(
                    self._endpoint,
                    json=payload,
                    timeout=self._timeout,
                )
            except requests.ConnectionError as exc:
                raise OllamaUnavailableError(
                    f"Cannot reach Ollama at {self._host}. "
                    f"Is 'ollama serve' running? Error: {exc}"
                ) from exc
            except requests.Timeout as exc:
                raise EmbeddingError(
                    f"Ollama request timed out after {self._timeout}s: {exc}"
                ) from exc
            except requests.RequestException as exc:
                raise EmbeddingError(
                    f"HTTP error calling Ollama: {exc}"
                ) from exc

            if response.status_code != 200:
                raise EmbeddingError(
                    f"Ollama returned HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            data = response.json()
            embedding = data.get("embedding")
            if embedding is None:
                raise EmbeddingError(
                    f"Ollama response missing 'embedding' key. "
                    f"Is the model '{self._model}' pulled? "
                    f"Response: {data}"
                )
            embeddings.append(embedding)

        return embeddings
