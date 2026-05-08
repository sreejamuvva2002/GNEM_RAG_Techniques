"""Custom exception hierarchy for the Simple RAG pipeline.

All domain-specific exceptions inherit from ``SimpleRAGError`` so callers
can catch the entire family with a single handler when needed.
"""


class SimpleRAGError(Exception):
    """Base exception for all Simple RAG errors."""


class ConfigError(SimpleRAGError):
    """Raised when configuration loading or validation fails.

    Examples:
        Missing required keys, malformed YAML, invalid value ranges.
    """


class EmbeddingError(SimpleRAGError):
    """Raised when embedding generation fails.

    Examples:
        Timeout, unexpected response shape, model not loaded.
    """


class OllamaUnavailableError(EmbeddingError):
    """Raised when the Ollama server cannot be reached.

    This is a specialisation of ``EmbeddingError`` because the root
    cause is the embedding provider being offline.
    """


class RetrievalError(SimpleRAGError):
    """Raised when the retrieval pipeline encounters an error.

    Examples:
        Empty index, score fusion failure, candidate pool exhaustion.
    """


class DataLoaderError(SimpleRAGError):
    """Raised when data loading or parsing fails.

    Examples:
        File not found, unexpected schema, corrupt workbook.
    """
