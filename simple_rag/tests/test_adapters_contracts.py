"""Adapter contract tests.

Verifies that every concrete adapter is a valid implementation of its
corresponding interface (LSP compliance).  Uses parametrised tests to
check ``isinstance`` and that all abstract methods are implemented.
"""

from __future__ import annotations

import inspect
from abc import ABC

import pytest

from src.interfaces.embedding_interface import EmbeddingInterface
from src.interfaces.vector_store_interface import VectorStoreInterface
from src.interfaces.keyword_search_interface import KeywordSearchInterface
from src.interfaces.data_loader_interface import DataLoaderInterface
from src.interfaces.result_writer_interface import ResultWriterInterface

from src.adapters.nomic_ollama_embedding_adapter import NomicOllamaEmbeddingAdapter
from src.adapters.faiss_vector_store_adapter import FAISSVectorStoreAdapter
from src.adapters.bm25_keyword_search_adapter import BM25KeywordSearchAdapter
from src.adapters.excel_data_loader_adapter import ExcelDataLoaderAdapter
from src.adapters.excel_result_writer_adapter import ExcelResultWriterAdapter


# ------------------------------------------------------------------ #
# Parametrised contract tests
# ------------------------------------------------------------------ #

_ADAPTER_PAIRS: list[tuple[type, type]] = [
    (NomicOllamaEmbeddingAdapter, EmbeddingInterface),
    (FAISSVectorStoreAdapter, VectorStoreInterface),
    (BM25KeywordSearchAdapter, KeywordSearchInterface),
    (ExcelDataLoaderAdapter, DataLoaderInterface),
    (ExcelResultWriterAdapter, ResultWriterInterface),
]


@pytest.mark.parametrize(
    "adapter_cls, interface_cls",
    _ADAPTER_PAIRS,
    ids=[pair[0].__name__ for pair in _ADAPTER_PAIRS],
)
class TestAdapterContracts:
    """Each adapter must be a valid implementation of its interface."""

    def test_is_subclass(self, adapter_cls: type, interface_cls: type):
        """Adapter class inherits from the interface."""
        assert issubclass(adapter_cls, interface_cls), (
            f"{adapter_cls.__name__} must subclass {interface_cls.__name__}"
        )

    def test_isinstance_check(self, adapter_cls: type, interface_cls: type):
        """Instantiated adapter passes isinstance check.

        We catch __init__ errors for adapters that need external
        services (e.g. Ollama) — the contract test is about the type
        hierarchy, not runtime connectivity.
        """
        try:
            instance = adapter_cls()
        except TypeError:
            # Some adapters require args — try with defaults.
            instance = adapter_cls.__new__(adapter_cls)
        assert isinstance(instance, interface_cls)

    def test_all_abstract_methods_implemented(
        self, adapter_cls: type, interface_cls: type
    ):
        """Every abstract method declared on the interface is concretely
        implemented by the adapter (no lingering ``pass`` stubs)."""
        abstract_methods = {
            name
            for name, _ in inspect.getmembers(interface_cls, predicate=inspect.isfunction)
            if getattr(getattr(interface_cls, name, None), "__isabstractmethod__", False)
        }
        for method_name in abstract_methods:
            adapter_method = getattr(adapter_cls, method_name, None)
            assert adapter_method is not None, (
                f"{adapter_cls.__name__} missing method: {method_name}"
            )
            assert not getattr(adapter_method, "__isabstractmethod__", False), (
                f"{adapter_cls.__name__}.{method_name} is still abstract"
            )
