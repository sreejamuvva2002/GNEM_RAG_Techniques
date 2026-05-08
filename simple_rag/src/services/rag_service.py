"""RAG service — query 50 questions → write output.

This service orchestrates the query phase of the pipeline.  It loads
questions, runs each through the hybrid retriever, formats the results,
and writes them via the result writer adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.core.hybrid_retriever import HybridRetriever
from src.interfaces.result_writer_interface import ResultWriterInterface
from src.exceptions import DataLoaderError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGService:
    """Orchestrates the query-and-output phase of the RAG pipeline.

    Args:
        retriever: A configured ``HybridRetriever`` instance.
        result_writer: Adapter for writing output files.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        result_writer: ResultWriterInterface,
    ) -> None:
        self._retriever = retriever
        self._result_writer = result_writer

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        questions_path: Path,
        output_path: Path,
    ) -> None:
        """Load questions, retrieve contexts, and write results.

        Args:
            questions_path: Path to the questions ``.xlsx`` file.
            output_path: Path to write the output ``.xlsx`` file.
        """
        # Step 1: Load questions.
        questions = self._load_questions(questions_path)
        logger.info("Loaded %d questions from %s", len(questions), questions_path)

        # Step 2: Process each question.
        all_results: list[dict[str, Any]] = []
        for i, question in enumerate(questions):
            logger.info("Processing question %d/%d: %s", i + 1, len(questions), question[:80])
            hits = self._retriever.retrieve(question)
            formatted = self._format_result(question, hits)
            all_results.append(formatted)

        # Step 3: Write output.
        self._result_writer.write(all_results, output_path)
        logger.info("Pipeline complete — results written to %s", output_path)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_questions(path: Path) -> list[str]:
        """Read questions from an Excel file.

        Expects a column named ``question``.

        Args:
            path: Path to the ``.xlsx`` file.

        Returns:
            List of question strings.

        Raises:
            DataLoaderError: If the file or column is missing.
        """
        path = Path(path)
        if not path.exists():
            raise DataLoaderError(f"Questions file not found: {path}")

        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception as exc:
            raise DataLoaderError(f"Failed to read questions file: {exc}") from exc

        # Normalise column names to lowercase for case-insensitive matching.
        df.columns = [c.strip().lower() for c in df.columns]

        if "question" not in df.columns:
            raise DataLoaderError(
                f"Questions file must have a 'question' column. "
                f"Found: {list(df.columns)}"
            )

        questions = df["question"].dropna().astype(str).tolist()
        return questions

    @staticmethod
    def _format_result(
        question: str,
        hits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Format retrieval hits into a single output row.

        Args:
            question: The original query.
            hits: List of result dicts from the hybrid retriever.

        Returns:
            A dict matching the output schema (question,
            retrieved_context, parent_ids, companies, scores).
        """
        separator = "\n\n---\n\n"

        contexts = [h["parent_text"] for h in hits]
        parent_ids = [h["parent_id"] for h in hits]
        companies = [h["company"] for h in hits]
        sim_scores = [str(h["similarity_score"]) for h in hits]
        kw_scores = [str(h["keyword_score"]) for h in hits]
        comb_scores = [str(h["combined_score"]) for h in hits]

        return {
            "question": question,
            "retrieved_context": separator.join(contexts),
            "parent_ids": " | ".join(parent_ids),
            "companies": " | ".join(companies),
            "similarity_scores": " | ".join(sim_scores),
            "keyword_scores": " | ".join(kw_scores),
            "combined_scores": " | ".join(comb_scores),
        }
