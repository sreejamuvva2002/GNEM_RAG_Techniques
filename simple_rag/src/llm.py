from __future__ import annotations
"""LLM client and strict KB-only prompt builder for the GNEM RAG pipeline."""

from src.utils import get_logger

import os
import re
from typing import Any

import requests


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenAI-compatible LLM client.

    Reads base_url / api_key / model from constructor args first,
    then falls back to env vars LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
        ).rstrip("/")
        self._api_key = api_key or os.environ.get("LLM_API_KEY", "ollama")
        self._model = model or os.environ.get("LLM_MODEL") or ""
        if not self._model:
            raise ValueError(
                "LLM model must be supplied via the model arg or LLM_MODEL env var."
            )
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._endpoint = f"{self._base_url}/chat/completions"

        logger.info("LLMClient: model=%s  base_url=%s", self._model, self._base_url)

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            response = requests.post(
                self._endpoint, json=payload, headers=headers, timeout=self._timeout
            )
        except requests.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot reach LLM at {self._base_url}. Is the server running? {exc}"
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError(f"LLM request timed out after {self._timeout}s: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"LLM returned HTTP {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM response has no choices: {data}")

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError(f"LLM response has empty content: {data}")

        logger.debug("LLM response: %d chars, model=%s", len(content), self._model)
        return content


# ---------------------------------------------------------------------------
# System prompt — strict KB-only answer generator
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an evidence-grounded answer generator for a RAG system.

The retrieved context may contain both highly relevant and partially relevant information. Carefully determine which information directly answers the question.

Your task is to answer the question using ONLY the retrieved context provided in the user message.

INSTRUCTIONS:
- Use only information supported by the retrieved context.
- Do not use outside or pretrained knowledge.
- Internally verify whether the retrieved information truly satisfies the question before using it.
- Ignore information that is related but does not directly answer the question.
- If multiple context sections support the same fact, combine them carefully.
- If the context is incomplete or insufficient, clearly state:
  "Not available in the provided context."
- If the retrieved context contains conflicting information, mention the conflict explicitly.
- Do not guess or infer unsupported facts.

RESPONSE FORMAT:

Direct Answer:
[Final grounded answer]

Supporting Information:
[Brief explanation of the evidence used]

Missing or Uncertain Information:
[Explain any missing, ambiguous, or conflicting information if applicable]
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_evidence_prompt(
    question: str,
    evidence_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the system + user messages for the LLM.

    Each evidence row is formatted as a structured KB entry so the LLM can
    easily validate constraints against the question.
    """
    kb_entries: list[str] = []
    for ev in evidence_rows:
        eid          = ev.get("evidence_id", "")
        company      = ev.get("company", "")          or "Not available"
        county       = ev.get("county", "")           or "Not available"
        role         = ev.get("ev_supply_chain_role", "") or "Not available"
        tier         = ev.get("tier_level", "")       or "Not available"
        employment   = ev.get("employment", "")       or "Not available"
        primary_oems = ev.get("primary_oems", "")     or "Not available"
        facility     = ev.get("facility_type", "")    or "Not available"
        industry     = ev.get("industry_name", "")    or "Not available"
        ev_relevant  = ev.get("ev_relevant", "")      or "Not available"
        product_svc  = ev.get("product_service", "")  or "Not available"
        supplier_type = ev.get("supplier_type", "")   or "Not available"

        kb_entries.append(
            f"[{eid}]\n"
            f"Company: {company}\n"
            f"Location (County): {county}\n"
            f"EV Supply Chain Role: {role}\n"
            f"Tier: {tier}\n"
            f"Employment: {employment}\n"
            f"Primary OEMs: {primary_oems}\n"
            f"Facility Type: {facility}\n"
            f"Industry Group: {industry}\n"
            f"EV / Battery Relevant: {ev_relevant}\n"
            f"Product / Service: {product_svc}\n"
            f"Supplier / Affiliation Type: {supplier_type}"
        )

    context_block = "\n\n".join(kb_entries)

    user_content = (
        f"User Question:\n{question}\n\n"
        f"Retrieved KB Context ({len(evidence_rows)} entries):\n\n"
        f"{context_block}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(
    response_text: str,
    all_evidence_ids: list[str],
) -> dict[str, Any]:
    """Parse the LLM response to extract structured metadata.

    Looks for evidence IDs in:
    - "Source Row / Chunk: [Exxx]" lines inside Matching KB Entries.
    - Inline [Exxx] references anywhere in the response.

    Returns:
        Dict with ``used_evidence_ids``, ``llm_selected_evidence_summary``,
        ``insufficient_evidence_flag``.
    """
    # Collect cited IDs from "Source Row / Chunk:" lines.
    chunk_ids = re.findall(
        r"Source Row\s*/\s*Chunk\s*:\s*\[?(E\d{3})\]?",
        response_text,
        re.IGNORECASE,
    )

    # Collect inline [E001] style citations.
    inline_ids = re.findall(r"\[(E\d{3})\]", response_text)

    # Merge, preserving order, deduplicating.
    all_cited = list(dict.fromkeys(chunk_ids + inline_ids))

    # Keep only IDs that were actually provided.
    valid_set = set(all_evidence_ids)
    used_ids = [eid for eid in all_cited if eid in valid_set]

    # Detect insufficient-evidence signals.
    insufficient_patterns = [
        r"not available in the provided kb context",
        r"insufficient.*to answer",
        r"evidence.*insufficient",
        r"not.*sufficient.*evidence",
        r"cannot.*answer.*based on.*evidence",
        r"no.*evidence.*provided.*for",
    ]
    insufficient = any(
        re.search(p, response_text, re.IGNORECASE)
        for p in insufficient_patterns
    )

    # Build a summary of which evidence was cited.
    if used_ids:
        summary = (
            f"LLM cited {len(used_ids)} of {len(all_evidence_ids)} KB entries: "
            + ", ".join(used_ids)
        )
    else:
        summary = "LLM did not cite any specific KB entry IDs."

    return {
        "used_evidence_ids": ", ".join(used_ids) if used_ids else "None",
        "llm_selected_evidence_summary": summary,
        "insufficient_evidence_flag": "Yes" if insufficient else "No",
    }
