"""End-to-end RAG orchestration: retrieval -> grounding -> structured output -> validation."""
from __future__ import annotations

import logging
from typing import Callable

from langchain_openai import ChatOpenAI

from src.config import Settings
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retriever import Retriever
from src.schemas import FinancialAnalysis, RagResponse, RetrievedChunk

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Insufficient evidence in the retrieved documents to answer this query."
)
VALIDATION_FAILED_MESSAGE = (
    "The generated analysis could not be validated against the retrieved sources. "
    "Please refine the query or provide additional documents."
)

# Extension point: swap in any callable (e.g. a different provider) as long as it
# returns a validated FinancialAnalysis for a query + retrieved chunks.
AnalysisGenerator = Callable[[str, list[RetrievedChunk]], FinancialAnalysis]


def build_llm_generator(settings: Settings) -> AnalysisGenerator:
    """Create the default LLM-backed analysis generator using structured output."""
    llm = ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key)
    structured_llm = llm.with_structured_output(FinancialAnalysis)

    def generate(query: str, chunks: list[RetrievedChunk]) -> FinancialAnalysis:
        user_prompt = build_user_prompt(query, chunks)
        result = structured_llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        if not isinstance(result, FinancialAnalysis):
            raise TypeError(f"LLM did not return a FinancialAnalysis instance, got {type(result)!r}")
        return result

    return generate


def validate_citations(analysis: FinancialAnalysis, retrieved_chunks: list[RetrievedChunk]) -> bool:
    """Ensure every cited source_id in the analysis was actually retrieved for this query.

    Rejects the analysis if `evidence` cites an unknown source_id, or if `sources` lists a
    source_id that was not among the retrieved chunks.
    """
    valid_ids = {chunk.source_id for chunk in retrieved_chunks}

    for evidence in analysis.evidence:
        for source_id in evidence.source_ids:
            if source_id not in valid_ids:
                logger.warning("Rejecting response: evidence cites unknown source_id '%s'.", source_id)
                return False

    for source in analysis.sources:
        if source.source_id not in valid_ids:
            logger.warning("Rejecting response: sources list contains unknown source_id '%s'.", source.source_id)
            return False

    return True


def answer_query(
    query: str,
    retriever: Retriever,
    generate_analysis: AnalysisGenerator,
) -> RagResponse:
    """Run the full grounded RAG pipeline for a single user query."""
    logger.info("Handling query: %s", query)
    retrieved_chunks = retriever.retrieve(query)

    if not retrieved_chunks:
        logger.info("No chunks passed the relevance threshold; skipping LLM call.")
        return RagResponse(status="INSUFFICIENT_EVIDENCE", message=INSUFFICIENT_EVIDENCE_MESSAGE)

    try:
        analysis = generate_analysis(query, retrieved_chunks)
    except Exception:
        logger.exception("LLM generation failed.")
        return RagResponse(
            status="VALIDATION_FAILED",
            message=VALIDATION_FAILED_MESSAGE,
            retrieved_sources=retrieved_chunks,
        )

    if not validate_citations(analysis, retrieved_chunks):
        return RagResponse(
            status="VALIDATION_FAILED",
            message=VALIDATION_FAILED_MESSAGE,
            retrieved_sources=retrieved_chunks,
        )

    return RagResponse(status="OK", analysis=analysis, retrieved_sources=retrieved_chunks)
