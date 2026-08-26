"""Tests for grounding safeguards: insufficient-evidence handling, citation validation,
and hallucination resistance.

These tests use stub retrievers and stub LLM generators (dependency injection) so they
never call a real model or require an OpenAI API key.
"""
from __future__ import annotations

from src.rag_chain import answer_query, validate_citations
from src.schemas import Evidence, FinancialAnalysis, RetrievedChunk, Source


def _make_chunk(source_id: str = "SRC_001") -> RetrievedChunk:
    return RetrievedChunk(
        source_id=source_id,
        file_name="apple.txt",
        source_path="apple.txt",
        page=None,
        chunk_id="apple_p0_c000",
        content="Apple revenue increased according to Source A.",
        relevance_score=0.9,
    )


def _make_analysis(source_ids: list[str]) -> FinancialAnalysis:
    return FinancialAnalysis(
        ticker="AAPL",
        summary="Apple revenue increased per the retrieved document.",
        future_outlook="Insufficient evidence in the retrieved documents.",
        positive_factors=["Revenue increased."],
        risks=[],
        evidence=[Evidence(claim="Apple revenue increased.", source_ids=source_ids)],
        confidence="LOW",
        investment_view="INSUFFICIENT_EVIDENCE",
        investment_reasoning="No valuation or price data is present in the retrieved documents.",
        sources=[
            Source(source_id=sid, file_name="apple.txt", page=None, chunk_id="apple_p0_c000", citation="apple.txt")
            for sid in source_ids
        ],
        limitations=["No live market data was used."],
    )


def test_validate_citations_accepts_known_source_id() -> None:
    chunk = _make_chunk("SRC_001")
    analysis = _make_analysis(["SRC_001"])

    assert validate_citations(analysis, [chunk]) is True


def test_validate_citations_rejects_fabricated_source_id() -> None:
    chunk = _make_chunk("SRC_001")
    analysis = _make_analysis(["SRC_999"])  # fabricated, never retrieved

    assert validate_citations(analysis, [chunk]) is False


def test_financial_analysis_supports_recording_conflicting_evidence() -> None:
    """Structural check: conflicting sources must be representable, never silently resolved."""
    analysis = _make_analysis(["SRC_001"])
    analysis.conflicting_evidence = [
        "Source A states revenue increased; Source B states revenue decreased for the same period.",
    ]

    assert analysis.conflicting_evidence


class _EmptyRetriever:
    """Stub retriever that always returns no chunks (simulates no relevant evidence)."""

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return []


def test_answer_query_returns_insufficient_evidence_without_calling_llm() -> None:
    def fake_generate(query: str, chunks: list[RetrievedChunk]) -> FinancialAnalysis:
        raise AssertionError("LLM must not be called when no chunks pass the relevance threshold.")

    response = answer_query("What is the outlook for AAPL?", _EmptyRetriever(), fake_generate)

    assert response.status == "INSUFFICIENT_EVIDENCE"


class _SingleChunkRetriever:
    """Stub retriever returning one chunk about revenue, with no price information."""

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return [_make_chunk("SRC_001")]


def test_answer_query_does_not_invent_a_stock_price_when_absent_from_context() -> None:
    """Hallucination-resistance: the analysis must not fabricate a price the context lacks."""

    def fake_generate(query: str, chunks: list[RetrievedChunk]) -> FinancialAnalysis:
        analysis = _make_analysis(["SRC_001"])
        analysis.future_outlook = "Insufficient evidence in the retrieved documents to determine a stock price."
        return analysis

    response = answer_query("What is Apple's current stock price?", _SingleChunkRetriever(), fake_generate)

    assert response.status == "OK"
    assert response.analysis is not None
    assert "$" not in response.analysis.future_outlook
    assert "insufficient evidence" in response.analysis.future_outlook.lower()


def test_answer_query_rejects_response_with_fabricated_citation() -> None:
    def fake_generate(query: str, chunks: list[RetrievedChunk]) -> FinancialAnalysis:
        return _make_analysis(["SRC_999"])  # cites a source that was never retrieved

    response = answer_query("What are the risks facing AAPL?", _SingleChunkRetriever(), fake_generate)

    assert response.status == "VALIDATION_FAILED"
