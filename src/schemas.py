"""Pydantic v2 schemas for structured, source-attributed financial analysis output."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single retrieved document chunk that may be cited as evidence."""

    source_id: str
    file_name: str
    page: int | None = None
    chunk_id: str
    citation: str


class Evidence(BaseModel):
    """A claim tied to one or more retrieved sources."""

    claim: str
    source_ids: list[str] = Field(default_factory=list)


class FinancialAnalysis(BaseModel):
    """Structured, strictly-grounded financial analysis produced by the LLM.

    `investment_view` reflects only what retrieved evidence supports — it is never a
    trading recommendation, and a positive business/financial outlook does not imply
    a supported investment view (valuation/market evidence may still be absent).
    """

    ticker: str | None = None
    summary: str
    future_outlook: str
    positive_factors: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    investment_view: Literal["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_EVIDENCE"]
    investment_reasoning: str
    sources: list[Source] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    conflicting_evidence: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever, enriched with a stable source id and score."""

    source_id: str
    file_name: str
    source_path: str
    page: int | None
    chunk_id: str
    content: str
    relevance_score: float


class RagResponse(BaseModel):
    """Top-level result of a RAG query: either a validated analysis or a safe fallback."""

    status: Literal["OK", "INSUFFICIENT_EVIDENCE", "VALIDATION_FAILED"]
    message: str | None = None
    analysis: FinancialAnalysis | None = None
    retrieved_sources: list[RetrievedChunk] = Field(default_factory=list)
