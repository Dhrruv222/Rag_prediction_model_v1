"""Semantic retrieval over the FAISS vectorstore with relevance-score filtering."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_community.vectorstores import FAISS

from src.schemas import RetrievedChunk
from src.utils import expand_ticker_query, is_ticker_query, make_source_id

logger = logging.getLogger(__name__)


@dataclass
class Retriever:
    """Wraps a FAISS vectorstore to produce relevance-filtered, source-tagged chunks.

    The vectorstore must have been built with normalize_L2=True and
    distance_strategy=MAX_INNER_PRODUCT (see src.ingest.build_vectorstore) so that the
    similarity score returned here is real cosine similarity, not a fabricated value.
    """

    vectorstore: FAISS
    top_k: int = 5
    relevance_threshold: float = 0.25
    fetch_k: int = 20

    def preprocess_query(self, query: str) -> str:
        """Expand bare ticker queries (e.g. 'AAPL') into a richer retrieval query."""
        query = query.strip()
        if is_ticker_query(query):
            expanded = expand_ticker_query(query)
            logger.info("Expanded ticker query '%s' -> '%s'", query, expanded)
            return expanded
        return query

    def retrieve(self, query: str, metadata_filter: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        """Retrieve, score, and relevance-filter the top-k chunks for a query."""
        processed_query = self.preprocess_query(query)
        fetch_k = max(self.top_k, self.fetch_k)

        results = self.vectorstore.similarity_search_with_score(
            processed_query, k=fetch_k, filter=metadata_filter
        )
        logger.info("Retrieved %d candidate chunk(s) for query.", len(results))

        chunks: list[RetrievedChunk] = []
        for document, score in results:
            if score < self.relevance_threshold:
                continue
            metadata = document.metadata
            chunks.append(
                RetrievedChunk(
                    source_id=make_source_id(len(chunks) + 1),
                    file_name=metadata.get("file_name", "unknown"),
                    source_path=metadata.get("source", "unknown"),
                    page=metadata.get("page"),
                    chunk_id=metadata.get("chunk_id", "unknown"),
                    content=document.page_content,
                    relevance_score=float(score),
                )
            )
            if len(chunks) >= self.top_k:
                break

        logger.info(
            "%d chunk(s) passed relevance threshold %.2f.", len(chunks), self.relevance_threshold
        )
        return chunks
