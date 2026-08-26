"""Tests for the Retriever: relevance-based filtering over FAISS results.

Uses a deterministic, dependency-free fake embeddings implementation so these tests
never require network access or an OpenAI API key.
"""
from __future__ import annotations

import hashlib
import warnings

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.retriever import Retriever


class DeterministicFakeEmbeddings(Embeddings):
    """Deterministic embeddings for tests: same text always maps to the same vector."""

    _dim = 32

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._dim]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _build_test_vectorstore(texts_and_metadata: list[tuple[str, dict]]) -> FAISS:
    documents = [Document(page_content=text, metadata=meta) for text, meta in texts_and_metadata]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Normalizing L2 is not applicable.*")
        return FAISS.from_documents(
            documents,
            DeterministicFakeEmbeddings(),
            normalize_L2=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )


def test_retriever_returns_relevant_chunk_for_matching_query() -> None:
    vectorstore = _build_test_vectorstore(
        [
            (
                "Apple revenue increased due to strong iPhone sales in the latest quarter.",
                {
                    "file_name": "apple.txt",
                    "source": "apple.txt",
                    "document_type": "txt",
                    "page": None,
                    "chunk_id": "apple_p0_c000",
                },
            ),
            (
                "Unrelated notes about gardening and houseplants.",
                {
                    "file_name": "garden.txt",
                    "source": "garden.txt",
                    "document_type": "txt",
                    "page": None,
                    "chunk_id": "garden_p0_c000",
                },
            ),
        ]
    )
    retriever = Retriever(vectorstore=vectorstore, top_k=5, relevance_threshold=0.0)

    results = retriever.retrieve("Apple revenue increased due to strong iPhone sales in the latest quarter.")

    assert len(results) >= 1
    assert results[0].file_name == "apple.txt"
    assert results[0].source_id == "SRC_001"


def test_retriever_filters_out_low_relevance_chunks() -> None:
    vectorstore = _build_test_vectorstore(
        [
            (
                "Apple revenue increased due to strong iPhone sales.",
                {
                    "file_name": "apple.txt",
                    "source": "apple.txt",
                    "document_type": "txt",
                    "page": None,
                    "chunk_id": "apple_p0_c000",
                },
            ),
        ]
    )
    # Cosine similarity is bounded by 1.0, so this threshold is unreachable by any result.
    retriever = Retriever(vectorstore=vectorstore, top_k=5, relevance_threshold=1.01)

    results = retriever.retrieve("Apple revenue increased due to strong iPhone sales.")

    assert results == []
