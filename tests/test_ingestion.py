"""Tests for document ingestion: loading, metadata preservation, and chunking."""
from __future__ import annotations

from pathlib import Path

from src.ingest import chunk_documents, load_documents


def test_load_documents_extracts_txt_with_metadata(tmp_path: Path) -> None:
    doc_path = tmp_path / "note.txt"
    doc_path.write_text("Apple reported strong iPhone revenue growth this quarter.")

    pages = load_documents(tmp_path)

    assert len(pages) == 1
    assert pages[0].file_name == "note.txt"
    assert pages[0].document_type == "txt"
    assert pages[0].page is None
    assert "iPhone" in pages[0].text


def test_chunk_documents_preserves_metadata_and_stable_ids(tmp_path: Path) -> None:
    doc_path = tmp_path / "report.txt"
    doc_path.write_text("Sentence one. " * 200)

    pages = load_documents(tmp_path)
    chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["file_name"] == "report.txt"
        assert chunk.metadata["document_type"] == "txt"
        assert chunk.metadata["chunk_id"].startswith("report_p0_c")

    chunk_ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
