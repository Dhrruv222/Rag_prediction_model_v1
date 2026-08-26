"""Document ingestion: load PDFs/TXT from data/, chunk them, and build the FAISS index."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.config import Settings
from src.utils import make_chunk_id

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


@dataclass(frozen=True)
class RawPage:
    """One page (PDF) or whole file (TXT) of extracted text before chunking."""

    text: str
    source_path: Path
    file_name: str
    document_type: str
    page: int | None


class IngestionError(RuntimeError):
    """Raised for ingestion-level failures that should stop or flag the pipeline."""


def scan_data_dir(data_dir: Path) -> list[Path]:
    """Recursively find supported document files under data_dir.

    Raises:
        IngestionError: If the directory is missing or contains no supported documents.
    """
    if not data_dir.exists():
        raise IngestionError(
            f"Data directory '{data_dir}' does not exist. Create it and add PDF/TXT files."
        )
    files = [
        path
        for path in sorted(data_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise IngestionError(
            f"No supported documents (.pdf, .txt) found in '{data_dir}'. "
            "Add documents before running ingestion."
        )
    return files


def _load_pdf(path: Path) -> list[RawPage]:
    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError) as exc:
        raise IngestionError(f"Failed to open PDF '{path}': {exc}") from exc

    pages: list[RawPage] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:  # pypdf can raise a variety of per-page errors
            logger.error("Failed to extract text from '%s' page %s: %s", path, page_number, exc)
            continue
        if text:
            pages.append(
                RawPage(
                    text=text,
                    source_path=path,
                    file_name=path.name,
                    document_type="pdf",
                    page=page_number,
                )
            )
    if not pages:
        logger.warning("PDF '%s' produced no extractable text.", path)
    return pages


def _load_txt(path: Path) -> list[RawPage]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1").strip()
    except OSError as exc:
        raise IngestionError(f"Failed to read TXT file '{path}': {exc}") from exc

    if not text:
        logger.warning("TXT file '%s' is empty.", path)
        return []
    return [RawPage(text=text, source_path=path, file_name=path.name, document_type="txt", page=None)]


def load_documents(data_dir: Path) -> list[RawPage]:
    """Load all supported documents under data_dir.

    Failures on individual files are logged clearly (never silently swallowed) and do not
    stop ingestion of the remaining files, unless every single file fails.
    """
    files = scan_data_dir(data_dir)
    pages: list[RawPage] = []
    failures: list[str] = []

    for path in files:
        try:
            if path.suffix.lower() == ".pdf":
                pages.extend(_load_pdf(path))
            else:
                pages.extend(_load_txt(path))
        except IngestionError as exc:
            logger.error(str(exc))
            failures.append(str(exc))

    if failures and not pages:
        raise IngestionError("All documents failed to load. First error: " + failures[0])

    logger.info(
        "Loaded %d page(s)/document(s) from %d file(s) (%d failure(s)).",
        len(pages), len(files), len(failures),
    )
    return pages


def chunk_documents(pages: list[RawPage], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split loaded pages into overlapping chunks, preserving metadata and stable IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for page in pages:
        file_stem = Path(page.file_name).stem
        for index, split_text in enumerate(splitter.split_text(page.text)):
            chunk_id = make_chunk_id(file_stem, page.page, index)
            metadata = {
                "source": str(page.source_path),
                "file_name": page.file_name,
                "document_type": page.document_type,
                "page": page.page,
                "chunk_id": chunk_id,
            }
            documents.append(Document(page_content=split_text, metadata=metadata))

    logger.info("Created %d chunk(s) from %d loaded page(s).", len(documents), len(pages))
    return documents


def _vectorstore_exists(vectorstore_dir: Path) -> bool:
    return (vectorstore_dir / "index.faiss").exists() and (vectorstore_dir / "index.pkl").exists()


def build_vectorstore(
    settings: Settings,
    embeddings: Embeddings | None = None,
    force_rebuild: bool = False,
) -> FAISS:
    """Build (or reuse) the FAISS vectorstore from documents in settings.data_dir.

    Embeddings are not regenerated if a vectorstore already exists, unless
    force_rebuild=True (e.g. via `python main.py --ingest --rebuild`).
    """
    if not force_rebuild and _vectorstore_exists(settings.vectorstore_dir):
        logger.info("Existing vectorstore found at '%s'; skipping rebuild.", settings.vectorstore_dir)
        return load_vectorstore(settings, embeddings)

    pages = load_documents(settings.data_dir)
    chunks = chunk_documents(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise IngestionError("No chunks were produced from the documents in the data directory.")

    embeddings = embeddings or OpenAIEmbeddings(model=settings.embedding_model)
    # Normalized vectors + inner product => raw score is real cosine similarity, not fabricated.
    # langchain-community warns that normalize_L2 is "not applicable" for this distance
    # strategy, but it still normalizes on add/search regardless of distance_strategy; the
    # warning is a known upstream inaccuracy, not a functional issue.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Normalizing L2 is not applicable.*")
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings,
            normalize_L2=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(settings.vectorstore_dir))
    logger.info("Vectorstore built and saved to '%s' (%d vectors).", settings.vectorstore_dir, len(chunks))
    return vectorstore


def load_vectorstore(settings: Settings, embeddings: Embeddings | None = None) -> FAISS:
    """Load a previously built FAISS vectorstore from disk.

    Raises:
        IngestionError: If no vectorstore has been built yet.
    """
    if not _vectorstore_exists(settings.vectorstore_dir):
        raise IngestionError(
            f"No vectorstore found at '{settings.vectorstore_dir}'. Run 'python main.py --ingest' first."
        )
    embeddings = embeddings or OpenAIEmbeddings(model=settings.embedding_model)
    # allow_dangerous_deserialization is safe here only because we only ever load a
    # vectorstore that this same application built locally (see README security note).
    return FAISS.load_local(
        str(settings.vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
