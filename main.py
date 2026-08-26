"""CLI entry point for the Financial RAG V1 system."""
from __future__ import annotations

import argparse
import logging
import sys

from src.config import ConfigError, load_settings
from src.ingest import IngestionError, build_vectorstore, load_vectorstore
from src.rag_chain import answer_query, build_llm_generator
from src.retriever import Retriever
from src.schemas import RagResponse
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Financial RAG V1 - strictly grounded RAG for financial analysis."
    )
    parser.add_argument("--ingest", action="store_true", help="Build/update the FAISS index from data/.")
    parser.add_argument("--rebuild", action="store_true", help="Force a full rebuild of the FAISS index.")
    parser.add_argument("--query", type=str, default=None, help="Run a single query non-interactively.")
    return parser.parse_args(argv)


def print_analysis(response: RagResponse) -> None:
    """Pretty-print a RagResponse to stdout."""
    if response.status != "OK" or response.analysis is None:
        print(f"\n[{response.status}] {response.message}\n")
        return

    analysis = response.analysis
    print("\n" + "=" * 70)
    print(f"Ticker: {analysis.ticker or 'N/A'}")
    print(f"Confidence: {analysis.confidence}   Investment view: {analysis.investment_view}")
    print("-" * 70)
    print("Summary:\n" + analysis.summary)
    print("\nFuture outlook:\n" + analysis.future_outlook)

    if analysis.positive_factors:
        print("\nPositive factors:")
        for item in analysis.positive_factors:
            print(f"  - {item}")

    if analysis.risks:
        print("\nRisks:")
        for item in analysis.risks:
            print(f"  - {item}")

    if analysis.conflicting_evidence:
        print("\nConflicting evidence:")
        for item in analysis.conflicting_evidence:
            print(f"  - {item}")

    if analysis.evidence:
        print("\nEvidence:")
        for item in analysis.evidence:
            print(f"  - {item.claim} [{', '.join(item.source_ids)}]")

    print("\nInvestment reasoning:\n" + analysis.investment_reasoning)

    if analysis.limitations:
        print("\nLimitations:")
        for item in analysis.limitations:
            print(f"  - {item}")

    if analysis.sources:
        print("\nSources:")
        for source in analysis.sources:
            page = f"p.{source.page}" if source.page is not None else "n/a"
            print(f"  [{source.source_id}] {source.file_name} ({page}) - {source.citation}")

    print("=" * 70 + "\n")


def run_ingest(force_rebuild: bool) -> None:
    """Build or update the FAISS index from the data directory."""
    settings = load_settings()
    build_vectorstore(settings, force_rebuild=force_rebuild)
    print("Ingestion complete. Vectorstore is ready.")


def _build_retriever_and_generator(settings) -> tuple[Retriever, "callable"]:
    vectorstore = load_vectorstore(settings)
    retriever = Retriever(
        vectorstore=vectorstore,
        top_k=settings.top_k,
        relevance_threshold=settings.relevance_threshold,
    )
    generate_analysis = build_llm_generator(settings)
    return retriever, generate_analysis


def run_query(query: str) -> None:
    """Answer a single query non-interactively."""
    settings = load_settings()
    retriever, generate_analysis = _build_retriever_and_generator(settings)
    response = answer_query(query, retriever, generate_analysis)
    print_analysis(response)


def run_interactive() -> None:
    """Start an interactive query loop."""
    settings = load_settings()
    retriever, generate_analysis = _build_retriever_and_generator(settings)

    print("Financial RAG V1")
    print("This analysis is based only on documents in the local knowledge base.")
    print("It does not use live market prices or real-time financial data.")
    print("Type a ticker or question, or 'exit' to quit.\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        response = answer_query(query, retriever, generate_analysis)
        print_analysis(response)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    setup_logging()
    args = parse_args(argv)

    try:
        if args.ingest:
            run_ingest(force_rebuild=args.rebuild)
            return 0
        if args.query:
            run_query(args.query)
            return 0
        run_interactive()
        return 0
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except IngestionError as exc:
        logger.error("Ingestion error: %s", exc)
        print(f"Ingestion error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
