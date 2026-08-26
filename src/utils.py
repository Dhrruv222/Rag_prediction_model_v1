"""Small shared helpers: logging setup, ticker detection, and stable id generation."""
from __future__ import annotations

import hashlib
import logging
import re

_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once. Never logs API keys or other secrets."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def is_ticker_query(query: str) -> bool:
    """Return True if the query looks like a bare ticker symbol, e.g. 'AAPL'."""
    return bool(_TICKER_PATTERN.match(query.strip()))


def expand_ticker_query(ticker: str) -> str:
    """Expand a bare ticker into a richer retrieval query covering common analysis angles."""
    ticker = ticker.strip().upper()
    return (
        f"Financial outlook, risks, growth, valuation, earnings, "
        f"competitive position and investment considerations for {ticker}."
    )


def make_chunk_id(file_stem: str, page: int | None, index: int) -> str:
    """Build a stable, human-readable chunk identifier."""
    page_part = f"p{page}" if page is not None else "p0"
    return f"{file_stem}_{page_part}_c{index:03d}"


def make_source_id(index: int) -> str:
    """Build a stable source identifier referenced in LLM context and citations."""
    return f"SRC_{index:03d}"


def stable_hash(*parts: str) -> str:
    """Deterministic short hash, useful for de-duplication or debugging."""
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
