"""Application configuration loaded from environment variables.

No secrets are hardcoded here. Values are read from the process environment (populated
from a local .env file via python-dotenv) with sensible defaults where appropriate.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _get_str(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise ConfigError(
            f"Missing required environment variable '{name}'. "
            "Set it in your .env file (see .env.example) or in the environment."
        )
    return value or ""


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable '{name}' must be an integer, got {raw!r}.") from exc


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable '{name}' must be a float, got {raw!r}.") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable, validated application settings."""

    openai_api_key: str
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    relevance_threshold: float
    data_dir: Path
    vectorstore_dir: Path


def load_settings() -> Settings:
    """Load and validate application settings from environment variables.

    Raises:
        ConfigError: If OPENAI_API_KEY is missing, or a numeric setting is invalid.
    """
    return Settings(
        openai_api_key=_get_str("OPENAI_API_KEY", required=True),
        chat_model=_get_str("OPENAI_CHAT_MODEL", default="gpt-4o-mini"),
        embedding_model=_get_str("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small"),
        chunk_size=_get_int("CHUNK_SIZE", default=1000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", default=150),
        top_k=_get_int("TOP_K", default=5),
        relevance_threshold=_get_float("RELEVANCE_THRESHOLD", default=0.25),
        data_dir=Path(_get_str("DATA_DIR", default="data")),
        vectorstore_dir=Path(_get_str("VECTORSTORE_DIR", default="vectorstore")),
    )
