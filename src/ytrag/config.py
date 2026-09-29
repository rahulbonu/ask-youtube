"""Settings, resolved from the environment or Streamlit secrets.

Values are read in this order, first hit wins:

1. process environment (including anything ``python-dotenv`` loaded from ``.env``)
2. ``st.secrets`` (how Streamlit Cloud injects configuration)
3. the default baked in below

Reading ``st.secrets`` is wrapped in ``try``/``except`` on purpose: it raises when
no secrets file exists, and importing this module must work outside Streamlit
(tests, scripts) where there is no script run context at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _from_secrets(name: str) -> str | None:
    try:
        import streamlit as st

        value = st.secrets.get(name)  # type: ignore[union-attr]
    except Exception:
        return None
    return None if value is None else str(value)


def get_setting(name: str, default: str | None = None) -> str | None:
    """Resolve a single setting from the environment, then Streamlit secrets."""
    for candidate in (os.environ.get(name), _from_secrets(name)):
        if candidate is not None and candidate.strip():
            return candidate.strip()
    return default


def _get_bool(name: str, default: bool) -> bool:
    raw = get_setting(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = get_setting(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_list(name: str, default: list[str]) -> list[str]:
    raw = get_setting(name)
    if not raw:
        return list(default)
    items = [part.strip().lower() for part in raw.split(",")]
    return [item for item in items if item] or list(default)


@dataclass(frozen=True)
class Settings:
    """Everything the pipeline needs to know, in one immutable object."""

    llm_provider: str = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    hf_token: str | None = None
    hf_model: str = DEFAULT_HF_MODEL

    embed_model: str = DEFAULT_EMBED_MODEL
    cache_dir: Path = field(default_factory=lambda: Path(".cache/ytrag"))

    languages: list[str] = field(default_factory=lambda: ["en"])
    youtube_proxy: str | None = None

    chunk_chars: int = 1000
    chunk_overlap_chars: int = 200
    top_k: int = 5
    condense_question: bool = True
    max_history_turns: int = 6

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            llm_provider=(get_setting("YTRAG_LLM_PROVIDER", "anthropic") or "anthropic").lower(),
            anthropic_api_key=get_setting("ANTHROPIC_API_KEY"),
            anthropic_model=get_setting("YTRAG_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
            or DEFAULT_ANTHROPIC_MODEL,
            # HF_TOKEN is the name the huggingface_hub client itself reads; the
            # other two are accepted because older deployments used them.
            hf_token=get_setting("HF_TOKEN")
            or get_setting("HUGGINGFACEHUB_ACCESS_TOKEN")
            or get_setting("HUGGINGFACE_API_TOKEN"),
            hf_model=get_setting("YTRAG_HF_MODEL", DEFAULT_HF_MODEL) or DEFAULT_HF_MODEL,
            embed_model=get_setting("YTRAG_EMBED_MODEL", DEFAULT_EMBED_MODEL)
            or DEFAULT_EMBED_MODEL,
            cache_dir=Path(get_setting("YTRAG_CACHE_DIR", ".cache/ytrag") or ".cache/ytrag"),
            languages=_get_list("YTRAG_LANGUAGES", ["en"]),
            youtube_proxy=get_setting("YTRAG_YOUTUBE_PROXY"),
            chunk_chars=_get_int("YTRAG_CHUNK_CHARS", 1000),
            chunk_overlap_chars=_get_int("YTRAG_CHUNK_OVERLAP_CHARS", 200),
            top_k=_get_int("YTRAG_TOP_K", 5),
            condense_question=_get_bool("YTRAG_CONDENSE_QUESTION", True),
            max_history_turns=_get_int("YTRAG_MAX_HISTORY_TURNS", 6),
        )
