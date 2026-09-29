from __future__ import annotations

from pathlib import Path

from ytrag.config import Settings


def test_defaults_are_sane() -> None:
    settings = Settings()
    assert settings.llm_provider == "anthropic"
    assert settings.top_k == 5
    assert settings.chunk_overlap_chars < settings.chunk_chars
    assert settings.condense_question is True


def test_reads_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("YTRAG_LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-test  ")
    monkeypatch.setenv("YTRAG_TOP_K", "9")
    monkeypatch.setenv("YTRAG_LANGUAGES", "HI, en ,")
    monkeypatch.setenv("YTRAG_CONDENSE_QUESTION", "false")
    monkeypatch.setenv("YTRAG_CACHE_DIR", "/tmp/ytrag-test")

    settings = Settings.from_env()
    assert settings.llm_provider == "huggingface"
    assert settings.anthropic_api_key == "sk-test"
    assert settings.top_k == 9
    assert settings.languages == ["hi", "en"]
    assert settings.condense_question is False
    assert settings.cache_dir == Path("/tmp/ytrag-test")


def test_invalid_numbers_fall_back_to_the_default(monkeypatch) -> None:
    monkeypatch.setenv("YTRAG_TOP_K", "not-a-number")
    assert Settings.from_env().top_k == 5


def test_hf_token_accepts_legacy_variable_names(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACEHUB_ACCESS_TOKEN", "hf-legacy")
    assert Settings.from_env().hf_token == "hf-legacy"


def test_empty_language_list_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("YTRAG_LANGUAGES", " , ,")
    assert Settings.from_env().languages == ["en"]
