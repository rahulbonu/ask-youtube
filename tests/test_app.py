"""Smoke tests for the Streamlit layer.

These run the real ``app.py`` through Streamlit's own test harness, which
executes the script exactly as the server would. They catch the mistakes that
unit tests cannot: a bad widget argument, a stale import, a crash on first load.

Nothing here touches the network - no video is ever loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

# Absolute, because AppTest resolves relative paths against this file.
APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def clean_env(monkeypatch):
    """Run the app as a new user would: no keys configured."""
    for name in (
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACEHUB_ACCESS_TOKEN",
        "HUGGINGFACE_API_TOKEN",
        "YTRAG_LLM_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_app_loads_without_error(clean_env) -> None:
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "🎬 Ask a YouTube Video"


def test_landing_page_asks_for_a_url(clean_env) -> None:
    app = AppTest.from_file(APP, default_timeout=60).run()
    labels = [widget.label for widget in app.text_input]
    assert "YouTube URL or video ID" in labels


def test_missing_key_is_flagged_not_crashed(clean_env) -> None:
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert any("ANTHROPIC_API_KEY" in warning.value for warning in app.warning)
    assert not app.exception


def test_invalid_url_shows_an_error_and_keeps_running(clean_env) -> None:
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.text_input[0].set_value("https://vimeo.com/123456789")
    app.button[0].click().run()

    assert not app.exception
    assert app.error, "an invalid URL should produce a visible error"
    assert "youtube" in app.error[0].value.lower()


def test_sidebar_defaults_match_settings(clean_env) -> None:
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert app.selectbox[0].value == "anthropic"
    assert app.slider[0].value == 5
