"""Streamlit front end.

Deliberately thin: this module handles widgets, session state, and rendering.
Every decision about transcripts, retrieval, and answering lives in ``ytrag`` so
it can be tested without a browser.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

# Streamlit Cloud runs this file directly rather than installing the project, so
# the package directory has to be made importable by hand.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

from ytrag.config import Settings  # noqa: E402
from ytrag.embeddings import SentenceTransformerEmbedder  # noqa: E402
from ytrag.engine import ChatEngine  # noqa: E402
from ytrag.errors import (  # noqa: E402
    InvalidVideoInput,
    ProviderError,
    ProviderNotConfigured,
    TranscriptUnavailable,
)
from ytrag.index import load_or_build_index  # noqa: E402
from ytrag.llm import PROVIDERS, build_provider  # noqa: E402
from ytrag.render import linkify_citations, sources_markdown  # noqa: E402
from ytrag.transcript import fetch_transcript  # noqa: E402
from ytrag.video import extract_video_id, format_timestamp, watch_url  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Ask a YouTube Video", page_icon="🎬", layout="centered")

SETTINGS = Settings.from_env()


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    """One embedding model per process. The first call downloads the weights."""
    return SentenceTransformerEmbedder(model_name)


def init_state() -> None:
    defaults: dict[str, object] = {
        "messages": [],
        "index": None,
        "video_id": "",
        "video_meta": {},
        "url_input": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Answer model")

    provider_name = st.selectbox(
        "Provider",
        PROVIDERS,
        index=PROVIDERS.index(SETTINGS.llm_provider)
        if SETTINGS.llm_provider in PROVIDERS
        else 0,
        help="Set YTRAG_LLM_PROVIDER to change the default.",
    )

    key_present = {
        "anthropic": bool(SETTINGS.anthropic_api_key),
        "huggingface": bool(SETTINGS.hf_token),
    }
    key_name = {"anthropic": "ANTHROPIC_API_KEY", "huggingface": "HF_TOKEN"}[provider_name]
    model_name = {
        "anthropic": SETTINGS.anthropic_model,
        "huggingface": SETTINGS.hf_model,
    }[provider_name]

    if key_present[provider_name]:
        st.caption(f"Model: `{model_name}`")
    else:
        st.warning(f"`{key_name}` is not set. Add it to `.env`, or to Settings → Secrets.")

    st.divider()
    st.subheader("Transcript")

    language_text = st.text_input(
        "Preferred languages",
        value=",".join(SETTINGS.languages),
        help="Comma-separated language codes, most preferred first. Falls back to "
        "any available track, translating where YouTube allows it.",
    )
    languages = [part.strip().lower() for part in language_text.split(",") if part.strip()]

    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("Excerpts per answer", 3, 10, SETTINGS.top_k)
    condense = st.toggle(
        "Rewrite follow-up questions",
        value=SETTINGS.condense_question,
        help="Turns “what about the second one?” into a standalone question before "
        "searching. Costs one extra short model call per follow-up.",
    )

    meta = st.session_state.video_meta
    if meta:
        st.divider()
        st.subheader("Loaded video")
        st.caption(
            f"Language: `{meta['language']}`  \n"
            f"Source: `{meta['source']}`  \n"
            f"Length: `{format_timestamp(meta['duration'])}`  \n"
            f"Chunks: `{meta['chunks']}`  \n"
            f"Index: `{'cached' if meta['cached'] else 'freshly built'}`"
        )

    st.divider()
    if st.button("Clear chat", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()
    if st.button(
        "Load a different video",
        use_container_width=True,
        disabled=not st.session_state.index,
    ):
        st.session_state.update(messages=[], index=None, video_id="", video_meta={})
        st.rerun()


# Settings the sidebar can override at runtime.
active_settings = replace(
    SETTINGS,
    llm_provider=provider_name,
    languages=languages or ["en"],
    top_k=top_k,
    condense_question=condense,
)


# --------------------------------------------------------------------------
# Header and video loading
# --------------------------------------------------------------------------

st.title("🎬 Ask a YouTube Video")
st.caption("Answers come from the video's transcript, cited back to the exact second.")


def load_video(raw_input: str) -> None:
    """Fetch, chunk, and embed a video, then store the index in session state."""
    video_id = extract_video_id(raw_input)

    with st.status("Preparing the video…", expanded=True) as status:
        st.write("Fetching the transcript…")
        transcript = fetch_transcript(
            video_id,
            languages=active_settings.languages,
            proxy=active_settings.youtube_proxy,
        )
        st.write(
            f"Got {len(transcript.segments):,} caption cues "
            f"in `{transcript.language}` via `{transcript.source}`."
        )

        st.write("Building the search index…")
        embedder = get_embedder(active_settings.embed_model)
        index, cached = load_or_build_index(
            transcript,
            embedder,
            active_settings.cache_dir,
            active_settings.chunk_chars,
            active_settings.chunk_overlap_chars,
        )
        st.write(
            f"{len(index)} chunks "
            + ("loaded from cache." if cached else "embedded and cached for next time.")
        )

        st.session_state.index = index
        st.session_state.video_id = video_id
        st.session_state.messages = []
        st.session_state.video_meta = {
            "language": transcript.language,
            "source": transcript.source,
            "duration": transcript.duration,
            "chunks": len(index),
            "cached": cached,
        }
        status.update(label="Ready.", state="complete", expanded=False)


if st.session_state.index is None:
    with st.form("load_video", clear_on_submit=False):
        raw_input = st.text_input(
            "YouTube URL or video ID",
            placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )
        submitted = st.form_submit_button("Load video", type="primary", use_container_width=True)

    if submitted:
        try:
            load_video(raw_input)
            st.rerun()
        except InvalidVideoInput as exc:
            st.error(str(exc))
        except TranscriptUnavailable as exc:
            st.error("Could not get a transcript for this video.")
            with st.expander("What was tried, and what to do next"):
                for source, reason in exc.attempts:
                    st.markdown(f"- **{source}** — {reason}")
                st.markdown(
                    "---\n"
                    "Most likely causes:\n"
                    "- The video genuinely has captions disabled.\n"
                    "- YouTube is blocking this machine's IP. This is common on cloud "
                    "hosts; set `YTRAG_YOUTUBE_PROXY` to a proxy to work around it.\n"
                    "- The video is private, age-restricted, or region-locked."
                )
        except Exception as exc:  # noqa: BLE001 - last resort, shown to the user
            st.error(f"Something went wrong preparing this video: {exc}")

    st.stop()


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------

st.video(watch_url(st.session_state.video_id))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(linkify_citations(message["content"], message.get("citations", [])))
            if message.get("citations"):
                with st.expander(f"Sources ({len(message['citations'])})"):
                    st.markdown(sources_markdown(message["citations"]))
        else:
            st.markdown(message["content"])

question = st.chat_input("Ask something about this video…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Completed turns only - the question just asked is not part of its own history.
    history = [
        (st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["content"])
        for i in range(0, len(st.session_state.messages) - 1, 2)
        if st.session_state.messages[i + 1]["role"] == "assistant"
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            provider = build_provider(active_settings, provider_name)
            engine = ChatEngine(
                st.session_state.index, get_embedder(active_settings.embed_model),
                provider, active_settings,
            )

            stream = engine.stream(question, history)
            buffer = ""
            for piece in stream:
                buffer += piece
                placeholder.markdown(buffer + "▌")

            answer = stream.result
            placeholder.markdown(linkify_citations(answer.text, answer.citations))

            if answer.citations:
                with st.expander(f"Sources ({len(answer.citations)})"):
                    st.markdown(sources_markdown(answer.citations))
            if answer.was_rewritten:
                st.caption(f"Searched for: _{answer.standalone_question}_")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer.text, "citations": answer.citations}
            )

        except ProviderNotConfigured as exc:
            placeholder.empty()
            st.error(str(exc))
            st.session_state.messages.pop()
        except ProviderError as exc:
            placeholder.empty()
            st.error(str(exc))
            st.session_state.messages.pop()
        except Exception as exc:  # noqa: BLE001 - last resort, shown to the user
            placeholder.empty()
            st.error(f"Could not generate an answer: {exc}")
            st.session_state.messages.pop()
