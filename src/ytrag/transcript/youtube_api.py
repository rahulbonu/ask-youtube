"""Transcript source backed by ``youtube-transcript-api``.

This is the fast path: one HTTP request to YouTube's own caption endpoint, no
subprocess, no media parsing.
"""

from __future__ import annotations

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig

from .models import Transcript

SOURCE = "youtube-transcript-api"


def _build_client(proxy: str | None) -> YouTubeTranscriptApi:
    if not proxy:
        return YouTubeTranscriptApi()
    return YouTubeTranscriptApi(
        proxy_config=GenericProxyConfig(http_url=proxy, https_url=proxy)
    )


def fetch(video_id: str, languages: list[str], proxy: str | None = None) -> Transcript:
    """Fetch a transcript, preferring ``languages`` in the order given.

    Falls back through progressively looser strategies: an exact language match,
    then any track YouTube will translate into the preferred language, then
    whatever track exists at all. A video in the wrong language still beats no
    answer, and the UI reports which language was actually used.
    """
    api = _build_client(proxy)

    # 1. Exact match on a preferred language.
    try:
        fetched = api.fetch(video_id, languages=languages or ["en"])
        return Transcript.from_items(
            video_id, fetched.language_code, SOURCE, fetched.snippets
        )
    except Exception:
        pass

    # Listing is a separate request, so only do it once the direct path failed.
    listing = api.list(video_id)
    # TranscriptList iterates manually-created tracks before generated ones,
    # which is the quality order we want anyway.
    available = list(listing)
    if not available:
        raise RuntimeError("YouTube reported no caption tracks")

    wanted = list(dict.fromkeys([*(languages or []), "en"]))

    # 2. Exact language code among the listed tracks.
    for code in wanted:
        for track in available:
            if track.language_code == code:
                fetched = track.fetch()
                return Transcript.from_items(
                    video_id, fetched.language_code, SOURCE, fetched.snippets
                )

    # 3. Ask YouTube to translate a track we can get into one we want.
    target = wanted[0]
    for track in available:
        if not track.is_translatable:
            continue
        codes = {
            language.language_code
            for language in getattr(track, "translation_languages", []) or []
        }
        if target in codes:
            fetched = track.translate(target).fetch()
            return Transcript.from_items(
                video_id, fetched.language_code, f"{SOURCE} (translated)", fetched.snippets
            )

    # 4. Anything at all.
    last_error: Exception | None = None
    for track in available:
        try:
            fetched = track.fetch()
            return Transcript.from_items(
                video_id, fetched.language_code, SOURCE, fetched.snippets
            )
        except Exception as exc:  # noqa: PERF203 - we want to try every track
            last_error = exc

    raise RuntimeError(f"caption tracks were listed but none could be fetched: {last_error}")
