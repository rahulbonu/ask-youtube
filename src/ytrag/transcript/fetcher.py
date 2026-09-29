"""The transcript fallback chain."""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..errors import TranscriptUnavailable
from . import youtube_api, ytdlp
from .models import Transcript

logger = logging.getLogger(__name__)

Source = Callable[[str, list[str], str | None], Transcript]

# Ordered cheapest and most reliable first.
SOURCES: tuple[tuple[str, Source], ...] = (
    (youtube_api.SOURCE, youtube_api.fetch),
    (ytdlp.SOURCE, ytdlp.fetch),
)


def fetch_transcript(
    video_id: str,
    languages: list[str] | None = None,
    proxy: str | None = None,
    sources: tuple[tuple[str, Source], ...] = SOURCES,
) -> Transcript:
    """Try each transcript source in turn, returning the first that succeeds.

    Raises:
        TranscriptUnavailable: if every source failed. The exception carries the
            individual reasons so the UI can tell the user what to try next.
    """
    languages = languages or ["en"]
    attempts: list[tuple[str, str]] = []

    for name, source in sources:
        try:
            transcript = source(video_id, languages, proxy)
        except Exception as exc:
            logger.info("transcript source %s failed for %s: %s", name, video_id, exc)
            attempts.append((name, str(exc) or exc.__class__.__name__))
            continue
        logger.info(
            "transcript for %s from %s (%s, %d segments)",
            video_id,
            transcript.source,
            transcript.language,
            len(transcript.segments),
        )
        return transcript

    raise TranscriptUnavailable(video_id, attempts)
