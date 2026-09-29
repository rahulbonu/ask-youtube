"""Domain errors.

Every failure the user can plausibly cause or fix gets its own type, so the UI
can render an actionable message instead of a stack trace.
"""

from __future__ import annotations


class YtragError(Exception):
    """Base class for every error this package raises deliberately."""


class InvalidVideoInput(YtragError):
    """The supplied text is not a YouTube video ID or a recognisable URL."""


class TranscriptUnavailable(YtragError):
    """No transcript could be retrieved for a video.

    Carries the per-source failures so the UI can explain what was tried.
    """

    def __init__(self, video_id: str, attempts: list[tuple[str, str]]) -> None:
        self.video_id = video_id
        self.attempts = attempts
        detail = "\n".join(f"  - {source}: {reason}" for source, reason in attempts)
        super().__init__(f"No transcript available for {video_id!r}. Tried:\n{detail}")


class ProviderNotConfigured(YtragError):
    """An answer provider was selected but its credentials are missing."""


class ProviderError(YtragError):
    """An answer provider was reachable but failed to produce an answer."""
