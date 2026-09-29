"""Transcript data model.

The whole point of keeping ``Segment`` around is that every piece of text stays
attached to the moment it was spoken. Flattening a transcript to one string
early - which is the obvious shortcut - makes timestamped citations impossible
later, because there is no way back from a character offset to a video time.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Collapse whitespace so chunk sizes mean the same thing across sources."""
    return _WHITESPACE.sub(" ", text or "").strip()


@dataclass(frozen=True, slots=True)
class Segment:
    """One caption cue: some text, and when it was said."""

    text: str
    start: float
    duration: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True, slots=True)
class Transcript:
    """An ordered set of caption cues for one video."""

    video_id: str
    language: str
    source: str
    segments: tuple[Segment, ...] = field(default_factory=tuple)

    @classmethod
    def from_items(
        cls,
        video_id: str,
        language: str,
        source: str,
        items: object,
    ) -> Transcript:
        """Build a transcript from raw cues.

        Accepts the two shapes ``youtube-transcript-api`` has used across
        versions - dicts and objects with ``.text`` / ``.start`` / ``.duration``
        attributes - as well as our own ``Segment`` instances.
        """
        segments: list[Segment] = []
        for item in items or ():  # type: ignore[union-attr]
            if isinstance(item, Segment):
                text, start, duration = item.text, item.start, item.duration
            elif isinstance(item, dict):
                text = item.get("text", "")
                start = item.get("start", 0.0)
                duration = item.get("duration", 0.0)
            else:
                text = getattr(item, "text", "")
                start = getattr(item, "start", 0.0)
                duration = getattr(item, "duration", 0.0)

            cleaned = normalise(str(text))
            if not cleaned:
                continue
            segments.append(
                Segment(text=cleaned, start=float(start or 0.0), duration=float(duration or 0.0))
            )

        if not segments:
            raise ValueError("transcript contained no usable text")

        segments.sort(key=lambda seg: seg.start)
        return cls(video_id=video_id, language=language, source=source, segments=tuple(segments))

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments)

    @property
    def duration(self) -> float:
        return max((segment.end for segment in self.segments), default=0.0)

    def fingerprint(self) -> str:
        """Stable hash of the transcript content, used as part of a cache key."""
        digest = hashlib.sha256()
        digest.update(self.video_id.encode())
        for segment in self.segments:
            digest.update(f"{segment.start:.3f}|{segment.text}\n".encode())
        return digest.hexdigest()[:16]
