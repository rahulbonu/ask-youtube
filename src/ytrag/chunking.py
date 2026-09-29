"""Group caption segments into retrievable chunks that remember their times.

A generic character-based text splitter would work on the flattened transcript,
but the resulting chunks could not be traced back to a moment in the video.
Chunking over segments instead means every chunk knows exactly when it starts
and ends, which is what makes timestamped citations possible.
"""

from __future__ import annotations

from dataclasses import dataclass

from .transcript.models import Segment, Transcript


@dataclass(frozen=True, slots=True)
class Chunk:
    """A window of transcript text, with the span of video it covers."""

    index: int
    text: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return {"index": self.index, "text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(
            index=int(data["index"]),
            text=data["text"],
            start=float(data["start"]),
            end=float(data["end"]),
        )


def chunk_segments(
    segments: list[Segment] | tuple[Segment, ...],
    chunk_chars: int = 1000,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Pack ``segments`` into chunks of roughly ``chunk_chars`` characters.

    Consecutive chunks overlap by about ``overlap_chars`` so a sentence that
    straddles a boundary is still retrievable as a whole.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    overlap_chars = max(0, min(overlap_chars, chunk_chars - 1))

    units = _split_oversized(list(segments), chunk_chars)
    chunks: list[Chunk] = []
    current: list[Segment] = []
    current_chars = 0

    def emit() -> None:
        if not current:
            return
        chunks.append(
            Chunk(
                index=len(chunks),
                text=" ".join(segment.text for segment in current),
                start=current[0].start,
                end=max(segment.end for segment in current),
            )
        )

    for segment in units:
        addition = len(segment.text) + (1 if current else 0)
        if current and current_chars + addition > chunk_chars:
            emit()
            current = _tail_for_overlap(current, overlap_chars)
            current_chars = sum(len(seg.text) + 1 for seg in current)
        current.append(segment)
        current_chars += addition

    emit()
    return chunks


def chunk_transcript(
    transcript: Transcript,
    chunk_chars: int = 1000,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Chunk a whole transcript."""
    return chunk_segments(transcript.segments, chunk_chars, overlap_chars)


def _tail_for_overlap(segments: list[Segment], overlap_chars: int) -> list[Segment]:
    """Return the trailing segments worth about ``overlap_chars`` characters."""
    if overlap_chars <= 0:
        return []
    tail: list[Segment] = []
    total = 0
    for segment in reversed(segments):
        if total + len(segment.text) > overlap_chars and tail:
            break
        tail.insert(0, segment)
        total += len(segment.text) + 1
    # Never carry the entire chunk forward, or packing would not advance.
    if len(tail) >= len(segments):
        tail = tail[1:]
    return tail


def _split_oversized(segments: list[Segment], limit: int) -> list[Segment]:
    """Split any single segment longer than ``limit`` into timed pieces.

    Caption cues are normally a few words, but ``json3`` tracks occasionally
    carry a whole paragraph in one event. Times are interpolated across the
    pieces by character position, which is accurate enough to cite.
    """
    result: list[Segment] = []
    for segment in segments:
        if len(segment.text) <= limit:
            result.append(segment)
            continue

        words = segment.text.split()
        pieces: list[str] = []
        buffer: list[str] = []
        length = 0
        for word in words:
            if buffer and length + len(word) + 1 > limit:
                pieces.append(" ".join(buffer))
                buffer, length = [], 0
            buffer.append(word)
            length += len(word) + 1
        if buffer:
            pieces.append(" ".join(buffer))

        total_chars = sum(len(piece) for piece in pieces) or 1
        elapsed = 0
        for piece in pieces:
            fraction_start = elapsed / total_chars
            elapsed += len(piece)
            fraction_end = elapsed / total_chars
            result.append(
                Segment(
                    text=piece,
                    start=segment.start + segment.duration * fraction_start,
                    duration=segment.duration * (fraction_end - fraction_start),
                )
            )
    return result
