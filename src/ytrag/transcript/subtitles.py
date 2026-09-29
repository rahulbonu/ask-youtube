"""Parsers for the two subtitle formats yt-dlp exposes for YouTube.

Both return ``Segment`` lists with real times. The original version of this
project stripped timestamps out with a regex and kept only the words, which is
why it could never cite anything - these parsers exist to keep that information.
"""

from __future__ import annotations

import json
import re

from .models import Segment, normalise

# Inline tags: <c.colorE5E5E5>, </c>, and karaoke cues like <00:00:12.345>.
_TAG_RE = re.compile(r"<[^>]*>")
# Positioning/style blobs that some tracks append to the text.
_BRACE_RE = re.compile(r"\{[^}]*\}")
_CUE_TIME_RE = re.compile(
    r"(?P<start>\d{1,3}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,3}:\d{2}[.,]\d{1,3})"
    r"\s*-->\s*"
    r"(?P<end>\d{1,3}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,3}:\d{2}[.,]\d{1,3})"
)


def parse_timestamp(value: str) -> float:
    """Parse ``HH:MM:SS.mmm`` or ``MM:SS.mmm`` into seconds."""
    cleaned = value.strip().replace(",", ".")
    parts = cleaned.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = "0", parts[0], parts[1]
    else:
        raise ValueError(f"unrecognised timestamp: {value!r}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _clean_cue_text(lines: list[str]) -> str:
    text = " ".join(lines)
    text = _TAG_RE.sub(" ", text)
    text = _BRACE_RE.sub(" ", text)
    return normalise(text)


def parse_vtt(content: str) -> list[Segment]:
    """Parse a WebVTT track into timed segments.

    YouTube's auto-generated VTT uses rolling captions, where each cue repeats
    the tail of the previous one. Those repeats are dropped so the transcript
    reads once rather than three times.
    """
    segments: list[Segment] = []
    block: list[str] = []

    def flush(block_lines: list[str]) -> None:
        timing_index = next(
            (i for i, line in enumerate(block_lines) if "-->" in line),
            None,
        )
        if timing_index is None:
            return
        match = _CUE_TIME_RE.search(block_lines[timing_index])
        if not match:
            return
        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        text = _clean_cue_text(block_lines[timing_index + 1 :])
        if not text:
            return
        segments.append(Segment(text=text, start=start, duration=max(0.0, end - start)))

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if line.strip():
            block.append(line)
            continue
        flush(block)
        block = []
    flush(block)

    return _dedupe_rolling(segments)


def parse_json3(content: str) -> list[Segment]:
    """Parse YouTube's ``json3`` caption format into timed segments.

    Preferred over VTT when available: times are given in milliseconds and the
    text carries no positioning markup to strip.
    """
    data = json.loads(content)
    segments: list[Segment] = []

    for event in data.get("events") or ():
        segs = event.get("segs") or ()
        text = normalise("".join(seg.get("utf8", "") for seg in segs))
        if not text:
            continue
        start_ms = event.get("tStartMs")
        if start_ms is None:
            continue
        duration_ms = event.get("dDurationMs") or 0
        segments.append(
            Segment(text=text, start=float(start_ms) / 1000.0, duration=float(duration_ms) / 1000.0)
        )

    return _dedupe_rolling(segments)


def _dedupe_rolling(segments: list[Segment]) -> list[Segment]:
    """Drop cues that only restate what the previous cue already said.

    Comparison is against the previous cue as it arrived, not as it was emitted.
    Once a rolling cue has been reduced to its new tail, the emitted text no
    longer resembles the next cue's prefix, so comparing against the output would
    let the third repeat of a rolling caption through.
    """
    result: list[Segment] = []
    previous = ""

    for segment in segments:
        text = segment.text

        if previous:
            # Exact repeat of the previous cue.
            if text == previous:
                continue
            # Rolling caption: the previous cue plus more words. Keep only the
            # new tail, timed from this cue's start.
            if text.startswith(previous) and len(text) > len(previous):
                remainder = text[len(previous) :].strip()
                previous = text
                if remainder:
                    result.append(
                        Segment(text=remainder, start=segment.start, duration=segment.duration)
                    )
                continue
            # Wholly contained in what we already have; nothing new to add.
            if text in previous:
                continue

        result.append(segment)
        previous = text

    return result
