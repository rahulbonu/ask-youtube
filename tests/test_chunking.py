from __future__ import annotations

import pytest

from ytrag.chunking import chunk_segments, chunk_transcript
from ytrag.transcript.models import Segment


def test_chunks_carry_the_time_span_they_cover(transcript) -> None:
    chunks = chunk_transcript(transcript, chunk_chars=100, overlap_chars=0)
    assert chunks
    assert chunks[0].start == 0.0
    # The last chunk must end at the end of the last segment it contains.
    assert chunks[-1].end == pytest.approx(15.0)
    for chunk in chunks:
        assert chunk.end >= chunk.start


def test_chunks_respect_the_size_budget(segments) -> None:
    chunks = chunk_segments(segments, chunk_chars=80, overlap_chars=0)
    assert len(chunks) > 1
    for chunk in chunks:
        # A chunk may exceed the budget only when a single segment does.
        assert len(chunk.text) <= 80 or " " not in chunk.text


def test_overlap_repeats_trailing_text(segments) -> None:
    without = chunk_segments(segments, chunk_chars=90, overlap_chars=0)
    with_overlap = chunk_segments(segments, chunk_chars=90, overlap_chars=45)
    assert len(with_overlap) >= len(without)
    # The second chunk should begin earlier in the video than it would with no
    # overlap, because it carries the tail of the first.
    assert with_overlap[1].start <= without[1].start


def test_indexes_are_sequential_from_zero(segments) -> None:
    chunks = chunk_segments(segments, chunk_chars=60, overlap_chars=10)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_single_segment_longer_than_budget_is_split_with_interpolated_times() -> None:
    long_text = " ".join(["word"] * 200)  # ~999 chars
    chunks = chunk_segments(
        [Segment(text=long_text, start=10.0, duration=20.0)],
        chunk_chars=100,
        overlap_chars=0,
    )
    assert len(chunks) > 1
    assert chunks[0].start == pytest.approx(10.0)
    # Times must advance across the pieces and stay inside the original cue.
    assert chunks[1].start > chunks[0].start
    assert chunks[-1].end <= 30.01


def test_empty_input_produces_no_chunks() -> None:
    assert chunk_segments([], chunk_chars=100) == []


def test_rejects_invalid_budget(segments) -> None:
    with pytest.raises(ValueError):
        chunk_segments(segments, chunk_chars=0)


def test_overlap_larger_than_budget_still_terminates(segments) -> None:
    """Clamping overlap is what stops packing from looping forever."""
    chunks = chunk_segments(segments, chunk_chars=50, overlap_chars=500)
    assert chunks
    assert len(chunks) < 100
