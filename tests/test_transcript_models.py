from __future__ import annotations

import pytest

from ytrag.transcript.models import Segment, Transcript, normalise


def test_normalise_collapses_whitespace() -> None:
    assert normalise("  hello \n\t world  ") == "hello world"
    assert normalise(None) == ""


def test_from_items_accepts_dicts() -> None:
    transcript = Transcript.from_items(
        "vid00000001",
        "en",
        "test",
        [{"text": "hello", "start": 1.0, "duration": 2.0}],
    )
    assert transcript.segments[0].text == "hello"
    assert transcript.segments[0].end == 3.0


def test_from_items_accepts_objects_with_attributes() -> None:
    class Snippet:
        text, start, duration = "spoken words", 5.0, 1.0

    transcript = Transcript.from_items("vid00000001", "en", "test", [Snippet()])
    assert transcript.text == "spoken words"


def test_from_items_drops_blank_cues_and_sorts_by_time() -> None:
    transcript = Transcript.from_items(
        "vid00000001",
        "en",
        "test",
        [
            {"text": "second", "start": 5.0, "duration": 1.0},
            {"text": "   ", "start": 2.0, "duration": 1.0},
            {"text": "first", "start": 1.0, "duration": 1.0},
        ],
    )
    assert [s.text for s in transcript.segments] == ["first", "second"]


def test_from_items_rejects_an_empty_transcript() -> None:
    with pytest.raises(ValueError):
        Transcript.from_items("vid00000001", "en", "test", [])


def test_duration_is_the_latest_end(transcript) -> None:
    assert transcript.duration == pytest.approx(15.0)


def test_fingerprint_tracks_content(transcript) -> None:
    same = Transcript(
        video_id=transcript.video_id,
        language="en",
        source="different-source",
        segments=transcript.segments,
    )
    # Source and language are presentation, not content.
    assert same.fingerprint() == transcript.fingerprint()

    changed = Transcript(
        video_id=transcript.video_id,
        language="en",
        source="test",
        segments=(*transcript.segments, Segment(text="extra", start=99.0, duration=1.0)),
    )
    assert changed.fingerprint() != transcript.fingerprint()
