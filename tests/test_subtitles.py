from __future__ import annotations

import json

import pytest

from ytrag.transcript.subtitles import parse_json3, parse_timestamp, parse_vtt

VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:04.000
Hello and welcome

00:00:04.000 --> 00:00:07.500 align:start position:0%
<c.colorE5E5E5>Today we discuss</c> <00:00:05.100>photosynthesis

00:00:07.500 --> 00:00:09.000
{\\an8}Plants use sunlight
"""


def test_parse_vtt_keeps_times_and_strips_markup() -> None:
    segments = parse_vtt(VTT)
    assert [s.text for s in segments] == [
        "Hello and welcome",
        "Today we discuss photosynthesis",
        "Plants use sunlight",
    ]
    assert segments[0].start == 1.0
    assert segments[0].duration == pytest.approx(3.0)
    assert segments[1].start == 4.0
    assert segments[1].duration == pytest.approx(3.5)


def test_parse_vtt_collapses_rolling_captions() -> None:
    """Auto-captions re-emit the previous cue with more words appended."""
    rolling = """WEBVTT

00:00:00.000 --> 00:00:02.000
the quick brown

00:00:02.000 --> 00:00:04.000
the quick brown fox jumps

00:00:04.000 --> 00:00:06.000
the quick brown fox jumps
"""
    segments = parse_vtt(rolling)
    assert [s.text for s in segments] == ["the quick brown", "fox jumps"]
    # The new words are timed from the cue that introduced them.
    assert segments[1].start == 2.0


def test_parse_vtt_ignores_cues_without_text() -> None:
    assert parse_vtt("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n\n") == []


def test_parse_vtt_handles_short_timestamps() -> None:
    segments = parse_vtt("WEBVTT\n\n01:02.500 --> 01:05.000\nhello\n")
    assert segments[0].start == pytest.approx(62.5)


def test_parse_json3() -> None:
    payload = json.dumps(
        {
            "events": [
                {
                    "tStartMs": 0,
                    "dDurationMs": 2000,
                    "segs": [{"utf8": "Hello "}, {"utf8": "there"}],
                },
                {"tStartMs": 2000, "dDurationMs": 1500, "segs": [{"utf8": "\n"}]},
                {"tStartMs": 3500, "dDurationMs": 2000, "segs": [{"utf8": "second line"}]},
                {"dDurationMs": 1000, "segs": [{"utf8": "no start time"}]},
            ]
        }
    )
    segments = parse_json3(payload)
    assert [s.text for s in segments] == ["Hello there", "second line"]
    assert segments[1].start == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("00:00:01.000", 1.0),
        ("00:01:30.500", 90.5),
        ("01:00:00.000", 3600.0),
        ("00:00:01,250", 1.25),
        ("02:30.000", 150.0),
    ],
)
def test_parse_timestamp(value: str, expected: float) -> None:
    assert parse_timestamp(value) == pytest.approx(expected)


def test_parse_timestamp_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        parse_timestamp("banana")
