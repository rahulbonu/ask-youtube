from __future__ import annotations

import pytest

from ytrag.errors import InvalidVideoInput
from ytrag.video import extract_video_id, format_timestamp, watch_url

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "value",
    [
        VIDEO_ID,
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"http://youtube.com/watch?v={VIDEO_ID}",
        f"www.youtube.com/watch?v={VIDEO_ID}",
        f"youtube.com/watch?v={VIDEO_ID}",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"https://music.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?si=abcdefg",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}",
        f"https://www.youtube.com/v/{VIDEO_ID}",
        f"https://www.youtube-nocookie.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/watch?v={VIDEO_ID}&t=42s&list=PLxyz",
        f"https://www.youtube.com/watch?app=desktop&v={VIDEO_ID}",
        f"  https://www.youtube.com/watch?v={VIDEO_ID}  ",
    ],
)
def test_accepts_every_common_form(value: str) -> None:
    assert extract_video_id(value) == VIDEO_ID


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "not a url",
        "https://vimeo.com/123456789",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=tooshort",
        "https://www.youtube.com/results?search_query=cats",
    ],
)
def test_rejects_non_youtube_input(value: str) -> None:
    with pytest.raises(InvalidVideoInput):
        extract_video_id(value)


def test_rejects_lookalike_host() -> None:
    """A host that merely contains "youtube.com" is not YouTube."""
    with pytest.raises(InvalidVideoInput):
        extract_video_id("https://notyoutube.com/watch?v=dQw4w9WgXcQ")


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00"), (5, "0:05"), (65, "1:05"), (600, "10:00"), (3661, "1:01:01"), (-5, "0:00")],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_watch_url_deep_links_to_the_second() -> None:
    assert watch_url(VIDEO_ID) == f"https://www.youtube.com/watch?v={VIDEO_ID}"
    assert watch_url(VIDEO_ID, 90.7) == f"https://www.youtube.com/watch?v={VIDEO_ID}&t=90s"
