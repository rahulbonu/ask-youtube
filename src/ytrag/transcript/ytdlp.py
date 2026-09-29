"""Transcript source backed by yt-dlp.

Used when ``youtube-transcript-api`` is blocked. yt-dlp negotiates with YouTube
differently and often still gets through; it also exposes the ``json3`` caption
format, which carries cleaner timing data than WebVTT.
"""

from __future__ import annotations

import requests

from .models import Transcript
from .subtitles import parse_json3, parse_vtt

SOURCE = "yt-dlp"

# json3 first: millisecond timings, no positioning markup to strip.
_FORMAT_PRIORITY = ("json3", "vtt", "srv1")
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


def fetch(video_id: str, languages: list[str], proxy: str | None = None) -> Transcript:
    """Fetch captions for ``video_id`` without downloading the video."""
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("yt-dlp is not installed") from exc

    options: dict[str, object] = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if proxy:
        options["proxy"] = proxy

    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

    manual = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}

    language, track = _select_track(manual, automatic, languages)
    if track is None:
        raise RuntimeError("yt-dlp found no caption tracks")

    response = requests.get(
        track["url"],
        timeout=30,
        headers={"User-Agent": _USER_AGENT},
        proxies={"http": proxy, "https": proxy} if proxy else None,
    )
    response.raise_for_status()

    extension = track.get("ext")
    if extension == "json3":
        segments = parse_json3(response.text)
    else:
        segments = parse_vtt(response.text)

    return Transcript.from_items(video_id, language, SOURCE, segments)


def _select_track(
    manual: dict[str, list[dict]],
    automatic: dict[str, list[dict]],
    languages: list[str],
) -> tuple[str, dict | None]:
    """Pick the best available track.

    Preference order: requested language before any other, human-written
    captions before auto-generated ones, and ``json3`` before ``vtt``.
    """
    wanted = list(dict.fromkeys([*(languages or []), "en"]))

    def candidates() -> list[tuple[str, dict[str, list[dict]]]]:
        pairs: list[tuple[str, dict[str, list[dict]]]] = []
        for code in wanted:
            pairs.append((code, manual))
        for code in wanted:
            pairs.append((code, automatic))
        # Fall back to every language the video offers, manual first.
        for source in (manual, automatic):
            for code in source:
                pairs.append((code, source))
        return pairs

    for code, source in candidates():
        entries = source.get(code)
        if not entries:
            continue
        for extension in _FORMAT_PRIORITY:
            for entry in entries:
                if entry.get("ext") == extension and entry.get("url"):
                    return code, entry
        # Unknown format, but a URL is better than nothing - treat it as VTT.
        for entry in entries:
            if entry.get("url"):
                return code, entry

    return "", None
