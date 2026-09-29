"""YouTube identifier parsing and timestamp formatting."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from .errors import InvalidVideoInput

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Hosts we accept. Anything else is rejected rather than regex-scraped, so a
# random string containing "v=..." cannot masquerade as a YouTube link.
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}
_SHORT_HOSTS = {"youtu.be", "www.youtu.be"}

# Path prefixes that carry the ID as the segment straight after them.
_PATH_PREFIXES = ("embed", "shorts", "live", "v")


def extract_video_id(value: str) -> str:
    """Return the 11-character video ID for a bare ID or a YouTube URL.

    Raises:
        InvalidVideoInput: if the value is neither a video ID nor a YouTube URL.
    """
    text = (value or "").strip()
    if not text:
        raise InvalidVideoInput("Enter a YouTube video ID or URL.")

    if VIDEO_ID_RE.match(text):
        return text

    # urlparse needs a scheme to populate netloc rather than path.
    candidate = text if "//" in text else f"https://{text}"
    parsed = urlparse(candidate)
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]

    if host in _SHORT_HOSTS:
        found = _first_id(parsed.path.strip("/").split("/"))
        if found:
            return found

    if host in _YOUTUBE_HOSTS:
        query_id = parse_qs(parsed.query).get("v", [""])[0].strip()
        if VIDEO_ID_RE.match(query_id):
            return query_id

        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if parts and parts[0].lower() in _PATH_PREFIXES:
            found = _first_id(parts[1:])
            if found:
                return found
        # /watch/<id> and bare /<id> both appear in the wild.
        found = _first_id(parts[1:] if parts[:1] == ["watch"] else parts)
        if found:
            return found

    raise InvalidVideoInput(
        f"Could not find a YouTube video ID in {text!r}. "
        "Paste a full YouTube URL or the 11-character video ID."
    )


def _first_id(parts: list[str]) -> str | None:
    for part in parts:
        if VIDEO_ID_RE.match(part):
            return part
    return None


def watch_url(video_id: str, start: float | None = None) -> str:
    """Build a watch URL, optionally deep-linked to ``start`` seconds."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    if start is None:
        return url
    return f"{url}&t={max(0, int(start))}s"


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``m:ss``, or ``h:mm:ss`` once past an hour."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
