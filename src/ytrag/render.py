"""Turning answers into display markdown.

Kept out of the Streamlit module so it can be tested without a UI.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .engine import Citation

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def linkify_citations(text: str, citations: Sequence[Citation]) -> str:
    """Turn ``[1]`` markers into markdown links that jump into the video.

    Grouped markers like ``[1, 2]`` become two separate links. A marker with no
    matching citation is left as plain text rather than becoming a broken link.
    """
    by_marker = {citation.marker: citation for citation in citations}

    def replace(match: re.Match[str]) -> str:
        rendered: list[str] = []
        for part in match.group(1).split(","):
            try:
                marker = int(part.strip())
            except ValueError:
                return match.group(0)
            citation = by_marker.get(marker)
            rendered.append(f"[[{marker}]]({citation.url})" if citation else f"[{marker}]")
        return "".join(rendered)

    return _CITATION_RE.sub(replace, text)


def sources_markdown(citations: Sequence[Citation]) -> str:
    """Render the cited excerpts as a readable list."""
    if not citations:
        return "_This answer cited no specific moment in the video._"
    lines = []
    for citation in citations:
        lines.append(f"**[{citation.marker}]** [{citation.span}]({citation.url})")
        lines.append(f"> {citation.chunk.text}")
        lines.append("")
    return "\n".join(lines)
