from __future__ import annotations

from ytrag.chunking import Chunk
from ytrag.engine import Citation
from ytrag.render import linkify_citations, sources_markdown


def make_citation(marker: int, start: float = 30.0) -> Citation:
    return Citation(
        marker=marker,
        chunk=Chunk(index=marker - 1, text=f"excerpt {marker}", start=start, end=start + 5),
        url=f"https://www.youtube.com/watch?v=vid00000001&t={int(start)}s",
    )


def test_linkifies_a_single_marker() -> None:
    citations = [make_citation(1)]
    result = linkify_citations("Plants use sunlight [1].", citations)
    assert result == (
        "Plants use sunlight [[1]](https://www.youtube.com/watch?v=vid00000001&t=30s)."
    )


def test_linkifies_grouped_markers_separately() -> None:
    citations = [make_citation(1, 10), make_citation(2, 20)]
    result = linkify_citations("Both [1, 2] agree.", citations)
    assert "[[1]](https://www.youtube.com/watch?v=vid00000001&t=10s)" in result
    assert "[[2]](https://www.youtube.com/watch?v=vid00000001&t=20s)" in result
    assert "[1, 2]" not in result


def test_unknown_marker_stays_plain_text() -> None:
    assert linkify_citations("See [7].", [make_citation(1)]) == "See [7]."


def test_non_numeric_marker_is_untouched() -> None:
    assert linkify_citations("An [aside] here.", [make_citation(1)]) == "An [aside] here."


def test_text_without_markers_is_unchanged() -> None:
    assert linkify_citations("No citations.", [make_citation(1)]) == "No citations."


def test_sources_markdown_lists_each_citation() -> None:
    rendered = sources_markdown([make_citation(1, 65), make_citation(2, 130)])
    assert "**[1]**" in rendered
    assert "1:05" in rendered
    assert "> excerpt 1" in rendered


def test_sources_markdown_handles_no_citations() -> None:
    assert "cited no specific moment" in sources_markdown([])
