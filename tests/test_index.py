from __future__ import annotations

import numpy as np
import pytest

from ytrag.index import VideoIndex, build_index, cache_key, load_or_build_index
from ytrag.transcript.models import Segment, Transcript


def test_search_finds_the_relevant_chunk(transcript, embedder) -> None:
    index = build_index(transcript, embedder, chunk_chars=60, overlap_chars=0)
    hits = index.search(embedder.encode(["volcanoes magma crust"])[0], k=1)
    assert hits
    assert "magma" in hits[0].chunk.text.lower() or "volcano" in hits[0].chunk.text.lower()


def test_search_returns_at_most_k(transcript, embedder) -> None:
    index = build_index(transcript, embedder, chunk_chars=50, overlap_chars=0)
    assert len(index.search(embedder.encode(["light"])[0], k=2)) <= 2


def test_search_on_empty_query_or_zero_k(transcript, embedder) -> None:
    index = build_index(transcript, embedder, chunk_chars=60, overlap_chars=0)
    assert index.search(embedder.encode(["light"])[0], k=0) == []


def test_mmr_avoids_returning_duplicates(embedder) -> None:
    """With five near-identical chunks, diversity-weighted search still varies."""
    duplicated = Transcript(
        video_id="vid00000001",
        language="en",
        source="test",
        segments=tuple(
            Segment(text="photosynthesis uses sunlight", start=float(i * 3), duration=3.0)
            for i in range(5)
        ),
    )
    index = build_index(duplicated, embedder, chunk_chars=40, overlap_chars=0)
    hits = index.search(embedder.encode(["photosynthesis"])[0], k=3, lambda_mult=0.5)
    starts = [hit.chunk.start for hit in hits]
    assert len(set(starts)) == len(starts)


def test_rejects_mismatched_chunks_and_vectors(transcript, embedder) -> None:
    index = build_index(transcript, embedder, chunk_chars=60, overlap_chars=0)
    with pytest.raises(ValueError):
        VideoIndex(
            video_id="x",
            language="en",
            source="t",
            embedder_id="e",
            chunks=index.chunks,
            vectors=np.zeros((len(index.chunks) + 1, 4), dtype=np.float32),
        )


def test_save_and_load_round_trip(tmp_path, transcript, embedder) -> None:
    index = build_index(transcript, embedder, chunk_chars=60, overlap_chars=0)
    destination = tmp_path / "idx"
    index.save(destination)
    restored = VideoIndex.load(destination)

    assert restored.video_id == index.video_id
    assert restored.language == index.language
    assert restored.embedder_id == index.embedder_id
    assert [c.to_dict() for c in restored.chunks] == [c.to_dict() for c in index.chunks]
    np.testing.assert_allclose(restored.vectors, index.vectors)


def test_second_build_is_served_from_cache(tmp_path, transcript, embedder) -> None:
    first, cached = load_or_build_index(transcript, embedder, tmp_path, 60, 0)
    assert cached is False

    second, cached = load_or_build_index(transcript, embedder, tmp_path, 60, 0)
    assert cached is True
    assert [c.to_dict() for c in second.chunks] == [c.to_dict() for c in first.chunks]


def test_cache_can_be_disabled(tmp_path, transcript, embedder) -> None:
    _, cached = load_or_build_index(transcript, embedder, None, 60, 0)
    assert cached is False
    assert not list(tmp_path.iterdir())


def test_corrupt_cache_entry_is_rebuilt(tmp_path, transcript, embedder) -> None:
    load_or_build_index(transcript, embedder, tmp_path, 60, 0)
    entry = next(tmp_path.iterdir())
    (entry / "meta.json").write_text("{ not json", encoding="utf-8")

    index, cached = load_or_build_index(transcript, embedder, tmp_path, 60, 0)
    assert cached is False
    assert len(index) > 0


@pytest.mark.parametrize(
    "change",
    ["transcript", "embedder", "chunk_chars", "overlap"],
)
def test_cache_key_changes_when_inputs_change(transcript, embedder, change) -> None:
    base = cache_key(transcript, embedder.id, 1000, 200)

    if change == "transcript":
        other = Transcript(
            video_id=transcript.video_id,
            language="en",
            source="test",
            segments=(Segment(text="entirely different words", start=0.0, duration=1.0),),
        )
        assert cache_key(other, embedder.id, 1000, 200) != base
    elif change == "embedder":
        assert cache_key(transcript, "other-model", 1000, 200) != base
    elif change == "chunk_chars":
        assert cache_key(transcript, embedder.id, 500, 200) != base
    else:
        assert cache_key(transcript, embedder.id, 1000, 100) != base


def test_cache_key_is_stable_for_identical_input(transcript, embedder) -> None:
    assert cache_key(transcript, embedder.id, 1000, 200) == cache_key(
        transcript, embedder.id, 1000, 200
    )
