"""Shared fixtures.

Everything here is offline. No test in this suite touches YouTube or an LLM API.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ytrag.transcript.models import Segment, Transcript


class HashingEmbedder:
    """A deterministic bag-of-words embedder.

    Real sentence embeddings would make retrieval assertions fuzzy and would pull
    torch into the test run. Hashing each word into a fixed-width vector keeps
    lexical overlap - which is all these tests assert on - while staying exact,
    instant, and dependency-free.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    @property
    def id(self) -> str:
        return f"hashing-{self.dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                digest = hashlib.md5(word.encode()).digest()
                matrix[row, int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def segments() -> list[Segment]:
    return [
        Segment(text="Today we are talking about photosynthesis", start=0.0, duration=3.0),
        Segment(text="Plants convert sunlight into chemical energy", start=3.0, duration=3.0),
        Segment(text="Chlorophyll absorbs red and blue light", start=6.0, duration=3.0),
        Segment(text="Now a completely different topic, volcanoes", start=9.0, duration=3.0),
        Segment(text="Magma rises through the crust and erupts", start=12.0, duration=3.0),
    ]


@pytest.fixture
def transcript(segments: list[Segment]) -> Transcript:
    return Transcript(
        video_id="dQw4w9WgXcQ",
        language="en",
        source="test",
        segments=tuple(segments),
    )
