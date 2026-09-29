"""Text embedding.

Embeddings run locally through ``sentence-transformers``. That keeps the app
usable with only one API key (for answers) and avoids one network round trip per
chunk, which is what made the hosted-embedding approach slow and rate-limited.

The ``Embedder`` protocol exists so tests can swap in a deterministic stub and
so a hosted embedding service can be dropped in without touching the pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Turns text into L2-normalised float32 vectors."""

    @property
    def id(self) -> str:
        """Stable identifier, used as part of the index cache key."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` array of unit vectors."""


class SentenceTransformerEmbedder:
    """Local embeddings via ``sentence-transformers``.

    The model is loaded on first use rather than in ``__init__`` so that
    constructing this object stays cheap - importing torch and pulling a few
    hundred megabytes of weights should not happen until someone asks for a
    vector.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    @property
    def id(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
