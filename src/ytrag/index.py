"""Vector index over one video's chunks, with an on-disk cache.

Search is exact brute-force cosine similarity in numpy rather than an ANN
library. For this workload that is the right call: even a ten-hour video is only
a few hundred chunks, where a dot product costs microseconds, and an approximate
index would add a heavyweight native dependency to save nothing measurable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_transcript
from .embeddings import Embedder
from .transcript.models import Transcript

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Retrieved:
    """A chunk that matched a query, and how well it matched."""

    chunk: Chunk
    score: float


class VideoIndex:
    """Chunks plus their embeddings, searchable by cosine similarity."""

    def __init__(
        self,
        video_id: str,
        language: str,
        source: str,
        embedder_id: str,
        chunks: list[Chunk],
        vectors: np.ndarray,
    ) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} chunks, {vectors.shape[0]} vectors"
            )
        self.video_id = video_id
        self.language = language
        self.source = source
        self.embedder_id = embedder_id
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
    ) -> list[Retrieved]:
        """Return up to ``k`` chunks, balancing relevance against redundancy.

        Uses maximal marginal relevance: it pulls the ``fetch_k`` closest chunks,
        then greedily picks ones that are similar to the query but dissimilar to
        what it has already picked. Without this, overlapping neighbouring chunks
        tend to fill every slot and the answer sees the same sentence five times.

        ``lambda_mult`` of 1.0 is pure relevance; 0.0 is pure diversity.
        """
        if len(self.chunks) == 0 or k <= 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        similarities = self.vectors @ query

        fetch_k = max(k, min(fetch_k, len(self.chunks)))
        candidates = list(np.argsort(-similarities)[:fetch_k])

        selected: list[int] = []
        while candidates and len(selected) < k:
            if not selected:
                best = max(candidates, key=lambda i: similarities[i])
            else:
                chosen = self.vectors[selected]
                best, best_score = None, -np.inf
                for position in candidates:
                    redundancy = float(np.max(chosen @ self.vectors[position]))
                    score = lambda_mult * float(similarities[position]) - (
                        1.0 - lambda_mult
                    ) * redundancy
                    if score > best_score:
                        best, best_score = position, score
                assert best is not None
            selected.append(best)
            candidates.remove(best)

        return [
            Retrieved(chunk=self.chunks[position], score=float(similarities[position]))
            for position in selected
        ]

    # -- persistence ------------------------------------------------------

    def save(self, directory: Path) -> None:
        """Write the index to ``directory`` atomically."""
        staging = directory.with_name(directory.name + ".tmp")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        (staging / "meta.json").write_text(
            json.dumps(
                {
                    "format": CACHE_FORMAT_VERSION,
                    "video_id": self.video_id,
                    "language": self.language,
                    "source": self.source,
                    "embedder_id": self.embedder_id,
                    "chunks": [chunk.to_dict() for chunk in self.chunks],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        np.save(staging / "vectors.npy", self.vectors)

        if directory.exists():
            shutil.rmtree(directory)
        staging.replace(directory)

    @classmethod
    def load(cls, directory: Path) -> VideoIndex:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format") != CACHE_FORMAT_VERSION:
            raise ValueError("cache written by a different version")
        vectors = np.load(directory / "vectors.npy")
        return cls(
            video_id=meta["video_id"],
            language=meta["language"],
            source=meta["source"],
            embedder_id=meta["embedder_id"],
            chunks=[Chunk.from_dict(item) for item in meta["chunks"]],
            vectors=vectors,
        )


def cache_key(
    transcript: Transcript,
    embedder_id: str,
    chunk_chars: int,
    overlap_chars: int,
) -> str:
    """Identify an index by everything that would change its contents.

    The transcript fingerprint is included so that a re-fetched transcript -
    auto-captions get revised, or a different language is selected - produces a
    different key rather than silently serving a stale index.
    """
    digest = hashlib.sha256(
        "|".join(
            [
                str(CACHE_FORMAT_VERSION),
                transcript.video_id,
                transcript.fingerprint(),
                embedder_id,
                str(chunk_chars),
                str(overlap_chars),
            ]
        ).encode()
    )
    return digest.hexdigest()[:16]


def build_index(
    transcript: Transcript,
    embedder: Embedder,
    chunk_chars: int = 1000,
    overlap_chars: int = 200,
) -> VideoIndex:
    """Chunk and embed a transcript. This is the expensive step."""
    chunks = chunk_transcript(transcript, chunk_chars, overlap_chars)
    if not chunks:
        raise ValueError("transcript produced no chunks")
    vectors = embedder.encode([chunk.text for chunk in chunks])
    return VideoIndex(
        video_id=transcript.video_id,
        language=transcript.language,
        source=transcript.source,
        embedder_id=embedder.id,
        chunks=chunks,
        vectors=vectors,
    )


def load_or_build_index(
    transcript: Transcript,
    embedder: Embedder,
    cache_dir: Path | None,
    chunk_chars: int = 1000,
    overlap_chars: int = 200,
) -> tuple[VideoIndex, bool]:
    """Return an index for ``transcript``, reusing a cached one when possible.

    Returns:
        The index, and whether it came from the cache.
    """
    if cache_dir is None:
        return build_index(transcript, embedder, chunk_chars, overlap_chars), False

    key = cache_key(transcript, embedder.id, chunk_chars, overlap_chars)
    directory = Path(cache_dir) / f"{transcript.video_id}-{key}"

    if directory.is_dir():
        try:
            index = VideoIndex.load(directory)
            logger.info("index cache hit for %s (%d chunks)", transcript.video_id, len(index))
            return index, True
        except Exception as exc:
            # A corrupt or stale cache entry must never break the app.
            logger.warning("ignoring unreadable cache at %s: %s", directory, exc)
            shutil.rmtree(directory, ignore_errors=True)

    index = build_index(transcript, embedder, chunk_chars, overlap_chars)
    try:
        index.save(directory)
    except Exception as exc:  # pragma: no cover - disk problems are environmental
        logger.warning("could not write index cache to %s: %s", directory, exc)
    return index, False
