"""RAG orchestration: rewrite, retrieve, answer, resolve citations."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from .chunking import Chunk
from .config import Settings
from .embeddings import Embedder
from .errors import ProviderError
from .index import Retrieved, VideoIndex
from .llm.base import ChatMessage, LLMProvider
from .prompts import (
    ANSWER_SYSTEM,
    CONDENSE_SYSTEM,
    build_answer_prompt,
    build_condense_prompt,
)
from .video import format_timestamp, watch_url

logger = logging.getLogger(__name__)

NO_ANSWER = "I couldn't find the answer to that in this video."

# Matches [3] and [1, 2] alike.
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True, slots=True)
class Citation:
    """A transcript excerpt the answer actually cited."""

    marker: int
    chunk: Chunk
    url: str

    @property
    def label(self) -> str:
        return format_timestamp(self.chunk.start)

    @property
    def span(self) -> str:
        return f"{format_timestamp(self.chunk.start)}-{format_timestamp(self.chunk.end)}"


@dataclass
class Answer:
    """A generated answer plus everything needed to show its provenance."""

    text: str
    question: str
    standalone_question: str
    retrieved: list[Retrieved] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)

    @property
    def was_rewritten(self) -> bool:
        return self.standalone_question.strip() != self.question.strip()


class ChatEngine:
    """Answers questions about one indexed video."""

    def __init__(
        self,
        index: VideoIndex,
        embedder: Embedder,
        provider: LLMProvider,
        settings: Settings | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.provider = provider
        self.settings = settings or Settings()

    # -- steps ------------------------------------------------------------

    def condense(self, question: str, history: list[tuple[str, str]]) -> str:
        """Rewrite a follow-up into a standalone question.

        This is what makes "what about the second one?" work: the retriever only
        ever sees a self-contained query, so it can embed something meaningful.
        A failure here is never fatal - we fall back to the raw question, since a
        slightly worse retrieval beats no answer at all.
        """
        if not history or not self.settings.condense_question:
            return question

        recent = history[-self.settings.max_history_turns :]
        try:
            rewritten = self.provider.complete(
                CONDENSE_SYSTEM,
                [ChatMessage(role="user", content=build_condense_prompt(question, recent))],
                max_tokens=200,
            ).strip()
        except ProviderError as exc:
            logger.info("question rewriting failed, using the original: %s", exc)
            return question

        # Guard against a chatty or empty rewrite.
        if not rewritten or len(rewritten) > 4 * max(len(question), 80):
            return question
        return rewritten.strip().strip('"')

    def retrieve(self, query: str) -> list[Retrieved]:
        """Find the most relevant excerpts for a query."""
        vector = self.embedder.encode([query])
        if vector.size == 0:
            return []
        return self.index.search(
            vector[0],
            k=self.settings.top_k,
            fetch_k=max(self.settings.top_k * 3, 15),
            lambda_mult=0.7,
        )

    # -- answering --------------------------------------------------------

    def answer(self, question: str, history: list[tuple[str, str]] | None = None) -> Answer:
        """Answer a question, waiting for the whole response."""
        stream = self.stream(question, history)
        for _ in stream:
            pass
        return stream.result

    def stream(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
    ) -> AnswerStream:
        """Answer a question, yielding text as it arrives.

        The returned object is iterable for display and exposes ``.result`` with
        the resolved citations once iteration finishes.
        """
        history = history or []
        standalone = self.condense(question, history)
        retrieved = self.retrieve(standalone)
        return AnswerStream(self, question, standalone, retrieved)


class AnswerStream:
    """Iterable answer text, with a resolved ``Answer`` available afterwards."""

    def __init__(
        self,
        engine: ChatEngine,
        question: str,
        standalone_question: str,
        retrieved: list[Retrieved],
    ) -> None:
        self._engine = engine
        self._question = question
        self._standalone = standalone_question
        self._retrieved = retrieved
        self._pieces: list[str] = []
        self._result: Answer | None = None

    def __iter__(self) -> Iterator[str]:
        # Nothing retrieved means nothing to ground an answer in; say so rather
        # than asking the model to speculate.
        if not self._retrieved:
            self._pieces.append(NO_ANSWER)
            self._finalise()
            yield NO_ANSWER
            return

        prompt = build_answer_prompt(self._standalone, self._retrieved)
        try:
            for piece in self._engine.provider.stream(
                ANSWER_SYSTEM,
                [ChatMessage(role="user", content=prompt)],
            ):
                self._pieces.append(piece)
                yield piece
        finally:
            self._finalise()

    @property
    def result(self) -> Answer:
        """The finished answer. Available once iteration has completed."""
        if self._result is None:
            self._finalise()
        assert self._result is not None
        return self._result

    def _finalise(self) -> None:
        text = "".join(self._pieces).strip()
        self._result = Answer(
            text=text,
            question=self._question,
            standalone_question=self._standalone,
            retrieved=self._retrieved,
            citations=resolve_citations(text, self._retrieved, self._engine.index.video_id),
        )


def resolve_citations(text: str, retrieved: list[Retrieved], video_id: str) -> list[Citation]:
    """Map the ``[n]`` markers in an answer to the excerpts they refer to.

    Only markers that actually appear in the text are returned, and only those
    within range - a model that hallucinates ``[9]`` against five excerpts should
    not produce a broken link. Results are ordered by position in the video,
    which reads more naturally than citation order.
    """
    markers: set[int] = set()
    for match in _CITATION_RE.finditer(text):
        for part in match.group(1).split(","):
            try:
                markers.add(int(part.strip()))
            except ValueError:
                continue

    citations = [
        Citation(
            marker=marker,
            chunk=retrieved[marker - 1].chunk,
            url=watch_url(video_id, retrieved[marker - 1].chunk.start),
        )
        for marker in sorted(markers)
        if 1 <= marker <= len(retrieved)
    ]
    citations.sort(key=lambda citation: citation.chunk.start)
    return citations
