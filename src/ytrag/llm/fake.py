"""A scripted provider, so the pipeline can be tested without a network call."""

from __future__ import annotations

from collections.abc import Iterator

from .base import ChatMessage, LLMProvider


class FakeProvider(LLMProvider):
    """Returns canned replies and records what it was asked."""

    id = "fake"

    def __init__(self, replies: list[str] | None = None) -> None:
        self._replies = list(replies or ["I couldn't find the answer to that in the video."])
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    @property
    def model(self) -> str:
        return "fake-1"

    def stream(
        self,
        system: str,
        messages: list[ChatMessage],
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        self.calls.append((system, list(messages)))
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        # Yield in pieces so callers exercise their streaming path.
        for word in reply.split(" "):
            yield word + " "
