"""The provider interface every answer backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn of conversation."""

    role: Role
    content: str


class LLMProvider(ABC):
    """Generates an answer from a system prompt and a conversation.

    Streaming is the primitive rather than the extra: a chat UI wants tokens as
    they arrive, and a non-streaming ``complete`` is trivially derived from a
    stream while the reverse is not.
    """

    #: Short name used in the UI and in settings, e.g. ``"anthropic"``.
    id: str = "base"

    @property
    @abstractmethod
    def model(self) -> str:
        """The specific model this provider will call."""

    @abstractmethod
    def stream(
        self,
        system: str,
        messages: list[ChatMessage],
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """Yield the answer in fragments as they are generated."""

    def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        max_tokens: int = 2048,
    ) -> str:
        """Return the whole answer as one string."""
        return "".join(self.stream(system, messages, max_tokens))

    def describe(self) -> str:
        return f"{self.id}:{self.model}"
