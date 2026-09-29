"""Answer generation via the Claude API. This is the default provider."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from ..errors import ProviderError, ProviderNotConfigured
from .base import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)

# Server-side refusal fallback: if a safety classifier declines the request, the
# API retries it on another model inside the same call instead of just stopping.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(LLMProvider):
    """Streams answers from Claude.

    Effort is set to ``low`` deliberately. Answering a question from supplied
    transcript context is a reading-comprehension task, not a reasoning problem -
    higher effort spends tokens and latency without improving the answer.
    """

    id = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        model: str = "claude-opus-5-5",
        effort: str = "low",
        use_refusal_fallback: bool = True,
    ) -> None:
        if not api_key:
            raise ProviderNotConfigured(
                "ANTHROPIC_API_KEY is not set. Add it to .env locally, or to "
                "Settings -> Secrets on Streamlit Cloud."
            )
        self._api_key = api_key
        self._model = model
        self._effort = effort
        self._use_refusal_fallback = use_refusal_fallback
        self._client = None

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def stream(
        self,
        system: str,
        messages: list[ChatMessage],
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        import anthropic

        client = self._get_client()
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "output_config": {"effort": self._effort},
        }

        try:
            manager = self._open(client, payload)
        except anthropic.AuthenticationError as exc:
            raise ProviderNotConfigured("Claude rejected the API key.") from exc
        except anthropic.NotFoundError as exc:
            raise ProviderError(f"Model {self._model!r} is not available to this key.") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError("Rate limited by the Claude API. Try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Claude API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("Could not reach the Claude API. Check your connection.") from exc

        try:
            with manager as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except anthropic.APIError as exc:
            raise ProviderError(f"Claude stopped mid-answer: {exc}") from exc

        if final.stop_reason == "refusal":
            detail = getattr(final.stop_details, "explanation", None)
            raise ProviderError(
                "Claude declined to answer this question"
                + (f": {detail}" if detail else ".")
            )

    def _open(self, client, payload: dict):
        """Open a stream, preferring the refusal-fallback beta when available.

        Not every account has the beta enabled, and sending an unrecognised beta
        flag fails the whole request - so a rejection downgrades to the plain
        endpoint once and stays downgraded for this provider instance.
        """
        import anthropic

        if self._use_refusal_fallback:
            try:
                return client.beta.messages.stream(
                    **payload,
                    betas=[FALLBACK_BETA],
                    fallbacks="default",
                )
            except anthropic.BadRequestError as exc:
                logger.info("refusal fallback unavailable, continuing without it: %s", exc)
                self._use_refusal_fallback = False

        return client.messages.stream(**payload)
