"""Answer generation via the Hugging Face inference API.

Kept as an alternative for people who would rather use an open-weights model.
Be aware that most instruct models on the serverless API are gated - you have to
accept the model's licence on huggingface.co before a token can call it.
"""

from __future__ import annotations

from collections.abc import Iterator

from ..errors import ProviderError, ProviderNotConfigured
from .base import ChatMessage, LLMProvider


class HuggingFaceProvider(LLMProvider):
    """Streams answers from a Hugging Face chat-completion endpoint."""

    id = "huggingface"

    def __init__(
        self,
        token: str | None,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> None:
        if not token:
            raise ProviderNotConfigured(
                "HF_TOKEN is not set. Create a token at "
                "huggingface.co/settings/tokens and add it to .env, or to "
                "Settings -> Secrets on Streamlit Cloud."
            )
        self._token = token
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._client = None

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(model=self._model, token=self._token, timeout=120)
        return self._client

    def stream(
        self,
        system: str,
        messages: list[ChatMessage],
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        client = self._get_client()
        payload = [
            {"role": "system", "content": system},
            *({"role": m.role, "content": m.content} for m in messages),
        ]

        try:
            chunks = client.chat_completion(
                messages=payload,
                max_tokens=max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                stream=True,
            )
            for chunk in chunks:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                piece = getattr(choices[0].delta, "content", None)
                if piece:
                    yield piece
        except Exception as exc:
            message = str(exc)
            if "gated" in message.lower() or "awaiting approval" in message.lower():
                raise ProviderError(
                    f"{self._model} is gated. Accept its licence at "
                    f"huggingface.co/{self._model} with the same account as your token, "
                    "or set YTRAG_HF_MODEL to an ungated model."
                ) from exc
            raise ProviderError(f"Hugging Face request failed: {message}") from exc
