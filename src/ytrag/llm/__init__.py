"""Answer providers and the factory that picks one."""

from __future__ import annotations

from ..config import Settings
from ..errors import ProviderNotConfigured
from .anthropic_provider import AnthropicProvider
from .base import ChatMessage, LLMProvider
from .fake import FakeProvider
from .huggingface_provider import HuggingFaceProvider

PROVIDERS = ("anthropic", "huggingface")

__all__ = [
    "AnthropicProvider",
    "ChatMessage",
    "FakeProvider",
    "HuggingFaceProvider",
    "LLMProvider",
    "PROVIDERS",
    "build_provider",
]


def build_provider(settings: Settings, name: str | None = None) -> LLMProvider:
    """Construct the configured provider.

    Raises:
        ProviderNotConfigured: if the provider is unknown, or its key is missing.
    """
    provider = (name or settings.llm_provider or "anthropic").lower()

    if provider == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )
    if provider == "huggingface":
        return HuggingFaceProvider(token=settings.hf_token, model=settings.hf_model)

    raise ProviderNotConfigured(
        f"Unknown provider {provider!r}. Set YTRAG_LLM_PROVIDER to one of: "
        + ", ".join(PROVIDERS)
    )
