from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider settings shared by the agents and optional benchmark judge.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize common spellings and reject unsupported providers."""
    provider = value.strip().lower().replace("-", "_")
    aliases = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google_genai": "gemini",
        "open_router": "openrouter",
        "local": "ollama",
    }
    provider = aliases.get(provider, provider)
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if provider not in supported:
        raise ValueError(f"Unsupported provider {value!r}; choose one of {', '.join(sorted(supported))}.")
    return provider


def build_chat_model(config: ProviderConfig):
    """Create a LangChain chat model, importing only the selected integration."""
    provider = normalize_provider(config.provider)
    options = {"model": config.model_name, "temperature": config.temperature}
    if config.api_key:
        options["api_key"] = config.api_key

    if provider in {"openai", "custom"}:
        from langchain_openai import ChatOpenAI

        if provider == "custom":
            if not config.base_url:
                raise ValueError("CUSTOM_BASE_URL is required for the custom provider.")
            options["base_url"] = config.base_url
        return ChatOpenAI(**options)
    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError:
            from langchain_openai import ChatOpenAI

            options["base_url"] = config.base_url or "https://openrouter.ai/api/v1"
            return ChatOpenAI(**options)
        if config.base_url:
            options["base_url"] = config.base_url
        return ChatOpenRouter(**options)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(**options)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**options)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        options.pop("api_key", None)
        if config.base_url:
            options["base_url"] = config.base_url
        return ChatOllama(**options)
    raise AssertionError(f"Unhandled provider: {provider}")
