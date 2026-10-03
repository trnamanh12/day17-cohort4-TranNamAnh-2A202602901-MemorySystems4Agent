from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared paths, memory limits, and model provider settings."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    def path_setting(name: str, default: Path) -> Path:
        value = os.getenv(name)
        path = Path(value).expanduser() if value else default
        return (path if path.is_absolute() else root / path).resolve()

    def integer_setting(name: str, default: int, minimum: int = 0) -> int:
        try:
            return max(minimum, int(os.getenv(name, str(default))))
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer.") from exc

    defaults = {
        "openai": "gpt-4o-mini",
        "custom": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash",
        "anthropic": "claude-3-5-haiku-latest",
        "ollama": "llama3.1",
        "openrouter": "openai/gpt-4o-mini",
    }
    credential_vars = {
        "openai": "OPENAI_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": "",
        "openrouter": "OPENROUTER_API_KEY",
    }
    base_url_vars = {
        "custom": "CUSTOM_BASE_URL",
        "ollama": "OLLAMA_BASE_URL",
        "openrouter": "OPENROUTER_BASE_URL",
    }

    def provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
        raw_provider = os.getenv(f"{prefix}LLM_PROVIDER")
        provider = normalize_provider(raw_provider or (fallback.provider if fallback else "openai"))
        model_name = os.getenv(f"{prefix}LLM_MODEL") or (
            fallback.model_name if fallback and not raw_provider else defaults[provider]
        )
        temperature = float(os.getenv(f"{prefix}LLM_TEMPERATURE", "0"))
        key_var = credential_vars[provider]
        api_key = os.getenv(f"{prefix}{key_var}") if key_var else None
        if prefix and not api_key and key_var:
            api_key = os.getenv(key_var)
        base_var = base_url_vars.get(provider)
        base_url = os.getenv(f"{prefix}{base_var}") if base_var else None
        if prefix and not base_url and base_var:
            base_url = os.getenv(base_var)
        return ProviderConfig(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    model = provider_config("")
    judge_model = provider_config("JUDGE_", fallback=model)
    data_dir = path_setting("DATA_DIR", root / "data")
    state_dir = path_setting("STATE_DIR", root / "state")
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=integer_setting("COMPACT_THRESHOLD_TOKENS", 1200, 1),
        compact_keep_messages=integer_setting("COMPACT_KEEP_MESSAGES", 4),
        model=model,
        judge_model=judge_model,
    )
