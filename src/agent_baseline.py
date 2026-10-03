from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model, normalize_provider


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


def _is_memory_question(message: str) -> bool:
    text = message.lower()
    return any(
        cue in text
        for cue in (
            "nhắc lại",
            "mình tên gì",
            "tên mình là gì",
            "hiện tại mình làm",
            "mình làm nghề gì",
            "nghề hiện tại",
            "nghề cũ",
            "nghề mới",
            "nghề nghiệp hiện tại",
            "đang ở đâu",
            "nơi ở hiện tại",
            "style trả lời",
            "kiểu trả lời",
            "đồ uống yêu thích",
            "món ăn yêu thích",
            "mình nuôi con gì",
            "tóm tắt ngắn về mình",
            "bạn có biết",
            "mối quan tâm",
            "nhắc ngắn",
        )
    )


def _answer_from_facts(facts: dict[str, str]) -> str:
    labels = {
        "name": "tên",
        "current_location": "nơi ở hiện tại",
        "profession": "nghề nghiệp hiện tại",
        "favorite_drink": "đồ uống yêu thích",
        "favorite_food": "món ăn yêu thích",
        "pet": "thú cưng",
        "response_style": "style trả lời",
        "interests": "mối quan tâm",
    }
    entries = [f"{labels[key]}: {facts[key]}" for key in labels if facts.get(key)]
    if not entries:
        return "Mình chưa có thông tin đó trong cuộc trò chuyện này."
    return "Mình nhớ: " + "; ".join(entries) + "."


def _response_text(result: Any) -> str:
    content = getattr(result, "content", result)
    return content if isinstance(content, str) else str(content)


class BaselineAgent:
    """A fair baseline with raw within-thread history and no durable profile."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt = [("system", "You are a helpful assistant. Use only this thread's conversation history.")]
        prompt.extend((item["role"], item["content"]) for item in session.messages)
        session.prompt_tokens_processed += estimate_tokens(" ".join(content for _, content in prompt))
        response = _response_text(self.langchain_agent.invoke(prompt))
        session.messages.append({"role": "assistant", "content": response})
        output_tokens = estimate_tokens(response)
        session.token_usage += output_tokens
        return {
            "response": response,
            "agent_tokens_only": output_tokens,
            "prompt_tokens_processed": session.prompt_tokens_processed,
        }

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        context = " ".join(item["content"] for item in session.messages)
        prompt_tokens = estimate_tokens(
            "You are a helpful assistant. Use only this thread's conversation history. " + context
        )
        session.prompt_tokens_processed += prompt_tokens

        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        if _is_memory_question(message):
            response = _answer_from_facts(facts)
        else:
            response = "Mình đã ghi nhận thông tin trong cuộc trò chuyện này."

        session.messages.append({"role": "assistant", "content": response})
        output_tokens = estimate_tokens(response)
        session.token_usage += output_tokens
        return {
            "response": response,
            "agent_tokens_only": output_tokens,
            "prompt_tokens_processed": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        if self.force_offline:
            return None
        config = self.config.model
        provider = normalize_provider(config.provider)
        configured = (
            provider == "ollama"
            or (provider == "custom" and bool(config.base_url))
            or bool(config.api_key)
        )
        if not configured:
            return None
        try:
            return build_chat_model(config)
        except ImportError:
            return None
