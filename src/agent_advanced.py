from __future__ import annotations

from typing import Any

from agent_baseline import _answer_from_facts, _is_memory_question, _response_text
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model, normalize_provider


class AdvancedAgent:
    """Combine thread memory, durable profile facts, and compacted history."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)

        self._update_profile(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        response = _response_text(self.langchain_agent.invoke(self._live_prompt(user_id, thread_id)))
        self.compact_memory.append(thread_id, "assistant", response)
        output_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + output_tokens
        return {
            "response": response,
            "agent_tokens_only": output_tokens,
            "prompt_tokens_processed": prompt_tokens,
        }

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._update_profile(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)
        output_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + output_tokens
        return {
            "response": response,
            "agent_tokens_only": output_tokens,
            "prompt_tokens_processed": prompt_tokens,
        }

    def _update_profile(self, user_id: str, message: str) -> None:
        updates = extract_profile_updates(message)
        current = self.profile_store.facts(user_id)
        for key, value in updates.items():
            if key == "response_style":
                old_items = current.get(key, "").split(", ")
                new_items = value.split(", ")
                for item in new_items:
                    if item in {"bullet", "3 bullet"}:
                        old_items = [old for old in old_items if old not in {"bullet", "3 bullet"}]
                    elif item.startswith("ví dụ"):
                        old_items = [old for old in old_items if not old.startswith("ví dụ")]
                value = ", ".join(dict.fromkeys(item for item in old_items + new_items if item))
            elif key == "interests" and current.get(key):
                value = ", ".join(
                    dict.fromkeys(current[key].split(", ") + value.split(", "))
                )
            current[key] = value
            self.profile_store.upsert_fact(user_id, key, value)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        summary = str(context["summary"])
        messages = context["messages"]
        content = [
            "You are a helpful assistant. Use the profile and recent conversation context.",
            self.profile_store.read_text(user_id),
            summary,
        ]
        content.extend(item["content"] for item in messages)
        return estimate_tokens("\n".join(part for part in content if part))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        if _is_memory_question(message):
            return _answer_from_facts(self.profile_store.facts(user_id))
        return "Mình đã ghi nhận thông tin vào bộ nhớ cuộc trò chuyện."

    def _live_prompt(self, user_id: str, thread_id: str) -> list[tuple[str, str]]:
        context = self.compact_memory.context(thread_id)
        system = "You are a helpful assistant. " + self.profile_store.read_text(user_id)
        if context["summary"]:
            system += "\nEarlier conversation summary:\n" + str(context["summary"])
        prompt = [("system", system)]
        prompt.extend(
            (item["role"], item["content"]) for item in context["messages"]
        )
        return prompt

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
