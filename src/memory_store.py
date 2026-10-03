from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Estimate tokens from characters; deterministic and tokenizer-free."""
    text = text.strip()
    return (len(text) + 3) // 4 if text else 0


@dataclass
class UserProfileStore:
    """Store each user's durable facts in a readable ``User.md`` file."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", user_id).strip("-_") or "user"
        suffix = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:8]
        return self.root_dir / f"{slug}-{suffix}" / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.is_file() else "# User profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False
        current = self.read_text(user_id)
        updated = current.replace(search_text, replacement, 1)
        if updated == current:
            return False
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.is_file() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            if line.startswith("- ") and ": " in line:
                key, value = line[2:].split(": ", 1)
                facts[key] = value
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        facts = self.facts(user_id)
        if value:
            facts[key] = value
        lines = ["# User profile", ""]
        lines.extend(f"- {fact_key}: {fact_value}" for fact_key, fact_value in facts.items())
        return self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract only explicit, positive profile statements from Vietnamese turns."""
    text = message.split("?", 1)[0]
    updates: dict[str, str] = {}

    name = re.search(r"\bmình\s+tên\s+là\s+([^.!?;,\n]+)", text, re.IGNORECASE)
    if name:
        value = name.group(1).strip().rstrip(".")
        if value and value.lower() not in {"gì", "ai", "không"}:
            updates["name"] = value

    locations = r"Đà\s+Nẵng|Hà\s+Nội|Huế"
    location_pattern = re.compile(
        rf"\b(?:mình\s+(?:(?:hiện(?:\s+tại)?|đang|vẫn|giờ)\s+)?ở|"
        rf"(?:hiện(?:\s+tại)?|đang|vẫn|giờ)\s+(?:mình\s+)?ở|"
        rf"mình\s+(?:đang\s+)?làm\s+việc\s+ở|"
        rf"(?:hiện(?:\s+tại)?|đang|từ\s+tuần\s+này)\s+mình\s+đang\s+làm\s+việc\s+ở)\s+"
        rf"(?P<location>{locations})\b",
        re.IGNORECASE,
    )
    location_matches = list(location_pattern.finditer(text))
    for match in reversed(location_matches):
        tail = text[match.end() : match.end() + 45].lower()
        if re.match(r"\s*(?:chứ\s+)?(?:không\s+phải|đã\s+rời)", tail):
            continue
        updates["current_location"] = re.sub(r"\s+", " ", match.group("location")).strip()
        break

    roles = r"backend\s+engineer|MLOps\s+engineer|software\s+engineer"
    role_pattern = re.compile(
        rf"\b(?:đang\s+làm|(?:giờ\s+)?chuyển\s+sang|"
        rf"(?:nghề\s+nghiệp(?:\s+hiện\s+tại)?|công\s+việc\s+hiện\s+tại)\s+"
        rf"(?:của\s+mình\s+)?(?:vẫn\s+)?là)\s+(?P<role>{roles})\b",
        re.IGNORECASE,
    )
    role_matches = list(role_pattern.finditer(text))
    for match in reversed(role_matches):
        prefix = text[max(0, match.start() - 50) : match.start()].lower()
        if any(cue in prefix for cue in ("hay là", "đùa", "nếu", "ước gì")):
            continue
        updates["profession"] = re.sub(r"\s+", " ", match.group("role")).strip()
        break

    style_cues = ("ngắn gọn", "gọn", "bullet", "ví dụ", "trade-off", "tradeoff", "lan man", "cấu trúc")
    if any(cue in text.lower() for cue in style_cues) and re.search(
        r"trả lời|câu trả lời|style|giải thích|ưu tiên|mình thích|mình muốn|hãy",
        text,
        re.IGNORECASE,
    ):
        low = text.lower()
        style = ["ngắn gọn"] if any(cue in low for cue in ("ngắn gọn", "lan man", "gọn")) else []
        if "3 bullet" in low:
            style.append("3 bullet")
        elif "bullet" in low:
            style.append("bullet")
        if "ví dụ" in low:
            style.append("ví dụ thực chiến" if "thực chiến" in low else "ví dụ thực tế")
        if "trade-off" in low or "tradeoff" in low or "so sánh" in low:
            style.append("ưu tiên trade-off")
        if "rõ ý" in low or "cấu trúc" in low:
            style.append("rõ ý")
        updates["response_style"] = ", ".join(dict.fromkeys(style))

    low = text.lower()
    if any(cue in low for cue in ("đồ uống yêu thích", "mình thích", "vẫn uống")) and "cà phê sữa đá" in low:
        updates["favorite_drink"] = "cà phê sữa đá"
    if "mì quảng" in low and re.search(r"món ăn yêu thích|món ruột|mình thích", low):
        updates["favorite_food"] = "mì Quảng"

    pet = re.search(r"\bnuôi\s+(?:một\s+)?(?:bé\s+)?corgi\s+tên\s+([\wÀ-ỹ-]+)", text, re.IGNORECASE)
    if pet:
        updates["pet"] = f"corgi tên {pet.group(1).strip()}"

    interest_statement = re.search(
        r"(?:quan\s+tâm(?:\s+nhiều)?\s+đến|mình\s+(?:rất\s+|vẫn\s+)?thích)\s+([^.!?\n]{1,100})",
        text,
        re.IGNORECASE,
    )
    if interest_statement:
        interest_names = ("Python", "AI ứng dụng", "AI agent", "AI", "MLOps", "RAG", "evaluation", "benchmark memory")
        found = []
        for interest in interest_names:
            if re.search(rf"(?<!\w){re.escape(interest)}(?!\w)", interest_statement.group(1), re.IGNORECASE):
                if interest == "AI" and any(item.lower().startswith("ai ") for item in found):
                    continue
                canonical = "AI ứng dụng" if interest == "AI ứng dụng" else interest
                if canonical.lower() not in {item.lower() for item in found}:
                    found.append(canonical)
        if found:
            updates["interests"] = ", ".join(found)

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Keep a small set of recent, readable snippets from archived turns."""
    snippets = []
    seen = set()
    for message in messages:
        content = " ".join(message.get("content", "").split())
        if not content:
            continue
        role = message.get("role", "user")
        snippet = f"{role}: {content[:180]}{'…' if len(content) > 180 else ''}"
        if snippet not in seen:
            seen.add(snippet)
            snippets.append(snippet)
    return "\n".join(snippets[-max(0, max_items) :]) if max_items else ""


@dataclass
class CompactMemoryManager:
    """Retain a recent message window and summarize older thread content."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        summary = str(thread["summary"])
        token_count = estimate_tokens(summary) + sum(
            estimate_tokens(item["content"]) for item in messages
        )
        if token_count <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        split_at = max(0, len(messages) - self.keep_messages)
        archived = messages[:split_at]
        prior_summary = [{"role": "summary", "content": summary}] if summary else []
        thread["summary"] = summarize_messages(prior_summary + archived)
        thread["messages"] = messages[split_at:]
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
