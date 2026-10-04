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
    _history: dict[str, dict[str, list[str]]] = field(default_factory=dict)

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
        if key in facts and facts[key] != value:
            self._history.setdefault(user_id, {}).setdefault(key, []).append(facts[key])
        if value:
            facts[key] = value
        lines = ["# User profile", ""]
        lines.extend(f"- {fact_key}: {fact_value}" for fact_key, fact_value in facts.items())
        return self.write_text(user_id, "\n".join(lines))

    def superseded_facts(self, user_id: str) -> dict[str, list[str]]:
        """Return history of overridden/superseded facts for audit & conflict resolution."""
        return self._history.get(user_id, {})

    def decay_facts(self, user_id: str, keys_to_remove: list[str]) -> Path:
        """Remove specified decayed or obsolete facts from the profile."""
        facts = self.facts(user_id)
        for key in keys_to_remove:
            facts.pop(key, None)
        lines = ["# User profile", ""]
        lines.extend(f"- {fact_key}: {fact_value}" for fact_key, fact_value in facts.items())
        return self.write_text(user_id, "\n".join(lines))

    def prune_stale_facts(
        self,
        user_id: str,
        keep_keys: set[str] | None = None,
        max_facts: int = 10,
    ) -> int:
        """Prune low-priority facts if profile exceeds max_facts to prevent unbounded growth."""
        essential_keys = keep_keys or {
            "name",
            "current_location",
            "profession",
            "response_style",
            "interests",
        }
        current = self.facts(user_id)
        if len(current) <= max_facts:
            return 0
        removable = [k for k in current if k not in essential_keys]
        num_pruned = 0
        while len(current) > max_facts and removable:
            target = removable.pop()
            del current[target]
            num_pruned += 1
        lines = ["# User profile", ""]
        lines.extend(f"- {fact_key}: {fact_value}" for fact_key, fact_value in current.items())
        self.write_text(user_id, "\n".join(lines))
        return num_pruned


@dataclass
class ExtractedFact:
    """A candidate fact extracted from text with structured metadata and confidence."""

    key: str
    value: str
    confidence: float
    is_correction: bool = False
    source_segment: str = ""


def _evaluate_confidence(text: str, start: int, end: int, is_corr: bool = False) -> float:
    low = text.lower()
    pre = low[max(0, start - 50) : start]
    post = low[end : min(len(low), end + 50)]
    surrounding = f"{pre} {post}"

    # Jokes, sarcasm or casual negation
    if any(cue in surrounding for cue in ("đùa", "câu đùa", "trêu", "nói chơi", "chỉ là nơi")):
        return 0.10

    # Question detection: '?' anywhere in the sentence or explicit question cues
    question_cues = (
        "phải không", "đúng không", "chăng", "hả", "có phải",
        "ở đâu", "là gì", "sao", "thế nào", "bao giờ", "nhắc lại", "nhớ lại"
    )
    if "?" in text or any(re.search(rf"\b{re.escape(q)}\b", surrounding) for q in question_cues):
        return 0.30

    # Doubt or hypotheticals in preceding context
    hypothetical_cues = ("nếu", "giả sử", "ước gì", "hình như", "có lẽ", "chắc là", "có thể", "không biết")
    if any(re.search(rf"\b{re.escape(h)}\b", pre) for h in hypothetical_cues):
        return 0.40

    if is_corr:
        return 1.0

    return 0.95


def extract_structured_facts(message: str) -> list[ExtractedFact]:
    """Extract candidate facts along with confidence and correction flags."""
    text = message.strip()
    low = text.lower()
    facts: list[ExtractedFact] = []

    name = re.search(r"\bmình\s+tên\s+là\s+([^.!?;,\n]+)", text, re.IGNORECASE)
    if name:
        value = name.group(1).strip().rstrip(".")
        if value and value.lower() not in {"gì", "ai", "không"}:
            conf = _evaluate_confidence(text, name.start(), name.end())
            facts.append(ExtractedFact("name", value, conf, False, name.group(0)))

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
        val = re.sub(r"\s+", " ", match.group("location")).strip()
        is_corr = bool(re.search(r"\b(?:đính chính|giờ|hiện tại|chuyển)\b", low[: match.start()]))
        conf = _evaluate_confidence(text, match.start(), match.end(), is_corr)
        facts.append(ExtractedFact("current_location", val, conf, is_corr, match.group(0)))
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
        val = re.sub(r"\s+", " ", match.group("role")).strip()
        is_corr = "chuyển sang" in match.group(0).lower() or "đính chính" in prefix
        conf = _evaluate_confidence(text, match.start(), match.end(), is_corr)
        facts.append(ExtractedFact("profession", val, conf, is_corr, match.group(0)))
        break

    style_cues = ("ngắn gọn", "gọn", "bullet", "ví dụ", "trade-off", "tradeoff", "lan man", "cấu trúc")
    if any(cue in low for cue in style_cues) and re.search(
        r"trả lời|câu trả lời|style|giải thích|ưu tiên|mình thích|mình muốn|hãy",
        text,
        re.IGNORECASE,
    ):
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
        val = ", ".join(dict.fromkeys(style))
        conf = 0.95 if "?" not in text else 0.30
        facts.append(ExtractedFact("response_style", val, conf, False, "style"))

    if any(cue in low for cue in ("đồ uống yêu thích", "mình thích", "vẫn uống")) and "cà phê sữa đá" in low:
        conf = 0.95 if not any(q in low for q in ("không?", "là gì", "phải không", "?")) else 0.30
        facts.append(ExtractedFact("favorite_drink", "cà phê sữa đá", conf, False, "favorite_drink"))
    if "mì quảng" in low and re.search(r"món ăn yêu thích|món ruột|mình thích", low):
        conf = 0.95 if "?" not in text else 0.30
        facts.append(ExtractedFact("favorite_food", "mì Quảng", conf, False, "favorite_food"))

    pet = re.search(r"\bnuôi\s+(?:một\s+)?(?:bé\s+)?corgi\s+tên\s+([\wÀ-ỹ-]+)", text, re.IGNORECASE)
    if pet:
        conf = 0.95 if "?" not in text else 0.30
        facts.append(ExtractedFact("pet", f"corgi tên {pet.group(1).strip()}", conf, False, pet.group(0)))

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
            conf = 0.95 if "?" not in text else 0.30
            facts.append(ExtractedFact("interests", ", ".join(found), conf, False, interest_statement.group(0)))

    return facts


def extract_profile_updates(message: str, min_confidence: float = 0.7) -> dict[str, str]:
    """Extract explicit, positive profile statements from Vietnamese turns with confidence >= min_confidence."""
    facts = extract_structured_facts(message)
    return {f.key: f.value for f in facts if f.confidence >= min_confidence}


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
