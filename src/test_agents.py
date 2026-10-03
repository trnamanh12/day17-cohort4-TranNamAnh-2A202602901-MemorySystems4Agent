from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path, threshold: int = 1200, keep_messages: int = 4) -> LabConfig:
    model = ProviderConfig("openai", "gpt-4o-mini", 0.0)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep_messages,
        model=model,
        judge_model=model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("user-1") == "# User profile\n"

    path = store.write_text("user-1", "# User profile\n\n- name: An")
    assert path.name == "User.md"
    assert "- name: An" in store.read_text("user-1")
    assert store.file_size("user-1") > 0
    assert store.edit_text("user-1", "An", "Anh")
    assert "- name: Anh" in store.read_text("user-1")
    assert not store.edit_text("user-1", "missing", "value")


def test_compact_trigger_keeps_recent_messages_and_summary() -> None:
    memory = CompactMemoryManager(threshold_tokens=60, keep_messages=2)
    for index in range(5):
        memory.append("thread", "user", f"message {index}: " + "context " * 12)

    context = memory.context("thread")
    assert memory.compaction_count("thread") > 0
    assert len(context["messages"]) <= 2
    assert "message 0" in context["summary"]


def test_cross_session_recall_uses_latest_explicit_facts(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    turns = (
        "Mình ở Đà Nẵng và đang làm backend engineer.",
        "Mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng. Mình chuyển sang MLOps engineer.",
        "Hà Nội chỉ là nơi mình đi họp; product manager chỉ là câu đùa.",
    )
    for turn in turns:
        advanced.reply("u", "old-thread", turn)
        baseline.reply("u", "old-thread", turn)

    question = "Nhắc lại nơi ở hiện tại và nghề nghiệp hiện tại của mình."
    advanced_answer = advanced.reply("u", "new-thread", question)["response"]
    baseline_answer = baseline.reply("u", "new-thread", question)["response"]

    assert "Huế" in advanced_answer
    assert "MLOps engineer" in advanced_answer
    assert "Đà Nẵng" not in advanced_answer
    assert "backend engineer" not in advanced_answer
    assert "Hà Nội" not in advanced_answer
    assert "product manager" not in advanced_answer
    assert "Huế" not in baseline_answer
    assert "MLOps engineer" not in baseline_answer


def test_compaction_reduces_processed_prompt_load(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=800, keep_messages=2)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for index in range(12):
        message = f"Turn {index}: " + "long context detail " * 50
        advanced.reply("u", "long-thread", message)
        baseline.reply("u", "long-thread", message)

    assert advanced.prompt_token_usage("long-thread") < baseline.prompt_token_usage("long-thread")
