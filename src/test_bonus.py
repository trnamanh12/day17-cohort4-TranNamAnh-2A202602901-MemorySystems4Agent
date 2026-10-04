from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from config import LabConfig
from memory_store import UserProfileStore
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


def test_confidence_threshold_ignores_questions_and_hypotheticals(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)

    # Questions and hypotheticals should have confidence < 0.7 and NOT be stored in User.md
    advanced.reply("user-test", "t1", "Mình đang ở Hà Nội phải không?")
    advanced.reply("user-test", "t1", "Nếu mình làm product manager thì sao?")
    facts = advanced.profile_store.facts("user-test")
    assert "current_location" not in facts
    assert "profession" not in facts

    # Explicit affirmative statements should have confidence >= 0.7 and BE stored
    advanced.reply("user-test", "t1", "Mình đang ở Huế và đang làm MLOps engineer.")
    facts = advanced.profile_store.facts("user-test")
    assert facts.get("current_location") == "Huế"
    assert facts.get("profession") == "MLOps engineer"


def test_conflict_handling_supersedes_and_audits_facts(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.upsert_fact("user-c", "current_location", "Đà Nẵng")
    assert store.facts("user-c")["current_location"] == "Đà Nẵng"

    # When location changes to Huế, conflict resolution updates fact and logs superseded value
    store.upsert_fact("user-c", "current_location", "Huế")
    assert store.facts("user-c")["current_location"] == "Huế"
    superseded = store.superseded_facts("user-c")
    assert "current_location" in superseded
    assert "Đà Nẵng" in superseded["current_location"]


def test_memory_decay_and_pruning(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.upsert_fact("user-d", "name", "Nam")
    store.upsert_fact("user-d", "current_location", "Hà Nội")
    store.upsert_fact("user-d", "profession", "software engineer")
    store.upsert_fact("user-d", "temp_fact_1", "đang học git")
    store.upsert_fact("user-d", "temp_fact_2", "uống trà chanh")
    store.upsert_fact("user-d", "temp_fact_3", "thích đọc sách")

    assert len(store.facts("user-d")) == 6
    # Prune non-essential facts if exceeding max_facts=4
    pruned_count = store.prune_stale_facts("user-d", max_facts=4)
    assert pruned_count == 2
    facts = store.facts("user-d")
    assert len(facts) == 4
    # Core essential facts must still remain
    assert "name" in facts
    assert "current_location" in facts
    assert "profession" in facts

    # Test explicit fact decay
    store.decay_facts("user-d", ["temp_fact_1"])
    assert "temp_fact_1" not in store.facts("user-d")
