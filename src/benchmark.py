from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}.")
    for item in data:
        if not isinstance(item, dict) or not {"id", "user_id", "turns", "recall_questions"} <= item.keys():
            raise ValueError(f"Invalid conversation record in {path}.")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    matches = sum(value.casefold() in answer.casefold() for value in expected)
    if matches == 0:
        return 0.0
    return 1.0 if matches == len(expected) else 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Score expected-fact coverage (70%) and concise, non-empty output (30%)."""
    if not answer.strip():
        return 0.0
    word_count = len(answer.split())
    concision = 1.0 if word_count <= 80 else max(0.0, 1.0 - (word_count - 80) / 160)
    return round(0.7 * recall_points(answer, expected) + 0.3 * concision, 3)


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    thread_ids: set[str] = set()
    users = {str(item["user_id"]) for item in conversations}
    initial_memory = {
        user_id: agent.memory_file_size(user_id) if hasattr(agent, "memory_file_size") else 0
        for user_id in users
    }
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        user_id = str(conversation["user_id"])
        thread_id = str(conversation["id"])
        thread_ids.add(thread_id)
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, str(turn))

        for index, item in enumerate(conversation["recall_questions"]):
            recall_thread = f"{thread_id}:recall:{index}"
            thread_ids.add(recall_thread)
            question = str(item["question"])
            expected = [str(value) for value in item.get("expected_contains", [])]
            answer = agent.reply(user_id, recall_thread, question)["response"]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    agent_tokens = sum(agent.token_usage(thread_id) for thread_id in thread_ids)
    prompt_tokens = sum(agent.prompt_token_usage(thread_id) for thread_id in thread_ids)
    compactions = sum(agent.compaction_count(thread_id) for thread_id in thread_ids)
    memory_growth = sum(
        max(0, agent.memory_file_size(user_id) - initial_memory[user_id])
        for user_id in users
    ) if hasattr(agent, "memory_file_size") else 0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = (
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    )
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        output.append(
            "| "
            + " | ".join(
                (
                    row.agent_name,
                    f"{row.agent_tokens_only:,}",
                    f"{row.prompt_tokens_processed:,}",
                    f"{row.recall_score:.1%}",
                    f"{row.response_quality:.1%}",
                    f"{row.memory_growth_bytes:,}",
                    str(row.compactions),
                )
            )
            + " |"
        )
    return "\n".join(output)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    suites = (
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    )
    with tempfile.TemporaryDirectory(prefix="memory-benchmark-") as temporary_state:
        for suite_name, path in suites:
            conversations = load_conversations(path)
            suite_config = replace(config, state_dir=Path(temporary_state) / suite_name.lower().replace(" ", "-"))
            baseline = BaselineAgent(suite_config, force_offline=True)
            advanced = AdvancedAgent(suite_config, force_offline=True)
            rows = [
                run_agent_benchmark("Baseline", baseline, conversations, suite_config),
                run_agent_benchmark("Advanced", advanced, conversations, suite_config),
            ]
            print(f"\n## {suite_name}\n")
            print(format_rows(rows))


if __name__ == "__main__":
    main()
