"""JSON-shaped LangGraph state and parallel append reducers."""

from __future__ import annotations

import json
from typing import Annotated, Any

from typing_extensions import TypedDict


def merge_records(
    current: list[dict[str, Any]] | None,
    incoming: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in (current or []) + (incoming or []):
        key = str(
            item.get("evidence_id")
            or item.get("event_id")
            or json.dumps(item, sort_keys=True, default=str)
        )
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


class RepairState(TypedDict, total=False):
    thread_id: str
    task_path: str
    task: dict[str, Any]
    provider: str
    model: str | None
    run_config: dict[str, Any]
    workspace_path: str
    workspace_baseline_hash: str
    hidden_tests_hash: str
    current_diff: str
    tool_trace: Annotated[list[dict[str, Any]], merge_records]
    baseline_result: dict[str, Any] | None
    memory_hits: list[dict[str, Any]]
    plan: dict[str, Any] | None
    read_tasks: list[dict[str, Any]]
    evidence: Annotated[list[dict[str, Any]], merge_records]
    patch_proposal: dict[str, Any] | None
    patch_result: dict[str, Any] | None
    verification: dict[str, Any] | None
    status: str
    metrics: dict[str, Any]
    started_at_epoch: float
    deadline_at_epoch: float
    paused_at_epoch: float | None
    paused_after_node: str | None
    total_paused_s: float


class ReadWorkerState(TypedDict, total=False):
    thread_id: str
    read_task: dict[str, Any]
    deadline_at_epoch: float
    evidence: Annotated[list[dict[str, Any]], merge_records]
    tool_trace: Annotated[list[dict[str, Any]], merge_records]
