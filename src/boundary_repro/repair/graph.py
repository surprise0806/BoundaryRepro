"""LangGraph state machine for generic, stateful repository repair."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from boundary_repro.repair.memory import RepairMemoryStore
from boundary_repro.repair.models import (
    LoadedTask,
    PatchProposal,
    ReadTask,
    RepairMemoryRecord,
    RepairPlan,
    RepairRunConfig,
    utc_now,
)
from boundary_repro.repair.providers import (
    ProviderDeadlineError,
    ProviderOutputError,
    ProviderRetryError,
    RepairProvider,
    retry_with_deadline,
)
from boundary_repro.repair.state import ReadWorkerState, RepairState
from boundary_repro.repair.tools import (
    RepositoryRepairTools,
    RepositoryToolError,
)
from boundary_repro.repair.verifier import verify_behavior


@dataclass(frozen=True)
class RepairGraphServices:
    task: LoadedTask
    tools: RepositoryRepairTools
    provider: RepairProvider
    memory: RepairMemoryStore
    config: RepairRunConfig


class ReadConcurrency:
    def __init__(self, limit: int) -> None:
        self.semaphore = asyncio.Semaphore(limit)
        self.lock = asyncio.Lock()
        self.active = 0
        self.maximum = 0

    async def enter(self) -> int:
        await self.semaphore.acquire()
        async with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            return self.active

    async def leave(self) -> None:
        async with self.lock:
            self.active -= 1
        self.semaphore.release()


def build_repair_graph(
    services: RepairGraphServices,
    *,
    checkpointer: Any,
    interrupt_after: list[str] | None = None,
) -> Any:
    """Compile a graph whose task behavior never branches on task_id."""

    read_concurrency = ReadConcurrency(services.config.max_concurrency)

    async def load_task(state: RepairState) -> dict[str, Any]:
        started = time.perf_counter()
        result = services.tools.read_issue()
        return {
            "status": "task_loaded",
            "tool_trace": [
                _event(
                    "read_issue",
                    "success",
                    {},
                    result,
                    started,
                    actor="tool",
                )
            ],
            "metrics": _metrics(state, "load_task"),
        }

    async def reproduce_failure(state: RepairState) -> dict[str, Any]:
        started = time.perf_counter()
        result = await _run_test_with_deadline(
            services,
            state,
            command_kind="public",
            expose_output=True,
        )
        baseline_diff = services.tools.show_diff()
        reproduced = (
            result.get("status") == "fail"
            and not str(baseline_diff["diff"]).strip()
        )
        deadline_exhausted = (
            result.get("status") == "timeout"
            and time.time() >= float(state["deadline_at_epoch"])
        )
        return {
            "baseline_result": result,
            "current_diff": str(baseline_diff["diff"]),
            "status": (
                "deadline_exceeded"
                if deadline_exhausted
                else (
                    "failure_reproduced"
                    if reproduced
                    else "baseline_not_failed"
                )
            ),
            "tool_trace": [
                _event(
                    "run_tests",
                    str(result.get("status", "error")),
                    {"phase": "baseline", "command_kind": "public"},
                    result,
                    started,
                    actor="tool",
                ),
                _simple_event(
                    "show_diff",
                    "success",
                    {
                        "phase": "baseline_integrity",
                        **baseline_diff,
                    },
                    actor="verifier",
                ),
            ],
            "metrics": _metrics(
                state,
                "reproduce_failure",
                baseline_failed=reproduced,
            ),
        }

    def route_after_reproduction(state: RepairState) -> str:
        if state.get("status") == "failure_reproduced":
            return "retrieve_memory"
        return "finalize"

    async def retrieve_memory(state: RepairState) -> dict[str, Any]:
        try:
            records = await services.memory.search(
                str(state["task"]["issue_text"]),
                limit=5,
                requesting_provider=str(state["provider"]),
                exclude_task_id=str(state["task"]["task_id"]),
            )
            hits = [_memory_context(record) for record in records]
            memory_refs = [_memory_ref(record) for record in records]
            status = "memory_retrieved"
            trace = [
                _simple_event(
                    "memory_search",
                    "success",
                    {"memories": memory_refs},
                    actor="memory",
                )
            ]
        except Exception as exc:
            hits = []
            memory_refs = []
            status = "memory_degraded"
            trace = [
                _simple_event(
                    "memory_search",
                    "error",
                    {"error_type": type(exc).__name__},
                    actor="memory",
                )
            ]
        return {
            "memory_hits": hits,
            "status": status,
            "tool_trace": trace,
            "metrics": _metrics(
                state,
                "retrieve_memory",
                memory_hits=len(hits),
                memory_sources=memory_refs,
            ),
        }

    async def plan(state: RepairState) -> dict[str, Any]:
        try:
            result, audit = await retry_with_deadline(
                lambda: services.provider.aplan(
                    task=state["task"],
                    baseline=dict(state["baseline_result"] or {}),
                    memory_hits=list(state.get("memory_hits", [])),
                ),
                operation_name="provider_plan",
                absolute_deadline=float(state["deadline_at_epoch"]),
                per_attempt_timeout_s=services.config.llm_timeout_s,
                max_retries=services.config.max_retries,
                initial_delay_s=services.config.retry_initial_delay_s,
            )
        except ProviderDeadlineError as exc:
            audit = getattr(exc, "audit", [])
            return {
                "status": "deadline_exceeded",
                "plan": None,
                "read_tasks": [],
                "tool_trace": _provider_failure_events(
                    "provider_plan",
                    exc,
                    audit,
                    status="deadline_exceeded",
                ),
                "metrics": _metrics(state, "plan"),
            }
        except (ProviderOutputError, ProviderRetryError) as exc:
            audit = getattr(exc, "audit", [])
            return {
                "status": "provider_failed",
                "plan": None,
                "read_tasks": [],
                "tool_trace": _provider_failure_events(
                    "provider_plan",
                    exc,
                    audit,
                ),
                "metrics": _metrics(state, "plan"),
            }
        except Exception as exc:
            return {
                "status": "provider_failed",
                "plan": None,
                "read_tasks": [],
                "tool_trace": _provider_failure_events(
                    "provider_plan",
                    exc,
                    [],
                ),
                "metrics": _metrics(state, "plan"),
            }
        tasks = _bounded_read_tasks(result, services)
        return {
            "plan": result.model_dump(mode="json"),
            "read_tasks": [
                task.model_dump(mode="json") for task in tasks
            ],
            "status": "planned",
            "tool_trace": _provider_audit_events(audit),
            "metrics": _metrics(
                state,
                "plan",
                read_tasks=len(tasks),
            ),
        }

    def route_after_plan(state: RepairState) -> str:
        return (
            "dispatch_read_workers"
            if state.get("status") == "planned"
            else "finalize"
        )

    async def dispatch_read_workers(
        state: RepairState,
    ) -> dict[str, Any]:
        return {
            "status": "reading_repository",
            "metrics": _metrics(
                state,
                "dispatch_read_workers",
                dispatched_read_workers=len(state.get("read_tasks", [])),
            ),
        }

    def send_read_workers(state: RepairState) -> Any:
        tasks = list(state.get("read_tasks", []))
        if not tasks:
            return "aggregate"
        return [
            Send(
                "read_worker",
                {
                    "thread_id": state["thread_id"],
                    "read_task": task,
                    "deadline_at_epoch": state["deadline_at_epoch"],
                },
            )
            for task in tasks
        ]

    async def read_worker(state: ReadWorkerState) -> dict[str, Any]:
        task = ReadTask.model_validate(state["read_task"])
        started_at = utc_now()
        started = time.perf_counter()
        active = await read_concurrency.enter()
        try:
            remaining = float(state["deadline_at_epoch"]) - time.time()
            if remaining <= 0:
                raise TimeoutError("overall repair deadline reached")
            result = await asyncio.wait_for(
                services.tools.execute_read(task.tool, task.arguments),
                timeout=min(services.config.tool_timeout_s, remaining),
            )
            status = (
                "success"
                if result.get("status") == "ok"
                else "error"
            )
        except TimeoutError as exc:
            status = "timeout"
            result = {
                "status": "timeout",
                "reason": str(exc) or "read tool timed out",
            }
        except (RepositoryToolError, OSError, ValueError) as exc:
            status = "error"
            result = {
                "status": "error",
                "reason": str(exc),
                "error_type": type(exc).__name__,
            }
        except Exception as exc:
            status = "error"
            result = {
                "status": "error",
                "reason": str(exc),
                "error_type": type(exc).__name__,
            }
        finally:
            await read_concurrency.leave()
        finished_at = utc_now()
        duration_ms = int((time.perf_counter() - started) * 1000)
        digest = hashlib.sha256(
            json.dumps(
                {
                    "task": task.model_dump(mode="json"),
                    "status": status,
                    "result": result,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:20]
        evidence = {
            "evidence_id": f"repair-ev-{digest}",
            "task_id": task.task_id,
            "tool": task.tool,
            "arguments": task.arguments,
            "status": status,
            "result": result,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "active_read_workers": active,
        }
        return {
            "evidence": [evidence],
            "tool_trace": [
                {
                    "event_id": f"tool-{digest}",
                    "actor": "read_worker",
                    "name": task.tool,
                    "arguments": task.arguments,
                    "status": status,
                    "result": result,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_ms": duration_ms,
                    "active_read_workers": active,
                }
            ],
        }

    async def aggregate(state: RepairState) -> dict[str, Any]:
        successes = sum(
            item.get("status") == "success"
            for item in state.get("evidence", [])
        )
        metrics = _metrics(
            state,
            "aggregate",
            evidence_records=len(state.get("evidence", [])),
            successful_read_workers=successes,
            max_active_read_workers=read_concurrency.maximum,
        )
        metrics["node_runs"]["read_worker"] = len(
            state.get("evidence", [])
        )
        return {
            "status": "evidence_aggregated",
            "metrics": metrics,
        }

    async def patch(state: RepairState) -> dict[str, Any]:
        try:
            proposal, audit = await retry_with_deadline(
                lambda: services.provider.apatch(
                    task=state["task"],
                    baseline=dict(state["baseline_result"] or {}),
                    evidence=list(state.get("evidence", [])),
                    memory_hits=list(state.get("memory_hits", [])),
                ),
                operation_name="provider_patch",
                absolute_deadline=float(state["deadline_at_epoch"]),
                per_attempt_timeout_s=services.config.llm_timeout_s,
                max_retries=services.config.max_retries,
                initial_delay_s=services.config.retry_initial_delay_s,
            )
        except ProviderDeadlineError as exc:
            audit = getattr(exc, "audit", [])
            return {
                "status": "deadline_exceeded",
                "patch_proposal": None,
                "patch_result": None,
                "tool_trace": _provider_failure_events(
                    "provider_patch",
                    exc,
                    audit,
                    status="deadline_exceeded",
                ),
                "metrics": _metrics(state, "patch"),
            }
        except (ProviderOutputError, ProviderRetryError) as exc:
            audit = getattr(exc, "audit", [])
            return {
                "status": "provider_failed",
                "patch_proposal": None,
                "patch_result": None,
                "tool_trace": _provider_failure_events(
                    "provider_patch",
                    exc,
                    audit,
                ),
                "metrics": _metrics(state, "patch"),
            }
        except Exception as exc:
            return {
                "status": "provider_failed",
                "patch_proposal": None,
                "patch_result": None,
                "tool_trace": _provider_failure_events(
                    "provider_patch",
                    exc,
                    [],
                ),
                "metrics": _metrics(state, "patch"),
            }

        started = time.perf_counter()
        remaining = float(state["deadline_at_epoch"]) - time.time()
        if remaining <= 0:
            result = {
                "status": "timeout",
                "reason": "overall repair deadline reached before patch",
            }
        else:
            result = await services.tools.apply_patch(
                proposal,
                baseline_failed=(
                    state.get("baseline_result", {}).get("status")
                    == "fail"
                ),
            )
        diff_result = services.tools.show_diff()
        applied = result.get("status") == "applied"
        patch_status = (
            "patch_applied"
            if applied
            else (
                "deadline_exceeded"
                if result.get("status") == "timeout"
                and time.time() >= float(state["deadline_at_epoch"])
                else "patch_failed"
            )
        )
        return {
            "patch_proposal": proposal.model_dump(mode="json"),
            "patch_result": result,
            "current_diff": str(diff_result["diff"]),
            "status": patch_status,
            "tool_trace": _provider_audit_events(audit)
            + [
                _event(
                    "apply_patch",
                    str(result.get("status", "error")),
                    {
                        "path": proposal.path,
                        "old_text": proposal.old_text,
                        "new_text": proposal.new_text,
                    },
                    result,
                    started,
                    actor="tool",
                ),
                _simple_event(
                    "show_diff",
                    "success",
                    diff_result,
                    actor="tool",
                ),
            ],
            "metrics": _metrics(state, "patch"),
        }

    def route_after_patch(state: RepairState) -> str:
        return (
            "verify"
            if state.get("status") == "patch_applied"
            else "finalize"
        )

    async def verify(state: RepairState) -> dict[str, Any]:
        diff_result = services.tools.show_diff()
        current_diff = str(diff_result["diff"])
        paths = [str(item) for item in diff_result["changed_paths"]]
        legal_paths = bool(paths) and all(
            services.tools.is_editable(path) for path in paths
        )
        events: list[dict[str, Any]] = [
            _simple_event(
                "show_diff",
                "success",
                diff_result,
                actor="verifier",
            )
        ]

        public_after = await _run_test_with_deadline(
            services,
            state,
            command_kind="public",
            expose_output=True,
        )
        events.append(
            _simple_event(
                "run_tests",
                str(public_after.get("status", "error")),
                {
                    "phase": "after_patch",
                    "command_kind": "public",
                    "result": public_after,
                },
                actor="verifier",
            )
        )
        regression = await _run_test_with_deadline(
            services,
            state,
            command_kind="regression",
            expose_output=True,
        )
        events.append(
            _simple_event(
                "run_tests",
                str(regression.get("status", "error")),
                {
                    "phase": "regression",
                    "command_kind": "regression",
                    "result": regression,
                },
                actor="verifier",
            )
        )
        hidden = await _run_test_with_deadline(
            services,
            state,
            command_kind="hidden",
            expose_output=False,
        )
        events.append(
            _simple_event(
                "hidden_tests",
                str(hidden.get("status", "error")),
                {
                    "status": hidden.get("status"),
                    "returncode": hidden.get("returncode"),
                    "duration_ms": hidden.get("duration_ms"),
                    "output_sha256": hidden.get("output_sha256"),
                },
                actor="verifier",
            )
        )

        behavior = verify_behavior(
            baseline_result=dict(state.get("baseline_result") or {}),
            current_diff=current_diff,
            changed_paths=paths,
            legal_paths=legal_paths,
            public_after=public_after,
            regression=regression,
            hidden=hidden,
        )
        proposal = PatchProposal.model_validate(state["patch_proposal"])
        submission = services.tools.submit_solution(proposal, behavior)
        submitted = submission.get("status") == "accepted"
        result = behavior.model_copy(
            update={"submitted": submitted}
        )
        events.append(
            _simple_event(
                "submit_solution",
                str(submission.get("status", "error")),
                submission,
                actor="verifier",
            )
        )
        deadline_exhausted = (
            time.time() >= float(state["deadline_at_epoch"])
            and any(
                result.get("status") == "timeout"
                for result in (public_after, regression, hidden)
            )
        )
        return {
            "verification": result.model_dump(mode="json"),
            "current_diff": current_diff,
            "status": (
                "deadline_exceeded"
                if deadline_exhausted
                else (
                    "verified"
                    if result.passed and result.submitted
                    else "verification_failed"
                )
            ),
            "tool_trace": events,
            "metrics": _metrics(
                state,
                "verify",
                verification_passed=result.passed,
                max_workspace_operations=(
                    services.tools.max_active_workspace_operations
                ),
            ),
        }

    def route_after_verification(state: RepairState) -> str:
        verification = state.get("verification") or {}
        return (
            "commit_memory"
            if verification.get("passed")
            and verification.get("submitted")
            else "finalize"
        )

    async def commit_memory(state: RepairState) -> dict[str, Any]:
        proposal = PatchProposal.model_validate(state["patch_proposal"])
        paths = services.tools.show_diff()["changed_paths"]
        record = RepairMemoryRecord.create(
            task_id=str(state["task"]["task_id"]),
            issue_summary=str(state["task"]["issue_text"]),
            verified_root_cause=proposal.root_cause,
            patch_summary=proposal.summary,
            changed_paths=[str(path) for path in paths],
            current_diff=state["current_diff"],
            evidence=proposal.evidence,
            provider=state["provider"],
            model=state.get("model"),
            source_thread_id=state["thread_id"],
        )
        try:
            inserted = await services.memory.upsert(record)
        except Exception as exc:
            return {
                "status": "degraded",
                "tool_trace": [
                    _simple_event(
                        "memory_write",
                        "error",
                        {
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                        actor="memory",
                    )
                ],
                "metrics": _metrics(state, "commit_memory"),
            }
        return {
            "status": "memory_committed",
            "tool_trace": [
                _simple_event(
                    "memory_write",
                    "inserted" if inserted else "deduplicated",
                    {
                        "memory_id": record.memory_id,
                        "source_provider": record.provider,
                        "source_task_id": record.task_id,
                    },
                    actor="memory",
                )
            ],
            "metrics": _metrics(
                state,
                "commit_memory",
                memory_written=inserted,
                memory_id=record.memory_id,
                memory_source_provider=record.provider,
                memory_source_task_id=record.task_id,
            ),
        }

    async def finalize(state: RepairState) -> dict[str, Any]:
        current = str(state.get("status", "failed"))
        if current == "memory_committed":
            final_status = "completed"
        elif current == "deadline_exceeded" or time.time() >= float(
            state.get("deadline_at_epoch", 0)
        ):
            final_status = "deadline_exceeded"
        elif current in {
            "provider_failed",
            "degraded",
            "baseline_not_failed",
            "patch_failed",
            "verification_failed",
        }:
            final_status = current
        else:
            final_status = "failed"
        memory_assisted = bool(state.get("memory_hits", []))
        total_paused_s = float(state.get("total_paused_s", 0.0))
        return {
            "status": final_status,
            "metrics": _metrics(
                state,
                "finalize",
                elapsed_ms=int(
                    (
                        time.time()
                        - float(state.get("started_at_epoch", time.time()))
                        - total_paused_s
                    )
                    * 1000
                ),
                memory_assisted=memory_assisted,
                evaluation_eligible=(
                    state.get("provider") != "scripted"
                    and not memory_assisted
                ),
            ),
        }

    builder = StateGraph(RepairState)
    nodes = {
        "load_task": load_task,
        "reproduce_failure": reproduce_failure,
        "retrieve_memory": retrieve_memory,
        "plan": plan,
        "dispatch_read_workers": dispatch_read_workers,
        "read_worker": read_worker,
        "aggregate": aggregate,
        "patch": patch,
        "verify": verify,
        "commit_memory": commit_memory,
        "finalize": finalize,
    }
    for name, node in nodes.items():
        if interrupt_after and name in interrupt_after:
            node = _with_pause_checkpoint(node, name)
        builder.add_node(
            name,
            node,
            timeout=services.config.node_timeout_s,
        )
    builder.add_edge(START, "load_task")
    builder.add_edge("load_task", "reproduce_failure")
    builder.add_conditional_edges(
        "reproduce_failure",
        route_after_reproduction,
        {
            "retrieve_memory": "retrieve_memory",
            "finalize": "finalize",
        },
    )
    builder.add_edge("retrieve_memory", "plan")
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "dispatch_read_workers": "dispatch_read_workers",
            "finalize": "finalize",
        },
    )
    builder.add_conditional_edges(
        "dispatch_read_workers",
        send_read_workers,
    )
    builder.add_edge("read_worker", "aggregate")
    builder.add_edge("aggregate", "patch")
    builder.add_conditional_edges(
        "patch",
        route_after_patch,
        {"verify": "verify", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "commit_memory": "commit_memory",
            "finalize": "finalize",
        },
    )
    builder.add_edge("commit_memory", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )


def _bounded_read_tasks(
    plan: RepairPlan,
    services: RepairGraphServices,
) -> list[ReadTask]:
    tasks: list[ReadTask] = [
        ReadTask(
            task_id="generic-list-files",
            tool="list_files",
            arguments={},
        )
    ]
    for editable in services.task.spec.editable_paths:
        target = (services.tools.workspace / editable).resolve()
        if target.is_file():
            paths = [target]
        elif target.is_dir():
            paths = [
                path
                for path in sorted(target.rglob("*"))
                if path.is_file() and not path.is_symlink()
            ]
        else:
            paths = []
        for path in paths:
            tasks.append(
                ReadTask(
                    task_id=(
                        "editable-"
                        + hashlib.sha256(
                            path.relative_to(
                                services.tools.workspace
                            ).as_posix().encode("utf-8")
                        ).hexdigest()[:12]
                    ),
                    tool="read_file",
                    arguments={
                        "path": path.relative_to(
                            services.tools.workspace
                        ).as_posix()
                    },
                )
            )
    tasks.extend(plan.read_tasks)
    deduplicated: list[ReadTask] = []
    seen: set[str] = set()
    for task in tasks:
        key = json.dumps(
            {
                "tool": task.tool,
                "arguments": task.arguments,
            },
            sort_keys=True,
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(task)
    return deduplicated[: services.config.max_read_tasks]


async def _run_test_with_deadline(
    services: RepairGraphServices,
    state: RepairState,
    *,
    command_kind: str,
    expose_output: bool,
) -> dict[str, Any]:
    remaining = float(state["deadline_at_epoch"]) - time.time()
    if remaining <= 0:
        return {
            "status": "timeout",
            "reason": "overall repair deadline reached",
            "command_kind": command_kind,
        }
    timeout = min(
        float(services.task.spec.timeout),
        services.config.tool_timeout_s,
        remaining,
    )
    return await services.tools.run_tests(
        command_kind=command_kind,
        timeout_s=timeout,
        expose_output=expose_output,
    )


def _metrics(
    state: RepairState,
    node: str,
    **updates: Any,
) -> dict[str, Any]:
    metrics = dict(state.get("metrics", {}))
    runs = dict(metrics.get("node_runs", {}))
    runs[node] = int(runs.get(node, 0)) + 1
    metrics["node_runs"] = runs
    metrics.update(updates)
    return metrics


def _event(
    name: str,
    status: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    started: float,
    *,
    actor: str,
) -> dict[str, Any]:
    duration_ms = int((time.perf_counter() - started) * 1000)
    created = utc_now()
    payload = {
        "actor": actor,
        "name": name,
        "arguments": arguments,
        "status": status,
        "result": result,
        "created_at": created,
        "duration_ms": duration_ms,
    }
    payload["event_id"] = "event-" + hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:20]
    return payload


def _simple_event(
    name: str,
    status: str,
    result: dict[str, Any],
    *,
    actor: str,
) -> dict[str, Any]:
    return _event(
        name,
        status,
        {},
        result,
        time.perf_counter(),
        actor=actor,
    )


def _provider_audit_events(
    audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _simple_event(
            str(item.get("operation", "provider")),
            str(item.get("status", "unknown")),
            dict(item),
            actor="provider",
        )
        for item in audit
    ]


def _provider_failure_events(
    operation: str,
    error: BaseException,
    audit: list[dict[str, Any]],
    *,
    status: str = "provider_failed",
) -> list[dict[str, Any]]:
    return _provider_audit_events(audit) + [
        _simple_event(
            operation,
            status,
            {
                "error_type": type(error).__name__,
                "message": str(error),
            },
            actor="provider",
        )
    ]


def _memory_ref(record: RepairMemoryRecord) -> dict[str, str]:
    return {
        "memory_id": record.memory_id,
        "source_provider": record.provider,
        "source_task_id": record.task_id,
    }


def _memory_context(record: RepairMemoryRecord) -> dict[str, Any]:
    return {
        **_memory_ref(record),
        "issue_summary": record.issue_summary,
        "verified_root_cause": record.verified_root_cause,
        "patch_summary": record.patch_summary,
        "changed_paths": list(record.changed_paths),
        "evidence": record.evidence,
    }


def _with_pause_checkpoint(node: Any, name: str) -> Any:
    async def wrapped(state: RepairState) -> dict[str, Any]:
        update = await node(state)
        update["paused_at_epoch"] = time.time()
        update["paused_after_node"] = name
        return update

    return wrapped
