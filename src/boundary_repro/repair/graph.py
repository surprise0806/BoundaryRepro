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
    AttemptRecord,
    CandidateVerification,
    LoadedTask,
    PatchProposal,
    ReadTask,
    RepairFeedback,
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
from boundary_repro.repair.verifier import (
    verify_behavior,
    verify_candidate_behavior,
)


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
                    patch_attempt=int(state.get("patch_attempt", 1)),
                    repair_feedback=_repair_feedback(state),
                    attempt_history=_attempt_history(state),
                    evidence=list(state.get("evidence", [])),
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
        tasks = _bounded_read_tasks(
            result,
            services,
            existing_evidence=list(state.get("evidence", [])),
        )
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
        attempt = int(state.get("patch_attempt", 1))
        try:
            proposal, audit = await retry_with_deadline(
                lambda: services.provider.apatch(
                    task=state["task"],
                    baseline=dict(state["baseline_result"] or {}),
                    evidence=list(state.get("evidence", [])),
                    memory_hits=list(state.get("memory_hits", [])),
                    patch_attempt=attempt,
                    repair_feedback=_repair_feedback(state),
                    attempt_history=_attempt_history(state),
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
        current_diff = str(diff_result["diff"])
        diff_sha256 = _sha256(current_diff)
        diff_history = list(state.get("patch_diff_history", []))
        applied = result.get("status") == "applied"
        repeated = applied and bool(current_diff.strip()) and (
            diff_sha256 in diff_history
        )
        no_progress = applied and (not current_diff.strip() or repeated)
        if applied and not no_progress:
            patch_status = "patch_applied"
            diff_history.append(diff_sha256)
        elif no_progress:
            patch_status = "no_progress"
        elif (
            result.get("status") == "timeout"
            and time.time() >= float(state["deadline_at_epoch"])
        ):
            patch_status = "deadline_exceeded"
        else:
            patch_status = "retry_required"
        result = {
            **result,
            "attempt": attempt,
            "diff_sha256": diff_sha256,
            "repeated_diff": repeated,
        }
        metrics = _metrics(
            state,
            "patch",
            patch_attempts=max(
                int(state.get("metrics", {}).get("patch_attempts", 0)),
                attempt,
            ),
        )
        if no_progress:
            metrics["no_progress_count"] = int(
                metrics.get("no_progress_count", 0)
            ) + 1
        if repeated:
            metrics["repeated_diff_count"] = int(
                metrics.get("repeated_diff_count", 0)
            ) + 1
        return {
            "patch_proposal": proposal.model_dump(mode="json"),
            "patch_result": result,
            "current_diff": current_diff,
            "patch_diff_history": diff_history,
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
                _simple_event(
                    "diff_progress_guard",
                    "no_progress" if no_progress else "progress",
                    {
                        "attempt": attempt,
                        "diff_sha256": diff_sha256,
                        "repeated_diff": repeated,
                        "nonempty_diff": bool(current_diff.strip()),
                    },
                    actor="verifier",
                ),
            ],
            "metrics": metrics,
        }

    def route_after_patch(state: RepairState) -> str:
        if state.get("status") == "patch_applied":
            return "verify_candidate"
        if state.get("status") in {"retry_required", "no_progress"}:
            return "prepare_retry"
        return "finalize"

    async def verify_candidate(state: RepairState) -> dict[str, Any]:
        diff_result = services.tools.show_diff()
        current_diff = str(diff_result["diff"])
        diff_sha256 = _sha256(current_diff)
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
                    "phase": "candidate",
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
        behavior = verify_candidate_behavior(
            attempt=int(state.get("patch_attempt", 1)),
            current_diff=current_diff,
            changed_paths=paths,
            legal_paths=legal_paths,
            public_after=public_after,
            regression=regression,
            diff_sha256=diff_sha256,
        )
        deadline_exhausted = (
            time.time() >= float(state["deadline_at_epoch"])
            and any(
                result.get("status") == "timeout"
                for result in (public_after, regression)
            )
        )
        if deadline_exhausted:
            status = "deadline_exceeded"
        elif not behavior.nonempty_diff or not behavior.legal_paths:
            status = "patch_failed"
        elif behavior.passed:
            status = "candidate_verified"
        else:
            status = "retry_required"
        history = list(state.get("attempt_history", []))
        if behavior.passed:
            history.append(
                _attempt_record(
                    state,
                    failure_stage=None,
                    public_status=behavior.public_test_status,
                    regression_status=behavior.regression_test_status,
                    changed_paths=paths,
                    diff_sha256=diff_sha256,
                ).model_dump(mode="json")
            )
        metrics = _metrics(state, "verify_candidate")
        metrics["candidate_verification_runs"] = int(
            metrics.get("candidate_verification_runs", 0)
        ) + 1
        return {
            "candidate_verification": behavior.model_dump(mode="json"),
            "attempt_history": history,
            "current_diff": current_diff,
            "status": status,
            "tool_trace": events,
            "metrics": {
                **metrics,
                "candidate_verification_passed": behavior.passed,
                "max_workspace_operations": (
                    services.tools.max_active_workspace_operations
                ),
            },
        }

    def route_after_candidate(state: RepairState) -> str:
        if state.get("status") == "candidate_verified":
            return "verify_hidden"
        if state.get("status") == "retry_required":
            return "prepare_retry"
        return "finalize"

    async def prepare_retry(state: RepairState) -> dict[str, Any]:
        attempt = int(state.get("patch_attempt", 1))
        candidate_payload = state.get("candidate_verification")
        candidate = (
            CandidateVerification.model_validate(candidate_payload)
            if candidate_payload
            else None
        )
        if state.get("status") == "no_progress":
            failure_stage = "no_progress"
            summary = "The applied patch repeated a prior diff or made no change."
        elif candidate is not None and candidate.failure_stage is not None:
            failure_stage = candidate.failure_stage
            summary = ", ".join(candidate.reasons) or "candidate failed"
        else:
            failure_stage = "patch_apply"
            summary = str(
                (state.get("patch_result") or {}).get(
                    "reason", "patch application failed"
                )
            )
        diff_result = services.tools.show_diff()
        paths = [str(item) for item in diff_result["changed_paths"]]
        diff_sha256 = str(
            (state.get("patch_result") or {}).get(
                "diff_sha256", _sha256(str(diff_result["diff"]))
            )
        )
        public_status = (
            candidate.public_test_status if candidate is not None else None
        )
        regression_status = (
            candidate.regression_test_status if candidate is not None else None
        )
        feedback = RepairFeedback(
            attempt=attempt,
            failure_stage=failure_stage,
            summary=summary,
            patch_status=str(
                (state.get("patch_result") or {}).get("status", "unknown")
            ),
            public_test_status=public_status,
            regression_test_status=regression_status,
            public_test_output=(
                candidate.public_test_output if candidate is not None else ""
            ),
            regression_test_output=(
                candidate.regression_test_output if candidate is not None else ""
            ),
            changed_paths=paths,
            previous_diff_sha256=diff_sha256,
        )
        history = list(state.get("attempt_history", []))
        history.append(
            _attempt_record(
                state,
                failure_stage=failure_stage,
                public_status=public_status,
                regression_status=regression_status,
                changed_paths=paths,
                diff_sha256=diff_sha256,
            ).model_dump(mode="json")
        )
        rollback_started = time.perf_counter()
        rollback = await services.tools.rollback_workspace()
        rolled_back = rollback.get("status") == "rolled_back"
        rollback_count = int(state.get("rollback_count", 0)) + int(
            rolled_back
        )
        exhausted = attempt >= services.config.max_patch_attempts
        deadline_exhausted = time.time() >= float(state["deadline_at_epoch"])
        if not rolled_back:
            status = "rollback_failed"
            termination_reason = "rollback_failed"
            next_attempt = attempt
        elif deadline_exhausted:
            status = "deadline_exceeded"
            termination_reason = "deadline_exceeded_before_retry"
            next_attempt = attempt
        elif exhausted:
            status = "repair_exhausted"
            termination_reason = "max_patch_attempts_exhausted"
            next_attempt = attempt
        else:
            status = "retry_planned"
            termination_reason = None
            next_attempt = attempt + 1
        metrics = _metrics(state, "prepare_retry")
        retry_reasons = list(metrics.get("retry_reasons", []))
        retry_reasons.append(failure_stage)
        metrics.update(
            {
                "repair_retries": int(metrics.get("repair_retries", 0))
                + int(status == "retry_planned"),
                "rollback_count": rollback_count,
                "retry_reasons": retry_reasons,
                "max_workspace_operations": (
                    services.tools.max_active_workspace_operations
                ),
            }
        )
        if termination_reason is not None:
            metrics["termination_reason"] = termination_reason
        return {
            "repair_feedback": feedback.model_dump(mode="json"),
            "attempt_history": history,
            "candidate_verification": None,
            "patch_attempt": next_attempt,
            "patch_proposal": None,
            "patch_result": None,
            "rollback_count": rollback_count,
            "current_diff": "" if rolled_back else str(diff_result["diff"]),
            "status": status,
            "tool_trace": [
                _event(
                    "rollback_workspace",
                    str(rollback.get("status", "error")),
                    {"attempt": attempt},
                    rollback,
                    rollback_started,
                    actor="tool",
                )
            ],
            "metrics": metrics,
        }

    def route_after_retry(state: RepairState) -> str:
        return "plan" if state.get("status") == "retry_planned" else "finalize"

    async def verify_hidden(state: RepairState) -> dict[str, Any]:
        candidate = CandidateVerification.model_validate(
            state["candidate_verification"]
        )
        hidden = await _run_test_with_deadline(
            services,
            state,
            command_kind="hidden",
            expose_output=False,
        )
        hidden_audit = {
            "status": hidden.get("status"),
            "returncode": hidden.get("returncode"),
            "duration_ms": hidden.get("duration_ms"),
            "output_sha256": hidden.get("output_sha256"),
        }
        behavior = verify_behavior(
            baseline_result=dict(state.get("baseline_result") or {}),
            current_diff=str(state.get("current_diff", "")),
            changed_paths=list(candidate.changed_paths),
            legal_paths=candidate.legal_paths,
            public_after={"status": candidate.public_test_status},
            regression={"status": candidate.regression_test_status},
            hidden=hidden_audit,
        )
        proposal = PatchProposal.model_validate(state["patch_proposal"])
        submission = services.tools.submit_solution(proposal, behavior)
        submitted = submission.get("status") == "accepted"
        result = behavior.model_copy(update={"submitted": submitted})
        deadline_exhausted = (
            hidden.get("status") == "timeout"
            and time.time() >= float(state["deadline_at_epoch"])
        )
        if deadline_exhausted:
            status = "deadline_exceeded"
            termination_reason = "deadline_exceeded_during_hidden_verification"
        elif result.passed and result.submitted:
            status = "verified"
            termination_reason = None
        else:
            status = "hidden_verification_failed"
            termination_reason = "hidden_verification_failed"
        metrics = _metrics(state, "verify_hidden")
        metrics.update(
            {
                "hidden_verification_runs": int(
                    metrics.get("hidden_verification_runs", 0)
                )
                + 1,
                "verification_passed": result.passed,
                "successful_attempt": (
                    int(state.get("patch_attempt", 1))
                    if result.passed and result.submitted
                    else None
                ),
            }
        )
        if termination_reason is not None:
            metrics["termination_reason"] = termination_reason
        events = [
            _simple_event(
                "hidden_tests",
                str(hidden.get("status", "error")),
                hidden_audit,
                actor="verifier",
            )
        ]
        if submitted:
            events.append(
                _simple_event(
                    "submit_solution",
                    "accepted",
                    submission,
                    actor="verifier",
                )
            )
        return {
            "verification": result.model_dump(mode="json"),
            "status": status,
            "tool_trace": events,
            "metrics": metrics,
        }

    def route_after_hidden(state: RepairState) -> str:
        verification = state.get("verification") or {}
        if verification.get("passed") and verification.get("submitted"):
            return "commit_memory"
        return "finalize"

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
            "repair_exhausted",
            "hidden_verification_failed",
            "rollback_failed",
        }:
            final_status = current
        else:
            final_status = "failed"
        memory_assisted = bool(state.get("memory_hits", []))
        total_paused_s = float(state.get("total_paused_s", 0.0))
        existing_metrics = dict(state.get("metrics", {}))
        termination_reason = existing_metrics.get("termination_reason")
        if final_status == "completed":
            termination_reason = "completed"
        elif termination_reason is None:
            termination_reason = final_status
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
                patch_attempts=int(
                    existing_metrics.get("patch_attempts", 0)
                ),
                repair_retries=int(
                    existing_metrics.get("repair_retries", 0)
                ),
                rollback_count=int(state.get("rollback_count", 0)),
                retry_reasons=list(
                    existing_metrics.get("retry_reasons", [])
                ),
                candidate_verification_runs=int(
                    existing_metrics.get("candidate_verification_runs", 0)
                ),
                hidden_verification_runs=int(
                    existing_metrics.get("hidden_verification_runs", 0)
                ),
                successful_attempt=existing_metrics.get(
                    "successful_attempt"
                ),
                termination_reason=termination_reason,
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
        "verify_candidate": verify_candidate,
        "prepare_retry": prepare_retry,
        "verify_hidden": verify_hidden,
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
        {
            "verify_candidate": "verify_candidate",
            "prepare_retry": "prepare_retry",
            "finalize": "finalize",
        },
    )
    builder.add_conditional_edges(
        "verify_candidate",
        route_after_candidate,
        {
            "verify_hidden": "verify_hidden",
            "prepare_retry": "prepare_retry",
            "finalize": "finalize",
        },
    )
    builder.add_conditional_edges(
        "prepare_retry",
        route_after_retry,
        {"plan": "plan", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "verify_hidden",
        route_after_hidden,
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
    *,
    existing_evidence: list[dict[str, Any]],
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
    seen: set[str] = {
        _read_key(
            str(item.get("tool", "")),
            dict(item.get("arguments", {})),
        )
        for item in existing_evidence
        if _read_evidence_is_complete(item)
    }
    for task in tasks:
        key = _read_key(task.tool, task.arguments)
        if key not in seen:
            seen.add(key)
            deduplicated.append(task)
    return deduplicated[: services.config.max_read_tasks]


def _read_evidence_is_complete(evidence: dict[str, Any]) -> bool:
    """Return whether one read result should suppress the same future read.

    Successful reads and deterministic boundary errors are final for a tool
    plus arguments key. Timeouts, OS errors, and unclassified handler errors
    may be transient, so a later semantic patch attempt may schedule them
    again. The graph's patch-attempt, read-task, and deadline budgets still
    bound those retries.
    """

    status = evidence.get("status")
    if status == "success":
        return True
    if status == "error":
        result = evidence.get("result")
        error_type = (
            result.get("error_type") if isinstance(result, dict) else None
        )
        return error_type in {"RepositoryToolError", "ValueError"}
    return status != "timeout"


def _read_key(tool: str, arguments: dict[str, Any]) -> str:
    return json.dumps(
        {"tool": tool, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
    )


def _repair_feedback(state: RepairState) -> RepairFeedback | None:
    payload = state.get("repair_feedback")
    return RepairFeedback.model_validate(payload) if payload else None


def _attempt_history(state: RepairState) -> list[AttemptRecord]:
    return [
        AttemptRecord.model_validate(item)
        for item in state.get("attempt_history", [])
    ]


def _attempt_record(
    state: RepairState,
    *,
    failure_stage: Any,
    public_status: str | None,
    regression_status: str | None,
    changed_paths: list[str],
    diff_sha256: str,
) -> AttemptRecord:
    proposal = PatchProposal.model_validate(state["patch_proposal"])
    patch_result = dict(state.get("patch_result") or {})
    return AttemptRecord(
        attempt=int(state.get("patch_attempt", 1)),
        proposal_summary=proposal.summary,
        changed_paths=changed_paths,
        diff_sha256=diff_sha256,
        patch_status=str(patch_result.get("status", "unknown")),
        public_test_status=public_status,
        regression_test_status=regression_status,
        failure_stage=failure_stage,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
