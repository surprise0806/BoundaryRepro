"""SQLite checkpoint lifecycle and resume validation for repair threads."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from boundary_repro.repair.graph import (
    RepairGraphServices,
    build_repair_graph,
)
from boundary_repro.repair.memory import (
    RepairMemoryStore,
    SqliteRepairMemoryStore,
)
from boundary_repro.repair.models import (
    LoadedTask,
    RepairRunConfig,
)
from boundary_repro.repair.providers import (
    GroqRepairProvider,
    RepairProvider,
    ScriptedRepairProvider,
)
from boundary_repro.repair.tools import (
    ReadHandler,
    RepositoryRepairTools,
    prepare_workspace,
    tree_hash,
    workspace_diff,
)

_THREAD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PAUSE_NODES = {
    "load_task",
    "reproduce_failure",
    "retrieve_memory",
    "plan",
    "dispatch_read_workers",
    "aggregate",
    "patch",
    "verify_candidate",
    "prepare_retry",
    "verify_hidden",
    "commit_memory",
}
TERMINAL_STATUSES = {
    "completed",
    "provider_failed",
    "degraded",
    "baseline_not_failed",
    "patch_failed",
    "verification_failed",
    "repair_exhausted",
    "hidden_verification_failed",
    "rollback_failed",
    "deadline_exceeded",
    "failed",
}


class RepairRuntime:
    """Run or resume one generic repository task without changing identity."""

    def __init__(
        self,
        project_root: Path,
        *,
        state_dir: Path | None = None,
        provider: RepairProvider | None = None,
        config: RepairRunConfig | None = None,
        memory: RepairMemoryStore | None = None,
        read_handlers: dict[str, ReadHandler] | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.state_dir = (
            state_dir.resolve()
            if state_dir is not None
            else (self.project_root / ".boundary_state").resolve()
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_root = self.state_dir / "repair-workspaces"
        self.checkpoint_path = self.state_dir / "repair-checkpoints.sqlite"
        self.memory_path = self.state_dir / "repair-memory.sqlite"
        self.provider = provider
        self.config = config
        self.memory = memory or SqliteRepairMemoryStore(self.memory_path)
        self.read_handlers = dict(read_handlers or {})
        self.last_tools: RepositoryRepairTools | None = None
        os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

    async def run(
        self,
        task_path: Path,
        thread_id: str,
        *,
        pause_after: str | None = None,
    ) -> dict[str, Any]:
        self._validate_thread_id(thread_id)
        if self.provider is None:
            raise ValueError("run requires an explicitly selected provider")
        config = self.config or RepairRunConfig()
        if pause_after is not None and pause_after not in PAUSE_NODES:
            raise ValueError(f"Unknown pause node: {pause_after}")
        manifest = (
            task_path
            if task_path.is_absolute()
            else self.project_root / task_path
        )
        task = LoadedTask.load(manifest)
        await self.memory.setup()

        async with AsyncSqliteSaver.from_conn_string(
            str(self.checkpoint_path)
        ) as checkpointer:
            await checkpointer.setup()
            if await checkpointer.aget_tuple(
                self._runnable_config(thread_id, config)
            ):
                raise ValueError(
                    f"Thread {thread_id!r} already exists; use resume."
                )
            workspace = prepare_workspace(
                task,
                self.workspace_root / thread_id,
            )
            baseline_hash = tree_hash(task.repository_path)
            hidden_tests_hash = tree_hash(task.hidden_tests_path)
            tools = RepositoryRepairTools(
                task,
                workspace,
                read_handlers=self.read_handlers,
            )
            self.last_tools = tools
            graph = build_repair_graph(
                RepairGraphServices(
                    task=task,
                    tools=tools,
                    provider=self.provider,
                    memory=self.memory,
                    config=config,
                ),
                checkpointer=checkpointer,
                interrupt_after=(
                    [pause_after] if pause_after is not None else None
                ),
            )
            started = time.time()
            initial_state = {
                "thread_id": thread_id,
                "task_path": str(task.manifest_path),
                "task": task.spec.model_dump(mode="json"),
                "provider": self.provider.name,
                "model": self.provider.model,
                "run_config": config.model_dump(mode="json"),
                "workspace_path": str(workspace),
                "workspace_baseline_hash": baseline_hash,
                "hidden_tests_hash": hidden_tests_hash,
                "current_diff": "",
                "tool_trace": [],
                "baseline_result": None,
                "memory_hits": [],
                "plan": None,
                "read_tasks": [],
                "evidence": [],
                "patch_proposal": None,
                "patch_result": None,
                "patch_attempt": 1,
                "repair_feedback": None,
                "attempt_history": [],
                "candidate_verification": None,
                "patch_diff_history": [],
                "rollback_count": 0,
                "verification": None,
                "status": "initialized",
                "metrics": {},
                "started_at_epoch": started,
                "deadline_at_epoch": started + config.run_deadline_s,
                "paused_at_epoch": None,
                "paused_after_node": None,
                "total_paused_s": 0.0,
            }
            runnable = self._runnable_config(thread_id, config)
            await graph.ainvoke(initial_state, config=runnable)
            return await self._inspect_graph(graph, runnable)

    async def resume(self, thread_id: str) -> dict[str, Any]:
        self._validate_thread_id(thread_id)
        await self.memory.setup()
        async with AsyncSqliteSaver.from_conn_string(
            str(self.checkpoint_path)
        ) as checkpointer:
            await checkpointer.setup()
            values = await self._checkpoint_values(
                checkpointer,
                thread_id,
            )
            config = RepairRunConfig.model_validate(values["run_config"])
            if self.config is not None and self.config != config:
                raise ValueError(
                    "Resume configuration differs from the checkpoint."
                )
            provider = self.provider or provider_from_identity(
                str(values["provider"]),
                values.get("model"),
                request_timeout_s=config.llm_timeout_s,
            )
            if (
                provider.name != values["provider"]
                or provider.model != values.get("model")
            ):
                raise ValueError(
                    "Resume provider/model does not match the checkpoint; "
                    "a Groq thread cannot continue with scripted."
                )
            task, workspace = self._validate_resume_files(values)
            tools = RepositoryRepairTools(
                task,
                workspace,
                read_handlers=self.read_handlers,
            )
            self.last_tools = tools
            graph = build_repair_graph(
                RepairGraphServices(
                    task=task,
                    tools=tools,
                    provider=provider,
                    memory=self.memory,
                    config=config,
                ),
                checkpointer=checkpointer,
            )
            runnable = self._runnable_config(thread_id, config)
            snapshot = await graph.aget_state(runnable)
            if not snapshot.next:
                return await self._inspect_graph(graph, runnable)
            paused_at = values.get("paused_at_epoch")
            paused_after_node = values.get("paused_after_node")
            if isinstance(paused_at, (int, float)):
                resumed_at = time.time()
                paused_duration = max(0.0, resumed_at - float(paused_at))
                deadline_before_resume = float(values["deadline_at_epoch"])
                next_before_resume = tuple(snapshot.next)
                await graph.aupdate_state(
                    runnable,
                    {
                        "deadline_at_epoch": (
                            deadline_before_resume + paused_duration
                        ),
                        "paused_at_epoch": None,
                        "total_paused_s": (
                            float(values.get("total_paused_s", 0.0))
                            + paused_duration
                        ),
                    },
                    as_node=(
                        str(paused_after_node)
                        if paused_after_node is not None
                        else None
                    ),
                )
                snapshot = await graph.aget_state(runnable)
                if tuple(snapshot.next) != next_before_resume:
                    raise ValueError(
                        "Deadline compensation changed the checkpoint's "
                        "next node; refusing an unsafe resume."
                    )
            await graph.ainvoke(None, config=runnable)
            return await self._inspect_graph(graph, runnable)

    async def status(self, thread_id: str) -> dict[str, Any]:
        values, checkpoint_id = await self._read_latest(thread_id)
        return {
            "thread_id": thread_id,
            "status": values.get("status"),
            "paused": values.get("status") not in TERMINAL_STATUSES,
            "checkpoint_id": checkpoint_id,
            "state": values,
        }

    async def history(
        self,
        thread_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self._validate_thread_id(thread_id)
        async with AsyncSqliteSaver.from_conn_string(
            str(self.checkpoint_path)
        ) as checkpointer:
            await checkpointer.setup()
            items: list[dict[str, Any]] = []
            async for checkpoint in checkpointer.alist(
                {"configurable": {"thread_id": thread_id}},
                limit=max(1, min(limit, 1000)),
            ):
                values = dict(
                    checkpoint.checkpoint.get("channel_values", {})
                )
                items.append(
                    {
                        "checkpoint_id": checkpoint.config.get(
                            "configurable", {}
                        ).get("checkpoint_id"),
                        "step": checkpoint.metadata.get("step"),
                        "source": checkpoint.metadata.get("source"),
                        "status": values.get("status"),
                        "tool_trace_count": len(
                            values.get("tool_trace", [])
                        ),
                        "evidence_count": len(values.get("evidence", [])),
                        "current_diff_sha256": _sha(
                            str(values.get("current_diff", ""))
                        ),
                    }
                )
            if not items:
                raise ValueError(f"Unknown repair thread: {thread_id}")
            return items

    async def list_memory(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in await self.memory.list(limit=limit)
        ]

    async def search_memory(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in await self.memory.search(query, limit=limit)
        ]

    async def _read_latest(
        self,
        thread_id: str,
    ) -> tuple[dict[str, Any], str | None]:
        self._validate_thread_id(thread_id)
        async with AsyncSqliteSaver.from_conn_string(
            str(self.checkpoint_path)
        ) as checkpointer:
            await checkpointer.setup()
            checkpoint = await checkpointer.aget_tuple(
                {"configurable": {"thread_id": thread_id}}
            )
            if checkpoint is None:
                raise ValueError(f"Unknown repair thread: {thread_id}")
            return (
                dict(checkpoint.checkpoint.get("channel_values", {})),
                checkpoint.config.get("configurable", {}).get(
                    "checkpoint_id"
                ),
            )

    async def _checkpoint_values(
        self,
        checkpointer: AsyncSqliteSaver,
        thread_id: str,
    ) -> dict[str, Any]:
        checkpoint = await checkpointer.aget_tuple(
            {"configurable": {"thread_id": thread_id}}
        )
        if checkpoint is None:
            raise ValueError(f"Unknown repair thread: {thread_id}")
        return dict(checkpoint.checkpoint.get("channel_values", {}))

    def _validate_resume_files(
        self,
        values: dict[str, Any],
    ) -> tuple[LoadedTask, Path]:
        task = LoadedTask.load(Path(str(values["task_path"])))
        if task.spec.model_dump(mode="json") != values.get("task"):
            raise ValueError("TaskSpec changed after the checkpoint.")
        baseline_hash = tree_hash(task.repository_path)
        if baseline_hash != values.get("workspace_baseline_hash"):
            raise ValueError(
                "Public repository template changed after the checkpoint."
            )
        if tree_hash(task.hidden_tests_path) != values.get(
            "hidden_tests_hash"
        ):
            raise ValueError(
                "Private hidden tests changed after the checkpoint."
            )
        workspace = Path(str(values["workspace_path"])).resolve()
        expected_root = self.workspace_root.resolve()
        if (
            not workspace.is_dir()
            or expected_root not in workspace.parents
        ):
            raise ValueError("Checkpoint workspace path is missing or unsafe.")
        observed_diff = workspace_diff(task.repository_path, workspace)
        if observed_diff != values.get("current_diff", ""):
            raise ValueError(
                "Workspace diff no longer matches the checkpoint; "
                "refusing an unsafe resume."
            )
        return task, workspace

    async def _inspect_graph(
        self,
        graph: Any,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot = await graph.aget_state(config)
        history_count = 0
        async for _ in graph.aget_state_history(config):
            history_count += 1
        return {
            "thread_id": snapshot.values.get("thread_id"),
            "status": snapshot.values.get("status"),
            "paused": bool(snapshot.next),
            "next": list(snapshot.next),
            "history_count": history_count,
            "state": dict(snapshot.values),
        }

    @staticmethod
    def _runnable_config(
        thread_id: str,
        config: RepairRunConfig,
    ) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "max_concurrency": config.max_concurrency,
            "recursion_limit": 100,
        }

    @staticmethod
    def _validate_thread_id(thread_id: str) -> None:
        if not _THREAD_ID.fullmatch(thread_id):
            raise ValueError(
                "thread_id must be 1-128 characters using letters, digits, "
                "dot, underscore, or hyphen."
            )


def provider_from_identity(
    name: str,
    model: Any,
    *,
    request_timeout_s: float,
) -> RepairProvider:
    if name == "scripted":
        if model is not None:
            raise ValueError("Scripted checkpoints cannot have a model ID.")
        return ScriptedRepairProvider()
    if name == "groq":
        if not isinstance(model, str) or not model:
            raise ValueError("Groq checkpoint is missing its model ID.")
        return GroqRepairProvider(
            model=model,
            request_timeout_s=request_timeout_s,
        )
    raise ValueError(
        f"Cannot reconstruct custom provider {name!r}; pass the original "
        "provider instance to RepairRuntime.resume()."
    )


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
