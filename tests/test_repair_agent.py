from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from boundary_repro.repair.models import (
    LoadedTask,
    PatchProposal,
    RepairPlan,
    RepairRunConfig,
    TaskSpec,
    VerificationResult,
)
from boundary_repro.repair.cli import build_parser
from boundary_repro.repair.providers import (
    ProviderOutputError,
    ScriptedRepairProvider,
    TransientProviderError,
    retry_with_deadline,
)
from boundary_repro.repair.runtime import RepairRuntime
from boundary_repro.repair.tools import (
    RepositoryRepairTools,
    prepare_workspace,
    repair_tool_schemas,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLIND_TASK_ROOT = PROJECT_ROOT / "benchmarks" / "blind-python-dotenv-207"
BLIND_TASK = BLIND_TASK_ROOT / "task.json"


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def copy_task(tmp_path: Path, *, task_id: str = "never-registered-task") -> Path:
    destination = tmp_path / "blind-task"
    shutil.copytree(BLIND_TASK_ROOT, destination)
    manifest = destination / "task.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["task_id"] = task_id
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


class AlwaysBrokenProvider:
    name = "always-broken"
    model = "broken-v1"

    def __init__(self) -> None:
        self.calls = 0

    async def aplan(self, **_: Any) -> RepairPlan:
        self.calls += 1
        raise ProviderOutputError("synthetic malformed provider output")

    async def apatch(self, **_: Any) -> PatchProposal:
        raise AssertionError("patch must not run after provider plan failure")


class GroqIdentityScriptedProvider(ScriptedRepairProvider):
    name = "groq"
    model = "fake-groq-model"

    def __init__(self) -> None:
        self.memory_inputs: list[list[dict[str, Any]]] = []

    async def aplan(self, **kwargs: Any) -> RepairPlan:
        self.memory_inputs.append(list(kwargs["memory_hits"]))
        return await super().aplan(**kwargs)

    async def apatch(self, **kwargs: Any) -> PatchProposal:
        self.memory_inputs.append(list(kwargs["memory_hits"]))
        return await super().apatch(**kwargs)


class RecordingScriptedProvider(GroqIdentityScriptedProvider):
    name = "scripted"
    model = None


class SlowGroqIdentityProvider(GroqIdentityScriptedProvider):
    async def aplan(self, **kwargs: Any) -> RepairPlan:
        await asyncio.sleep(1)
        return await super().aplan(**kwargs)


def test_always_broken_provider_cannot_complete_or_write_memory(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        provider = AlwaysBrokenProvider()
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=provider,
            config=RepairRunConfig(max_retries=5),
        )
        result = await runtime.run(BLIND_TASK, "broken-provider")

        assert result["status"] == "provider_failed"
        assert provider.calls == 1
        assert result["state"]["verification"] is None
        assert await runtime.list_memory() == []
        assert not any(
            event["name"] == "memory_write"
            for event in result["state"]["tool_trace"]
        )

    run(scenario())


def test_unregistered_taskspec_runs_without_graph_registration(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = copy_task(tmp_path)
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=ScriptedRepairProvider(),
            config=RepairRunConfig(),
        )
        result = await runtime.run(manifest, "generic-task")

        assert result["status"] == "completed"
        assert result["state"]["task"]["task_id"] == "never-registered-task"
        assert result["state"]["verification"]["passed"] is True
        assert result["state"]["metrics"]["evaluation_eligible"] is False
        assert len(await runtime.list_memory()) == 1

    run(scenario())


def test_scripted_memory_is_not_injected_into_groq_identity_provider(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        source = copy_task(
            tmp_path / "source",
            task_id="scripted-memory-source",
        )
        target = copy_task(
            tmp_path / "target",
            task_id="groq-memory-target",
        )
        state_dir = tmp_path / "state"
        scripted = RepairRuntime(
            PROJECT_ROOT,
            state_dir=state_dir,
            provider=ScriptedRepairProvider(),
        )
        seeded = await scripted.run(source, "scripted-memory-seed")
        assert seeded["status"] == "completed"

        provider = GroqIdentityScriptedProvider()
        real_identity = RepairRuntime(
            PROJECT_ROOT,
            state_dir=state_dir,
            provider=provider,
        )
        result = await real_identity.run(target, "groq-memory-isolation")

        assert result["status"] == "completed"
        assert result["state"]["memory_hits"] == []
        assert provider.memory_inputs == [[], []]
        assert result["state"]["metrics"]["evaluation_eligible"] is True

    run(scenario())


def test_same_task_memory_is_not_reinjected(tmp_path: Path) -> None:
    async def scenario() -> None:
        state_dir = tmp_path / "state"
        seeded = RepairRuntime(
            PROJECT_ROOT,
            state_dir=state_dir,
            provider=ScriptedRepairProvider(),
        )
        first = await seeded.run(BLIND_TASK, "same-task-source")
        assert first["status"] == "completed"

        provider = RecordingScriptedProvider()
        repeated = RepairRuntime(
            PROJECT_ROOT,
            state_dir=state_dir,
            provider=provider,
        )
        result = await repeated.run(BLIND_TASK, "same-task-target")

        assert result["status"] == "completed"
        assert result["state"]["memory_hits"] == []
        assert provider.memory_inputs == [[], []]

    run(scenario())


def test_memory_assisted_real_provider_run_is_not_evaluation_eligible(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        source = copy_task(
            tmp_path / "source",
            task_id="real-memory-source",
        )
        target = copy_task(
            tmp_path / "target",
            task_id="real-memory-target",
        )
        state_dir = tmp_path / "state"
        source_provider = GroqIdentityScriptedProvider()
        source_runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=state_dir,
            provider=source_provider,
        )
        clean = await source_runtime.run(source, "real-memory-seed")
        assert clean["status"] == "completed"
        assert clean["state"]["metrics"]["evaluation_eligible"] is True

        target_provider = GroqIdentityScriptedProvider()
        target_runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=state_dir,
            provider=target_provider,
        )
        assisted = await target_runtime.run(target, "real-memory-assisted")

        assert assisted["status"] == "completed"
        hits = assisted["state"]["memory_hits"]
        assert len(hits) == 1
        assert hits[0]["source_provider"] == "groq"
        assert hits[0]["source_task_id"] == "real-memory-source"
        assert assisted["state"]["metrics"]["memory_assisted"] is True
        assert (
            assisted["state"]["metrics"]["evaluation_eligible"] is False
        )
        memory_events = [
            event
            for event in assisted["state"]["tool_trace"]
            if event["name"] == "memory_search"
        ]
        assert memory_events[-1]["result"]["memories"] == [
            {
                "memory_id": hits[0]["memory_id"],
                "source_provider": "groq",
                "source_task_id": "real-memory-source",
            }
        ]
        assert assisted["state"]["metrics"]["memory_sources"] == (
            memory_events[-1]["result"]["memories"]
        )

    run(scenario())


def test_clean_real_provider_identity_run_is_evaluation_eligible(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        provider = GroqIdentityScriptedProvider()
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=provider,
        )
        result = await runtime.run(BLIND_TASK, "clean-real-identity")

        assert result["status"] == "completed"
        assert result["state"]["memory_hits"] == []
        assert result["state"]["metrics"]["memory_assisted"] is False
        assert result["state"]["metrics"]["evaluation_eligible"] is True

    run(scenario())


def test_pause_resume_does_not_rerun_completed_tools(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=ScriptedRepairProvider(),
            config=RepairRunConfig(),
        )
        paused = await runtime.run(
            BLIND_TASK,
            "resume-repair",
            pause_after="aggregate",
        )
        before_trace = list(paused["state"]["tool_trace"])
        before_runs = dict(paused["state"]["metrics"]["node_runs"])

        assert paused["paused"] is True
        assert paused["next"] == ["patch"]
        assert before_runs["reproduce_failure"] == 1
        assert before_runs["aggregate"] == 1
        assert paused["state"]["provider"] == "scripted"
        assert paused["state"]["model"] is None
        assert paused["state"]["run_config"]
        assert paused["state"]["workspace_path"]
        assert paused["state"]["workspace_baseline_hash"]
        assert paused["state"]["current_diff"] == ""

        completed = await runtime.resume("resume-repair")

        assert completed["status"] == "completed"
        after_runs = completed["state"]["metrics"]["node_runs"]
        for node, count in before_runs.items():
            assert after_runs[node] == count
        after_read_events = [
            item
            for item in completed["state"]["tool_trace"]
            if item["actor"] == "read_worker"
        ]
        before_read_events = [
            item
            for item in before_trace
            if item["actor"] == "read_worker"
        ]
        assert len(after_read_events) == len(before_read_events)
        assert completed["state"]["current_diff"]

    run(scenario())


def test_pause_time_does_not_consume_deadline_or_rerun_nodes(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        config = RepairRunConfig(run_deadline_s=3.0)
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=ScriptedRepairProvider(),
            config=config,
        )
        paused = await runtime.run(
            BLIND_TASK,
            "deadline-pause-resume",
            pause_after="aggregate",
        )
        assert paused["paused"] is True
        paused_state = paused["state"]
        paused_at = paused_state["paused_at_epoch"]
        deadline_before = paused_state["deadline_at_epoch"]
        runs_before = dict(paused_state["metrics"]["node_runs"])
        assert isinstance(paused_at, float)
        assert paused_state["paused_after_node"] == "aggregate"

        await asyncio.sleep(config.run_deadline_s + 0.05)
        assert time.time() > deadline_before
        completed = await runtime.resume("deadline-pause-resume")

        assert completed["status"] == "completed"
        completed_state = completed["state"]
        assert completed_state["paused_at_epoch"] is None
        assert completed_state["total_paused_s"] >= config.run_deadline_s
        assert completed_state["deadline_at_epoch"] > deadline_before
        runs_after = completed_state["metrics"]["node_runs"]
        for node in (
            "load_task",
            "reproduce_failure",
            "read_worker",
            "aggregate",
        ):
            assert runs_after[node] == runs_before[node]

    run(scenario())


def test_active_provider_deadline_is_not_provider_failure(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=SlowGroqIdentityProvider(),
            config=RepairRunConfig(
                run_deadline_s=0.5,
                llm_timeout_s=2,
                max_retries=0,
            ),
        )
        result = await runtime.run(BLIND_TASK, "active-deadline")

        assert result["status"] == "deadline_exceeded"
        assert result["status"] != "provider_failed"
        assert await runtime.list_memory() == []

    run(scenario())


def test_per_attempt_provider_timeout_is_not_deadline_exceeded(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=SlowGroqIdentityProvider(),
            config=RepairRunConfig(
                run_deadline_s=3,
                llm_timeout_s=0.05,
                max_retries=0,
            ),
        )
        result = await runtime.run(BLIND_TASK, "per-attempt-timeout")

        assert result["status"] == "provider_failed"
        assert result["status"] != "deadline_exceeded"
        assert await runtime.list_memory() == []

    run(scenario())


def test_read_workers_really_overlap_with_send_and_semaphore(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        active = 0
        maximum = 0
        release = asyncio.Event()

        async def overlap(arguments: dict[str, Any]) -> dict[str, Any]:
            nonlocal active, maximum
            del arguments
            active += 1
            maximum = max(maximum, active)
            if active == 3:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=1)
            await asyncio.sleep(0.01)
            active -= 1
            return {"status": "ok"}

        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=ScriptedRepairProvider(),
            config=RepairRunConfig(max_concurrency=3),
            read_handlers={
                "list_files": overlap,
                "search_code": overlap,
                "read_file": overlap,
            },
        )
        paused = await runtime.run(
            BLIND_TASK,
            "parallel-reads",
            pause_after="aggregate",
        )

        assert maximum == 3
        assert (
            paused["state"]["metrics"]["max_active_read_workers"] == 3
        )
        assert max(
            item["active_read_workers"]
            for item in paused["state"]["evidence"]
        ) == 3

    run(scenario())


def test_workspace_patch_and_tests_share_one_serial_lock(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        task = LoadedTask.load(BLIND_TASK)
        workspace = prepare_workspace(task, tmp_path / "workspace")
        tools = RepositoryRepairTools(task, workspace)
        proposal = PatchProposal(
            path="dotenv_loader.py",
            old_text="DEFAULT_ENCODING: str | None = None",
            new_text='DEFAULT_ENCODING: str | None = "utf-8"',
            root_cause="locale-dependent default",
            summary="set a deterministic UTF-8 default",
            evidence="public test failure",
        )
        patch_result, test_result = await asyncio.gather(
            tools.apply_patch(proposal, baseline_failed=True),
            tools.run_tests(
                command_kind="public",
                timeout_s=10,
            ),
        )

        assert patch_result["status"] == "applied"
        assert test_result["status"] in {"fail", "pass"}
        assert tools.max_active_workspace_operations == 1

    run(scenario())


def test_baseline_that_does_not_fail_blocks_submission(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = copy_task(tmp_path, task_id="already-passing-task")
        public_test = (
            manifest.parent
            / "repository"
            / "tests"
            / "test_public.py"
        )
        public_test.write_text(
            "import unittest\n\n"
            "class Passing(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=ScriptedRepairProvider(),
            config=RepairRunConfig(),
        )
        result = await runtime.run(manifest, "baseline-pass")

        assert result["status"] == "baseline_not_failed"
        assert result["state"]["patch_proposal"] is None
        assert not any(
            event["name"] == "submit_solution"
            for event in result["state"]["tool_trace"]
        )
        assert await runtime.list_memory() == []

    run(scenario())


def test_test_paths_and_unauthorized_paths_are_rejected(
    tmp_path: Path,
) -> None:
    payload = json.loads(BLIND_TASK.read_text(encoding="utf-8"))
    payload["editable_paths"] = ["tests/test_public.py"]
    with pytest.raises(ValueError, match="test"):
        TaskSpec.model_validate(payload)

    async def scenario() -> None:
        task = LoadedTask.load(BLIND_TASK)
        workspace = prepare_workspace(task, tmp_path / "workspace")
        tools = RepositoryRepairTools(task, workspace)
        proposal = PatchProposal(
            path="tests/test_public.py",
            old_text="import unittest",
            new_text="import unittest  # changed",
            root_cause="bad",
            summary="bad",
            evidence="bad",
        )
        result = await tools.apply_patch(
            proposal,
            baseline_failed=True,
        )
        assert result["status"] == "blocked"

        traversal = PatchProposal(
            path="README.md",
            old_text="x",
            new_text="y",
            root_cause="bad",
            summary="bad",
            evidence="bad",
        )
        outside = await tools.apply_patch(
            traversal,
            baseline_failed=True,
        )
        assert outside["status"] == "blocked"

    run(scenario())


def test_hidden_test_failure_prevents_memory_write(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = copy_task(tmp_path, task_id="hidden-failure-task")
        hidden = manifest.parent / "hidden_tests" / "test_hidden.py"
        hidden.write_text(
            "import unittest\n\n"
            "class HiddenFailure(unittest.TestCase):\n"
            "    def test_private_failure(self):\n"
            "        self.fail('private oracle rejected patch')\n",
            encoding="utf-8",
        )
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=ScriptedRepairProvider(),
            config=RepairRunConfig(),
        )
        result = await runtime.run(manifest, "hidden-fails")

        assert result["status"] == "hidden_verification_failed"
        verification = result["state"]["verification"]
        assert verification["hidden_tests_passed"] is False
        assert verification["passed"] is False
        assert await runtime.list_memory() == []
        hidden_events = [
            event
            for event in result["state"]["tool_trace"]
            if event["name"] == "hidden_tests"
        ]
        assert hidden_events
        assert "output" not in hidden_events[-1]["result"]

    run(scenario())


def test_all_behavioral_checks_pass_before_memory_write(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=ScriptedRepairProvider(),
            config=RepairRunConfig(),
        )
        result = await runtime.run(BLIND_TASK, "all-checks")
        verification = VerificationResult.model_validate(
            result["state"]["verification"]
        )

        assert result["status"] == "completed"
        assert verification.passed
        assert verification.baseline_failed
        assert verification.nonempty_diff
        assert verification.legal_paths
        assert verification.public_tests_passed
        assert verification.regression_passed
        assert verification.hidden_tests_passed
        assert verification.submitted
        records = await runtime.list_memory()
        assert len(records) == 1
        assert records[0]["source_thread_id"] == "all-checks"

    run(scenario())


def test_deadline_covers_every_retry_and_backoff() -> None:
    async def scenario() -> None:
        calls = 0

        async def always_transient() -> None:
            nonlocal calls
            calls += 1
            raise TransientProviderError("temporary outage")

        started = time.perf_counter()
        with pytest.raises(Exception) as captured:
            await retry_with_deadline(
                always_transient,
                operation_name="deadline-test",
                absolute_deadline=time.time() + 0.08,
                per_attempt_timeout_s=1,
                max_retries=20,
                initial_delay_s=0.05,
            )
        elapsed = time.perf_counter() - started

        assert elapsed < 0.11
        assert calls <= 2
        audit = captured.value.audit
        assert any(item["status"] == "backoff" for item in audit)
        assert any(
            item["status"] == "backoff_skipped_deadline"
            for item in audit
        )

    run(scenario())


def test_resume_rejects_provider_model_switch_and_workspace_tamper(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        initial = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=GroqIdentityScriptedProvider(),
            config=RepairRunConfig(),
        )
        paused = await initial.run(
            BLIND_TASK,
            "identity-check",
            pause_after="aggregate",
        )
        switched = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=ScriptedRepairProvider(),
            config=RepairRunConfig(),
        )
        with pytest.raises(ValueError, match="provider/model"):
            await switched.resume("identity-check")

        workspace = Path(paused["state"]["workspace_path"])
        source = workspace / "dotenv_loader.py"
        source.write_text(
            source.read_text(encoding="utf-8") + "\n# tampered\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="diff"):
            await initial.resume("identity-check")

    run(scenario())


def test_taskspec_forbids_answer_bearing_extra_fields() -> None:
    payload = json.loads(BLIND_TASK.read_text(encoding="utf-8"))
    payload["fix_url"] = "https://example.invalid/fix"
    with pytest.raises(ValueError):
        TaskSpec.model_validate(payload)


def test_public_tool_surface_is_exactly_generic() -> None:
    names = [
        item["function"]["name"]
        for item in repair_tool_schemas()
    ]
    assert names == [
        "read_issue",
        "list_files",
        "search_code",
        "read_file",
        "run_tests",
        "apply_patch",
        "show_diff",
        "submit_solution",
    ]


def test_minimal_run_cli_defaults_to_real_provider_not_scripted() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--task",
            "task.json",
            "--thread-id",
            "minimal-cli",
        ]
    )
    assert args.brain == "groq"
    assert args.model == "openai/gpt-oss-120b"
    assert args.max_patch_attempts == 3
