from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from boundary_repro.repair.models import (
    PatchProposal,
    RepairPlan,
    RepairRunConfig,
)
from boundary_repro.repair.runtime import RepairRuntime
from boundary_repro.repair.tools import workspace_diff

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def make_task(
    root: Path,
    *,
    public_expression: str = "model.VALUE == 2",
    regression_expression: str = "model.VALUE == 2",
    hidden_expression: str = "model.VALUE == 2",
    slow_candidate: bool = False,
) -> Path:
    task_root = root / "iterative-task"
    repository = task_root / "repository"
    tests = repository / "tests"
    hidden = task_root / "hidden_tests"
    tests.mkdir(parents=True)
    hidden.mkdir(parents=True)
    (repository / "model.py").write_text("VALUE = 0\n", encoding="utf-8")
    (tests / "__init__.py").write_text("", encoding="utf-8")
    sleep_line = (
        "        if model.VALUE == 1:\n"
        "            time.sleep(2)\n"
        if slow_candidate
        else ""
    )
    imports = "import time\n" if slow_candidate else ""
    (tests / "test_public.py").write_text(
        "import unittest\n"
        f"{imports}"
        "import model\n\n"
        "class PublicTest(unittest.TestCase):\n"
        "    def test_behavior(self):\n"
        f"{sleep_line}"
        f"        self.assertTrue({public_expression})\n",
        encoding="utf-8",
    )
    (tests / "test_regression.py").write_text(
        "import unittest\n"
        "import model\n\n"
        "class RegressionTest(unittest.TestCase):\n"
        "    def test_behavior(self):\n"
        f"        self.assertTrue({regression_expression})\n",
        encoding="utf-8",
    )
    (hidden / "test_hidden.py").write_text(
        "import unittest\n"
        "import model\n\n"
        "class HiddenTest(unittest.TestCase):\n"
        "    def test_behavior(self):\n"
        f"        self.assertTrue({hidden_expression}, "
        "'private oracle rejected patch')\n",
        encoding="utf-8",
    )
    manifest = task_root / "task.json"
    manifest.write_text(
        json.dumps(
            {
                "task_id": "iterative-repair-test",
                "issue_text": "Change the public behavior to the required value.",
                "repository": "repository",
                "test_command": [
                    "{python}",
                    "-m",
                    "unittest",
                    "tests.test_public",
                    "-v",
                ],
                "regression_command": [
                    "{python}",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-v",
                ],
                "editable_paths": ["model.py"],
                "timeout": 10,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def proposal(old_text: str, value: int) -> PatchProposal:
    return PatchProposal(
        path="model.py",
        old_text=old_text,
        new_text=f"VALUE = {value}",
        root_cause="The baseline value does not meet the public contract.",
        summary=f"Set the value to {value}.",
        evidence="The public and regression tests define the behavior.",
    )


class SequenceProvider:
    name = "groq"
    model = "offline-iterative-double"

    def __init__(
        self,
        proposals: list[PatchProposal],
        *,
        template: Path | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.proposals = proposals
        self.template = template
        self.workspace = workspace
        self.plan_inputs: list[dict[str, Any]] = []
        self.patch_inputs: list[dict[str, Any]] = []
        self.diff_before_patch: list[str] = []

    async def aplan(self, **kwargs: Any) -> RepairPlan:
        self.plan_inputs.append(kwargs)
        return RepairPlan(summary="Inspect the editable source.", read_tasks=[])

    async def apatch(self, **kwargs: Any) -> PatchProposal:
        self.patch_inputs.append(kwargs)
        if self.template is not None and self.workspace is not None:
            self.diff_before_patch.append(
                workspace_diff(self.template, self.workspace)
            )
        attempt = int(kwargs["patch_attempt"])
        return self.proposals[attempt - 1]


def test_public_failure_then_second_attempt_completes(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = make_task(tmp_path)
        state_dir = tmp_path / "state"
        provider = SequenceProvider(
            [proposal("VALUE = 0", 1), proposal("VALUE = 0", 2)],
            template=manifest.parent / "repository",
            workspace=state_dir / "repair-workspaces" / "public-retry",
        )
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=state_dir,
            provider=provider,
            config=RepairRunConfig(max_patch_attempts=2),
        )
        result = await runtime.run(manifest, "public-retry")
        state = result["state"]
        metrics = state["metrics"]

        assert result["status"] == "completed"
        assert metrics["patch_attempts"] == 2
        assert metrics["repair_retries"] == 1
        assert metrics["rollback_count"] == 1
        assert metrics["successful_attempt"] == 2
        assert state["attempt_history"][0]["failure_stage"] == "public_tests"
        assert provider.patch_inputs[1]["repair_feedback"].failure_stage == (
            "public_tests"
        )
        assert provider.diff_before_patch == ["", ""]
        assert sum(
            event["name"] == "run_tests"
            and event["arguments"].get("phase") == "baseline"
            for event in state["tool_trace"]
        ) == 1
        assert len(await runtime.list_memory()) == 1
        assert metrics["evaluation_eligible"] is True
        assert runtime.last_tools is not None
        assert runtime.last_tools.max_active_workspace_operations == 1
        rollback = next(
            event["result"]
            for event in state["tool_trace"]
            if event["name"] == "rollback_workspace"
        )
        assert rollback["changed_paths_before"] == ["model.py"]
        assert rollback["changed_paths_after"] == []
        assert rollback["diff_before_sha256"] != rollback["post_diff_sha256"]

    run(scenario())


def test_regression_failure_feedback_reaches_second_attempt(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = make_task(
            tmp_path,
            public_expression="model.VALUE in {1, 2}",
        )
        state_dir = tmp_path / "state"
        provider = SequenceProvider(
            [proposal("VALUE = 0", 1), proposal("VALUE = 0", 2)],
            template=manifest.parent / "repository",
            workspace=state_dir / "repair-workspaces" / "regression-retry",
        )
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=state_dir,
            provider=provider,
            config=RepairRunConfig(max_patch_attempts=2),
        )
        result = await runtime.run(manifest, "regression-retry")

        assert result["status"] == "completed"
        feedback = provider.patch_inputs[1]["repair_feedback"]
        assert feedback.failure_stage == "regression_tests"
        assert feedback.public_test_status == "pass"
        assert feedback.regression_test_status == "fail"
        assert provider.diff_before_patch == ["", ""]

    run(scenario())


def test_patch_mismatch_is_recoverable(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = make_task(tmp_path)
        provider = SequenceProvider(
            [proposal("VALUE = 99", 1), proposal("VALUE = 0", 2)]
        )
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=provider,
            config=RepairRunConfig(max_patch_attempts=2),
        )
        result = await runtime.run(manifest, "patch-mismatch")

        assert result["status"] == "completed"
        assert result["state"]["attempt_history"][0]["failure_stage"] == (
            "patch_apply"
        )
        assert result["state"]["metrics"]["rollback_count"] == 1

    run(scenario())


def test_repeated_diff_is_no_progress_and_exhausts_budget(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = make_task(tmp_path)
        repeated = proposal("VALUE = 0", 1)
        provider = SequenceProvider([repeated, repeated])
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=provider,
            config=RepairRunConfig(max_patch_attempts=2),
        )
        result = await runtime.run(manifest, "repeated-diff")
        metrics = result["state"]["metrics"]

        assert result["status"] == "repair_exhausted"
        assert metrics["patch_attempts"] == 2
        assert metrics["candidate_verification_runs"] == 1
        assert metrics["repeated_diff_count"] == 1
        assert metrics["no_progress_count"] == 1
        assert result["state"]["attempt_history"][-1]["failure_stage"] == (
            "no_progress"
        )
        assert metrics["hidden_verification_runs"] == 0
        assert await runtime.list_memory() == []

    run(scenario())


def test_failed_candidates_exhaust_without_hidden_or_memory(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = make_task(tmp_path)
        provider = SequenceProvider(
            [proposal("VALUE = 0", 1), proposal("VALUE = 0", 3)]
        )
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=provider,
            config=RepairRunConfig(max_patch_attempts=2),
        )
        result = await runtime.run(manifest, "failed-candidates")
        state = result["state"]

        assert result["status"] == "repair_exhausted"
        assert state["metrics"]["candidate_verification_runs"] == 2
        assert state["metrics"]["hidden_verification_runs"] == 0
        assert not any(
            event["name"] == "hidden_tests" for event in state["tool_trace"]
        )
        assert await runtime.list_memory() == []

    run(scenario())


def test_hidden_failure_is_terminal_and_private_output_is_not_exposed(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = make_task(tmp_path, hidden_expression="model.VALUE == 9")
        provider = SequenceProvider([proposal("VALUE = 0", 2)])
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=provider,
        )
        result = await runtime.run(manifest, "hidden-terminal")
        serialized = json.dumps(result["state"])

        assert result["status"] == "hidden_verification_failed"
        assert len(provider.patch_inputs) == 1
        assert result["state"]["metrics"]["repair_retries"] == 0
        assert "private oracle rejected patch" not in serialized
        hidden_event = next(
            event
            for event in result["state"]["tool_trace"]
            if event["name"] == "hidden_tests"
        )
        assert set(hidden_event["result"]) == {
            "status",
            "returncode",
            "duration_ms",
            "output_sha256",
        }
        assert await runtime.list_memory() == []

    run(scenario())


def test_deadline_exhaustion_stops_before_second_provider_call(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = make_task(tmp_path, slow_candidate=True)
        provider = SequenceProvider(
            [proposal("VALUE = 0", 1), proposal("VALUE = 0", 2)]
        )
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=provider,
            config=RepairRunConfig(
                max_patch_attempts=2,
                run_deadline_s=0.5,
                tool_timeout_s=5,
            ),
        )
        result = await runtime.run(manifest, "deadline-retry")

        assert result["status"] == "deadline_exceeded"
        assert len(provider.patch_inputs) == 1

    run(scenario())


def test_pause_after_prepare_retry_resumes_without_duplicate_work(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = make_task(tmp_path)
        state_dir = tmp_path / "state"
        first_provider = SequenceProvider(
            [proposal("VALUE = 0", 1), proposal("VALUE = 0", 2)]
        )
        first_runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=state_dir,
            provider=first_provider,
            config=RepairRunConfig(max_patch_attempts=2),
        )
        paused = await first_runtime.run(
            manifest,
            "resume-retry",
            pause_after="prepare_retry",
        )
        assert paused["paused"] is True
        assert paused["next"] == ["plan"]
        assert paused["state"]["patch_attempt"] == 2
        assert paused["state"]["rollback_count"] == 1
        assert len(paused["state"]["attempt_history"]) == 1
        completed_read_workers = paused["state"]["metrics"]["node_runs"][
            "read_worker"
        ]

        second_provider = SequenceProvider(
            [proposal("VALUE = 0", 1), proposal("VALUE = 0", 2)]
        )
        second_runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=state_dir,
            provider=second_provider,
            config=RepairRunConfig(max_patch_attempts=2),
        )
        resumed = await second_runtime.resume("resume-retry")
        state = resumed["state"]

        assert resumed["status"] == "completed"
        assert [item["patch_attempt"] for item in second_provider.patch_inputs] == [2]
        assert state["metrics"]["patch_attempts"] == 2
        assert state["metrics"]["rollback_count"] == 1
        assert len(state["attempt_history"]) == 2
        assert state["metrics"]["node_runs"]["read_worker"] == (
            completed_read_workers
        )
        assert sum(
            event["name"] == "run_tests"
            and event["arguments"].get("phase") == "baseline"
            for event in state["tool_trace"]
        ) == 1

    run(scenario())


def test_rollback_failure_is_terminal_and_does_not_modify_git(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = make_task(tmp_path)
        public_test = manifest.parent / "repository/tests/test_public.py"
        public_test.write_text(
            "import pathlib\n"
            "import unittest\n"
            "import model\n\n"
            "class PublicTest(unittest.TestCase):\n"
            "    def test_behavior(self):\n"
            "        if model.VALUE == 1:\n"
            "            git_dir = pathlib.Path('.git')\n"
            "            git_dir.mkdir(exist_ok=True)\n"
            "            (git_dir / 'marker').write_text('untouched')\n"
            "        self.assertEqual(model.VALUE, 2)\n",
            encoding="utf-8",
        )
        provider = SequenceProvider([proposal("VALUE = 0", 1)])
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=provider,
            config=RepairRunConfig(max_patch_attempts=2),
        )
        result = await runtime.run(manifest, "rollback-failure")

        assert result["status"] == "rollback_failed"
        assert runtime.last_tools is not None
        marker = runtime.last_tools.workspace / ".git/marker"
        assert marker.read_text(encoding="utf-8") == "untouched"
        assert not (manifest.parent / "repository/.git").exists()
        assert result["state"]["metrics"]["rollback_count"] == 0
        assert result["state"]["metrics"]["termination_reason"] == (
            "rollback_failed"
        )
        assert await runtime.list_memory() == []

    run(scenario())


def test_timed_out_read_retries_but_successful_read_stays_deduplicated(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = make_task(tmp_path)
        read_file_calls = 0

        async def transient_read(arguments: dict[str, Any]) -> dict[str, Any]:
            nonlocal read_file_calls
            read_file_calls += 1
            if read_file_calls == 1:
                raise TimeoutError("synthetic transient read timeout")
            return {
                "status": "ok",
                "path": arguments["path"],
                "content": "VALUE = 0",
            }

        provider = SequenceProvider(
            [
                proposal("VALUE = 0", 1),
                proposal("VALUE = 0", 3),
                proposal("VALUE = 0", 2),
            ]
        )
        runtime = RepairRuntime(
            PROJECT_ROOT,
            state_dir=tmp_path / "state",
            provider=provider,
            config=RepairRunConfig(
                max_patch_attempts=3,
            ),
            read_handlers={"read_file": transient_read},
        )
        result = await runtime.run(manifest, "retry-timeout-read")
        state = result["state"]

        assert result["status"] == "completed"
        assert state["patch_attempt"] == 3
        assert read_file_calls == 2
        read_events = [
            event
            for event in state["tool_trace"]
            if event["actor"] == "read_worker"
            and event["name"] == "read_file"
        ]
        assert [event["status"] for event in read_events] == [
            "timeout",
            "success",
        ]
        list_events = [
            event
            for event in state["tool_trace"]
            if event["actor"] == "read_worker"
            and event["name"] == "list_files"
        ]
        assert len(list_events) == 1
        assert len(provider.plan_inputs) == 3
        assert any(
            item["tool"] == "read_file" and item["status"] == "timeout"
            for item in provider.plan_inputs[1]["evidence"]
        )
        assert any(
            item["tool"] == "read_file" and item["status"] == "success"
            for item in provider.plan_inputs[2]["evidence"]
        )

    run(scenario())


def test_max_patch_attempts_defaults_when_old_checkpoint_config_is_loaded() -> None:
    payload = RepairRunConfig().model_dump(mode="json")
    payload.pop("max_patch_attempts")

    restored = RepairRunConfig.model_validate(payload)

    assert restored.max_patch_attempts == 3
