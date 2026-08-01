from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import groq
import pytest

from boundary_repro.repair.models import (
    AttemptRecord,
    PatchProposal,
    ReadTask,
    RepairFeedback,
    RepairPlan,
)
from boundary_repro.repair.providers import (
    GroqRepairProvider,
    ProviderOutputError,
)
from boundary_repro.repair.runtime import RepairRuntime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLIND_TASK = (
    PROJECT_ROOT / "benchmarks" / "blind-python-dotenv-207" / "task.json"
)

VALID_PLAN = {
    "summary": "Collect source evidence before proposing a patch.",
    "read_tasks": [
        {
            "task_id": "list-root",
            "tool": "list_files",
            "arguments": {},
        },
        {
            "task_id": "search-symbol",
            "tool": "search_code",
            "arguments": {"query": "DEFAULT_ENCODING"},
        },
        {
            "task_id": "read-source",
            "tool": "read_file",
            "arguments": {"path": "dotenv_loader.py"},
        },
    ],
}

VALID_PATCH = {
    "path": "dotenv_loader.py",
    "old_text": "DEFAULT_ENCODING: str | None = None",
    "new_text": 'DEFAULT_ENCODING: str | None = "utf-8"',
    "root_cause": "The default falls back to a locale-dependent codec.",
    "summary": "Use UTF-8 by default while preserving explicit encodings.",
    "evidence": "The public failure and source read identify the default.",
}


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


class RecordingGroq:
    def __init__(self, responses: list[dict[str, Any] | str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.client_arguments: list[dict[str, Any]] = []

    def client(self, **kwargs: Any) -> Any:
        self.client_arguments.append(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=self.create),
            )
        )

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        content = (
            response if isinstance(response, str) else json.dumps(response)
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ]
        )


def install_recording_groq(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[dict[str, Any] | str],
) -> RecordingGroq:
    recorder = RecordingGroq(responses)
    monkeypatch.setattr(groq, "AsyncGroq", recorder.client)
    return recorder


def assert_groq_strict_schema(value: Any, path: str = "$") -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_groq_strict_schema(item, f"{path}[{index}]")
        return
    if not isinstance(value, dict):
        return
    if value.get("type") == "object":
        assert value.get("additionalProperties") is False, path
        properties = set(value.get("properties", {}))
        required = set(value.get("required", []))
        assert properties <= required, path
    for key, item in value.items():
        assert_groq_strict_schema(item, f"{path}.{key}")


def plan_context() -> dict[str, Any]:
    feedback = RepairFeedback(
        attempt=1,
        failure_stage="public_tests",
        summary="The first candidate failed its public assertion.",
        patch_status="applied",
        public_test_status="fail",
        regression_test_status="fail",
        public_test_output="public failure",
        regression_test_output="regression failure",
        changed_paths=["dotenv_loader.py"],
        previous_diff_sha256="a" * 64,
    )
    history = AttemptRecord(
        attempt=1,
        proposal_summary="First candidate.",
        changed_paths=["dotenv_loader.py"],
        diff_sha256="a" * 64,
        patch_status="applied",
        public_test_status="fail",
        regression_test_status="fail",
        failure_stage="public_tests",
    )
    return {
        "task": {"issue_text": "Investigate the public failure."},
        "baseline": {"status": "fail"},
        "memory_hits": [{"memory_id": "verified-memory"}],
        "patch_attempt": 2,
        "repair_feedback": feedback,
        "attempt_history": [history],
        "evidence": [{"tool": "read_file", "status": "success"}],
    }


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
)
def test_gpt_oss_plan_uses_strict_schema_without_native_tools(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    recorder = install_recording_groq(monkeypatch, [VALID_PLAN])
    provider = GroqRepairProvider(model=model, api_key="offline-test-key")
    context = plan_context()

    plan = run(provider.aplan(**context))

    assert isinstance(plan, RepairPlan)
    assert all(isinstance(task, ReadTask) for task in plan.read_tasks)
    assert [task.tool for task in plan.read_tasks] == [
        "list_files",
        "search_code",
        "read_file",
    ]
    request = recorder.calls[0]
    assert request["model"] == model
    assert "tools" not in request
    assert "tool_choice" not in request
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    strict_format = response_format["json_schema"]
    assert strict_format["name"] == "repair_plan"
    assert strict_format["strict"] is True
    assert_groq_strict_schema(strict_format["schema"])

    user_payload = json.loads(request["messages"][1]["content"])
    assert user_payload["public_task"] == context["task"]
    assert user_payload["failing_public_test"] == context["baseline"]
    assert user_payload[
        "verified_memories_for_prioritization_only"
    ] == context["memory_hits"]
    assert user_payload["patch_attempt"] == 2
    assert user_payload["repair_feedback"]["failure_stage"] == (
        "public_tests"
    )
    assert user_payload["prior_attempt_history"][0]["attempt"] == 1
    assert user_payload["existing_read_only_evidence"] == context["evidence"]
    assert "schema" not in user_payload
    assert "output_schema" not in user_payload
    assert "allowed_read_tools" not in user_payload
    constraints = " ".join(user_payload["constraints"])
    assert "No tools are callable" in constraints
    assert "LangGraph Harness" in constraints
    assert "provider-native tool/function call" in constraints


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
)
def test_gpt_oss_patch_uses_checked_patch_schema(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    recorder = install_recording_groq(monkeypatch, [VALID_PATCH])
    provider = GroqRepairProvider(model=model, api_key="offline-test-key")
    evidence = [{"tool": "read_file", "status": "success"}]

    proposal = run(
        provider.apatch(
            task={"issue_text": "Fix the public failure."},
            baseline={"status": "fail"},
            evidence=evidence,
            memory_hits=[],
            patch_attempt=2,
            repair_feedback=plan_context()["repair_feedback"],
            attempt_history=plan_context()["attempt_history"],
        )
    )

    assert proposal == PatchProposal.model_validate(VALID_PATCH)
    request = recorder.calls[0]
    assert "tools" not in request
    assert "tool_choice" not in request
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    strict_format = response_format["json_schema"]
    assert strict_format["name"] == "patch_proposal"
    assert strict_format["strict"] is True
    assert strict_format["schema"] == PatchProposal.model_json_schema()
    assert_groq_strict_schema(strict_format["schema"])

    user_payload = json.loads(request["messages"][1]["content"])
    assert user_payload["read_only_evidence"] == evidence
    assert user_payload["patch_attempt"] == 2
    assert user_payload["repair_feedback"]["failure_stage"] == (
        "public_tests"
    )
    assert user_payload["prior_attempt_history"][0]["attempt"] == 1
    assert "schema" not in user_payload
    assert "output_schema" not in user_payload
    assert "No tools are callable" in " ".join(user_payload["constraints"])


@pytest.mark.parametrize(
    "bad_task",
    [
        {
            "task_id": "illegal-tool",
            "tool": "repo_browser.list_files",
            "arguments": {},
        },
        {
            "task_id": "extra-argument",
            "tool": "list_files",
            "arguments": {"depth": 2},
        },
        {
            "task_id": "missing-query",
            "tool": "search_code",
            "arguments": {},
        },
        {
            "task_id": "missing-arguments",
            "tool": "read_file",
        },
        {
            "task_id": "extra-read-argument",
            "tool": "read_file",
            "arguments": {"path": "file.py", "line": 10},
        },
    ],
)
def test_plan_boundary_rejects_illegal_tools_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
    bad_task: dict[str, Any],
) -> None:
    response = {"summary": "invalid plan", "read_tasks": [bad_task]}
    install_recording_groq(monkeypatch, [response])
    provider = GroqRepairProvider(
        model="openai/gpt-oss-120b",
        api_key="offline-test-key",
    )

    with pytest.raises(ProviderOutputError, match="invalid Groq repair plan"):
        run(provider.aplan(**plan_context()))


def test_non_gpt_oss_model_keeps_explicit_json_object_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = install_recording_groq(monkeypatch, [VALID_PLAN])
    provider = GroqRepairProvider(
        model="llama-3.1-8b-instant",
        api_key="offline-test-key",
    )

    plan = run(provider.aplan(**plan_context()))

    assert isinstance(plan, RepairPlan)
    request = recorder.calls[0]
    assert request["response_format"] == {"type": "json_object"}
    assert "tools" not in request
    user_payload = json.loads(request["messages"][1]["content"])
    assert "output_schema" in user_payload
    assert_groq_strict_schema(user_payload["output_schema"])


def test_groq_schema_failure_stops_before_patch_verify_or_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorder = install_recording_groq(
        monkeypatch,
        [
            {
                "summary": "invalid provider response",
                "read_tasks": [
                    {
                        "task_id": "native-tool-shape",
                        "tool": "repo_browser.list_files",
                        "arguments": {"path": "", "depth": 2},
                    }
                ],
            }
        ],
    )
    provider = GroqRepairProvider(
        model="openai/gpt-oss-120b",
        api_key="offline-test-key",
    )
    runtime = RepairRuntime(
        PROJECT_ROOT,
        state_dir=tmp_path / "state",
        provider=provider,
    )

    result = run(runtime.run(BLIND_TASK, "groq-schema-failure"))

    assert result["status"] == "provider_failed"
    assert result["state"]["provider"] == "groq"
    assert result["state"]["current_diff"] == ""
    assert result["state"]["patch_proposal"] is None
    assert result["state"]["patch_result"] is None
    assert result["state"]["verification"] is None
    assert "patch" not in result["state"]["metrics"]["node_runs"]
    assert "verify" not in result["state"]["metrics"]["node_runs"]
    assert len(recorder.calls) == 1
    assert not any(
        event["name"] in {"apply_patch", "memory_write"}
        for event in result["state"]["tool_trace"]
    )
    assert run(runtime.list_memory()) == []


def test_groq_patch_schema_failure_does_not_apply_or_verify(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorder = install_recording_groq(
        monkeypatch,
        [
            VALID_PLAN,
            {
                "path": "dotenv_loader.py",
                "old_text": "DEFAULT_ENCODING: str | None = None",
                "new_text": 'DEFAULT_ENCODING: str | None = "utf-8"',
                "root_cause": "missing required report fields",
            },
        ],
    )
    provider = GroqRepairProvider(
        model="openai/gpt-oss-120b",
        api_key="offline-test-key",
    )
    runtime = RepairRuntime(
        PROJECT_ROOT,
        state_dir=tmp_path / "state",
        provider=provider,
    )

    result = run(runtime.run(BLIND_TASK, "groq-patch-schema-failure"))

    assert result["status"] == "provider_failed"
    assert result["state"]["plan"] is not None
    assert result["state"]["current_diff"] == ""
    assert result["state"]["patch_proposal"] is None
    assert result["state"]["patch_result"] is None
    assert result["state"]["verification"] is None
    assert result["state"]["metrics"]["node_runs"]["patch"] == 1
    assert "verify" not in result["state"]["metrics"]["node_runs"]
    assert len(recorder.calls) == 2
    assert not any(
        event["name"] in {"apply_patch", "memory_write"}
        for event in result["state"]["tool_trace"]
    )
    assert run(runtime.list_memory()) == []


def test_groq_provider_default_model_is_gpt_oss_120b() -> None:
    provider = GroqRepairProvider(api_key="offline-test-key")
    assert provider.model == "openai/gpt-oss-120b"
