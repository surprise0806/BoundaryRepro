"""Strict public and internal contracts for repository repair runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_TASK_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
_TEST_PATH_PARTS = {
    "test",
    "tests",
    "testing",
    "hidden",
    "hidden_tests",
}

ReadToolName = Literal["list_files", "search_code", "read_file"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskSpec(StrictModel):
    """The complete public task schema; no answer-bearing fields are allowed."""

    task_id: str
    issue_text: str
    repository: str
    test_command: list[str]
    regression_command: list[str]
    editable_paths: list[str]
    timeout: int = Field(ge=1, le=300)

    @field_validator("task_id")
    @classmethod
    def valid_task_id(cls, value: str) -> str:
        if not _TASK_ID.fullmatch(value):
            raise ValueError("task_id must be a lowercase kebab-case ID")
        return value

    @field_validator("issue_text")
    @classmethod
    def nonempty_issue(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("issue_text must be non-empty")
        return value

    @field_validator("repository")
    @classmethod
    def safe_repository(cls, value: str) -> str:
        return normalize_relative(value)

    @field_validator("test_command", "regression_command")
    @classmethod
    def safe_command(cls, value: list[str]) -> list[str]:
        if (
            not value
            or not all(isinstance(item, str) and item for item in value)
            or value[0] != "{python}"
        ):
            raise ValueError(
                "commands must be non-empty argv lists starting with {python}"
            )
        return value

    @field_validator("editable_paths")
    @classmethod
    def safe_editable_paths(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("editable_paths must not be empty")
        normalized = [normalize_relative(item) for item in value]
        for item in normalized:
            parts = [part.casefold() for part in PurePosixPath(item).parts]
            if any(
                part in _TEST_PATH_PARTS
                or part.startswith("test_")
                or part.endswith("_test.py")
                for part in parts
            ):
                raise ValueError(
                    f"tests and hidden-test paths cannot be editable: {item}"
                )
        return list(dict.fromkeys(normalized))


@dataclass(frozen=True)
class LoadedTask:
    """Trusted paths kept outside the public TaskSpec payload."""

    spec: TaskSpec
    manifest_path: Path
    repository_path: Path
    hidden_tests_path: Path

    @classmethod
    def load(cls, manifest_path: Path) -> "LoadedTask":
        manifest = manifest_path.resolve()
        if not manifest.is_file():
            raise ValueError(f"Task manifest does not exist: {manifest}")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            spec = TaskSpec.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid repair task: {exc}") from exc
        repository = resolve_beneath(manifest.parent, spec.repository)
        if not repository.is_dir():
            raise ValueError(f"Task repository does not exist: {repository}")
        hidden = (manifest.parent / "hidden_tests").resolve()
        if (
            not hidden.is_dir()
            or manifest.parent.resolve() not in hidden.parents
            or any(path.is_symlink() for path in hidden.rglob("*"))
        ):
            raise ValueError(
                "A repair task requires a private hidden_tests directory "
                "beside task.json."
            )
        if any(path.is_symlink() for path in repository.rglob("*")):
            raise ValueError("Task repositories may not contain symlinks")
        return cls(
            spec=spec,
            manifest_path=manifest,
            repository_path=repository,
            hidden_tests_path=hidden,
        )


class RepairRunConfig(StrictModel):
    tool_timeout_s: float = Field(default=20, gt=0, le=300)
    llm_timeout_s: float = Field(default=60, gt=0, le=300)
    node_timeout_s: float = Field(default=90, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=10)
    retry_initial_delay_s: float = Field(default=0.1, ge=0, le=30)
    max_concurrency: int = Field(default=3, ge=1, le=16)
    run_deadline_s: float = Field(default=180, gt=0, le=3600)
    max_read_tasks: int = Field(default=12, ge=1, le=50)


class ReadTask(StrictModel):
    task_id: str
    tool: ReadToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class RepairPlan(StrictModel):
    summary: str
    read_tasks: list[ReadTask]


class ListFilesArguments(StrictModel):
    """The only valid argument object for a planned list_files read."""


class SearchCodeArguments(StrictModel):
    """The only valid argument object for a planned search_code read."""

    query: str


class ReadFileArguments(StrictModel):
    """The only valid argument object for a planned read_file read."""

    path: str


class ListFilesReadTask(StrictModel):
    """Strict provider-boundary representation of a list_files request."""

    task_id: str
    tool: Literal["list_files"]
    arguments: ListFilesArguments


class SearchCodeReadTask(StrictModel):
    """Strict provider-boundary representation of a search_code request."""

    task_id: str
    tool: Literal["search_code"]
    arguments: SearchCodeArguments


class ReadFileReadTask(StrictModel):
    """Strict provider-boundary representation of a read_file request."""

    task_id: str
    tool: Literal["read_file"]
    arguments: ReadFileArguments


class StrictRepairPlanOutput(StrictModel):
    """Groq-compatible plan schema before conversion to RepairPlan."""

    summary: str
    read_tasks: list[
        ListFilesReadTask | SearchCodeReadTask | ReadFileReadTask
    ]


class PatchProposal(StrictModel):
    path: str
    old_text: str
    new_text: str
    root_cause: str
    summary: str
    evidence: str

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return normalize_relative(value)

    @field_validator(
        "old_text",
        "root_cause",
        "summary",
        "evidence",
    )
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("patch/report fields must be non-empty")
        return value


class VerificationResult(StrictModel):
    passed: bool
    baseline_failed: bool
    nonempty_diff: bool
    legal_paths: bool
    public_tests_passed: bool
    regression_passed: bool
    hidden_tests_passed: bool
    submitted: bool
    reasons: list[str] = Field(default_factory=list)


class RepairMemoryRecord(StrictModel):
    memory_id: str
    task_id: str
    issue_summary: str
    verified_root_cause: str
    patch_summary: str
    changed_paths: list[str]
    diff_sha256: str
    evidence: str
    provider: str
    model: str | None
    source_thread_id: str
    created_at: str
    content_hash: str

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        issue_summary: str,
        verified_root_cause: str,
        patch_summary: str,
        changed_paths: list[str],
        current_diff: str,
        evidence: str,
        provider: str,
        model: str | None,
        source_thread_id: str,
    ) -> "RepairMemoryRecord":
        canonical = json.dumps(
            {
                "task_id": task_id,
                "issue_summary": issue_summary,
                "verified_root_cause": verified_root_cause,
                "patch_summary": patch_summary,
                "changed_paths": sorted(changed_paths),
                "diff_sha256": hashlib.sha256(
                    current_diff.encode("utf-8")
                ).hexdigest(),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(
            memory_id=f"repair-mem-{content_hash[:20]}",
            task_id=task_id,
            issue_summary=issue_summary,
            verified_root_cause=verified_root_cause,
            patch_summary=patch_summary,
            changed_paths=sorted(changed_paths),
            diff_sha256=hashlib.sha256(
                current_diff.encode("utf-8")
            ).hexdigest(),
            evidence=evidence,
            provider=provider,
            model=model,
            source_thread_id=source_thread_id,
            created_at=utc_now(),
            content_hash=content_hash,
        )


def normalize_relative(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty relative path")
    candidate = PurePosixPath(value.replace("\\", "/"))
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or ".git" in candidate.parts
    ):
        raise ValueError(f"unsafe relative path: {value}")
    normalized = candidate.as_posix()
    if normalized in {"", "."}:
        raise ValueError("path must identify a repository entry")
    return normalized


def resolve_beneath(root: Path, value: str) -> Path:
    normalized = normalize_relative(value)
    resolved_root = root.resolve()
    target = (resolved_root / normalized).resolve()
    if target == resolved_root or resolved_root not in target.parents:
        raise ValueError(f"path escapes the allowed root: {value}")
    return target
