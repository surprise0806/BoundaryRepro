"""Generic repository tools shared by every repair task."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import Any

from boundary_repro.repair.models import (
    LoadedTask,
    PatchProposal,
    VerificationResult,
    normalize_relative,
    resolve_beneath,
)

_SKIPPED_DIRECTORIES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
_MAX_FILE_BYTES = 512_000
_MAX_PATCH_BYTES = 40_000
_MAX_TOOL_OUTPUT = 16_000
_MAX_DIFF_BYTES = 200_000

ReadHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class RepositoryToolError(RuntimeError):
    """Raised for deterministic, unsafe, or malformed tool requests."""


def prepare_workspace(task: LoadedTask, destination: Path) -> Path:
    """Copy only the public repository snapshot into a fresh workspace."""

    target = destination.resolve()
    if target.exists():
        raise ValueError(f"Workspace already exists: {target}")
    if target == task.repository_path or task.repository_path in target.parents:
        raise ValueError("Workspace cannot be inside the task template")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        task.repository_path,
        target,
        ignore=shutil.ignore_patterns(*_SKIPPED_DIRECTORIES),
    )
    return target


def tree_hash(root: Path) -> str:
    """Hash stable public repository contents, excluding generated paths."""

    digest = hashlib.sha256()
    for path in _repository_files(root):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def workspace_diff(template: Path, workspace: Path) -> str:
    """Produce one deterministic unified diff against the clean template."""

    template_files = {
        path.relative_to(template).as_posix(): path
        for path in _repository_files(template)
    }
    workspace_files = {
        path.relative_to(workspace).as_posix(): path
        for path in _repository_files(workspace)
    }
    chunks: list[str] = []
    for relative in sorted(template_files.keys() | workspace_files.keys()):
        before_path = template_files.get(relative)
        after_path = workspace_files.get(relative)
        before = _read_diff_text(before_path)
        after = _read_diff_text(after_path)
        if before == after:
            continue
        chunks.append(
            "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                )
            )
        )
    result = "".join(chunks)
    if len(result.encode("utf-8")) > _MAX_DIFF_BYTES:
        raise RepositoryToolError("workspace diff exceeds the audit limit")
    return result


def changed_paths(template: Path, workspace: Path) -> list[str]:
    template_files = {
        path.relative_to(template).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in _repository_files(template)
    }
    workspace_files = {
        path.relative_to(workspace).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in _repository_files(workspace)
    }
    return sorted(
        relative
        for relative in template_files.keys() | workspace_files.keys()
        if template_files.get(relative) != workspace_files.get(relative)
    )


class RepositoryRepairTools:
    """Eight generic tools plus private regression/hidden-test execution."""

    def __init__(
        self,
        task: LoadedTask,
        workspace: Path,
        *,
        read_handlers: dict[str, ReadHandler] | None = None,
    ) -> None:
        self.task = task
        self.workspace = workspace.resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace does not exist: {self.workspace}")
        self._read_handlers = dict(read_handlers or {})
        self._workspace_lock = asyncio.Lock()
        self.active_workspace_operations = 0
        self.max_active_workspace_operations = 0

    def read_issue(self) -> dict[str, Any]:
        return {
            "status": "ok",
            **self.task.spec.model_dump(mode="json"),
        }

    def list_files(self, path: str | None = None) -> dict[str, Any]:
        root = (
            self.workspace
            if path is None
            else self._entry(normalize_relative(path), require_file=False)
        )
        if root.is_file():
            files = [root]
        else:
            files = list(_repository_files(root))
        visible = [
            item.relative_to(self.workspace).as_posix()
            for item in files[:500]
        ]
        return {
            "status": "ok",
            "path": path or ".",
            "files": visible,
            "truncated": len(files) > 500,
        }

    def search_code(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not 1 <= len(query) <= 200:
            raise RepositoryToolError(
                "query must contain between 1 and 200 characters"
            )
        matches: list[dict[str, Any]] = []
        lowered = query.casefold()
        files_searched = 0
        for path in _repository_files(self.workspace):
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            files_searched += 1
            for line_number, line in enumerate(lines, start=1):
                if lowered in line.casefold():
                    matches.append(
                        {
                            "path": path.relative_to(
                                self.workspace
                            ).as_posix(),
                            "line": line_number,
                            "preview": line.strip()[:300],
                        }
                    )
                    if len(matches) >= 50:
                        break
            if len(matches) >= 50:
                break
        return {
            "status": "ok",
            "query": query,
            "files_searched": files_searched,
            "matches": matches,
            "truncated": len(matches) >= 50,
        }

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 240,
    ) -> dict[str, Any]:
        if (
            not isinstance(start_line, int)
            or not isinstance(end_line, int)
            or start_line < 1
            or end_line < start_line
            or end_line - start_line >= 240
        ):
            raise RepositoryToolError(
                "read_file requires a valid range of at most 240 lines"
            )
        target = self._entry(normalize_relative(path), require_file=True)
        if target.stat().st_size > _MAX_FILE_BYTES:
            raise RepositoryToolError("file exceeds the read limit")
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise RepositoryToolError("file is not UTF-8 text") from exc
        selected = lines[start_line - 1 : end_line]
        return {
            "status": "ok",
            "path": target.relative_to(self.workspace).as_posix(),
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1,
            "total_lines": len(lines),
            "content": "\n".join(
                f"{number}: {line}"
                for number, line in enumerate(selected, start=start_line)
            ),
        }

    async def execute_read(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        handler = self._read_handlers.get(name)
        if handler is not None:
            return await handler(arguments)
        functions: dict[str, Callable[..., dict[str, Any]]] = {
            "list_files": self.list_files,
            "search_code": self.search_code,
            "read_file": self.read_file,
        }
        function = functions.get(name)
        if function is None:
            raise RepositoryToolError(f"unknown read tool: {name}")
        try:
            return await asyncio.to_thread(function, **arguments)
        except TypeError as exc:
            raise RepositoryToolError(str(exc)) from exc

    async def run_tests(
        self,
        *,
        command_kind: str,
        timeout_s: float,
        expose_output: bool = True,
    ) -> dict[str, Any]:
        if command_kind == "public":
            command = self.task.spec.test_command
        elif command_kind == "regression":
            command = self.task.spec.regression_command
        elif command_kind == "hidden":
            command = [
                "{python}",
                "-m",
                "unittest",
                "discover",
                "-s",
                str(self.task.hidden_tests_path),
                "-v",
            ]
        else:
            raise RepositoryToolError(
                f"unknown test command kind: {command_kind}"
            )
        async with self._workspace_operation():
            result = await _to_thread_holding_lock_on_cancellation(
                self._run_command,
                command,
                timeout_s,
                command_kind == "hidden",
            )
        if not expose_output:
            output = str(result.pop("output", ""))
            result["output_sha256"] = hashlib.sha256(
                output.encode("utf-8")
            ).hexdigest()
        result["command_kind"] = command_kind
        return result

    async def apply_patch(
        self,
        proposal: PatchProposal,
        *,
        baseline_failed: bool,
    ) -> dict[str, Any]:
        if not baseline_failed:
            return {
                "status": "blocked",
                "reason": "Run the unmodified public test and observe failure first.",
            }
        async with self._workspace_operation():
            return self._apply_patch_sync(proposal)

    async def rollback_workspace(self) -> dict[str, Any]:
        """Restore the public template without invoking a shell or VCS."""

        async with self._workspace_operation():
            return await _to_thread_holding_lock_on_cancellation(
                self._rollback_workspace_sync
            )

    def show_diff(self) -> dict[str, Any]:
        diff = workspace_diff(self.task.repository_path, self.workspace)
        return {
            "status": "ok",
            "diff": diff,
            "changed_paths": changed_paths(
                self.task.repository_path,
                self.workspace,
            ),
        }

    def submit_solution(
        self,
        proposal: PatchProposal,
        verification: VerificationResult,
    ) -> dict[str, Any]:
        if not verification.passed:
            return {
                "status": "rejected",
                "reason": "Behavioral verification has not passed.",
            }
        return {
            "status": "accepted",
            "report": {
                "root_cause": proposal.root_cause,
                "summary": proposal.summary,
                "evidence": proposal.evidence,
                "changed_files": changed_paths(
                    self.task.repository_path,
                    self.workspace,
                ),
            },
        }

    def is_editable(self, path: str) -> bool:
        candidate = PurePosixPath(normalize_relative(path))
        for allowed_value in self.task.spec.editable_paths:
            allowed = PurePosixPath(allowed_value)
            if candidate == allowed or allowed in candidate.parents:
                return not _looks_like_test(candidate)
        return False

    def _apply_patch_sync(self, proposal: PatchProposal) -> dict[str, Any]:
        if not self.is_editable(proposal.path):
            return {
                "status": "blocked",
                "reason": (
                    "path is outside editable_paths or is a test path: "
                    f"{proposal.path}"
                ),
            }
        size = len(proposal.old_text.encode("utf-8")) + len(
            proposal.new_text.encode("utf-8")
        )
        if size > _MAX_PATCH_BYTES:
            return {"status": "invalid", "reason": "patch exceeds size limit"}
        target = self._entry(proposal.path, require_file=True)
        original = target.read_text(encoding="utf-8")
        occurrences = original.count(proposal.old_text)
        if occurrences != 1:
            return {
                "status": "invalid",
                "reason": (
                    "old_text must match exactly once; "
                    f"found {occurrences}"
                ),
            }
        updated = original.replace(
            proposal.old_text,
            proposal.new_text,
            1,
        )
        target.write_text(updated, encoding="utf-8")
        relative = target.relative_to(self.workspace).as_posix()
        return {
            "status": "applied",
            "path": relative,
            "before_sha256": hashlib.sha256(
                original.encode("utf-8")
            ).hexdigest(),
            "after_sha256": hashlib.sha256(
                updated.encode("utf-8")
            ).hexdigest(),
        }

    def _rollback_workspace_sync(self) -> dict[str, Any]:
        template = self.task.repository_path.resolve()
        workspace = self.workspace.resolve()
        if (
            workspace == template
            or template in workspace.parents
            or workspace in template.parents
            or not workspace.is_dir()
            or not template.is_dir()
        ):
            return {
                "status": "error",
                "reason": "rollback source or destination is unsafe",
            }

        template_hash_before = tree_hash(template)
        workspace_hash_before = tree_hash(workspace)
        paths_before = changed_paths(template, workspace)
        diff_before_sha256 = hashlib.sha256(
            workspace_diff(template, workspace).encode("utf-8")
        ).hexdigest()
        try:
            for entry in workspace.iterdir():
                if entry.name == ".git":
                    return {
                        "status": "error",
                        "reason": "rollback refuses to modify .git",
                        "template_hash_before": template_hash_before,
                        "workspace_hash_before": workspace_hash_before,
                        "changed_paths_before": paths_before,
                        "diff_before_sha256": diff_before_sha256,
                    }
                if entry.is_symlink() or (
                    hasattr(entry, "is_junction") and entry.is_junction()
                ):
                    entry.unlink() if entry.is_symlink() else entry.rmdir()
                    continue
                safe_entry = resolve_beneath(workspace, entry.name)
                if safe_entry != entry.resolve():
                    raise RepositoryToolError(
                        "rollback entry resolves outside the workspace"
                    )
                if entry.is_file():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            shutil.copytree(
                template,
                workspace,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*_SKIPPED_DIRECTORIES),
            )
        except (OSError, RepositoryToolError, ValueError) as exc:
            return {
                "status": "error",
                "reason": f"filesystem rollback failed: {exc}",
                "template_hash_before": template_hash_before,
                "workspace_hash_before": workspace_hash_before,
                "changed_paths_before": paths_before,
                "diff_before_sha256": diff_before_sha256,
            }

        template_hash_after = tree_hash(template)
        workspace_hash_after = tree_hash(workspace)
        paths_after = changed_paths(template, workspace)
        post_diff = workspace_diff(template, workspace)
        if template_hash_after != template_hash_before:
            return {
                "status": "error",
                "reason": "public template changed during rollback",
                "template_hash_before": template_hash_before,
                "template_hash_after": template_hash_after,
                "workspace_hash_before": workspace_hash_before,
                "workspace_hash_after": workspace_hash_after,
                "changed_paths_before": paths_before,
                "changed_paths_after": paths_after,
                "diff_before_sha256": diff_before_sha256,
                "post_diff_sha256": hashlib.sha256(
                    post_diff.encode("utf-8")
                ).hexdigest(),
            }
        if post_diff or paths_after or workspace_hash_after != template_hash_after:
            return {
                "status": "error",
                "reason": "rollback did not restore the public template",
                "template_hash_before": template_hash_before,
                "template_hash_after": template_hash_after,
                "workspace_hash_before": workspace_hash_before,
                "workspace_hash_after": workspace_hash_after,
                "changed_paths_before": paths_before,
                "changed_paths_after": paths_after,
                "diff_before_sha256": diff_before_sha256,
                "post_diff_sha256": hashlib.sha256(
                    post_diff.encode("utf-8")
                ).hexdigest(),
            }
        return {
            "status": "rolled_back",
            "template_hash_before": template_hash_before,
            "template_hash_after": template_hash_after,
            "workspace_hash_before": workspace_hash_before,
            "workspace_hash_after": workspace_hash_after,
            "changed_paths_before": paths_before,
            "changed_paths_after": paths_after,
            "diff_before_sha256": diff_before_sha256,
            "post_diff_sha256": hashlib.sha256(b"").hexdigest(),
        }

    def _run_command(
        self,
        command_template: list[str],
        timeout_s: float,
        hidden: bool,
    ) -> dict[str, Any]:
        command = [
            sys.executable if item == "{python}" else item
            for item in command_template
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "NO_COLOR": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        if hidden:
            existing = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(self.workspace)
                if not existing
                else str(self.workspace) + os.pathsep + existing
            )
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace,
                text=True,
                capture_output=True,
                timeout=max(0.001, timeout_s),
                check=False,
                env=environment,
                shell=False,
            )
            status = "pass" if completed.returncode == 0 else "fail"
            returncode: int | None = completed.returncode
            output = _truncate_output(completed.stdout, completed.stderr)
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            returncode = None
            output = _truncate_output(exc.stdout or "", exc.stderr or "")
        return {
            "status": status,
            "returncode": returncode,
            "output": output,
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }

    def _entry(self, relative: str, *, require_file: bool) -> Path:
        try:
            target = resolve_beneath(self.workspace, relative)
        except ValueError as exc:
            raise RepositoryToolError(str(exc)) from exc
        if _is_skipped(target, self.workspace) or target.is_symlink():
            raise RepositoryToolError(
                "generated, dependency, and symlink paths are blocked"
            )
        if require_file and not target.is_file():
            raise RepositoryToolError(f"file does not exist: {relative}")
        if not require_file and not target.exists():
            raise RepositoryToolError(f"path does not exist: {relative}")
        return target

    class _Operation:
        def __init__(self, owner: "RepositoryRepairTools") -> None:
            self.owner = owner

        async def __aenter__(self) -> None:
            await self.owner._workspace_lock.acquire()
            self.owner.active_workspace_operations += 1
            self.owner.max_active_workspace_operations = max(
                self.owner.max_active_workspace_operations,
                self.owner.active_workspace_operations,
            )
            await asyncio.sleep(0)

        async def __aexit__(self, *_: object) -> None:
            self.owner.active_workspace_operations -= 1
            self.owner._workspace_lock.release()

    def _workspace_operation(self) -> "_Operation":
        return self._Operation(self)


def repair_tool_schemas() -> list[dict[str, Any]]:
    """Provider-neutral schemas for exactly the eight public tools."""

    empty = {"type": "object", "properties": {}, "additionalProperties": False}
    return [
        _schema("read_issue", "Read the public issue and task constraints.", empty),
        _schema(
            "list_files",
            "List public workspace files.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        _schema(
            "search_code",
            "Literal case-insensitive search of public workspace text.",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        _schema(
            "read_file",
            "Read at most 240 numbered lines from a public workspace file.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        _schema("run_tests", "Run the fixed public test argv.", empty),
        _schema(
            "apply_patch",
            "Apply one exact replacement in an allowlisted source path.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        ),
        _schema("show_diff", "Show the complete source diff.", empty),
        _schema(
            "submit_solution",
            "Submit only after all behavioral verifier checks pass.",
            {
                "type": "object",
                "properties": {
                    "root_cause": {"type": "string"},
                    "summary": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["root_cause", "summary", "evidence"],
                "additionalProperties": False,
            },
        ),
    ]


def _schema(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _repository_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and not path.is_symlink()
            and not _is_skipped(path, root)
        ):
            files.append(path)
    return files


def _is_skipped(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    return any(part in _SKIPPED_DIRECTORIES for part in relative.parts)


def _read_diff_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"<binary sha256={hashlib.sha256(path.read_bytes()).hexdigest()}>\n"


def _looks_like_test(path: PurePosixPath) -> bool:
    return any(
        part.casefold() in {"test", "tests", "hidden", "hidden_tests"}
        or part.casefold().startswith("test_")
        or part.casefold().endswith("_test.py")
        for part in path.parts
    )


def _truncate_output(stdout: str, stderr: str) -> str:
    combined = ""
    if stdout:
        combined += f"STDOUT:\n{stdout}"
    if stderr:
        combined += f"\nSTDERR:\n{stderr}"
    combined = combined.strip()
    if len(combined) <= _MAX_TOOL_OUTPUT:
        return combined
    omitted = len(combined) - _MAX_TOOL_OUTPUT
    return (
        combined[:_MAX_TOOL_OUTPUT].rstrip()
        + f"\n... <{omitted} characters omitted>"
    )


async def _to_thread_holding_lock_on_cancellation(
    function: Callable[..., dict[str, Any]],
    *arguments: Any,
) -> dict[str, Any]:
    """Do not release the workspace lock while a cancelled thread still runs."""

    task = asyncio.create_task(asyncio.to_thread(function, *arguments))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
