"""Repair providers with strict output validation and deadline-aware retries."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

from pydantic import ValidationError

from boundary_repro.repair.models import (
    PatchProposal,
    ReadTask,
    RepairPlan,
    StrictRepairPlanOutput,
)

T = TypeVar("T")

_GPT_OSS_STRICT_MODELS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }
)


class TransientProviderError(ConnectionError):
    """Network, rate-limit, server, or timeout failure safe to retry."""


class ProviderOutputError(ValueError):
    """Malformed or unsupported provider output; retrying will not repair it."""


class ProviderRetryError(RuntimeError):
    """A transient provider operation exhausted retries or its deadline."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        audit: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.audit = audit


class ProviderDeadlineError(ProviderRetryError):
    """A provider operation exhausted the active run deadline."""


class RepairProvider(Protocol):
    name: str
    model: str | None

    async def aplan(
        self,
        *,
        task: dict[str, Any],
        baseline: dict[str, Any],
        memory_hits: list[dict[str, Any]],
    ) -> RepairPlan: ...

    async def apatch(
        self,
        *,
        task: dict[str, Any],
        baseline: dict[str, Any],
        evidence: list[dict[str, Any]],
        memory_hits: list[dict[str, Any]],
    ) -> PatchProposal: ...


async def retry_with_deadline(
    operation: Callable[[], Awaitable[T]],
    *,
    operation_name: str,
    absolute_deadline: float,
    per_attempt_timeout_s: float,
    max_retries: int,
    initial_delay_s: float,
) -> tuple[T, list[dict[str, Any]]]:
    """Retry transient failures without any attempt/backoff crossing deadline."""

    audit: list[dict[str, Any]] = []
    delay = initial_delay_s
    attempts = 0
    while True:
        remaining_before_attempt = absolute_deadline - time.time()
        if remaining_before_attempt <= 0:
            raise ProviderDeadlineError(
                f"{operation_name} exceeded the overall deadline",
                attempts=attempts,
                audit=audit,
            )
        attempts += 1
        attempt_limited_by_deadline = (
            remaining_before_attempt <= per_attempt_timeout_s
        )
        timeout = min(per_attempt_timeout_s, remaining_before_attempt)
        started = time.perf_counter()
        try:
            value = await asyncio.wait_for(operation(), timeout=timeout)
            audit.append(
                {
                    "operation": operation_name,
                    "attempt": attempts,
                    "status": "success",
                    "timeout_s": timeout,
                    "remaining_before_attempt_s": remaining_before_attempt,
                    "attempt_limited_by_deadline": (
                        attempt_limited_by_deadline
                    ),
                    "duration_ms": int(
                        (time.perf_counter() - started) * 1000
                    ),
                }
            )
            return value, audit
        except (TransientProviderError, TimeoutError) as exc:
            audit.append(
                {
                    "operation": operation_name,
                    "attempt": attempts,
                    "status": "transient_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "timeout_s": timeout,
                    "remaining_before_attempt_s": remaining_before_attempt,
                    "attempt_limited_by_deadline": (
                        attempt_limited_by_deadline
                    ),
                    "duration_ms": int(
                        (time.perf_counter() - started) * 1000
                    ),
                }
            )
            deadline_exhausted = time.time() >= absolute_deadline
            timeout_limited_by_deadline = (
                isinstance(exc, TimeoutError)
                and attempt_limited_by_deadline
            )
            if attempts > max_retries:
                error_type = (
                    ProviderDeadlineError
                    if timeout_limited_by_deadline or deadline_exhausted
                    else ProviderRetryError
                )
                raise error_type(
                    f"{operation_name} failed after {attempts} attempts: {exc}",
                    attempts=attempts,
                    audit=audit,
                ) from exc

            remaining_before_backoff = absolute_deadline - time.time()
            if remaining_before_backoff <= 0 or delay >= remaining_before_backoff:
                audit.append(
                    {
                        "operation": operation_name,
                        "attempt": attempts,
                        "status": "backoff_skipped_deadline",
                        "requested_backoff_s": delay,
                        "remaining_before_backoff_s": max(
                            0.0,
                            remaining_before_backoff,
                        ),
                    }
                )
                raise ProviderDeadlineError(
                    f"{operation_name} has no deadline budget for backoff",
                    attempts=attempts,
                    audit=audit,
                ) from exc
            audit.append(
                {
                    "operation": operation_name,
                    "attempt": attempts,
                    "status": "backoff",
                    "backoff_s": delay,
                    "remaining_before_backoff_s": remaining_before_backoff,
                }
            )
            if delay:
                await asyncio.sleep(delay)
            delay *= 2


class ScriptedRepairProvider:
    """Deterministic offline demonstration provider, never an LLM score."""

    name = "scripted"
    model: str | None = None

    async def aplan(
        self,
        *,
        task: dict[str, Any],
        baseline: dict[str, Any],
        memory_hits: list[dict[str, Any]],
    ) -> RepairPlan:
        del baseline, memory_hits
        issue = str(task["issue_text"])
        tokens = [
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_-]{3,}", issue)
            if token.casefold()
            not in {
                "with",
                "when",
                "from",
                "should",
                "public",
                "tests",
                "default",
            }
        ]
        queries = list(dict.fromkeys(tokens))[:4] or ["TODO"]
        return RepairPlan(
            summary=(
                "Inspect editable source and search for symbols named in the "
                "public issue. This is deterministic demo behavior."
            ),
            read_tasks=[
                ReadTask(
                    task_id=f"search-{index}",
                    tool="search_code",
                    arguments={"query": query},
                )
                for index, query in enumerate(queries, start=1)
            ],
        )

    async def apatch(
        self,
        *,
        task: dict[str, Any],
        baseline: dict[str, Any],
        evidence: list[dict[str, Any]],
        memory_hits: list[dict[str, Any]],
    ) -> PatchProposal:
        del baseline, memory_hits
        issue = str(task["issue_text"])
        source_records = [
            item.get("result", {})
            for item in evidence
            if item.get("tool") == "read_file"
            and item.get("status") == "success"
        ]
        desired_encoding = (
            "utf-8"
            if re.search(r"utf[- ]?8", issue, re.IGNORECASE)
            else None
        )
        if desired_encoding:
            for result in source_records:
                path = str(result.get("path", ""))
                content = _strip_line_numbers(
                    str(result.get("content", ""))
                )
                match = re.search(
                    r"(?m)^(?P<line>[A-Z][A-Z0-9_]*(?:\s*:[^=\n]+)?"
                    r"\s*=\s*None)\s*$",
                    content,
                )
                if match:
                    old_text = match.group("line")
                    new_text = re.sub(
                        r"None\s*$",
                        f'"{desired_encoding}"',
                        old_text,
                    )
                    return PatchProposal(
                        path=path,
                        old_text=old_text,
                        new_text=new_text,
                        root_cause=(
                            "The repository leaves its documented text "
                            "encoding default unset, so behavior depends on "
                            "the process locale."
                        ),
                        summary=(
                            "Use the public issue's UTF-8 default while "
                            "preserving explicit encoding overrides."
                        ),
                        evidence=(
                            "The public test fails before the source change; "
                            "verification must rerun public, regression, and "
                            "private hidden tests."
                        ),
                    )
        raise ProviderOutputError(
            "The scripted demo provider could not derive a generic patch. "
            "Use --brain groq for non-demo tasks."
        )


class GroqRepairProvider:
    """Real Groq planner/patch proposer with no scripted fallback."""

    name = "groq"

    def __init__(
        self,
        *,
        model: str = "openai/gpt-oss-120b",
        api_key: str | None = None,
        request_timeout_s: float = 60,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ProviderOutputError(
                "GROQ_API_KEY is missing. Resume cannot change providers."
            )
        self.request_timeout_s = request_timeout_s

    async def aplan(
        self,
        *,
        task: dict[str, Any],
        baseline: dict[str, Any],
        memory_hits: list[dict[str, Any]],
    ) -> RepairPlan:
        payload = await self._complete_json(
            {
                "operation": "plan_read_only_repository_investigation",
                "output_contract": "RepairPlan",
                "public_task": task,
                "failing_public_test": baseline,
                "verified_memories_for_prioritization_only": memory_hits[:5],
                "constraints": [
                    "No tools are callable in this provider request.",
                    (
                        "read_tasks are data for the LangGraph Harness to "
                        "execute later, not tool calls to execute now."
                    ),
                    (
                        "Do not emit a provider-native tool/function call or "
                        "claim any read task has already run."
                    ),
                    "Do not invent files or hidden-test content.",
                    "Return JSON only.",
                ],
            },
            output_name="repair_plan",
            output_schema=_REPAIR_PLAN_SCHEMA,
        )
        try:
            strict_plan = StrictRepairPlanOutput.model_validate(payload)
            return RepairPlan.model_validate(strict_plan.model_dump())
        except ValidationError as exc:
            raise ProviderOutputError(f"invalid Groq repair plan: {exc}") from exc

    async def apatch(
        self,
        *,
        task: dict[str, Any],
        baseline: dict[str, Any],
        evidence: list[dict[str, Any]],
        memory_hits: list[dict[str, Any]],
    ) -> PatchProposal:
        payload = await self._complete_json(
            {
                "operation": "propose_one_minimal_exact_source_replacement",
                "output_contract": "PatchProposal",
                "public_task": task,
                "failing_public_test": baseline,
                "read_only_evidence": evidence,
                "verified_memories_for_prioritization_only": memory_hits[:5],
                "constraints": [
                    "No tools are callable in this provider request.",
                    (
                        "Repository reads in the evidence were performed by "
                        "the LangGraph Harness, not by this provider request."
                    ),
                    (
                        "Do not emit a provider-native tool/function call or "
                        "claim that a patch or verifier has already run."
                    ),
                    "old_text must appear exactly once in the cited source.",
                    "Do not modify tests or paths outside editable_paths.",
                    "Do not claim hidden tests passed.",
                    "Return JSON only; no private chain-of-thought.",
                ],
            },
            output_name="patch_proposal",
            output_schema=_PATCH_PROPOSAL_SCHEMA,
        )
        try:
            return PatchProposal.model_validate(payload)
        except ValidationError as exc:
            raise ProviderOutputError(
                f"invalid Groq patch proposal: {exc}"
            ) from exc

    async def _complete_json(
        self,
        context: dict[str, Any],
        *,
        output_name: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            from groq import (
                APIConnectionError,
                APITimeoutError,
                AsyncGroq,
                RateLimitError,
            )
        except ImportError as exc:
            raise ProviderOutputError("The Groq SDK is not installed") from exc
        try:
            client = AsyncGroq(
                api_key=self.api_key,
                timeout=self.request_timeout_s,
            )
            strict_json_schema = self.model in _GPT_OSS_STRICT_MODELS
            prompt_context = dict(context)
            if strict_json_schema:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_name,
                        "strict": True,
                        "schema": output_schema,
                    },
                }
            else:
                # JSON Object Mode needs the schema in the prompt because the
                # provider does not enforce it. Local Pydantic validation is
                # still mandatory and never falls back to scripted behavior.
                prompt_context["output_schema"] = output_schema
                response_format = {"type": "json_object"}
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You repair repositories from public evidence. "
                            "This request exposes no callable tools. Return "
                            "exactly one JSON data object matching the output "
                            "contract. Never emit a provider-native tool call "
                            "or claim a planned action already ran."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            prompt_context,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                ],
                response_format=response_format,
                temperature=0,
            )
            content = response.choices[0].message.content
        except (
            TimeoutError,
            ConnectionError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        ) as exc:
            raise TransientProviderError(str(exc)) from exc
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status == 429 or (
                isinstance(status, int) and status >= 500
            ):
                raise TransientProviderError(str(exc)) from exc
            raise ProviderOutputError(f"Groq request failed: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderOutputError("Groq returned empty content")
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderOutputError(
                f"Groq returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ProviderOutputError("Groq response must be a JSON object")
        return value


def _strip_line_numbers(content: str) -> str:
    return "\n".join(
        re.sub(r"^\d+:\s?", "", line)
        for line in content.splitlines()
    )


def _strict_schema(model: type[Any]) -> dict[str, Any]:
    """Build and verify the subset required by Groq strict JSON Schema."""

    schema = model.model_json_schema()
    _assert_strict_objects(schema)
    return schema


def _assert_strict_objects(value: Any, path: str = "$") -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_strict_objects(item, f"{path}[{index}]")
        return
    if not isinstance(value, dict):
        return
    if value.get("type") == "object":
        if value.get("additionalProperties") is not False:
            raise AssertionError(
                f"{path} must set additionalProperties=false"
            )
        properties = value.get("properties", {})
        required = set(value.get("required", []))
        missing = set(properties) - required
        if missing:
            raise AssertionError(
                f"{path} properties missing from required: {sorted(missing)}"
            )
    for key, item in value.items():
        _assert_strict_objects(item, f"{path}.{key}")


_REPAIR_PLAN_SCHEMA = _strict_schema(StrictRepairPlanOutput)
_PATCH_PROPOSAL_SCHEMA = _strict_schema(PatchProposal)
