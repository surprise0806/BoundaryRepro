"""Task-agnostic behavioral verifier for repository repairs."""

from __future__ import annotations

from typing import Any

from boundary_repro.repair.models import (
    CandidateVerification,
    VerificationResult,
)


def verify_candidate_behavior(
    *,
    attempt: int,
    current_diff: str,
    changed_paths: list[str],
    legal_paths: bool,
    public_after: dict[str, Any],
    regression: dict[str, Any],
    diff_sha256: str,
) -> CandidateVerification:
    """Evaluate only provider-visible public and regression behavior."""

    checks = {
        "nonempty_diff": bool(current_diff.strip()),
        "legal_paths": bool(changed_paths) and legal_paths,
        "public_tests_passed": public_after.get("status") == "pass",
        "regression_passed": regression.get("status") == "pass",
    }
    failure_stage = None
    if not checks["public_tests_passed"]:
        failure_stage = "public_tests"
    elif not checks["regression_passed"]:
        failure_stage = "regression_tests"
    return CandidateVerification(
        attempt=attempt,
        passed=all(checks.values()),
        public_test_status=str(public_after.get("status", "unknown")),
        regression_test_status=str(regression.get("status", "unknown")),
        public_test_output=str(public_after.get("output", "")),
        regression_test_output=str(regression.get("output", "")),
        changed_paths=changed_paths,
        diff_sha256=diff_sha256,
        failure_stage=failure_stage,
        reasons=[name for name, passed in checks.items() if not passed],
        **checks,
    )


def verify_behavior(
    *,
    baseline_result: dict[str, Any],
    current_diff: str,
    changed_paths: list[str],
    legal_paths: bool,
    public_after: dict[str, Any],
    regression: dict[str, Any],
    hidden: dict[str, Any],
) -> VerificationResult:
    """Evaluate behavior only; never inspect diagnosis text or task keywords."""

    checks = {
        "baseline_failed": baseline_result.get("status") == "fail",
        "nonempty_diff": bool(current_diff.strip()),
        "legal_paths": bool(changed_paths) and legal_paths,
        "public_tests_passed": public_after.get("status") == "pass",
        "regression_passed": regression.get("status") == "pass",
        "hidden_tests_passed": hidden.get("status") == "pass",
    }
    return VerificationResult(
        passed=all(checks.values()),
        submitted=False,
        reasons=[
            name for name, passed in checks.items() if not passed
        ],
        **checks,
    )
