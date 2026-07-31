"""Task-agnostic behavioral verifier for repository repairs."""

from __future__ import annotations

from typing import Any

from boundary_repro.repair.models import VerificationResult


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
