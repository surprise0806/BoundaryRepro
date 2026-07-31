from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = PROJECT_ROOT / "examples"
SAMPLE_TRACE = EXAMPLES / "stateful_repair_trace.sample.json"

WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9])[A-Z]:(?:\\\\|\\|/)"
)
WINDOWS_USER_DIRECTORY = re.compile(
    r"(?i)(?:\\\\|\\|/)Users(?:\\\\|\\|/)"
)
API_KEY = re.compile(
    r"\b(?:gsk_[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{12,}|"
    r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"AIza[A-Za-z0-9_-]{20,})\b"
)
NAMED_API_KEY = re.compile(
    r"(?i)(?:api[_-]?key|authorization)[\"'\s:=]+"
    r"(?:bearer\s+)?[\"']?(?!<)[A-Za-z0-9_-]{12,}"
)


def test_public_examples_contain_no_local_paths_or_api_keys() -> None:
    violations: list[str] = []
    local_username = Path.home().name if os.name == "nt" else ""

    examples = sorted(
        path for path in EXAMPLES.iterdir() if path.is_file()
    )
    for example in examples:
        text = example.read_text(encoding="utf-8")
        if WINDOWS_ABSOLUTE_PATH.search(text):
            violations.append(f"{example.name}: Windows absolute path")
        if WINDOWS_USER_DIRECTORY.search(text):
            violations.append(f"{example.name}: Windows user directory")
        if API_KEY.search(text) or NAMED_API_KEY.search(text):
            violations.append(f"{example.name}: API key pattern")
        if local_username and local_username.casefold() in text.casefold():
            violations.append(f"{example.name}: local username")

    assert violations == []


def test_scripted_sample_is_current_completed_and_ineligible() -> None:
    payload = json.loads(SAMPLE_TRACE.read_text(encoding="utf-8"))

    assert payload["status"] == "completed"
    assert payload["sample_metadata"]["boundary_repro_version"] == "0.5.1"
    assert payload["sample_metadata"]["provider"] == "scripted"
    assert payload["state"]["provider"] == "scripted"
    assert payload["state"]["verification"]["passed"] is True
    assert payload["state"]["metrics"]["evaluation_eligible"] is False
    assert "not an LLM score" in payload["sample_metadata"][
        "evaluation_note"
    ]
