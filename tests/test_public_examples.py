from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = PROJECT_ROOT / "examples"
SAMPLE_TRACE = EXAMPLES / "stateful_repair_trace.sample.json"
REAL_GROQ_TRACE = EXAMPLES / "groq_repair_trace.sample.json"

WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9])[A-Z]:(?:\\\\|\\|/)"
)
WINDOWS_USER_DIRECTORY = re.compile(
    r"(?i)(?:\\\\|\\|/)Users(?:\\\\|\\|/)"
)
APPDATA_DIRECTORY = re.compile(
    r"(?i)(?:\\\\|\\|/)AppData(?:\\\\|\\|/)"
)
UNIX_LOCAL_ABSOLUTE_PATH = re.compile(
    r"(?i)(?<![:A-Za-z0-9<])/"
    r"(?:Users|home|tmp|var/tmp|opt|workspace|mnt|private)(?:/|\\)"
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
PROXY_SETTING = re.compile(
    r"(?i)\b(?:https?_proxy|all_proxy|proxy_(?:url|host|port))"
    r"[\"'\s:=]+[^\s,\"'}]+"
)
PRIVATE_PROXY_URL = re.compile(
    r"(?i)\b(?:https?|socks5?)://"
    r"(?:[^/\s:@]+(?::[^/\s@]*)?@)?"
    r"(?:localhost|127(?:\.\d{1,3}){3}|10(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})"
    r"(?::\d+)?"
)
URL = re.compile(r"https?://[^\s\"<>]+")


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
        if APPDATA_DIRECTORY.search(text):
            violations.append(f"{example.name}: AppData directory")
        if UNIX_LOCAL_ABSOLUTE_PATH.search(text):
            violations.append(f"{example.name}: Unix absolute path")
        if API_KEY.search(text) or NAMED_API_KEY.search(text):
            violations.append(f"{example.name}: API key pattern")
        if local_username and local_username.casefold() in text.casefold():
            violations.append(f"{example.name}: local username")

    assert violations == []


def test_scripted_sample_is_current_completed_and_ineligible() -> None:
    text = SAMPLE_TRACE.read_text(encoding="utf-8")
    payload = json.loads(text)
    state = payload["state"]
    metrics = state["metrics"]

    assert payload["status"] == "completed"
    assert payload["sample_metadata"]["boundary_repro_version"] == "0.6.0"
    assert payload["sample_metadata"]["provider"] == "scripted"
    assert state["provider"] == "scripted"
    assert state["verification"]["passed"] is True
    assert metrics["evaluation_eligible"] is False
    assert state["patch_attempt"] >= 2
    assert metrics["patch_attempts"] >= 2
    assert state["rollback_count"] >= 1
    assert metrics["rollback_count"] >= 1
    assert state["repair_feedback"]["attempt"] == 1
    assert state["repair_feedback"]["failure_stage"] == "public_tests"
    assert state["attempt_history"][0]["failure_stage"] is not None
    assert state["attempt_history"][0]["public_test_status"] == "fail"
    assert state["attempt_history"][-1]["failure_stage"] is None
    assert state["attempt_history"][-1]["public_test_status"] == "pass"
    assert metrics["successful_attempt"] == state["patch_attempt"]
    candidate_statuses = [
        event["status"]
        for event in state["tool_trace"]
        if event["name"] == "run_tests"
        and event["result"].get("phase") == "candidate"
    ]
    assert candidate_statuses == ["fail", "pass"]
    rollback = next(
        event["result"]
        for event in state["tool_trace"]
        if event["name"] == "rollback_workspace"
    )
    assert rollback["changed_paths_after"] == []
    assert rollback["post_diff_sha256"] == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert "not an LLM score" in payload["sample_metadata"][
        "evaluation_note"
    ]
    assert WINDOWS_ABSOLUTE_PATH.search(text) is None
    assert UNIX_LOCAL_ABSOLUTE_PATH.search(text) is None


def test_real_groq_sample_is_completed_clean_and_sanitized() -> None:
    text = REAL_GROQ_TRACE.read_text(encoding="utf-8")
    payload = json.loads(text)
    state = payload["state"]
    metrics = state["metrics"]

    assert payload["status"] == "completed"
    assert state["provider"] == "groq"
    assert state["model"] == "openai/gpt-oss-120b"
    assert state["verification"]["passed"] is True
    assert metrics["evaluation_eligible"] is True
    assert metrics["memory_assisted"] is False
    assert state["memory_hits"] == []
    assert metrics["memory_hits"] == 0
    assert metrics["read_tasks"] == 5
    assert metrics["max_active_read_workers"] == 3
    assert metrics["successful_read_workers"] == 4
    assert metrics["elapsed_ms"] == 19923
    assert state["patch_attempt"] == 1
    assert metrics["patch_attempts"] == 1
    assert state["rollback_count"] == 0
    assert metrics["rollback_count"] == 0
    assert state["repair_feedback"] is None

    failed_evidence = [
        item for item in state["evidence"] if item["status"] != "success"
    ]
    assert len(failed_evidence) == 1
    assert failed_evidence[0]["arguments"] == {"path": "tests/.env"}
    assert failed_evidence[0]["result"]["reason"] == (
        "file does not exist: tests/.env"
    )
    assert len(state["evidence"]) == 5
    assert state["verification"]["passed"] is True

    hidden_events = [
        event
        for event in state["tool_trace"]
        if event["name"] == "hidden_tests"
    ]
    assert len(hidden_events) == 1
    assert set(hidden_events[0]["result"]) == {
        "status",
        "returncode",
        "duration_ms",
        "output_sha256",
    }
    assert "output" not in hidden_events[0]["result"]

    assert WINDOWS_ABSOLUTE_PATH.search(text) is None
    assert WINDOWS_USER_DIRECTORY.search(text) is None
    assert APPDATA_DIRECTORY.search(text) is None
    assert UNIX_LOCAL_ABSOLUTE_PATH.search(text) is None
    assert API_KEY.search(text) is None
    assert NAMED_API_KEY.search(text) is None
    assert PROXY_SETTING.search(text) is None
    assert PRIVATE_PROXY_URL.search(text) is None
    local_username = Path.home().name if os.name == "nt" else ""
    assert not local_username or (
        local_username.casefold() not in text.casefold()
    )
    assert all(
        urlsplit(url).hostname == "github.com" for url in URL.findall(text)
    )
