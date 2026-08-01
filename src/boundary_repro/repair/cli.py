"""boundary-repair command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from boundary_repro.repair.models import RepairRunConfig
from boundary_repro.repair.providers import (
    GroqRepairProvider,
    ProviderOutputError,
    ScriptedRepairProvider,
)
from boundary_repro.repair.runtime import (
    PAUSE_NODES,
    RepairRuntime,
    TERMINAL_STATUSES,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boundary-repair",
        description=(
            "Stateful, resumable repository issue agent with hidden verification."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Start a new repair thread.")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--thread-id", required=True)
    run.add_argument(
        "--brain",
        choices=("scripted", "groq"),
        default="groq",
        help=(
            "default: groq; scripted must be selected explicitly and is "
            "not a real-model score"
        ),
    )
    run.add_argument("--model", default="openai/gpt-oss-120b")
    run.add_argument("--pause-after", choices=sorted(PAUSE_NODES))
    run.add_argument(
        "--json-out",
        type=Path,
        help="Write the complete checkpoint state and tool trace.",
    )
    _add_state_dir(run)
    _add_config(run)

    resume = subparsers.add_parser(
        "resume",
        help="Resume with the provider/model/config stored in the checkpoint.",
    )
    resume.add_argument("--thread-id", required=True)
    _add_state_dir(resume)

    status = subparsers.add_parser("status")
    status.add_argument("--thread-id", required=True)
    _add_state_dir(status)

    history = subparsers.add_parser("history")
    history.add_argument("--thread-id", required=True)
    history.add_argument("--limit", type=int, default=100)
    _add_state_dir(history)

    memory = subparsers.add_parser("memory")
    memory_commands = memory.add_subparsers(
        dest="memory_command",
        required=True,
    )
    memory_list = memory_commands.add_parser("list")
    memory_list.add_argument("--limit", type=int, default=100)
    _add_state_dir(memory_list)
    memory_search = memory_commands.add_parser("search")
    memory_search.add_argument("--query", required=True)
    memory_search.add_argument("--limit", type=int, default=5)
    _add_state_dir(memory_search)
    return parser


def _add_state_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Defaults to CURRENT_DIRECTORY/.boundary_state",
    )


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tool-timeout", type=float, default=20)
    parser.add_argument("--llm-timeout", type=float, default=60)
    parser.add_argument("--node-timeout", type=float, default=90)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=0.1)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--deadline", type=float, default=180)
    parser.add_argument("--max-read-tasks", type=int, default=12)
    parser.add_argument("--max-patch-attempts", type=int, default=3)


async def _execute(args: argparse.Namespace) -> Any:
    root = Path.cwd().resolve()
    state_dir = (
        args.state_dir
        if args.state_dir is None or args.state_dir.is_absolute()
        else root / args.state_dir
    )
    if args.command == "run":
        config = RepairRunConfig(
            tool_timeout_s=args.tool_timeout,
            llm_timeout_s=args.llm_timeout,
            node_timeout_s=args.node_timeout,
            max_retries=args.max_retries,
            retry_initial_delay_s=args.retry_delay,
            max_concurrency=args.max_concurrency,
            run_deadline_s=args.deadline,
            max_read_tasks=args.max_read_tasks,
            max_patch_attempts=args.max_patch_attempts,
        )
        if args.brain == "scripted":
            provider = ScriptedRepairProvider()
        else:
            provider = GroqRepairProvider(
                model=args.model,
                request_timeout_s=args.llm_timeout,
            )
        runtime = RepairRuntime(
            root,
            state_dir=state_dir,
            provider=provider,
            config=config,
        )
        return await runtime.run(
            args.task,
            args.thread_id,
            pause_after=args.pause_after,
        )

    runtime = RepairRuntime(root, state_dir=state_dir)
    if args.command == "resume":
        return await runtime.resume(args.thread_id)
    if args.command == "status":
        return await runtime.status(args.thread_id)
    if args.command == "history":
        return await runtime.history(args.thread_id, limit=args.limit)
    if args.command == "memory" and args.memory_command == "list":
        return await runtime.list_memory(limit=args.limit)
    if args.command == "memory" and args.memory_command == "search":
        return await runtime.search_memory(args.query, limit=args.limit)
    raise AssertionError("unreachable command")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = asyncio.run(_execute(args))
    except (
        OSError,
        ProviderOutputError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    output = getattr(args, "json_out", None)
    if output is not None:
        destination = (
            output if output.is_absolute() else Path.cwd() / output
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if isinstance(result, dict):
        status = result.get("status")
        if result.get("paused") is True and status not in TERMINAL_STATUSES:
            return 0
        if status == "completed":
            return 0
        if status in TERMINAL_STATUSES:
            return 1
    return 1 if isinstance(result, dict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
