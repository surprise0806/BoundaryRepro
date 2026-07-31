"""Verified repository repair memory, separate from graph checkpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

import aiosqlite

from boundary_repro.repair.models import RepairMemoryRecord


class RepairMemoryStore(Protocol):
    async def setup(self) -> None: ...

    async def upsert(self, record: RepairMemoryRecord) -> bool: ...

    async def list(self, limit: int = 100) -> list[RepairMemoryRecord]: ...

    async def search(
        self,
        query: str,
        limit: int = 5,
        *,
        requesting_provider: str | None = None,
        exclude_task_id: str | None = None,
    ) -> list[RepairMemoryRecord]: ...


class SqliteRepairMemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    async def setup(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as database:
            await database.execute("PRAGMA journal_mode=WAL")
            await database.execute(
                """
                CREATE TABLE IF NOT EXISTS repair_memories (
                    memory_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    issue_summary TEXT NOT NULL,
                    verified_root_cause TEXT NOT NULL,
                    patch_summary TEXT NOT NULL,
                    changed_paths TEXT NOT NULL,
                    diff_sha256 TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT,
                    source_thread_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                )
                """
            )
            await database.commit()

    async def upsert(self, record: RepairMemoryRecord) -> bool:
        await self.setup()
        async with aiosqlite.connect(self.path) as database:
            cursor = await database.execute(
                """
                INSERT OR IGNORE INTO repair_memories (
                    memory_id, task_id, issue_summary, verified_root_cause,
                    patch_summary, changed_paths, diff_sha256, evidence,
                    provider, model, source_thread_id, created_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    record.task_id,
                    record.issue_summary,
                    record.verified_root_cause,
                    record.patch_summary,
                    json.dumps(record.changed_paths, ensure_ascii=False),
                    record.diff_sha256,
                    record.evidence,
                    record.provider,
                    record.model,
                    record.source_thread_id,
                    record.created_at,
                    record.content_hash,
                ),
            )
            await database.commit()
            return cursor.rowcount == 1

    async def list(self, limit: int = 100) -> list[RepairMemoryRecord]:
        await self.setup()
        bounded = max(1, min(limit, 1000))
        async with aiosqlite.connect(self.path) as database:
            database.row_factory = aiosqlite.Row
            cursor = await database.execute(
                """
                SELECT * FROM repair_memories
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (bounded,),
            )
            rows = await cursor.fetchall()
        return [self._from_row(dict(row)) for row in rows]

    async def search(
        self,
        query: str,
        limit: int = 5,
        *,
        requesting_provider: str | None = None,
        exclude_task_id: str | None = None,
    ) -> list[RepairMemoryRecord]:
        tokens = {
            token.casefold()
            for token in re.findall(r"[\w:+.-]+", query)
            if len(token) >= 2
        }
        records = [
            record
            for record in await self.list(limit=1000)
            if record.task_id != exclude_task_id
            and not (
                requesting_provider is not None
                and requesting_provider != "scripted"
                and record.provider == "scripted"
            )
        ]

        def score(record: RepairMemoryRecord) -> tuple[int, str]:
            text = " ".join(
                [
                    record.task_id,
                    record.issue_summary,
                    record.verified_root_cause,
                    record.patch_summary,
                    " ".join(record.changed_paths),
                ]
            ).casefold()
            return sum(token in text for token in tokens), record.created_at

        if not tokens:
            return records[: max(1, min(limit, 100))]
        ranked = sorted(records, key=score, reverse=True)
        return [
            item
            for item in ranked
            if score(item)[0] > 0
        ][: max(1, min(limit, 100))]

    @staticmethod
    def _from_row(row: dict[str, object]) -> RepairMemoryRecord:
        return RepairMemoryRecord(
            memory_id=str(row["memory_id"]),
            task_id=str(row["task_id"]),
            issue_summary=str(row["issue_summary"]),
            verified_root_cause=str(row["verified_root_cause"]),
            patch_summary=str(row["patch_summary"]),
            changed_paths=json.loads(str(row["changed_paths"])),
            diff_sha256=str(row["diff_sha256"]),
            evidence=str(row["evidence"]),
            provider=str(row["provider"]),
            model=(
                None if row["model"] is None else str(row["model"])
            ),
            source_thread_id=str(row["source_thread_id"]),
            created_at=str(row["created_at"]),
            content_hash=str(row["content_hash"]),
        )
