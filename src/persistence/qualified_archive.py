from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class QualifiedAlphaArchiveRecord:
    task_id: str
    archived_at: str
    grade: str | None
    platform_alpha_id: str
    sharpe: float
    fitness: float
    turnover: float


def initialize_qualified_archive_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS qualified_alpha_archive (
            task_id TEXT PRIMARY KEY REFERENCES backtest_results(task_id),
            archived_at TEXT NOT NULL,
            grade TEXT,
            platform_alpha_id TEXT,
            sharpe REAL,
            fitness REAL,
            turnover REAL
        )
        """
    )


def archive_qualified_alpha(
    connection: sqlite3.Connection, *, task_id: str, archived_at: str,
) -> QualifiedAlphaArchiveRecord:
    """Record retirement once; original results and budgets remain authoritative."""
    observed = datetime.fromisoformat(archived_at)
    if observed.utcoffset() is None:
        raise ValueError("qualified_archive_timestamp_invalid")
    source = connection.execute(
        "SELECT t.status, t.finished_at, t.platform_alpha_id, r.grade, r.sharpe, r.fitness, r.turnover FROM backtest_tasks t "
        "JOIN backtest_results r USING(task_id) WHERE t.task_id = ?",
        (task_id,),
    ).fetchone()
    if source is None or source["status"] != "completed":
        raise ValueError("qualified_archive_completed_source_required")
    if observed < datetime.fromisoformat(source["finished_at"]):
        raise ValueError("qualified_archive_timestamp_before_result")
    connection.execute(
        "INSERT INTO qualified_alpha_archive "
        "(task_id, archived_at, grade, platform_alpha_id, sharpe, fitness, turnover) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(task_id) DO NOTHING",
        (task_id, archived_at, source["grade"], source["platform_alpha_id"],
         source["sharpe"], source["fitness"], source["turnover"]),
    )
    row = connection.execute(
        "SELECT task_id, archived_at, grade, platform_alpha_id, sharpe, fitness, turnover "
        "FROM qualified_alpha_archive WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    return QualifiedAlphaArchiveRecord(**dict(row))


def list_qualified_alpha_archive(
    connection: sqlite3.Connection,
) -> tuple[QualifiedAlphaArchiveRecord, ...]:
    return tuple(
        QualifiedAlphaArchiveRecord(**dict(row))
        for row in connection.execute(
            "SELECT task_id, archived_at, grade, platform_alpha_id, sharpe, fitness, turnover "
            "FROM qualified_alpha_archive ORDER BY archived_at, task_id"
        )
    )


def delete_qualified_alpha_archive_item(connection: sqlite3.Connection, task_id: str) -> bool:
    """Consume membership only; the caller owns the transaction."""
    return connection.execute(
        "DELETE FROM qualified_alpha_archive WHERE task_id = ?", (task_id,),
    ).rowcount == 1
