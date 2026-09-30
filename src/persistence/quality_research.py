"""Immutable purpose of quality attempts; expenditure stays in backtest tasks."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


def initialize_quality_research_schema(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS quality_research_tasks (
        task_id TEXT PRIMARY KEY REFERENCES backtest_tasks(task_id),
        root_task_id TEXT NOT NULL REFERENCES backtest_tasks(task_id)
    )""")


def record_quality_research_task(connection: sqlite3.Connection, *, task_id: str, root_task_id: str) -> None:
    existing = connection.execute("SELECT root_task_id FROM quality_research_tasks WHERE task_id=?", (task_id,)).fetchone()
    if existing is not None:
        if existing[0] != root_task_id:
            raise ValueError("quality_research_root_conflict")
        return
    connection.execute("INSERT INTO quality_research_tasks VALUES (?, ?)", (task_id, root_task_id))


@dataclass(frozen=True, slots=True)
class QualityResearchTask:
    task_id: str
    account_scope: str
    settings_json: str
    root_task_id: str | None
    attempted: bool


def list_quality_research_tasks(connection: sqlite3.Connection) -> tuple[QualityResearchTask, ...]:
    # Older optimization runs identify their purpose unambiguously. Ordinary
    # historical mutations lacking a purpose record are not relabeled as quality.
    return tuple(QualityResearchTask(*row) for row in connection.execute("""
        SELECT t.task_id, t.account_scope, t.settings_json, q.root_task_id,
            t.submission_started_at IS NOT NULL
        FROM backtest_tasks t
        LEFT JOIN quality_research_tasks q USING(task_id)
        LEFT JOIN automated_run_backtests b USING(task_id)
        LEFT JOIN automated_runs r USING(run_id)
        WHERE (q.task_id IS NOT NULL OR r.optimization_only=1)
            AND NOT (t.status='failed' AND t.failure_code='automated_run_stopped_before_submission'
                AND t.submission_started_at IS NULL AND t.remote_id IS NULL AND t.platform_alpha_id IS NULL)
    """))
