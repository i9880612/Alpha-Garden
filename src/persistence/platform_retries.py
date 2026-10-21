"""Backtests resent once after the platform reported an internal simulation error.

The platform answers some simulations with status ERROR ("There was an error while
running the simulation. Please try again"); the same request usually succeeds later.
The task keeps its identity and is sent again; this record makes that happen at most
once and keeps the first attempt's platform id.
"""
from __future__ import annotations

import sqlite3


def initialize_platform_retry_schema(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS backtest_platform_retries (
        task_id TEXT PRIMARY KEY REFERENCES backtest_tasks(task_id),
        first_remote_id TEXT NOT NULL CHECK (length(trim(first_remote_id)) > 0),
        failure_message TEXT NOT NULL,
        retried_at TEXT NOT NULL CHECK (length(trim(retried_at)) > 0)
    )""")


def record_platform_retry(connection: sqlite3.Connection, *, task_id: str, first_remote_id: str,
                          failure_message: str, retried_at: str) -> None:
    connection.execute(
        "INSERT INTO backtest_platform_retries (task_id, first_remote_id, failure_message, retried_at) "
        "VALUES (?, ?, ?, ?)",
        (task_id, first_remote_id, failure_message, retried_at),
    )


def platform_retry_used(connection: sqlite3.Connection, task_id: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM backtest_platform_retries WHERE task_id = ?", (task_id,)
    ).fetchone() is not None
