from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persistence.database import open_database
from persistence.platform_retries import platform_retry_used, record_platform_retry
from persistence.schema import add_platform_retry_storage, initialize_database_schema


def _insert_task(connection, task_id):
    connection.execute(
        "INSERT INTO backtest_tasks (task_id, account_scope, formula, formula_fingerprint, "
        "settings_json, request_fingerprint, status, created_at) "
        "VALUES (?, 'acct', ?, ?, '{}', ?, 'created', '2026-10-20T00:00:00+00:00')",
        (task_id, f"rank({task_id})", task_id, task_id),
    )


def test_retry_is_recorded_once_per_task(tmp_path):
    with open_database(tmp_path / "db.sqlite3") as connection:
        initialize_database_schema(connection)
        _insert_task(connection, "task-a")
        assert not platform_retry_used(connection, "task-a")
        record_platform_retry(connection, task_id="task-a", first_remote_id="sim-1",
                              failure_message="ERROR", retried_at="2026-10-20T01:00:00+00:00")
        assert platform_retry_used(connection, "task-a")
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            record_platform_retry(connection, task_id="task-a", first_remote_id="sim-2",
                                  failure_message="ERROR", retried_at="2026-10-20T02:00:00+00:00")


def test_retry_storage_upgrade_is_explicit_atomic_and_keeps_history(tmp_path):
    with open_database(tmp_path / "upgrade.sqlite3") as connection:
        initialize_database_schema(connection)
        _insert_task(connection, "old-task")
        connection.execute("DROP TABLE backtest_platform_retries")
        connection.commit()
        with pytest.raises(ValueError, match="database_schema_incompatible"):
            initialize_database_schema(connection)
        before = tuple(connection.iterdump())
        with pytest.raises(ValueError, match="requires_transaction"):
            add_platform_retry_storage(connection)
        connection.execute("BEGIN IMMEDIATE")
        add_platform_retry_storage(connection)
        connection.rollback()
        assert tuple(connection.iterdump()) == before
        connection.execute("BEGIN IMMEDIATE")
        add_platform_retry_storage(connection)
        add_platform_retry_storage(connection)
        connection.commit()
        initialize_database_schema(connection)
        assert connection.execute("SELECT COUNT(*) FROM backtest_tasks").fetchone()[0] == 1
