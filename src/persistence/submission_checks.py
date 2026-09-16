from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SubmissionCheckRecord:
    task_id: str
    observed_at: str
    payload_json: str | None
    error_code: str | None
    attempt_count: int = 1
    retry_not_before: str | None = None


def initialize_submission_check_schema(connection: sqlite3.Connection) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS submission_checks (
            task_id TEXT PRIMARY KEY REFERENCES backtest_tasks(task_id),
            observed_at TEXT NOT NULL,
            payload_json TEXT,
            error_code TEXT,
            attempt_count INTEGER NOT NULL CHECK (attempt_count > 0),
            retry_not_before TEXT,
            CHECK ((payload_json IS NOT NULL AND error_code IS NULL)
                OR (payload_json IS NULL AND error_code IS NOT NULL))
        )
    """)


def get_submission_check(connection: sqlite3.Connection, task_id: str) -> SubmissionCheckRecord | None:
    row = connection.execute(
        "SELECT task_id, observed_at, payload_json, error_code, attempt_count, retry_not_before "
        "FROM submission_checks WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    return SubmissionCheckRecord(*row) if row is not None else None


def list_submission_checks(connection: sqlite3.Connection) -> tuple[SubmissionCheckRecord, ...]:
    rows = connection.execute(
        "SELECT task_id, observed_at, payload_json, error_code, attempt_count, retry_not_before "
        "FROM submission_checks ORDER BY task_id"
    ).fetchall()
    return tuple(SubmissionCheckRecord(*row) for row in rows)


def save_submission_check(connection: sqlite3.Connection, record: SubmissionCheckRecord) -> None:
    if not record.task_id or datetime.fromisoformat(record.observed_at).utcoffset() is None:
        raise ValueError("submission_check_identity_invalid")
    if (record.payload_json is None) == (record.error_code is None):
        raise ValueError("submission_check_observation_invalid")
    if record.payload_json is not None:
        json.loads(record.payload_json)
    if record.attempt_count < 1:
        raise ValueError("submission_check_attempt_count_invalid")
    if record.retry_not_before is not None:
        retry_at = datetime.fromisoformat(record.retry_not_before)
        if retry_at.utcoffset() is None or retry_at <= datetime.fromisoformat(record.observed_at):
            raise ValueError("submission_check_retry_time_invalid")
    existing = get_submission_check(connection, record.task_id)
    if existing is not None:
        if existing == record:
            return
        if (record.attempt_count != existing.attempt_count + 1
                or datetime.fromisoformat(record.observed_at) < datetime.fromisoformat(existing.retry_not_before or existing.observed_at)):
            raise ValueError("submission_check_observation_conflict")
    elif record.attempt_count != 1:
        raise ValueError("submission_check_attempt_count_invalid")
    connection.execute(
        "INSERT INTO submission_checks "
        "(task_id, observed_at, payload_json, error_code, attempt_count, retry_not_before) "
        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(task_id) DO UPDATE SET "
        "observed_at=excluded.observed_at, payload_json=excluded.payload_json, "
        "error_code=excluded.error_code, attempt_count=excluded.attempt_count, "
        "retry_not_before=excluded.retry_not_before",
        (record.task_id, record.observed_at, record.payload_json, record.error_code,
         record.attempt_count, record.retry_not_before),
    )
