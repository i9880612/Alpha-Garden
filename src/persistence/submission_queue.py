from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from persistence.submissions import normalize_submitted_formula


@dataclass(frozen=True, slots=True)
class FormalSubmissionQueueRecord:
    task_id: str
    account_scope: str
    run_id: str
    cycle_number: int
    family_root_task_id: str
    normalized_formula: str
    enqueued_at: str


def initialize_submission_queue_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS submission_queue (
            task_id TEXT PRIMARY KEY REFERENCES backtest_tasks(task_id),
            account_scope TEXT NOT NULL CHECK (
                length(trim(account_scope)) > 0
            ),
            run_id TEXT NOT NULL REFERENCES automated_runs(run_id),
            cycle_number INTEGER NOT NULL CHECK (cycle_number > 0),
            family_root_task_id TEXT NOT NULL REFERENCES backtest_tasks(task_id),
            normalized_formula TEXT NOT NULL CHECK (
                length(normalized_formula) > 0
            ),
            enqueued_at TEXT NOT NULL CHECK (
                length(trim(enqueued_at)) > 0
            ),
            UNIQUE (account_scope, normalized_formula)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_submission_queue_run
        ON submission_queue (run_id, cycle_number, task_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_submission_queue_family
        ON submission_queue (account_scope, family_root_task_id, task_id)
        """
    )


def create_formal_submission_queue_item(
    connection: sqlite3.Connection,
    record: FormalSubmissionQueueRecord,
) -> FormalSubmissionQueueRecord:
    _validate_record(record)
    _validate_source(connection, record)
    existing = get_formal_submission_queue_item(connection, record.task_id)
    if existing is not None:
        if _record_identity(existing) != _record_identity(record):
            raise ValueError("formal_submission_queue_identity_conflict")
        return existing
    duplicate = connection.execute(
        """
        SELECT task_id
        FROM submission_queue
        WHERE account_scope = ? AND normalized_formula = ?
        """,
        (record.account_scope, record.normalized_formula),
    ).fetchone()
    if duplicate is not None:
        raise ValueError("formal_submission_queue_formula_duplicate")
    attempted = connection.execute(
        "SELECT 1 FROM formal_submission_attempts WHERE task_id = ?",
        (record.task_id,),
    ).fetchone()
    if attempted is not None:
        raise ValueError("formal_submission_queue_task_already_claimed")
    connection.execute(
        """
        INSERT INTO submission_queue (
            task_id, account_scope, run_id, cycle_number,
            family_root_task_id, normalized_formula, enqueued_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.task_id,
            record.account_scope,
            record.run_id,
            record.cycle_number,
            record.family_root_task_id,
            record.normalized_formula,
            record.enqueued_at,
        ),
    )
    return record


def get_formal_submission_queue_item(
    connection: sqlite3.Connection,
    task_id: str,
) -> FormalSubmissionQueueRecord | None:
    _require_clean_text(task_id, "formal_submission_queue_task_id_invalid")
    row = connection.execute(
        "SELECT * FROM submission_queue WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    return _record_from_row(row) if row is not None else None


def list_formal_submission_queue(
    connection: sqlite3.Connection,
    *,
    account_scope: str | None = None,
    run_id: str | None = None,
) -> tuple[FormalSubmissionQueueRecord, ...]:
    clauses: list[str] = []
    parameters: list[str] = []
    if account_scope is not None:
        _require_clean_text(
            account_scope,
            "formal_submission_queue_account_scope_invalid",
        )
        clauses.append("account_scope = ?")
        parameters.append(account_scope)
    if run_id is not None:
        _require_clean_text(run_id, "formal_submission_queue_run_id_invalid")
        clauses.append("run_id = ?")
        parameters.append(run_id)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = connection.execute(
        f"""
        SELECT * FROM submission_queue{where}
        ORDER BY enqueued_at, task_id
        """,
        parameters,
    ).fetchall()
    return tuple(_record_from_row(row) for row in rows)


def delete_formal_submission_queue_item(
    connection: sqlite3.Connection,
    task_id: str,
) -> None:
    _require_clean_text(task_id, "formal_submission_queue_task_id_invalid")
    deleted = connection.execute(
        "DELETE FROM submission_queue WHERE task_id = ?",
        (task_id,),
    )
    if deleted.rowcount != 1:
        raise ValueError("formal_submission_queue_item_missing")


def _record_from_row(row: sqlite3.Row) -> FormalSubmissionQueueRecord:
    record = FormalSubmissionQueueRecord(
        task_id=row["task_id"],
        account_scope=row["account_scope"],
        run_id=row["run_id"],
        cycle_number=row["cycle_number"],
        family_root_task_id=row["family_root_task_id"],
        normalized_formula=row["normalized_formula"],
        enqueued_at=row["enqueued_at"],
    )
    _validate_record(record)
    return record


def _record_identity(record: FormalSubmissionQueueRecord) -> tuple[object, ...]:
    return (
        record.task_id,
        record.account_scope,
        record.run_id,
        record.cycle_number,
        record.family_root_task_id,
        record.normalized_formula,
    )


def _validate_record(record: FormalSubmissionQueueRecord) -> None:
    if not isinstance(record, FormalSubmissionQueueRecord):
        raise ValueError("formal_submission_queue_record_invalid")
    for value, error in (
        (record.task_id, "formal_submission_queue_task_id_invalid"),
        (record.account_scope, "formal_submission_queue_account_scope_invalid"),
        (record.run_id, "formal_submission_queue_run_id_invalid"),
        (
            record.family_root_task_id,
            "formal_submission_queue_family_root_invalid",
        ),
        (
            record.normalized_formula,
            "formal_submission_queue_formula_invalid",
        ),
    ):
        _require_clean_text(value, error)
    if (
        isinstance(record.cycle_number, bool)
        or not isinstance(record.cycle_number, int)
        or record.cycle_number <= 0
    ):
        raise ValueError("formal_submission_queue_cycle_invalid")
    _timestamp(record.enqueued_at)


def _validate_source(
    connection: sqlite3.Connection,
    record: FormalSubmissionQueueRecord,
) -> None:
    source = connection.execute(
        """
        SELECT tasks.account_scope, tasks.formula, tasks.status,
               links.cycle_number, runs.account_scope AS run_account_scope
        FROM backtest_tasks AS tasks
        JOIN automated_run_backtests AS links ON links.task_id = tasks.task_id
        JOIN automated_runs AS runs ON runs.run_id = links.run_id
        WHERE tasks.task_id = ? AND links.run_id = ?
        """,
        (record.task_id, record.run_id),
    ).fetchone()
    root = connection.execute(
        "SELECT account_scope FROM backtest_tasks WHERE task_id = ?",
        (record.family_root_task_id,),
    ).fetchone()
    if (
        source is None
        or source["status"] != "completed"
        or source["account_scope"] != record.account_scope
        or source["run_account_scope"] != record.account_scope
        or source["cycle_number"] != record.cycle_number
        or normalize_submitted_formula(source["formula"]) != record.normalized_formula
        or root is None
        or root["account_scope"] != record.account_scope
    ):
        raise ValueError("formal_submission_queue_source_invalid")


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("formal_submission_queue_enqueued_at_invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("formal_submission_queue_enqueued_at_invalid") from exc
    if parsed.utcoffset() is None:
        raise ValueError("formal_submission_queue_enqueued_at_invalid")
    return parsed


def _require_clean_text(value: object, error: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(error)
