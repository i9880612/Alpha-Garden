from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SignalSeedRecord:
    root_task_id: str
    promoted_at: str


def initialize_signal_seed_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_seeds (
            root_task_id TEXT PRIMARY KEY REFERENCES backtest_tasks(task_id),
            promoted_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_signal_seeds_promoted
        ON signal_seeds (promoted_at, root_task_id)
        """
    )


def create_signal_seed(
    connection: sqlite3.Connection,
    record: SignalSeedRecord,
) -> SignalSeedRecord:
    _validate_record(record)
    _require_completed_task(connection, record.root_task_id)
    existing = get_signal_seed(connection, record.root_task_id)
    if existing is not None:
        if existing != record:
            raise ValueError("signal_seed_identity_conflict")
        return existing
    connection.execute(
        """
        INSERT INTO signal_seeds (root_task_id, promoted_at)
        VALUES (?, ?)
        """,
        (record.root_task_id, record.promoted_at),
    )
    return record


def get_signal_seed(
    connection: sqlite3.Connection,
    root_task_id: str,
) -> SignalSeedRecord | None:
    row = connection.execute(
        "SELECT * FROM signal_seeds WHERE root_task_id = ?",
        (root_task_id,),
    ).fetchone()
    return _record_from_row(row) if row is not None else None


def delete_signal_seed(connection: sqlite3.Connection, root_task_id: str) -> bool:
    """Delete membership only; the caller owns the transaction and decision."""
    return connection.execute(
        "DELETE FROM signal_seeds WHERE root_task_id = ?", (root_task_id,),
    ).rowcount == 1


def list_signal_seeds(
    connection: sqlite3.Connection,
) -> tuple[SignalSeedRecord, ...]:
    rows = connection.execute(
        """
        SELECT * FROM signal_seeds
        ORDER BY promoted_at, root_task_id
        """
    ).fetchall()
    return tuple(_record_from_row(row) for row in rows)


def _require_completed_task(
    connection: sqlite3.Connection,
    task_id: str,
) -> None:
    task = connection.execute(
        "SELECT status FROM backtest_tasks WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if task is None:
        raise ValueError("signal_seed_task_missing")
    if task["status"] != "completed":
        raise ValueError("signal_seed_task_not_completed")


def _validate_record(record: SignalSeedRecord) -> None:
    if not isinstance(record, SignalSeedRecord):
        raise ValueError("signal_seed_record_invalid")
    if not isinstance(record.root_task_id, str) or not record.root_task_id.strip():
        raise ValueError("signal_seed_task_id_missing")
    _timestamp(record.promoted_at)


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("signal_seed_timestamp_invalid")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("signal_seed_timestamp_invalid") from exc
    if timestamp.utcoffset() is None:
        raise ValueError("signal_seed_timestamp_invalid")
    return timestamp


def _record_from_row(row: sqlite3.Row) -> SignalSeedRecord:
    return SignalSeedRecord(
        root_task_id=row["root_task_id"],
        promoted_at=row["promoted_at"],
    )
