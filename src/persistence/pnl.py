from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from worldquant.pnl import parse_pnl


@dataclass(frozen=True, slots=True)
class PnlSeriesRecord:
    account_scope: str
    platform_alpha_id: str
    observed_at: str
    points: tuple[tuple[str, float], ...] | None
    retry_not_before: str | None = None


def initialize_pnl_schema(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS platform_pnl_series (
        account_scope TEXT NOT NULL,
        platform_alpha_id TEXT NOT NULL,
        observed_at TEXT NOT NULL,
        points_json TEXT,
        retry_not_before TEXT,
        PRIMARY KEY (account_scope, platform_alpha_id)
    )""")


def list_pnl_series(connection: sqlite3.Connection) -> tuple[PnlSeriesRecord, ...]:
    return tuple(
        PnlSeriesRecord(
            row[0],
            row[1],
            row[2],
            tuple((day, value) for day, value in json.loads(row[3]))
            if row[3] is not None
            else None,
            row[4],
        )
        for row in connection.execute(
            "SELECT account_scope, platform_alpha_id, observed_at, points_json, retry_not_before FROM platform_pnl_series"
        )
    )


def save_pnl_series(connection: sqlite3.Connection, record: PnlSeriesRecord) -> None:
    if (
        not record.account_scope.strip()
        or not record.platform_alpha_id.strip()
        or datetime.fromisoformat(record.observed_at).utcoffset() is None
    ):
        raise ValueError("pnl_series_identity_invalid")
    if record.points is not None:
        if record.retry_not_before is not None:
            raise ValueError("pnl_series_retry_conflict")
        parse_pnl(
            {
                "schema": {"properties": [{"name": "date"}, {"name": "pnl"}]},
                "records": [list(point) for point in record.points],
            }
        )
    elif (
        record.retry_not_before is None
        or datetime.fromisoformat(record.retry_not_before).utcoffset() is None
        or datetime.fromisoformat(record.retry_not_before)
        <= datetime.fromisoformat(record.observed_at)
    ):
        raise ValueError("pnl_series_pending_retry_missing")
    # Immutable captured facts; a conflicting capture must not silently replace it.
    values = (
        record.account_scope,
        record.platform_alpha_id,
        record.observed_at,
        json.dumps(record.points, separators=(",", ":"))
        if record.points is not None
        else None,
        record.retry_not_before,
    )
    existing = connection.execute(
        "SELECT points_json FROM platform_pnl_series WHERE account_scope=? AND platform_alpha_id=?",
        values[:2],
    ).fetchone()
    if existing is not None and existing[0] is not None:
        if existing[0] != values[3]:
            raise ValueError("pnl_series_capture_conflict")
        return
    connection.execute(
        """INSERT INTO platform_pnl_series VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(account_scope, platform_alpha_id) DO UPDATE SET
        observed_at=excluded.observed_at, points_json=excluded.points_json,
        retry_not_before=excluded.retry_not_before""",
        values,
    )
