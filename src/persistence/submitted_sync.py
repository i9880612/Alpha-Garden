"""Account-scoped evidence that a complete submitted-alpha scan finished."""
from __future__ import annotations

import json
from datetime import datetime

from persistence.submissions import list_platform_submitted_alphas
from persistence.pnl import pnl_capture_states


def initialize_submitted_sync_schema(connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS platform_submitted_alpha_syncs (
        account_scope TEXT PRIMARY KEY CHECK (length(trim(account_scope)) > 0),
        completed_at TEXT NOT NULL,
        alpha_ids_json TEXT NOT NULL
    )""")


def invalidate_submitted_sync(connection, *, account_scope: str) -> None:
    connection.execute("DELETE FROM platform_submitted_alpha_syncs WHERE account_scope=?", (account_scope,))


def complete_submitted_sync(connection, *, account_scope: str,
                            alpha_ids: tuple[str, ...], completed_at: str) -> None:
    if (not account_scope.strip() or datetime.fromisoformat(completed_at).utcoffset() is None
            or len(set(alpha_ids)) != len(alpha_ids) or any(not value.strip() for value in alpha_ids)):
        raise ValueError("submitted_sync_identity_invalid")
    stored = {record.platform_alpha_id for record in
              list_platform_submitted_alphas(connection, account_scope=account_scope)}
    if stored != set(alpha_ids):
        raise ValueError("submitted_sync_record_set_mismatch")
    _require_captured_pnl(connection, account_scope, stored)
    connection.execute("""INSERT INTO platform_submitted_alpha_syncs VALUES (?, ?, ?)
        ON CONFLICT(account_scope) DO UPDATE SET completed_at=excluded.completed_at,
        alpha_ids_json=excluded.alpha_ids_json""",
        (account_scope, completed_at, json.dumps(sorted(alpha_ids), separators=(",", ":"))))


def require_submitted_baseline(connection, *, account_scope: str) -> None:
    exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                ("platform_submitted_alpha_syncs",)).fetchone()
    row = connection.execute("SELECT alpha_ids_json FROM platform_submitted_alpha_syncs WHERE account_scope=?",
                             (account_scope,)).fetchone() if exists else None
    if row is None:
        raise ValueError("submitted_baseline_incomplete:run sync-submitted first")
    stored = {record.platform_alpha_id for record in
              list_platform_submitted_alphas(connection, account_scope=account_scope)}
    # Locally confirmed submissions may extend the last complete scan.
    if not set(json.loads(row[0])) <= stored:
        raise ValueError("submitted_baseline_record_missing")
    _require_captured_pnl(connection, account_scope, stored)


def _require_captured_pnl(connection, account_scope, alpha_ids) -> None:
    captured = {alpha for alpha, (ready, _) in pnl_capture_states(
        connection, account_scope=account_scope, platform_alpha_ids=frozenset(alpha_ids)).items() if ready}
    if not alpha_ids <= captured:
        raise ValueError("submitted_baseline_pnl_incomplete:run sync-submitted first")
