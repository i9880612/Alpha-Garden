"""Storage measurements and PnL dependency facts; no cleanup or platform reads."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StorageMeasurements:
    page_size: int
    page_count: int
    free_pages: int
    object_bytes: tuple[tuple[str, int], ...] | None
    pnl_count: int
    captured_count: int
    payload_bytes: int


def measure_storage(connection: sqlite3.Connection) -> StorageMeasurements:
    page_size = connection.execute("PRAGMA page_size").fetchone()[0]
    page_count = connection.execute("PRAGMA page_count").fetchone()[0]
    free_pages = connection.execute("PRAGMA freelist_count").fetchone()[0]
    # dbstat is an optional SQLite build feature. Other SQL failures still escape.
    has_dbstat = any(row[0] == "dbstat" for row in connection.execute("PRAGMA module_list"))
    sizes = tuple((row[0], row[1]) for row in connection.execute(
        "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY SUM(pgsize) DESC, name"
    )) if has_dbstat else None
    counts = connection.execute(
        "SELECT COUNT(*), COUNT(points_json), COALESCE(SUM(length(CAST(points_json AS BLOB))), 0) "
        "FROM platform_pnl_series"
    ).fetchone()
    return StorageMeasurements(page_size, page_count, free_pages, sizes, *counts)


@dataclass(frozen=True, slots=True)
class PnlRetentionFacts:
    account_scope: str
    platform_alpha_id: str
    observed_at: str
    payload_bytes: int | None
    submitted: bool
    recovery_reference: bool
    task_count: int
    completed_count: int
    failed_check_count: int
    protected_task_count: int
    latest_task_observation: str | None
    unknown_task_time: bool


def iter_pnl_retention_facts(connection: sqlite3.Connection) -> Iterator[PnlRetentionFacts]:
    """Stream metadata only, joining alpha identities within their account.

    All lineage members and explicit research/submission dependencies are retained
    for the first preview policy, including retired parents and failed descendants.
    """
    rows = connection.execute("""
        WITH protected_tasks(task_id) AS (
            SELECT root_task_id FROM signal_seeds
            UNION SELECT parent_task_id FROM backtest_mutations
            UNION SELECT child_task_id FROM backtest_mutations
            UNION SELECT task_id FROM qualified_alpha_archive
            UNION SELECT task_id FROM submission_queue
            UNION SELECT family_root_task_id FROM submission_queue
            UNION SELECT task_id FROM formal_submission_attempts
            UNION SELECT family_root_task_id FROM formal_submission_attempts
            UNION SELECT task_id FROM submission_checks
            UNION SELECT b.task_id FROM automated_run_backtests b
                JOIN automated_runs r USING(run_id) WHERE r.status != 'completed'
        ), failed_tasks AS (
            SELECT DISTINCT task_id FROM backtest_checks
            WHERE status = 'FAIL' AND check_name != 'SELF_CORRELATION'
        ), task_facts AS (
            SELECT t.account_scope, t.platform_alpha_id, COUNT(*) AS task_count,
                SUM(t.status = 'completed') AS completed_count,
                SUM(f.task_id IS NOT NULL) AS failed_count,
                SUM(p.task_id IS NOT NULL) AS protected_count,
                MAX(julianday(t.last_observed_at)) AS latest_observation,
                SUM(julianday(t.last_observed_at) IS NULL OR julianday(t.finished_at) IS NULL) AS unknown_time
            FROM backtest_tasks t
            LEFT JOIN protected_tasks p USING(task_id)
            LEFT JOIN failed_tasks f USING(task_id)
            WHERE t.platform_alpha_id IS NOT NULL
            GROUP BY t.account_scope, t.platform_alpha_id
        ), recovery_references AS (
            SELECT DISTINCT t.account_scope, r.conflict_reference_alpha_id AS platform_alpha_id
            FROM backtest_mutation_references r JOIN backtest_tasks t ON t.task_id = r.child_task_id
        )
        SELECT p.account_scope, p.platform_alpha_id, p.observed_at,
            length(CAST(p.points_json AS BLOB)), s.platform_alpha_id IS NOT NULL,
            r.platform_alpha_id IS NOT NULL, COALESCE(t.task_count, 0),
            COALESCE(t.completed_count, 0), COALESCE(t.failed_count, 0),
            COALESCE(t.protected_count, 0),
            strftime('%Y-%m-%dT%H:%M:%f+00:00', t.latest_observation),
            COALESCE(t.unknown_time, 0) > 0
        FROM platform_pnl_series p
        LEFT JOIN platform_submitted_alphas s USING(account_scope, platform_alpha_id)
        LEFT JOIN recovery_references r USING(account_scope, platform_alpha_id)
        LEFT JOIN task_facts t USING(account_scope, platform_alpha_id)
        ORDER BY p.account_scope, p.platform_alpha_id
    """)
    for row in rows:
        yield PnlRetentionFacts(*row)
