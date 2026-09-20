"""Read-only facts for the local console; no schema creation or reconciliation."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def read_console_database(path: Path):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    try:
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()


def formula_page(connection, account_scope, *, run_id, execution, search, grade, page, page_size):
    scope = "t.account_scope=?" + (" AND l.run_id=?" if run_id else "")
    parameters = [account_scope, *([run_id] if run_id else [])]
    source = """FROM backtest_tasks t
        LEFT JOIN automated_run_backtests l ON l.task_id=t.task_id
        LEFT JOIN backtest_results r ON r.task_id=t.task_id"""
    group = """CASE WHEN t.status IN ('completed','failed') THEN 'finished'
        WHEN t.status='created' AND t.submission_started_at IS NULL THEN 'planned'
        ELSE 'inflight' END"""
    counts = dict(connection.execute(
        f"SELECT {group}, COUNT(*) {source} WHERE {scope} GROUP BY 1", parameters))
    if execution == "started":
        scope += " AND t.submission_started_at IS NOT NULL"
    elif execution != "all":
        scope += f" AND ({group})=?"
        parameters.append(execution)
    if search:
        scope += " AND instr(lower(COALESCE(NULLIF(t.platform_alpha_id,''),t.task_id)),?)>0"
        parameters.append(search.lower())
    if grade:
        scope += " AND COALESCE(NULLIF(r.grade,''),'UNKNOWN')=?"
        parameters.append(grade)
    total = connection.execute(f"SELECT COUNT(*) {source} WHERE {scope}", parameters).fetchone()[0]
    task_ids = tuple(row[0] for row in connection.execute(f"""
        SELECT t.task_id {source} WHERE {scope}
        ORDER BY COALESCE(t.finished_at,t.created_at) DESC,t.task_id LIMIT ? OFFSET ?
    """, (*parameters, page_size, (page-1)*page_size)))
    return task_ids, total, counts


def attempted_task_ids(connection, account_scope, task_ids):
    if not task_ids:
        return frozenset()
    return frozenset(row[0] for row in connection.execute("""
        SELECT task_id FROM backtest_tasks WHERE account_scope=? AND submission_started_at IS NOT NULL
        AND task_id IN (SELECT value FROM json_each(?))
    """, (account_scope, json.dumps(task_ids))))


def formula_facts(connection, account_scope, *, task_ids=None):
    if task_ids is not None and not task_ids:
        return []
    scope = " AND t.task_id IN (SELECT value FROM json_each(?))" if task_ids is not None else ""
    return [dict(row) for row in connection.execute("""
        SELECT t.task_id, t.platform_alpha_id AS alpha_id, t.status, t.created_at,
               t.finished_at, t.submission_started_at, t.failure_code,
               r.grade, r.sharpe, r.fitness, r.turnover, r.returns, r.drawdown, r.margin,
               m.parent_task_id, p.platform_alpha_id AS parent_alpha_id, m.action,
               l.run_id, l.cycle_number, a.archived_at, q.enqueued_at,
               s.observed_at AS check_at, s.payload_json AS check_payload, s.error_code AS check_error,
               f.check_observed_at AS formal_check_at, f.check_payload_json AS formal_check_payload,
               f.status AS submission_status
        FROM backtest_tasks t
        LEFT JOIN backtest_results r ON r.task_id=t.task_id
        LEFT JOIN backtest_mutations m ON m.child_task_id=t.task_id
        LEFT JOIN backtest_tasks p ON p.task_id=m.parent_task_id AND p.account_scope=t.account_scope
        LEFT JOIN automated_run_backtests l ON l.task_id=t.task_id
        LEFT JOIN qualified_alpha_archive a ON a.task_id=t.task_id
        LEFT JOIN submission_queue q ON q.task_id=t.task_id
        LEFT JOIN submission_checks s ON s.task_id=t.task_id
        LEFT JOIN formal_submission_attempts f ON f.task_id=t.task_id
        WHERE t.account_scope=?
    """ + scope + " ORDER BY COALESCE(t.finished_at,t.created_at) DESC, t.task_id",
        (account_scope, json.dumps(task_ids)) if task_ids is not None else (account_scope,))]


def run_backtest_counts(connection, account_scope):
    return {row["run_id"]: dict(row) for row in connection.execute("""
        SELECT l.run_id, COUNT(*) AS planned,
               SUM(t.status IN ('completed','failed')) AS finished,
               SUM(t.status='completed') AS completed
        FROM automated_run_backtests l JOIN backtest_tasks t ON t.task_id=l.task_id
        WHERE t.account_scope=? GROUP BY l.run_id
    """, (account_scope,))}


def run_facts(connection, account_scope, *, limit=None):
    rows = [dict(row) for row in connection.execute("""
        SELECT run_id, status, current_cycle, max_cycles, max_backtests, backtest_count,
               optimization_only, automatic_submissions_enabled, created_at, started_at,
               finished_at, stop_reason, last_request_failure_code, retry_not_before
        FROM automated_runs WHERE account_scope=? ORDER BY created_at DESC, run_id
    """ + (" LIMIT ?" if limit is not None else ""),
        (account_scope, limit) if limit is not None else (account_scope,))]
    for row in rows:
        row["optimization_only"] = bool(row["optimization_only"])
        row["automatic_submissions_enabled"] = bool(row["automatic_submissions_enabled"])
    return rows


def failed_backtest_checks(connection, account_scope, task_ids):
    if not task_ids:
        return {}
    checks = {}
    for row in connection.execute(f"""
        SELECT c.task_id, c.check_name AS name, c.actual_value AS actual,
               c.threshold_value AS threshold
        FROM backtest_checks c JOIN backtest_tasks t ON t.task_id=c.task_id
        WHERE t.account_scope=? AND c.task_id IN ({','.join('?' for _ in task_ids)})
              AND UPPER(c.status)='FAIL'
        ORDER BY c.task_id, c.check_name
    """, (account_scope, *task_ids)):
        item = dict(row)
        checks.setdefault(item.pop("task_id"), []).append(item)
    return checks


def completed_dates(connection, account_scope, *, since):
    return tuple(row[0] for row in connection.execute("""
        SELECT finished_at FROM backtest_tasks
        WHERE account_scope=? AND status='completed' AND julianday(finished_at)>=julianday(?)
    """, (account_scope, since)))


def archive_facts(connection, account_scope):
    return [dict(row) for row in connection.execute("""
        SELECT a.task_id, a.archived_at FROM qualified_alpha_archive a
        JOIN backtest_tasks t ON t.task_id=a.task_id WHERE t.account_scope=?
    """, (account_scope,))]


def latest_cycle_backtest_facts(connection, account_scope, run_id):
    return [dict(row) for row in connection.execute("""
        SELECT l.cycle_number, t.status, m.action FROM automated_run_backtests l
        JOIN backtest_tasks t ON t.task_id=l.task_id
        LEFT JOIN backtest_mutations m ON m.child_task_id=t.task_id
        WHERE t.account_scope=? AND l.run_id=? AND l.cycle_number=(
            SELECT MAX(cycle_number) FROM automated_run_backtests WHERE run_id=l.run_id
        )
    """, (account_scope, run_id))]


def recent_backtest_facts(connection, account_scope):
    return [dict(row) for row in connection.execute("""
        SELECT t.task_id, t.platform_alpha_id AS alpha_id, t.formula, r.grade, r.sharpe
        FROM backtest_tasks t LEFT JOIN backtest_results r ON r.task_id=t.task_id
        WHERE t.account_scope=? AND t.status='completed'
        ORDER BY t.finished_at DESC, t.task_id LIMIT 4
    """, (account_scope,))]


def submitted_facts(connection, account_scope):
    return [dict(row) for row in connection.execute("""
        SELECT platform_alpha_id AS alpha_id, status, date_submitted, observed_at,
               json_extract(raw_payload_json, '$.grade') AS grade
        FROM platform_submitted_alphas WHERE account_scope=?
        ORDER BY date_submitted DESC, platform_alpha_id
    """, (account_scope,))]


def submission_facts(connection, account_scope):
    return [dict(row) for row in connection.execute("""
        SELECT f.task_id, t.platform_alpha_id AS alpha_id, r.grade, f.status, f.source,
               f.created_at, f.updated_at, f.failure_code, f.check_attempt_count
        FROM formal_submission_attempts f JOIN backtest_tasks t ON t.task_id=f.task_id
        LEFT JOIN backtest_results r ON r.task_id=t.task_id
        WHERE t.account_scope=? ORDER BY f.updated_at DESC, f.task_id
    """, (account_scope,))]
