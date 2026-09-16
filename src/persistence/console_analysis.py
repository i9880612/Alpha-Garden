"""Account-scoped facts for quality diagnosis, without platform reads or writes."""
import json


def quality_facts(connection, account, *, since, until):
    rows = [dict(row) for row in connection.execute("""
        SELECT t.task_id, t.platform_alpha_id AS alpha_id, t.status,
               t.settings_json, t.finished_at, t.submission_started_at,
               r.grade, r.sharpe, r.fitness, r.turnover, m.action,
               l.run_id, a.optimization_only,
               s.observed_at AS check_at, s.payload_json AS check_payload, s.error_code AS check_error,
               f.check_observed_at AS formal_check_at, f.check_payload_json AS formal_check_payload
        FROM backtest_tasks t
        LEFT JOIN backtest_results r ON r.task_id=t.task_id
        LEFT JOIN backtest_mutations m ON m.child_task_id=t.task_id
        LEFT JOIN automated_run_backtests l ON l.task_id=t.task_id
        LEFT JOIN automated_runs a ON a.run_id=l.run_id AND a.account_scope=t.account_scope
        LEFT JOIN submission_checks s ON s.task_id=t.task_id
        LEFT JOIN formal_submission_attempts f ON f.task_id=t.task_id
        WHERE t.account_scope=? AND t.submission_started_at IS NOT NULL
          AND (? IS NULL OR julianday(COALESCE(t.finished_at,t.submission_started_at))>=julianday(?))
          AND julianday(COALESCE(t.finished_at,t.submission_started_at))<julianday(?)
        ORDER BY COALESCE(t.finished_at,t.submission_started_at) DESC,t.task_id
    """, (account, since, since, until))]
    return rows


def quality_checks(connection, account, task_ids):
    result = {task_id: [] for task_id in task_ids}
    if task_ids:
        for row in connection.execute("""
            SELECT c.task_id,c.check_name AS name,c.status,c.actual_value AS actual,c.threshold_value AS threshold
            FROM backtest_checks c JOIN backtest_tasks t ON t.task_id=c.task_id
            WHERE t.account_scope=? AND c.task_id IN (SELECT value FROM json_each(?))
            ORDER BY c.task_id,c.check_name
        """, (account, json.dumps(task_ids))):
            item = dict(row)
            result[item.pop("task_id")].append(item)
    return result
