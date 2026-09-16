from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime

from generation.parser import FormulaSyntaxError, parse_formula
from persistence.seeds import initialize_signal_seed_schema
from persistence.qualified_archive import initialize_qualified_archive_schema
from persistence.pnl import initialize_pnl_schema
from persistence.submission_checks import initialize_submission_check_schema


BACKTEST_ACTIVE_STATUSES = frozenset({"created", "submission_unknown", "pending"})
BACKTEST_TERMINAL_STATUSES = frozenset({"completed", "failed"})
BACKTEST_CANCELLED_BEFORE_SUBMISSION_SQL = (
    "status = 'failed' AND failure_code = 'automated_run_stopped_before_submission' "
    "AND submission_started_at IS NULL AND remote_id IS NULL AND platform_alpha_id IS NULL"
)


@dataclass(frozen=True, slots=True)
class BacktestTaskRecord:
    task_id: str
    account_scope: str
    formula: str
    formula_fingerprint: str
    settings_json: str
    request_fingerprint: str
    status: str
    remote_id: str | None
    platform_alpha_id: str | None
    created_at: str
    submission_started_at: str | None
    last_observed_at: str | None
    retry_not_before: str | None
    finished_at: str | None
    failure_code: str | None
    failure_message: str | None


@dataclass(frozen=True, slots=True)
class BacktestCheckRecord:
    name: str
    status: str
    threshold: float | None
    actual: float | None
    platform_date: str | None


@dataclass(frozen=True, slots=True)
class BacktestResultRecord:
    task_id: str
    sharpe: float
    fitness: float
    turnover: float
    returns: float
    drawdown: float
    margin: float
    book_size: float | None
    pnl: float | None
    long_count: int | None
    short_count: int | None
    check_details_captured: bool
    checks: tuple[BacktestCheckRecord, ...]
    grade: str | None = None


@dataclass(frozen=True, slots=True)
class BacktestYearlyStatRecord:
    task_id: str
    year: int
    pnl: float | None
    book_size: float | None
    long_count: int | None
    short_count: int | None
    turnover: float | None
    sharpe: float | None
    returns: float | None
    drawdown: float | None
    margin: float | None
    fitness: float | None
    stage: str


@dataclass(frozen=True, slots=True)
class BacktestMutationRecord:
    child_task_id: str
    parent_task_id: str
    action: str
    location: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class BacktestSnapshot:
    task: BacktestTaskRecord
    result: BacktestResultRecord | None
    yearly_stats: tuple[BacktestYearlyStatRecord, ...] | None


def backtest_was_cancelled_before_submission(task: BacktestTaskRecord) -> bool:
    return (task.status == "failed"
            and task.failure_code == "automated_run_stopped_before_submission"
            and task.submission_started_at is None
            and task.remote_id is None and task.platform_alpha_id is None)


def initialize_backtest_schema(connection: sqlite3.Connection) -> None:
    initialize_pnl_schema(connection)
    _create_backtest_task_table(connection)
    initialize_submission_check_schema(connection)
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_tasks_recovery
        ON backtest_tasks (status, created_at, task_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_tasks_formula
        ON backtest_tasks (formula_fingerprint, task_id)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_results (
            task_id TEXT PRIMARY KEY REFERENCES backtest_tasks(task_id),
            sharpe REAL NOT NULL,
            fitness REAL NOT NULL,
            turnover REAL NOT NULL,
            returns REAL NOT NULL,
            drawdown REAL NOT NULL,
            margin REAL NOT NULL,
            book_size REAL,
            pnl REAL,
            long_count INTEGER CHECK (long_count IS NULL OR long_count >= 0),
            short_count INTEGER CHECK (short_count IS NULL OR short_count >= 0),
            check_details_captured INTEGER NOT NULL CHECK (
                check_details_captured IN (0, 1)
            ),
            grade TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_yearly_stats_captures (
            task_id TEXT PRIMARY KEY REFERENCES backtest_tasks(task_id)
                ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_yearly_stats (
            task_id TEXT NOT NULL REFERENCES backtest_yearly_stats_captures(task_id)
                ON DELETE CASCADE,
            stage TEXT NOT NULL,
            year INTEGER NOT NULL CHECK (year > 0),
            pnl REAL,
            book_size REAL,
            long_count INTEGER CHECK (long_count IS NULL OR long_count >= 0),
            short_count INTEGER CHECK (short_count IS NULL OR short_count >= 0),
            turnover REAL,
            sharpe REAL,
            returns REAL,
            drawdown REAL,
            margin REAL,
            fitness REAL,
            PRIMARY KEY (task_id, stage, year)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_checks (
            task_id TEXT NOT NULL REFERENCES backtest_results(task_id)
                ON DELETE CASCADE,
            check_name TEXT NOT NULL,
            status TEXT NOT NULL,
            threshold_value REAL,
            actual_value REAL,
            platform_date TEXT,
            PRIMARY KEY (task_id, check_name)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_mutations (
            child_task_id TEXT PRIMARY KEY REFERENCES backtest_tasks(task_id),
            parent_task_id TEXT NOT NULL REFERENCES backtest_tasks(task_id),
            action TEXT NOT NULL,
            location TEXT NOT NULL,
            before TEXT NOT NULL,
            after TEXT NOT NULL,
            CHECK (child_task_id <> parent_task_id)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_mutations_parent
        ON backtest_mutations (parent_task_id, child_task_id)
        """
    )
    initialize_signal_seed_schema(connection)
    initialize_qualified_archive_schema(connection)


def create_backtest_task(
    connection: sqlite3.Connection,
    task: BacktestTaskRecord,
) -> BacktestTaskRecord:
    _validate_task(task)
    if task.status != "created":
        raise ValueError("backtest_initial_status_invalid")
    existing = get_backtest_task(connection, task.task_id)
    if existing is not None:
        if _reservation_identity(existing.task) != _reservation_identity(task):
            raise ValueError("backtest_task_identity_conflict")
        return existing.task
    reserved = get_backtest_task_by_identity(
        connection,
        account_scope=task.account_scope,
        formula_fingerprint=task.formula_fingerprint,
        settings_json=task.settings_json,
    )
    if reserved is not None and not backtest_was_cancelled_before_submission(reserved.task):
        return reserved.task
    connection.execute(
        """
        INSERT INTO backtest_tasks (
            task_id, account_scope, formula, formula_fingerprint, settings_json,
            request_fingerprint, status, remote_id, platform_alpha_id,
            created_at, submission_started_at, last_observed_at,
            retry_not_before, finished_at, failure_code, failure_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _task_values(task),
    )
    return task


def replace_backtest_task(
    connection: sqlite3.Connection,
    task: BacktestTaskRecord,
    *,
    expected_status: str,
) -> BacktestTaskRecord:
    _validate_task(task)
    existing = get_backtest_task(connection, task.task_id)
    if existing is None:
        raise ValueError("backtest_task_missing")
    if _task_identity(existing.task) != _task_identity(task):
        raise ValueError("backtest_task_identity_conflict")
    if existing.task == task:
        return existing.task
    if existing.task.status != expected_status:
        raise ValueError("backtest_transition_conflict")
    cursor = connection.execute(
        """
        UPDATE backtest_tasks SET
            status = ?, remote_id = ?, platform_alpha_id = ?,
            submission_started_at = ?, last_observed_at = ?,
            retry_not_before = ?, finished_at = ?,
            failure_code = ?, failure_message = ?
        WHERE task_id = ? AND status = ?
        """,
        (
            task.status,
            task.remote_id,
            task.platform_alpha_id,
            task.submission_started_at,
            task.last_observed_at,
            task.retry_not_before,
            task.finished_at,
            task.failure_code,
            task.failure_message,
            task.task_id,
            expected_status,
        ),
    )
    if cursor.rowcount != 1:
        raise ValueError("backtest_transition_conflict")
    return task


def save_backtest_result(
    connection: sqlite3.Connection,
    result: BacktestResultRecord,
) -> BacktestResultRecord:
    _validate_result(result)
    task = get_backtest_task(connection, result.task_id)
    if task is None:
        raise ValueError("backtest_task_missing")
    existing = get_backtest_result(connection, result.task_id)
    if existing is not None:
        if existing != result:
            raise ValueError("backtest_result_conflict")
        return existing
    if (
        task.task.status != "pending"
        or task.task.platform_alpha_id is None
        or task.yearly_stats is not None
    ):
        raise ValueError("backtest_result_task_not_ready")
    connection.execute(
        """
        INSERT INTO backtest_results (
            task_id, sharpe, fitness, turnover, returns, drawdown,
            margin, book_size, pnl, long_count, short_count,
            check_details_captured, grade
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.task_id,
            result.sharpe,
            result.fitness,
            result.turnover,
            result.returns,
            result.drawdown,
            result.margin,
            result.book_size,
            result.pnl,
            result.long_count,
            result.short_count,
            int(result.check_details_captured),
            result.grade,
        ),
    )
    connection.executemany(
        """
        INSERT INTO backtest_checks (
            task_id, check_name, status, threshold_value,
            actual_value, platform_date
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            (
                result.task_id,
                check.name,
                check.status,
                check.threshold,
                check.actual,
                check.platform_date,
            )
            for check in result.checks
        ),
    )
    return result


def save_backtest_yearly_stats(
    connection: sqlite3.Connection,
    task_id: str,
    yearly_stats: tuple[BacktestYearlyStatRecord, ...],
) -> tuple[BacktestYearlyStatRecord, ...]:
    _require_text(task_id, "backtest_yearly_stats_task_id_missing")
    if not isinstance(yearly_stats, tuple):
        raise ValueError("backtest_yearly_stats_invalid")
    for record in yearly_stats:
        _validate_yearly_stat(record)
        if record.task_id != task_id:
            raise ValueError("backtest_yearly_stats_task_identity_mismatch")
    keys = tuple((record.stage, record.year) for record in yearly_stats)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError("backtest_yearly_stats_invalid")

    snapshot = get_backtest_task(connection, task_id)
    if snapshot is None:
        raise ValueError("backtest_task_missing")
    existing = snapshot.yearly_stats
    if existing is not None:
        if existing != yearly_stats:
            raise ValueError("backtest_yearly_stats_conflict")
        return existing
    if (
        snapshot.task.status != "pending"
        or snapshot.task.platform_alpha_id is None
        or snapshot.result is None
    ):
        raise ValueError("backtest_yearly_stats_task_not_ready")

    connection.execute(
        "INSERT INTO backtest_yearly_stats_captures (task_id) VALUES (?)",
        (task_id,),
    )
    connection.executemany(
        """
        INSERT INTO backtest_yearly_stats (
            task_id, stage, year, pnl, book_size, long_count, short_count,
            turnover, sharpe, returns, drawdown, margin, fitness
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                record.task_id,
                record.stage,
                record.year,
                record.pnl,
                record.book_size,
                record.long_count,
                record.short_count,
                record.turnover,
                record.sharpe,
                record.returns,
                record.drawdown,
                record.margin,
                record.fitness,
            )
            for record in yearly_stats
        ),
    )
    return yearly_stats


def create_backtest_mutation(
    connection: sqlite3.Connection,
    mutation: BacktestMutationRecord,
) -> BacktestMutationRecord:
    _validate_mutation(mutation)
    child = get_backtest_task(connection, mutation.child_task_id)
    parent = get_backtest_task(connection, mutation.parent_task_id)
    if child is None:
        raise ValueError("backtest_mutation_child_missing")
    if parent is None:
        raise ValueError("backtest_mutation_parent_missing")
    existing = get_backtest_mutation(connection, mutation.child_task_id)
    if existing is not None:
        if existing != mutation:
            raise ValueError("backtest_mutation_conflict")
        return existing
    connection.execute(
        """
        INSERT INTO backtest_mutations (
            child_task_id, parent_task_id, action, location, before, after
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            mutation.child_task_id,
            mutation.parent_task_id,
            mutation.action,
            mutation.location,
            mutation.before,
            mutation.after,
        ),
    )
    return mutation


def get_backtest_task(
    connection: sqlite3.Connection,
    task_id: str,
) -> BacktestSnapshot | None:
    row = connection.execute(
        "SELECT * FROM backtest_tasks WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    return BacktestSnapshot(
        task=_task_from_row(row),
        result=get_backtest_result(connection, task_id),
        yearly_stats=get_backtest_yearly_stats(connection, task_id),
    )


def get_backtest_task_by_identity(
    connection: sqlite3.Connection,
    *,
    account_scope: str,
    formula_fingerprint: str,
    settings_json: str,
) -> BacktestSnapshot | None:
    row = connection.execute(
        """
        SELECT task_id FROM backtest_tasks
        WHERE account_scope = ?
          AND formula_fingerprint = ?
          AND settings_json = ?
        """ + f" ORDER BY ({BACKTEST_CANCELLED_BEFORE_SUBMISSION_SQL}), created_at DESC, task_id DESC LIMIT 1",
        (account_scope, formula_fingerprint, settings_json),
    ).fetchone()
    return get_backtest_task(connection, row["task_id"]) if row is not None else None


def list_backtest_tasks_by_platform_alpha_id(
    connection: sqlite3.Connection,
    *,
    account_scope: str,
    platform_alpha_id: str,
) -> tuple[BacktestTaskRecord, ...]:
    _require_text(account_scope, "backtest_account_scope_missing")
    _require_text(platform_alpha_id, "backtest_platform_alpha_id_invalid")
    rows = connection.execute(
        """
        SELECT * FROM backtest_tasks
        WHERE account_scope = ? AND platform_alpha_id = ?
        ORDER BY task_id
        """,
        (account_scope, platform_alpha_id),
    ).fetchall()
    return tuple(_task_from_row(row) for row in rows)


def get_backtest_result(
    connection: sqlite3.Connection,
    task_id: str,
) -> BacktestResultRecord | None:
    row = connection.execute(
        "SELECT * FROM backtest_results WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    check_rows = connection.execute(
        """
        SELECT * FROM backtest_checks
        WHERE task_id = ?
        ORDER BY check_name
        """,
        (task_id,),
    ).fetchall()
    return _result_from_row(row, check_rows)


def get_backtest_yearly_stats(
    connection: sqlite3.Connection,
    task_id: str,
) -> tuple[BacktestYearlyStatRecord, ...] | None:
    captured = connection.execute(
        "SELECT 1 FROM backtest_yearly_stats_captures WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if captured is None:
        return None
    rows = connection.execute(
        """
        SELECT * FROM backtest_yearly_stats
        WHERE task_id = ?
        ORDER BY stage, year
        """,
        (task_id,),
    ).fetchall()
    return tuple(_yearly_stat_from_row(row) for row in rows)


def get_backtest_mutation(
    connection: sqlite3.Connection,
    child_task_id: str,
) -> BacktestMutationRecord | None:
    row = connection.execute(
        "SELECT * FROM backtest_mutations WHERE child_task_id = ?",
        (child_task_id,),
    ).fetchone()
    return _mutation_from_row(row) if row is not None else None


def list_backtest_mutations(
    connection: sqlite3.Connection,
    *,
    parent_task_ids: frozenset[str] | None = None,
) -> tuple[BacktestMutationRecord, ...]:
    if parent_task_ids is not None and not parent_task_ids:
        return ()
    scope = "" if parent_task_ids is None else "WHERE parent_task_id IN (SELECT value FROM json_each(?))"
    parameters = () if parent_task_ids is None else (json.dumps(sorted(parent_task_ids)),)
    rows = connection.execute(
        f"SELECT * FROM backtest_mutations {scope} ORDER BY child_task_id", parameters,
    ).fetchall()
    return tuple(_mutation_from_row(row) for row in rows)


def list_backtest_formulas(connection: sqlite3.Connection) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT formula FROM backtest_tasks ORDER BY formula_fingerprint"
    ).fetchall()
    return tuple(row["formula"] for row in rows)


def list_backtest_tasks(
    connection: sqlite3.Connection,
    *,
    task_ids: frozenset[str] | None = None,
) -> tuple[BacktestTaskRecord, ...]:
    if task_ids is not None and not task_ids:
        return ()
    scope = "" if task_ids is None else "WHERE task_id IN (SELECT value FROM json_each(?))"
    parameters = () if task_ids is None else (json.dumps(sorted(task_ids)),)
    rows = connection.execute(
        f"SELECT * FROM backtest_tasks {scope} ORDER BY created_at, task_id", parameters,
    ).fetchall()
    return tuple(_task_from_row(row) for row in rows)


def list_started_backtest_task_ids(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(row[0] for row in connection.execute(
        "SELECT task_id FROM backtest_tasks WHERE submission_started_at IS NOT NULL"
    ))


def list_active_backtest_tasks(
    connection: sqlite3.Connection,
) -> tuple[BacktestTaskRecord, ...]:
    rows = connection.execute(
        """
        SELECT * FROM backtest_tasks
        WHERE status IN ('created', 'submission_unknown', 'pending')
        ORDER BY created_at, task_id
        """
    ).fetchall()
    return tuple(_task_from_row(row) for row in rows)


def list_completed_backtests(
    connection: sqlite3.Connection,
    *,
    task_ids: frozenset[str] | None = None,
) -> tuple[BacktestSnapshot, ...]:
    if task_ids is not None and not task_ids:
        return ()
    scope = "t.status = 'completed'"
    parameters = ()
    if task_ids is not None:
        scope += " AND t.task_id IN (SELECT value FROM json_each(?))"
        parameters = (json.dumps(sorted(task_ids)),)
    tasks = connection.execute(
        f"SELECT t.* FROM backtest_tasks t WHERE {scope} ORDER BY t.finished_at, t.task_id",
        parameters,
    ).fetchall()
    if not tasks:
        return ()
    results = {row["task_id"]: row for row in connection.execute(f"""
        SELECT r.* FROM backtest_results r
        JOIN backtest_tasks t ON t.task_id = r.task_id
        WHERE {scope}
    """, parameters).fetchall()}
    checks: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute(f"""
        SELECT c.* FROM backtest_checks c
        JOIN backtest_tasks t ON t.task_id = c.task_id
        WHERE {scope} ORDER BY c.task_id, c.check_name
    """, parameters).fetchall():
        checks[row["task_id"]].append(row)
    captured = {row["task_id"] for row in connection.execute(f"""
        SELECT c.task_id FROM backtest_yearly_stats_captures c
        JOIN backtest_tasks t ON t.task_id = c.task_id
        WHERE {scope}
    """, parameters).fetchall()}
    yearly: dict[str, list[BacktestYearlyStatRecord]] = defaultdict(list)
    for row in connection.execute(f"""
        SELECT s.* FROM backtest_yearly_stats s
        JOIN backtest_tasks t ON t.task_id = s.task_id
        WHERE {scope} ORDER BY s.task_id, s.stage, s.year
    """, parameters).fetchall():
        yearly[row["task_id"]].append(_yearly_stat_from_row(row))
    snapshots: list[BacktestSnapshot] = []
    for row in tasks:
        task_id = row["task_id"]
        if task_id not in results or task_id not in captured:
            raise ValueError("completed_backtest_result_missing")
        snapshots.append(BacktestSnapshot(
            task=_task_from_row(row),
            result=_result_from_row(results[task_id], checks[task_id]),
            yearly_stats=tuple(yearly[task_id]),
        ))
    return tuple(snapshots)


def list_terminal_backtests(
    connection: sqlite3.Connection,
) -> tuple[BacktestSnapshot, ...]:
    rows = connection.execute(
        """
        SELECT task_id FROM backtest_tasks
        WHERE status IN ('completed', 'failed')
        ORDER BY finished_at, task_id
        """
    ).fetchall()
    snapshots: list[BacktestSnapshot] = []
    for row in rows:
        snapshot = get_backtest_task(connection, row["task_id"])
        if snapshot is None:
            raise ValueError("terminal_backtest_missing")
        if snapshot.task.status == "completed" and snapshot.result is None:
            raise ValueError("completed_backtest_result_missing")
        if snapshot.task.status == "completed" and snapshot.yearly_stats is None:
            raise ValueError("completed_backtest_yearly_stats_missing")
        if (
            snapshot.task.status == "failed"
            and snapshot.result is not None
            and snapshot.task.failure_code != "platform_pending_timeout"
        ):
            raise ValueError("failed_backtest_result_present")
        if snapshot.task.status == "failed" and snapshot.yearly_stats is not None:
            raise ValueError("failed_backtest_yearly_stats_present")
        snapshots.append(snapshot)
    return tuple(snapshots)


def _validate_task(task: BacktestTaskRecord) -> None:
    for value, error in (
        (task.task_id, "backtest_task_id_missing"),
        (task.account_scope, "backtest_account_scope_missing"),
        (task.formula, "backtest_formula_missing"),
        (task.formula_fingerprint, "backtest_formula_fingerprint_missing"),
        (task.settings_json, "backtest_settings_missing"),
        (task.request_fingerprint, "backtest_request_fingerprint_missing"),
    ):
        _require_text(value, error)
    try:
        parsed = parse_formula(task.formula)
    except FormulaSyntaxError as exc:
        raise ValueError("backtest_formula_invalid") from exc
    if parsed.normalized != task.formula:
        raise ValueError("backtest_formula_not_normalized")
    if parsed.fingerprint != task.formula_fingerprint:
        raise ValueError("backtest_formula_identity_mismatch")
    _canonical_json_object(task.settings_json, "backtest_settings_invalid")
    created = _timestamp(task.created_at, "backtest_created_at_invalid")
    if task.status not in BACKTEST_ACTIVE_STATUSES | BACKTEST_TERMINAL_STATUSES:
        raise ValueError("backtest_status_invalid")
    for value, error in (
        (task.submission_started_at, "backtest_submission_started_at_invalid"),
        (task.last_observed_at, "backtest_last_observed_at_invalid"),
        (task.retry_not_before, "backtest_retry_not_before_invalid"),
        (task.finished_at, "backtest_finished_at_invalid"),
    ):
        if value is not None and _timestamp(value, error) < created:
            raise ValueError(error)
    if (
        task.submission_started_at is not None
        and task.last_observed_at is not None
        and _timestamp(task.last_observed_at, "backtest_last_observed_at_invalid")
        < _timestamp(
            task.submission_started_at, "backtest_submission_started_at_invalid"
        )
    ):
        raise ValueError("backtest_observed_before_submission")
    if (
        task.retry_not_before is not None
        and (
            task.status != "pending"
            or task.last_observed_at is None
            or _timestamp(
                task.retry_not_before,
                "backtest_retry_not_before_invalid",
            )
            <= _timestamp(
                task.last_observed_at,
                "backtest_last_observed_at_invalid",
            )
        )
    ):
        raise ValueError("backtest_retry_not_before_invalid")
    if (
        task.finished_at is not None
        and task.last_observed_at is not None
        and _timestamp(task.finished_at, "backtest_finished_at_invalid")
        < _timestamp(task.last_observed_at, "backtest_last_observed_at_invalid")
    ):
        raise ValueError("backtest_finished_before_observation")
    _validate_task_state(task)


def _validate_task_state(task: BacktestTaskRecord) -> None:
    if task.status == "created":
        expected = (None, None, None, None, None, None, None, None)
    elif task.status == "submission_unknown":
        expected = (
            None,
            None,
            task.submission_started_at,
            task.last_observed_at,
            None,
            None,
            None,
            None,
        )
    elif task.status == "pending":
        expected = (
            task.remote_id,
            task.platform_alpha_id,
            task.submission_started_at,
            task.last_observed_at,
            task.retry_not_before,
            None,
            None,
            None,
        )
    elif task.status == "completed":
        expected = (
            task.remote_id,
            task.platform_alpha_id,
            task.submission_started_at,
            task.last_observed_at,
            None,
            task.finished_at,
            None,
            None,
        )
    else:
        expected = (
            task.remote_id,
            task.platform_alpha_id,
            task.submission_started_at,
            task.last_observed_at,
            None,
            task.finished_at,
            task.failure_code,
            task.failure_message,
        )
    required_indexes = {
        "submission_unknown": (2, 3),
        "pending": (0, 2, 3),
        "completed": (0, 1, 2, 3, 5),
        "failed": (3, 5, 6, 7),
    }.get(task.status, ())
    if any(expected[index] is None for index in required_indexes):
        raise ValueError("backtest_state_fields_invalid")
    for index in required_indexes:
        if isinstance(expected[index], str) and not expected[index].strip():
            raise ValueError("backtest_state_fields_invalid")
    if task.status == "created" and any(value is not None for value in expected):
        raise ValueError("backtest_state_fields_invalid")
    if task.status == "submission_unknown" and any(
        value is not None for index, value in enumerate(expected) if index not in {2, 3}
    ):
        raise ValueError("backtest_state_fields_invalid")
    if task.status in {"pending", "completed"} and any(
        value is not None for value in (task.failure_code, task.failure_message)
    ):
        raise ValueError("backtest_state_fields_invalid")


def _validate_result(result: BacktestResultRecord) -> None:
    _require_text(result.task_id, "backtest_result_task_id_missing")
    if result.grade is not None:
        _require_text(result.grade, "backtest_result_grade_invalid")
    for name in (
        "sharpe",
        "fitness",
        "turnover",
        "returns",
        "drawdown",
        "margin",
        "book_size",
        "pnl",
    ):
        value = getattr(result, name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"backtest_result_{name}_invalid")
    for name in ("long_count", "short_count"):
        value = getattr(result, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"backtest_result_{name}_invalid")
    if not isinstance(result.check_details_captured, bool):
        raise ValueError("backtest_check_details_captured_invalid")
    if not isinstance(result.checks, tuple) or not result.checks:
        raise ValueError("backtest_checks_invalid")
    for check in result.checks:
        _validate_check(check)
    names = tuple(check.name for check in result.checks)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("backtest_checks_invalid")
    for check in result.checks:
        if not result.check_details_captured and any(
            value is not None
            for value in (check.threshold, check.actual, check.platform_date)
        ):
            raise ValueError("backtest_check_details_capture_conflict")


def _validate_check(check: BacktestCheckRecord) -> None:
    if not isinstance(check, BacktestCheckRecord):
        raise ValueError("backtest_check_invalid")
    for value, error in (
        (check.name, "backtest_check_name_invalid"),
        (check.status, "backtest_check_status_invalid"),
    ):
        _require_text(value, error)
        if value != value.strip():
            raise ValueError(error)
    for value, error in (
        (check.threshold, "backtest_check_threshold_invalid"),
        (check.actual, "backtest_check_actual_invalid"),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(error)
    if check.platform_date is not None:
        _require_text(check.platform_date, "backtest_check_platform_date_invalid")
        try:
            parsed = date.fromisoformat(check.platform_date)
        except ValueError as exc:
            raise ValueError("backtest_check_platform_date_invalid") from exc
        if check.platform_date != parsed.isoformat():
            raise ValueError("backtest_check_platform_date_invalid")


def _validate_yearly_stat(record: BacktestYearlyStatRecord) -> None:
    if not isinstance(record, BacktestYearlyStatRecord):
        raise ValueError("backtest_yearly_stat_invalid")
    _require_text(record.task_id, "backtest_yearly_stat_task_id_missing")
    _require_text(record.stage, "backtest_yearly_stat_stage_invalid")
    if record.stage != record.stage.strip():
        raise ValueError("backtest_yearly_stat_stage_invalid")
    if (
        isinstance(record.year, bool)
        or not isinstance(record.year, int)
        or record.year <= 0
    ):
        raise ValueError("backtest_yearly_stat_year_invalid")
    for name in (
        "pnl",
        "book_size",
        "turnover",
        "sharpe",
        "returns",
        "drawdown",
        "margin",
        "fitness",
    ):
        value = getattr(record, name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"backtest_yearly_stat_{name}_invalid")
    for name in ("long_count", "short_count"):
        value = getattr(record, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"backtest_yearly_stat_{name}_invalid")


def _validate_mutation(mutation: BacktestMutationRecord) -> None:
    for value, error in (
        (mutation.child_task_id, "backtest_mutation_child_missing"),
        (mutation.parent_task_id, "backtest_mutation_parent_missing"),
        (mutation.action, "backtest_mutation_action_missing"),
        (mutation.location, "backtest_mutation_location_missing"),
        (mutation.before, "backtest_mutation_before_missing"),
        (mutation.after, "backtest_mutation_after_missing"),
    ):
        _require_text(value, error)
    if mutation.child_task_id == mutation.parent_task_id:
        raise ValueError("backtest_mutation_self_parent")


def _task_identity(task: BacktestTaskRecord) -> tuple[str, ...]:
    return (
        task.task_id,
        task.account_scope,
        task.formula,
        task.formula_fingerprint,
        task.settings_json,
        task.request_fingerprint,
        task.created_at,
    )


def _reservation_identity(task: BacktestTaskRecord) -> tuple[str, str, str]:
    return (
        task.account_scope,
        task.formula_fingerprint,
        task.settings_json,
    )


def _task_values(task: BacktestTaskRecord) -> tuple[object, ...]:
    return (
        task.task_id,
        task.account_scope,
        task.formula,
        task.formula_fingerprint,
        task.settings_json,
        task.request_fingerprint,
        task.status,
        task.remote_id,
        task.platform_alpha_id,
        task.created_at,
        task.submission_started_at,
        task.last_observed_at,
        task.retry_not_before,
        task.finished_at,
        task.failure_code,
        task.failure_message,
    )


def _task_from_row(row: sqlite3.Row) -> BacktestTaskRecord:
    return BacktestTaskRecord(
        task_id=row["task_id"],
        account_scope=row["account_scope"],
        formula=row["formula"],
        formula_fingerprint=row["formula_fingerprint"],
        settings_json=row["settings_json"],
        request_fingerprint=row["request_fingerprint"],
        status=row["status"],
        remote_id=row["remote_id"],
        platform_alpha_id=row["platform_alpha_id"],
        created_at=row["created_at"],
        submission_started_at=row["submission_started_at"],
        last_observed_at=row["last_observed_at"],
        retry_not_before=row["retry_not_before"],
        finished_at=row["finished_at"],
        failure_code=row["failure_code"],
        failure_message=row["failure_message"],
    )


def _result_from_row(
    row: sqlite3.Row,
    check_rows: list[sqlite3.Row],
) -> BacktestResultRecord:
    return BacktestResultRecord(
        task_id=row["task_id"],
        sharpe=row["sharpe"],
        fitness=row["fitness"],
        turnover=row["turnover"],
        returns=row["returns"],
        drawdown=row["drawdown"],
        margin=row["margin"],
        book_size=row["book_size"],
        pnl=row["pnl"],
        long_count=row["long_count"],
        short_count=row["short_count"],
        check_details_captured=bool(row["check_details_captured"]),
        grade=row["grade"],
        checks=tuple(
            BacktestCheckRecord(
                name=check_row["check_name"],
                status=check_row["status"],
                threshold=check_row["threshold_value"],
                actual=check_row["actual_value"],
                platform_date=check_row["platform_date"],
            )
            for check_row in check_rows
        ),
    )


def _yearly_stat_from_row(row: sqlite3.Row) -> BacktestYearlyStatRecord:
    return BacktestYearlyStatRecord(
        task_id=row["task_id"],
        year=row["year"],
        pnl=row["pnl"],
        book_size=row["book_size"],
        long_count=row["long_count"],
        short_count=row["short_count"],
        turnover=row["turnover"],
        sharpe=row["sharpe"],
        returns=row["returns"],
        drawdown=row["drawdown"],
        margin=row["margin"],
        fitness=row["fitness"],
        stage=row["stage"],
    )


def _mutation_from_row(row: sqlite3.Row) -> BacktestMutationRecord:
    return BacktestMutationRecord(
        child_task_id=row["child_task_id"],
        parent_task_id=row["parent_task_id"],
        action=row["action"],
        location=row["location"],
        before=row["before"],
        after=row["after"],
    )


def _canonical_json_object(value: str, error: str) -> None:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if not isinstance(parsed, dict):
        raise ValueError(error)
    canonical = json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if value != canonical:
        raise ValueError(error)


def _timestamp(value: object, error: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(error) from exc
    if parsed.utcoffset() is None:
        raise ValueError(error)
    return parsed


def _require_text(value: object, error: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)


def _create_backtest_task_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_tasks (
            task_id TEXT PRIMARY KEY,
            account_scope TEXT NOT NULL,
            formula TEXT NOT NULL,
            formula_fingerprint TEXT NOT NULL,
            settings_json TEXT NOT NULL,
            request_fingerprint TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN (
                    'created', 'submission_unknown', 'pending',
                    'completed', 'failed'
                )
            ),
            remote_id TEXT,
            platform_alpha_id TEXT,
            created_at TEXT NOT NULL,
            submission_started_at TEXT,
            last_observed_at TEXT,
            retry_not_before TEXT,
            finished_at TEXT,
            failure_code TEXT,
            failure_message TEXT,
            UNIQUE (account_scope, remote_id),
            CHECK (
                (status = 'created'
                    AND remote_id IS NULL
                    AND platform_alpha_id IS NULL
                    AND submission_started_at IS NULL
                    AND last_observed_at IS NULL
                    AND retry_not_before IS NULL
                    AND finished_at IS NULL
                    AND failure_code IS NULL
                    AND failure_message IS NULL)
                OR
                (status = 'submission_unknown'
                    AND remote_id IS NULL
                    AND platform_alpha_id IS NULL
                    AND submission_started_at IS NOT NULL
                    AND last_observed_at IS NOT NULL
                    AND retry_not_before IS NULL
                    AND finished_at IS NULL
                    AND failure_code IS NULL
                    AND failure_message IS NULL)
                OR
                (status = 'pending'
                    AND remote_id IS NOT NULL
                    AND submission_started_at IS NOT NULL
                    AND last_observed_at IS NOT NULL
                    AND finished_at IS NULL
                    AND failure_code IS NULL
                    AND failure_message IS NULL)
                OR
                (status = 'completed'
                    AND remote_id IS NOT NULL
                    AND platform_alpha_id IS NOT NULL
                    AND submission_started_at IS NOT NULL
                    AND last_observed_at IS NOT NULL
                    AND retry_not_before IS NULL
                    AND finished_at IS NOT NULL
                    AND failure_code IS NULL
                    AND failure_message IS NULL)
                OR
                (status = 'failed'
                    AND last_observed_at IS NOT NULL
                    AND retry_not_before IS NULL
                    AND finished_at IS NOT NULL
                    AND failure_code IS NOT NULL
                    AND failure_message IS NOT NULL)
            )
        )
        """
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_backtest_request_reservation "
        f"ON backtest_tasks(request_fingerprint) WHERE NOT ({BACKTEST_CANCELLED_BEFORE_SUBMISSION_SQL})"
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_backtest_formula_reservation "
        "ON backtest_tasks(account_scope, formula_fingerprint, settings_json) "
        f"WHERE NOT ({BACKTEST_CANCELLED_BEFORE_SUBMISSION_SQL})"
    )
