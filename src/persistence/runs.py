from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

from persistence.run_diagnostics import (
    CANDIDATE_PLANNING_STOP_REASONS,
    CandidatePlanningDiagnostic,
)
from persistence.submissions import automated_run_has_active_formal_submission
from persistence.run_allocations import initialize_run_allocation_schema


ACTIVE_RUN_STATUSES = frozenset({"created", "running"})
TERMINAL_RUN_STATUSES = frozenset({"completed", "failed"})


@dataclass(frozen=True, slots=True)
class AutomatedRunRecord:
    run_id: str
    account_scope: str
    settings_policy_json: str
    settings_policy_key: str
    generation_count: int
    backtest_count: int
    minimum_exploration_backtests: int
    exploration_seed_attempt_multiplier: int
    max_cycles: int
    max_backtests: int
    max_pending_seconds: int
    max_consecutive_failures: int
    max_request_failures: int
    max_in_flight_backtests: int
    real_backtests_authorized: bool
    automatic_submissions_enabled: bool
    status: str
    current_cycle: int
    consecutive_failures: int
    request_failure_count: int
    last_request_failure_code: str | None
    last_request_status_code: int | None
    last_request_failure_at: str | None
    last_request_retry_after_seconds: float | None
    created_at: str
    started_at: str | None
    retry_not_before: str | None
    finished_at: str | None
    stop_reason: str | None
    candidate_planning_stop_diagnostic_json: str | None
    optimization_only: bool = False


@dataclass(frozen=True, slots=True)
class AutomatedRunBacktestRecord:
    run_id: str
    task_id: str
    cycle_number: int


@dataclass(frozen=True, slots=True)
class AutomatedCycleSettlementRecord:
    run_id: str
    cycle_number: int
    outcome: str
    frontier_advanced: bool
    settled_at: str


def initialize_run_schema(connection: sqlite3.Connection) -> None:
    _create_automated_run_table(connection)
    initialize_run_allocation_schema(connection)
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_automated_runs_single_active
        ON automated_runs ((1))
        WHERE status IN ('created', 'running')
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS automated_cycle_settlements (
            run_id TEXT NOT NULL REFERENCES automated_runs(run_id),
            cycle_number INTEGER NOT NULL CHECK (cycle_number > 0),
            outcome TEXT NOT NULL CHECK (
                outcome IN (
                    'qualified', 'frontier_advanced',
                    'not_qualified', 'failed'
                )
            ),
            frontier_advanced INTEGER NOT NULL CHECK (
                frontier_advanced IN (0, 1)
            ),
            settled_at TEXT NOT NULL,
            PRIMARY KEY (run_id, cycle_number)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS automated_run_backtests (
            run_id TEXT NOT NULL REFERENCES automated_runs(run_id),
            task_id TEXT NOT NULL UNIQUE REFERENCES backtest_tasks(task_id),
            cycle_number INTEGER NOT NULL CHECK (cycle_number > 0),
            PRIMARY KEY (run_id, task_id)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_automated_run_backtests_cycle
        ON automated_run_backtests (run_id, cycle_number, task_id)
        """
    )


def _create_automated_run_table(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS automated_runs (
            run_id TEXT PRIMARY KEY,
            account_scope TEXT NOT NULL,
            settings_policy_json TEXT NOT NULL,
            settings_policy_key TEXT NOT NULL,
            generation_count INTEGER NOT NULL CHECK (generation_count > 0),
            backtest_count INTEGER NOT NULL CHECK (
                backtest_count > 0 AND backtest_count <= generation_count
            ),
            minimum_exploration_backtests INTEGER NOT NULL CHECK (
                minimum_exploration_backtests > 0
                AND minimum_exploration_backtests <= backtest_count
            ),
            exploration_seed_attempt_multiplier INTEGER NOT NULL CHECK (
                exploration_seed_attempt_multiplier > 0
            ),
            max_cycles INTEGER NOT NULL CHECK (
                max_cycles = -1 OR max_cycles > 0
            ),
            max_backtests INTEGER NOT NULL CHECK (max_backtests >= 0),
            max_pending_seconds INTEGER NOT NULL CHECK (max_pending_seconds > 0),
            max_consecutive_failures INTEGER NOT NULL CHECK (
                max_consecutive_failures > 0
            ),
            max_request_failures INTEGER NOT NULL CHECK (
                max_request_failures > 0
            ),
            max_in_flight_backtests INTEGER NOT NULL CHECK (
                max_in_flight_backtests > 0
            ),
            real_backtests_authorized INTEGER NOT NULL CHECK (
                real_backtests_authorized IN (0, 1)
            ),
            automatic_submissions_enabled INTEGER NOT NULL CHECK (
                automatic_submissions_enabled IN (0, 1)
            ),
            status TEXT NOT NULL CHECK (
                status IN ('created', 'running', 'completed', 'failed')
            ),
            current_cycle INTEGER NOT NULL CHECK (current_cycle >= 0),
            consecutive_failures INTEGER NOT NULL CHECK (
                consecutive_failures >= 0
            ),
            request_failure_count INTEGER NOT NULL CHECK (
                request_failure_count >= 0
            ),
            last_request_failure_code TEXT,
            last_request_status_code INTEGER,
            last_request_failure_at TEXT,
            last_request_retry_after_seconds REAL,
            created_at TEXT NOT NULL,
            started_at TEXT,
            retry_not_before TEXT,
            finished_at TEXT,
            stop_reason TEXT,
            candidate_planning_stop_diagnostic_json TEXT,
            optimization_only INTEGER NOT NULL DEFAULT 0 CHECK (optimization_only IN (0, 1)),
            CHECK (
                CASE
                    WHEN stop_reason IN (
                        'generation_attempt_budget_exhausted',
                        'selection_rejection_shortfall'
                    ) THEN
                        status = 'completed'
                        AND candidate_planning_stop_diagnostic_json IS NOT NULL
                    ELSE candidate_planning_stop_diagnostic_json IS NULL
                END
            )
        )
        """
    )


def create_automated_run(
    connection: sqlite3.Connection,
    record: AutomatedRunRecord,
) -> AutomatedRunRecord:
    _validate_record(record)
    existing = get_automated_run(connection, record.run_id)
    if existing is not None:
        if existing != record:
            raise ValueError("automated_run_identity_conflict")
        return existing
    if list_active_automated_runs(connection):
        raise ValueError("automated_run_active_exists")
    connection.execute(
        """
        INSERT INTO automated_runs (
            run_id, account_scope, settings_policy_json, settings_policy_key,
            generation_count, backtest_count, minimum_exploration_backtests,
            exploration_seed_attempt_multiplier,
            max_cycles, max_backtests,
            max_pending_seconds, max_consecutive_failures, max_request_failures,
            max_in_flight_backtests,
            real_backtests_authorized,
            automatic_submissions_enabled,
            status, current_cycle, consecutive_failures, request_failure_count,
            last_request_failure_code, last_request_status_code,
            last_request_failure_at, last_request_retry_after_seconds,
            created_at,
            started_at, retry_not_before, finished_at, stop_reason,
            candidate_planning_stop_diagnostic_json, optimization_only
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _record_values(record),
    )
    return record


def replace_automated_run(
    connection: sqlite3.Connection,
    record: AutomatedRunRecord,
    *,
    expected_status: str,
) -> AutomatedRunRecord:
    _validate_record(record)
    existing = get_automated_run(connection, record.run_id)
    if existing is None:
        raise ValueError("automated_run_missing")
    if _record_identity(existing) != _record_identity(record):
        raise ValueError("automated_run_identity_conflict")
    if (
        record.status in {"completed", "failed"}
        and (
            (
                record.status == "completed"
                and automated_run_has_submission_unknown(connection, record.run_id)
            )
            or (
                record.automatic_submissions_enabled
                and automated_run_has_active_formal_submission(
                    connection,
                    record.run_id,
                )
            )
        )
        and (
            record.status != "failed"
            or record.stop_reason != "submission_reconciliation_required"
        )
    ):
        raise ValueError("automated_run_submission_reconciliation_required")
    if record.status == "completed" and _automated_run_has_active_backtests(
        connection,
        record.run_id,
    ):
        raise ValueError("automated_run_completion_has_active_backtests")
    if (
        record.status == "failed"
        and record.stop_reason != "submission_reconciliation_required"
        and _automated_run_has_created_backtests(connection, record.run_id)
    ):
        raise ValueError("automated_run_failure_has_unsubmitted_backtests")
    if existing == record:
        return existing
    if existing.status != expected_status:
        raise ValueError("automated_run_transition_conflict")
    cursor = connection.execute(
        """
        UPDATE automated_runs SET
            status = ?, current_cycle = ?, consecutive_failures = ?,
            request_failure_count = ?,
            last_request_failure_code = ?, last_request_status_code = ?,
            last_request_failure_at = ?, last_request_retry_after_seconds = ?,
            started_at = ?, retry_not_before = ?, finished_at = ?,
            stop_reason = ?, candidate_planning_stop_diagnostic_json = ?
        WHERE run_id = ? AND status = ? AND current_cycle = ?
        """,
        (
            record.status,
            record.current_cycle,
            record.consecutive_failures,
            record.request_failure_count,
            record.last_request_failure_code,
            record.last_request_status_code,
            record.last_request_failure_at,
            record.last_request_retry_after_seconds,
            record.started_at,
            record.retry_not_before,
            record.finished_at,
            record.stop_reason,
            record.candidate_planning_stop_diagnostic_json,
            record.run_id,
            expected_status,
            existing.current_cycle,
        ),
    )
    if cursor.rowcount != 1:
        raise ValueError("automated_run_transition_conflict")
    return record


def get_automated_run(
    connection: sqlite3.Connection,
    run_id: str,
) -> AutomatedRunRecord | None:
    row = connection.execute(
        "SELECT * FROM automated_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return _record_from_row(row) if row is not None else None


def list_failed_automated_runs(
    connection: sqlite3.Connection, *, account_scope: str
) -> tuple[AutomatedRunRecord, ...]:
    rows = connection.execute(
        "SELECT * FROM automated_runs WHERE status='failed' AND account_scope=? ORDER BY created_at,run_id",
        (account_scope,),
    ).fetchall()
    return tuple(_record_from_row(row) for row in rows)


def list_active_automated_runs(
    connection: sqlite3.Connection,
) -> tuple[AutomatedRunRecord, ...]:
    rows = connection.execute(
        """
        SELECT * FROM automated_runs
        WHERE status IN ('created', 'running')
        ORDER BY created_at, run_id
        """
    ).fetchall()
    return tuple(_record_from_row(row) for row in rows)


def list_submission_reconciliation_runs(
    connection: sqlite3.Connection,
) -> tuple[AutomatedRunRecord, ...]:
    rows = connection.execute(
        """
        SELECT runs.* FROM automated_runs AS runs
        WHERE (
            runs.status = 'failed'
            AND runs.stop_reason = 'submission_reconciliation_required'
        ) OR EXISTS (
            SELECT 1
            FROM automated_run_backtests AS links
            JOIN backtest_tasks AS tasks ON tasks.task_id = links.task_id
            WHERE links.run_id = runs.run_id
              AND tasks.status = 'submission_unknown'
        )
        OR EXISTS (
            SELECT 1
            FROM formal_submission_attempts AS attempts
            WHERE attempts.run_id = runs.run_id
              AND attempts.submission_mode = 'automatic'
              AND attempts.status IN (
                  'submitting', 'confirmation_pending', 'submission_unknown'
              )
        )
        ORDER BY runs.created_at, runs.run_id
        """
    ).fetchall()
    return tuple(_record_from_row(row) for row in rows)


def automated_run_has_submission_unknown(
    connection: sqlite3.Connection,
    run_id: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM automated_run_backtests AS links
        JOIN backtest_tasks AS tasks ON tasks.task_id = links.task_id
        WHERE links.run_id = ? AND tasks.status = 'submission_unknown'
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    return row is not None


def _automated_run_has_active_backtests(
    connection: sqlite3.Connection,
    run_id: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM automated_run_backtests AS links
        JOIN backtest_tasks AS tasks ON tasks.task_id = links.task_id
        WHERE links.run_id = ? AND tasks.status IN ('created', 'pending')
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    return row is not None


def _automated_run_has_created_backtests(
    connection: sqlite3.Connection,
    run_id: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM automated_run_backtests AS links
        JOIN backtest_tasks AS tasks ON tasks.task_id = links.task_id
        WHERE links.run_id = ? AND tasks.status = 'created'
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    return row is not None


def attach_backtest_to_automated_run(
    connection: sqlite3.Connection,
    record: AutomatedRunBacktestRecord,
) -> AutomatedRunBacktestRecord:
    _validate_backtest_record(record)
    existing = get_automated_run_backtest_by_task(connection, record.task_id)
    if existing is not None:
        if existing != record:
            raise ValueError("automated_run_backtest_identity_conflict")
        return existing
    run = get_automated_run(connection, record.run_id)
    if run is None:
        raise ValueError("automated_run_missing")
    if run.status not in ACTIVE_RUN_STATUSES:
        raise ValueError("automated_run_backtest_run_status_invalid")
    task = connection.execute(
        "SELECT account_scope FROM backtest_tasks WHERE task_id = ?",
        (record.task_id,),
    ).fetchone()
    if task is None:
        raise ValueError("automated_run_backtest_task_missing")
    if task["account_scope"] != run.account_scope:
        raise ValueError("automated_run_backtest_account_scope_mismatch")
    connection.execute(
        """
        INSERT INTO automated_run_backtests (run_id, task_id, cycle_number)
        VALUES (?, ?, ?)
        """,
        (record.run_id, record.task_id, record.cycle_number),
    )
    return record


def get_automated_run_backtest_by_task(
    connection: sqlite3.Connection,
    task_id: str,
) -> AutomatedRunBacktestRecord | None:
    row = connection.execute(
        """
        SELECT run_id, task_id, cycle_number
        FROM automated_run_backtests
        WHERE task_id = ?
        """,
        (task_id,),
    ).fetchone()
    return _backtest_record_from_row(row) if row is not None else None


def list_automated_run_backtests(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    cycle_number: int | None = None,
) -> tuple[AutomatedRunBacktestRecord, ...]:
    rows = connection.execute(
        """
        SELECT run_id, task_id, cycle_number
        FROM automated_run_backtests
        WHERE run_id = ?
        """ + (" AND cycle_number = ? " if cycle_number is not None else "") + """
        ORDER BY cycle_number, task_id
        """,
        (run_id, cycle_number) if cycle_number is not None else (run_id,),
    ).fetchall()
    return tuple(_backtest_record_from_row(row) for row in rows)


def list_all_automated_run_backtests(
    connection: sqlite3.Connection,
) -> tuple[AutomatedRunBacktestRecord, ...]:
    rows = connection.execute(
        """
        SELECT run_id, task_id, cycle_number
        FROM automated_run_backtests
        ORDER BY run_id, cycle_number, task_id
        """
    ).fetchall()
    return tuple(_backtest_record_from_row(row) for row in rows)


def create_automated_cycle_settlement(
    connection: sqlite3.Connection,
    record: AutomatedCycleSettlementRecord,
) -> AutomatedCycleSettlementRecord:
    _validate_cycle_settlement(record)
    if get_automated_run(connection, record.run_id) is None:
        raise ValueError("automated_cycle_settlement_run_missing")
    existing = get_automated_cycle_settlement(
        connection,
        record.run_id,
        record.cycle_number,
    )
    if existing is not None:
        if existing != record:
            raise ValueError("automated_cycle_settlement_conflict")
        return existing
    observed = _timestamp(
        record.settled_at, "automated_cycle_settlement_timestamp_invalid"
    )
    if any(
        _timestamp(row[0], "automated_cycle_settlement_timestamp_invalid") > observed
        for row in connection.execute(
            "SELECT settled_at FROM automated_cycle_settlements WHERE run_id = ?",
            (record.run_id,),
        )
    ):
        raise ValueError("automated_cycle_settlement_time_order_invalid")
    connection.execute(
        """
        INSERT INTO automated_cycle_settlements (
            run_id, cycle_number, outcome, frontier_advanced, settled_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            record.run_id,
            record.cycle_number,
            record.outcome,
            int(record.frontier_advanced),
            record.settled_at,
        ),
    )
    return record


def get_automated_cycle_settlement(
    connection: sqlite3.Connection,
    run_id: str,
    cycle_number: int,
) -> AutomatedCycleSettlementRecord | None:
    row = connection.execute(
        """
        SELECT * FROM automated_cycle_settlements
        WHERE run_id = ? AND cycle_number = ?
        """,
        (run_id, cycle_number),
    ).fetchone()
    return _cycle_settlement_from_row(row) if row is not None else None


def _validate_record(record: AutomatedRunRecord) -> None:
    _require_text(record.run_id, "automated_run_id_missing")
    _require_text(record.account_scope, "automated_run_account_scope_missing")
    _require_text(
        record.settings_policy_json,
        "automated_run_settings_policy_missing",
    )
    _require_text(
        record.settings_policy_key,
        "automated_run_settings_policy_key_missing",
    )
    _validate_settings_policy(
        record.settings_policy_json,
        record.settings_policy_key,
    )
    for name in (
        "generation_count",
        "backtest_count",
        "minimum_exploration_backtests",
        "exploration_seed_attempt_multiplier",
        "max_cycles",
        "max_backtests",
        "max_pending_seconds",
        "max_consecutive_failures",
        "max_request_failures",
        "max_in_flight_backtests",
        "current_cycle",
        "consecutive_failures",
        "request_failure_count",
    ):
        value = getattr(record, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"automated_run_{name}_invalid")
    if record.generation_count <= 0:
        raise ValueError("automated_run_generation_count_invalid")
    if not 0 < record.backtest_count <= record.generation_count:
        raise ValueError("automated_run_backtest_count_invalid")
    if not 0 < record.minimum_exploration_backtests <= record.backtest_count:
        raise ValueError("automated_run_exploration_reserve_invalid")
    if record.exploration_seed_attempt_multiplier <= 0:
        raise ValueError("automated_run_exploration_attempt_multiplier_invalid")
    if record.max_cycles == 0 or record.max_cycles < -1:
        raise ValueError("automated_run_max_cycles_invalid")
    if record.max_pending_seconds <= 0:
        raise ValueError("automated_run_max_pending_invalid")
    if record.max_consecutive_failures <= 0:
        raise ValueError("automated_run_failure_limit_invalid")
    if record.max_request_failures <= 0:
        raise ValueError("automated_run_request_failure_limit_invalid")
    if record.max_in_flight_backtests <= 0:
        raise ValueError("automated_run_in_flight_limit_invalid")
    if not isinstance(record.real_backtests_authorized, bool):
        raise ValueError("automated_run_real_authorization_invalid")
    if not isinstance(record.automatic_submissions_enabled, bool):
        raise ValueError("automated_run_automatic_submission_setting_invalid")
    if not isinstance(record.optimization_only, bool):
        raise ValueError("automated_run_optimization_mode_invalid")
    if record.max_backtests < 0:
        raise ValueError("automated_run_backtest_limit_invalid")
    if record.real_backtests_authorized:
        if record.max_cycles > 0 and record.max_backtests == 0:
            raise ValueError("automated_run_backtest_limit_invalid")
    elif record.max_backtests != 0:
        raise ValueError("automated_run_backtest_limit_invalid")
    if record.automatic_submissions_enabled and not record.real_backtests_authorized:
        raise ValueError("automated_run_automatic_submission_without_backtests")
    if record.status not in ACTIVE_RUN_STATUSES | TERMINAL_RUN_STATUSES:
        raise ValueError("automated_run_status_invalid")
    if record.current_cycle < 0 or (
        record.max_cycles > 0 and record.current_cycle > record.max_cycles
    ):
        raise ValueError("automated_run_cycle_invalid")
    if not 0 <= record.consecutive_failures <= record.current_cycle:
        raise ValueError("automated_run_failure_count_invalid")
    if record.request_failure_count < 0:
        raise ValueError("automated_run_request_failure_count_invalid")
    if record.last_request_failure_code is not None:
        _require_text(
            record.last_request_failure_code,
            "automated_run_request_failure_code_invalid",
        )
    if record.last_request_status_code is not None and (
        isinstance(record.last_request_status_code, bool)
        or not isinstance(record.last_request_status_code, int)
        or not 100 <= record.last_request_status_code <= 599
    ):
        raise ValueError("automated_run_request_status_code_invalid")
    if record.last_request_failure_at is not None:
        _timestamp(
            record.last_request_failure_at,
            "automated_run_request_failure_at_invalid",
        )
    if record.last_request_retry_after_seconds is not None and (
        isinstance(record.last_request_retry_after_seconds, bool)
        or not isinstance(record.last_request_retry_after_seconds, (int, float))
        or not math.isfinite(record.last_request_retry_after_seconds)
        or record.last_request_retry_after_seconds <= 0
    ):
        raise ValueError("automated_run_request_retry_delay_invalid")

    created = _timestamp(record.created_at, "automated_run_created_at_invalid")
    started = (
        _timestamp(record.started_at, "automated_run_started_at_invalid")
        if record.started_at is not None
        else None
    )
    finished = (
        _timestamp(record.finished_at, "automated_run_finished_at_invalid")
        if record.finished_at is not None
        else None
    )
    retry_not_before = (
        _timestamp(
            record.retry_not_before,
            "automated_run_retry_not_before_invalid",
        )
        if record.retry_not_before is not None
        else None
    )
    if started is not None and started < created:
        raise ValueError("automated_run_started_at_invalid")
    if finished is not None and finished < (started or created):
        raise ValueError("automated_run_finished_at_invalid")
    if retry_not_before is not None and (started is None or retry_not_before < started):
        raise ValueError("automated_run_retry_not_before_invalid")
    if record.status == "created":
        if any(
            value is not None
            for value in (
                record.started_at,
                record.retry_not_before,
                record.finished_at,
                record.stop_reason,
                record.candidate_planning_stop_diagnostic_json,
            )
        ) or any(
            value != 0
            for value in (
                record.current_cycle,
                record.consecutive_failures,
                record.request_failure_count,
            )
        ):
            raise ValueError("automated_run_state_invalid")
    elif record.status == "running":
        if (
            record.started_at is None
            or record.finished_at is not None
            or record.stop_reason is not None
            or record.candidate_planning_stop_diagnostic_json is not None
            or (record.max_cycles > 0 and record.current_cycle >= record.max_cycles)
            or (record.request_failure_count == 0) != (record.retry_not_before is None)
        ):
            raise ValueError("automated_run_state_invalid")
    elif (
        (record.started_at is None and (
            record.status != "failed" or record.current_cycle != 0
            or record.request_failure_count != 0 or record.consecutive_failures != 0
        ))
        or record.finished_at is None
        or not isinstance(record.stop_reason, str)
        or not record.stop_reason.strip()
        or record.retry_not_before is not None
    ):
        raise ValueError("automated_run_state_invalid")
    _validate_candidate_planning_diagnostic(record)


def _validate_candidate_planning_diagnostic(record: AutomatedRunRecord) -> None:
    value = record.candidate_planning_stop_diagnostic_json
    if value is None:
        if record.stop_reason in CANDIDATE_PLANNING_STOP_REASONS:
            raise ValueError("automated_run_candidate_planning_diagnostic_missing")
        return
    if (
        record.status != "completed"
        or record.stop_reason not in CANDIDATE_PLANNING_STOP_REASONS
    ):
        raise ValueError("automated_run_candidate_planning_diagnostic_unexpected")
    diagnostic = CandidatePlanningDiagnostic.from_canonical_json(
        value,
        stop_reason=record.stop_reason,
    )
    if diagnostic.cycle_number != record.current_cycle + 1:
        raise ValueError("automated_run_candidate_planning_cycle_invalid")
    if (
        diagnostic.exploration_generation_target_count > record.generation_count
        or diagnostic.exploration_backtest_target_count > record.backtest_count
    ):
        raise ValueError("automated_run_candidate_planning_target_invalid")
    if diagnostic.seed_attempt_limit != (
        diagnostic.exploration_generation_target_count
        * record.exploration_seed_attempt_multiplier
    ):
        raise ValueError("automated_run_candidate_planning_budget_invalid")


def _validate_backtest_record(record: AutomatedRunBacktestRecord) -> None:
    _require_text(record.run_id, "automated_run_id_missing")
    _require_text(record.task_id, "automated_run_backtest_task_id_missing")
    if (
        isinstance(record.cycle_number, bool)
        or not isinstance(record.cycle_number, int)
        or record.cycle_number <= 0
    ):
        raise ValueError("automated_run_backtest_cycle_invalid")


def _validate_cycle_settlement(record: AutomatedCycleSettlementRecord) -> None:
    if not isinstance(record, AutomatedCycleSettlementRecord):
        raise ValueError("automated_cycle_settlement_invalid")
    _require_text(record.run_id, "automated_cycle_settlement_run_id_missing")
    if (
        isinstance(record.cycle_number, bool)
        or not isinstance(record.cycle_number, int)
        or record.cycle_number <= 0
    ):
        raise ValueError("automated_cycle_settlement_cycle_invalid")
    if record.outcome not in {
        "qualified",
        "frontier_advanced",
        "not_qualified",
        "failed",
    }:
        raise ValueError("automated_cycle_settlement_outcome_invalid")
    if not isinstance(record.frontier_advanced, bool):
        raise ValueError("automated_cycle_settlement_frontier_invalid")
    if (record.outcome == "frontier_advanced" and not record.frontier_advanced) or (
        record.outcome in {"not_qualified", "failed"} and record.frontier_advanced
    ):
        raise ValueError("automated_cycle_settlement_frontier_conflict")
    _timestamp(
        record.settled_at,
        "automated_cycle_settlement_timestamp_invalid",
    )


def _validate_settings_policy(
    settings_policy_json: str,
    settings_policy_key: str,
) -> None:
    try:
        payload = json.loads(settings_policy_json)
    except json.JSONDecodeError as exc:
        raise ValueError("automated_run_settings_policy_invalid") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError("automated_run_settings_policy_invalid")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if canonical != settings_policy_json:
        raise ValueError("automated_run_settings_policy_invalid")
    if sha256(settings_policy_json.encode("utf-8")).hexdigest() != settings_policy_key:
        raise ValueError("automated_run_settings_policy_key_invalid")


def _record_identity(record: AutomatedRunRecord) -> tuple[object, ...]:
    return (
        record.run_id,
        record.account_scope,
        record.settings_policy_json,
        record.settings_policy_key,
        record.generation_count,
        record.backtest_count,
        record.minimum_exploration_backtests,
        record.exploration_seed_attempt_multiplier,
        record.max_cycles,
        record.max_backtests,
        record.max_pending_seconds,
        record.max_consecutive_failures,
        record.max_request_failures,
        record.max_in_flight_backtests,
        record.real_backtests_authorized,
        record.automatic_submissions_enabled,
        record.created_at,
        record.optimization_only,
    )


def _record_values(record: AutomatedRunRecord) -> tuple[object, ...]:
    return (
        record.run_id,
        record.account_scope,
        record.settings_policy_json,
        record.settings_policy_key,
        record.generation_count,
        record.backtest_count,
        record.minimum_exploration_backtests,
        record.exploration_seed_attempt_multiplier,
        record.max_cycles,
        record.max_backtests,
        record.max_pending_seconds,
        record.max_consecutive_failures,
        record.max_request_failures,
        record.max_in_flight_backtests,
        record.real_backtests_authorized,
        record.automatic_submissions_enabled,
        record.status,
        record.current_cycle,
        record.consecutive_failures,
        record.request_failure_count,
        record.last_request_failure_code,
        record.last_request_status_code,
        record.last_request_failure_at,
        record.last_request_retry_after_seconds,
        record.created_at,
        record.started_at,
        record.retry_not_before,
        record.finished_at,
        record.stop_reason,
        record.candidate_planning_stop_diagnostic_json,
        record.optimization_only,
    )


def _record_from_row(row: sqlite3.Row) -> AutomatedRunRecord:
    return AutomatedRunRecord(
        run_id=row["run_id"],
        account_scope=row["account_scope"],
        settings_policy_json=row["settings_policy_json"],
        settings_policy_key=row["settings_policy_key"],
        generation_count=row["generation_count"],
        backtest_count=row["backtest_count"],
        minimum_exploration_backtests=row["minimum_exploration_backtests"],
        exploration_seed_attempt_multiplier=row["exploration_seed_attempt_multiplier"],
        max_cycles=row["max_cycles"],
        max_backtests=row["max_backtests"],
        max_pending_seconds=row["max_pending_seconds"],
        max_consecutive_failures=row["max_consecutive_failures"],
        max_request_failures=row["max_request_failures"],
        max_in_flight_backtests=row["max_in_flight_backtests"],
        real_backtests_authorized=bool(row["real_backtests_authorized"]),
        automatic_submissions_enabled=bool(row["automatic_submissions_enabled"]),
        optimization_only=bool(row["optimization_only"]),
        status=row["status"],
        current_cycle=row["current_cycle"],
        consecutive_failures=row["consecutive_failures"],
        request_failure_count=row["request_failure_count"],
        last_request_failure_code=row["last_request_failure_code"],
        last_request_status_code=row["last_request_status_code"],
        last_request_failure_at=row["last_request_failure_at"],
        last_request_retry_after_seconds=row["last_request_retry_after_seconds"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        retry_not_before=row["retry_not_before"],
        finished_at=row["finished_at"],
        stop_reason=row["stop_reason"],
        candidate_planning_stop_diagnostic_json=row[
            "candidate_planning_stop_diagnostic_json"
        ],
    )


def _backtest_record_from_row(row: sqlite3.Row) -> AutomatedRunBacktestRecord:
    return AutomatedRunBacktestRecord(
        run_id=row["run_id"],
        task_id=row["task_id"],
        cycle_number=row["cycle_number"],
    )


def _cycle_settlement_from_row(
    row: sqlite3.Row,
) -> AutomatedCycleSettlementRecord:
    return AutomatedCycleSettlementRecord(
        run_id=row["run_id"],
        cycle_number=row["cycle_number"],
        outcome=row["outcome"],
        frontier_advanced=bool(row["frontier_advanced"]),
        settled_at=row["settled_at"],
    )


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
