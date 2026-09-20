from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path

from execution.catalog import load_generation_catalog
from execution.backtests import cancel_unsubmitted_backtest_task
from persistence.backtests import get_backtest_task, get_backtest_task_by_identity, list_active_backtest_tasks, replace_backtest_task
from persistence.catalog import FieldCatalogContext
from persistence.database import open_database
from persistence.run_allocations import RunAllocationRecord, create_run_allocation
from persistence.run_diagnostics import (
    CANDIDATE_PLANNING_STOP_REASONS,
    CandidatePlanningDiagnostic,
)
from persistence.runs import (
    AutomatedCycleSettlementRecord,
    AutomatedRunRecord,
    automated_run_has_submission_unknown,
    create_automated_cycle_settlement,
    create_automated_run,
    get_automated_cycle_settlement,
    get_automated_run,
    list_automated_run_backtests,
    list_active_automated_runs,
    get_automated_run_backtest_by_task,
    replace_automated_run,
)
from persistence.submissions import (
    account_has_active_formal_submission,
    automated_run_has_unresolved_formal_submission,
)
from selection.settings import BacktestSettingsPolicy, load_backtest_settings_policy


@dataclass(frozen=True, slots=True)
class AutomatedRunLimits:
    generation_count: int
    backtest_count: int
    max_cycles: int
    max_backtests: int
    max_pending_seconds: int
    max_consecutive_failures: int
    max_request_failures: int
    max_in_flight_backtests: int
    exploration_seed_attempt_multiplier: int
    exploration_percent: int
    self_correlation_percent: int
    mutation_percent: int
    direction_validation_percent: int
    real_backtests_authorized: bool = False
    automatic_submissions_enabled: bool = False
    optimization_only: bool = False


@dataclass(frozen=True, slots=True)
class AutomatedCycleTransition:
    status: str
    stop_reason: str | None
    consecutive_failures: int


class AutomatedRunPaused(Exception):
    """Cooperative stop at a boundary where observed results are already saved."""


def set_automated_run_paused(database_path: str | Path, run_id: str, *, paused: bool) -> None:
    """Called by the launch/resume owner while it holds the process lock.

    The unfinished plan keeps its tasks and budgets. Its stop reason records
    the pause; terminal outcomes and their original reasons are never changed.
    """
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        run = get_automated_run(connection, run_id)
        if run is None:
            raise ValueError("automated_run_missing")
        if run.status not in {"created", "running"}:
            return
        if paused or run.stop_reason == "user_paused":
            replace_automated_run(connection, replace(run, stop_reason="user_paused" if paused else None),
                                  expected_status=run.status)


def validate_automated_run_limits(limits: AutomatedRunLimits) -> None:
    if not isinstance(limits, AutomatedRunLimits):
        raise ValueError("automated_run_limits_invalid")
    for name in (
        "generation_count",
        "backtest_count",
        "max_cycles",
        "max_backtests",
        "max_pending_seconds",
        "max_consecutive_failures",
        "max_request_failures",
        "max_in_flight_backtests",
        "exploration_seed_attempt_multiplier",
        "exploration_percent",
        "self_correlation_percent",
        "mutation_percent",
        "direction_validation_percent",
    ):
        value = getattr(limits, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"automated_run_{name}_invalid")
    if limits.generation_count <= 0:
        raise ValueError("automated_run_generation_count_invalid")
    if not (
        0 < limits.exploration_percent <= 100
        and 0 <= limits.self_correlation_percent <= 100
        and 0 <= limits.mutation_percent <= 100
        and 0 <= limits.direction_validation_percent <= limits.mutation_percent
        and limits.exploration_percent + limits.self_correlation_percent
        + limits.mutation_percent == 100
    ):
        raise ValueError("automated_run_allocation_percent_invalid")
    if not 0 < limits.backtest_count <= limits.generation_count:
        raise ValueError("automated_run_backtest_count_invalid")
    if limits.max_cycles == 0 or limits.max_cycles < -1:
        raise ValueError("automated_run_max_cycles_invalid")
    if limits.max_pending_seconds <= 0:
        raise ValueError("automated_run_max_pending_invalid")
    if limits.max_consecutive_failures <= 0:
        raise ValueError("automated_run_failure_limit_invalid")
    if limits.max_request_failures <= 0:
        raise ValueError("automated_run_request_failure_limit_invalid")
    if limits.max_in_flight_backtests <= 0:
        raise ValueError("automated_run_in_flight_limit_invalid")
    if limits.exploration_seed_attempt_multiplier <= 0:
        raise ValueError("automated_run_exploration_attempt_multiplier_invalid")
    if not isinstance(limits.real_backtests_authorized, bool):
        raise ValueError("automated_run_real_authorization_invalid")
    if not isinstance(limits.automatic_submissions_enabled, bool):
        raise ValueError("automated_run_automatic_submission_setting_invalid")
    if not isinstance(limits.optimization_only, bool):
        raise ValueError("automated_run_optimization_mode_invalid")
    if limits.max_backtests < 0:
        raise ValueError("automated_run_backtest_limit_invalid")
    if limits.real_backtests_authorized:
        if limits.max_cycles > 0 and limits.max_backtests == 0:
            raise ValueError("automated_run_backtest_limit_invalid")
    elif limits.max_backtests != 0:
        raise ValueError("automated_run_backtest_limit_invalid")
    if limits.automatic_submissions_enabled and not limits.real_backtests_authorized:
        raise ValueError("automated_run_automatic_submission_without_backtests")


def automated_run_backtest_limit(record: AutomatedRunRecord) -> int | None:
    return record.max_backtests if record.max_backtests > 0 else None


def uses_continuous_recovery(record: AutomatedRunRecord) -> bool:
    """Unlimited research waits through temporary faults without ending the run."""
    return record.max_cycles == -1 and record.max_backtests == 0


def can_plan_automated_cycle(
    run: AutomatedRunRecord, *, latest_cycle: int, prepared_backtests: int
) -> bool:
    return (
        (run.max_cycles == -1 or latest_cycle < run.max_cycles)
        and (run.max_backtests == 0 or prepared_backtests < run.max_backtests)
    )


def automated_run_has_hard_stop_evidence(
    connection: sqlite3.Connection, run: AutomatedRunRecord
) -> bool:
    """Older unknown pauses may have masked a hard stop; confirmation cannot clear it."""
    if (
        run.last_request_failure_at == run.finished_at
        or (
            not uses_continuous_recovery(run)
            and (
                run.request_failure_count >= run.max_request_failures
                or run.consecutive_failures >= run.max_consecutive_failures
            )
        )
    ):
        return True
    links = list_automated_run_backtests(connection, run.run_id)
    snapshots = tuple(get_backtest_task(connection, link.task_id) for link in links)
    if any(snapshot is None for snapshot in snapshots):
        raise ValueError("automated_run_backtest_task_missing")
    return any(
        snapshot.task.failure_code in {
            "platform_pending_timeout", "submission_outcome_timeout"
        }
        and snapshot.task.platform_alpha_id is None
        for snapshot in snapshots
    )


def can_resume_independent_backtests(
    connection: sqlite3.Connection, run: AutomatedRunRecord
) -> bool:
    """Unknown requests retain slots; only independent authorized work may resume."""
    if (
        run.status != "failed"
        or run.stop_reason != "submission_reconciliation_required"
        or automated_run_has_hard_stop_evidence(connection, run)
    ):
        return False
    links = list_automated_run_backtests(connection, run.run_id)
    snapshots = tuple(get_backtest_task(connection, link.task_id) for link in links)
    statuses = [snapshot.task.status for snapshot in snapshots]
    if "pending" in statuses:
        return True
    return statuses.count("submission_unknown") < run.max_in_flight_backtests and (
        "created" in statuses
        or can_plan_automated_cycle(
            run,
            latest_cycle=max((link.cycle_number for link in links), default=0),
            prepared_backtests=len(links),
        )
    )


def remaining_automated_run_backtests(
    record: AutomatedRunRecord,
    used_backtests: int,
) -> int | None:
    if (
        isinstance(used_backtests, bool)
        or not isinstance(used_backtests, int)
        or used_backtests < 0
    ):
        raise ValueError("automated_run_backtest_usage_invalid")
    limit = automated_run_backtest_limit(record)
    return None if limit is None else max(0, limit - used_backtests)


def prepare_automated_run(
    database_path: str | Path,
    policy_path: str | Path,
    *,
    account_scope: str,
    limits: AutomatedRunLimits,
    created_at: str,
) -> AutomatedRunRecord:
    validate_automated_run_limits(limits)
    settings_policy = load_backtest_settings_policy(policy_path)
    settings_policy_json = settings_policy.canonical_json()
    settings_policy_key = settings_policy.fingerprint
    run_id = _run_id(
        account_scope=account_scope,
        settings_policy_json=settings_policy_json,
        limits=limits,
        created_at=created_at,
    )
    record = AutomatedRunRecord(
        run_id=run_id,
        account_scope=account_scope,
        settings_policy_json=settings_policy_json,
        settings_policy_key=settings_policy_key,
        generation_count=limits.generation_count,
        backtest_count=limits.backtest_count,
        minimum_exploration_backtests=(
            limits.backtest_count * limits.exploration_percent + 99
        ) // 100,
        exploration_seed_attempt_multiplier=(
            limits.exploration_seed_attempt_multiplier
        ),
        max_cycles=limits.max_cycles,
        max_backtests=limits.max_backtests,
        max_pending_seconds=limits.max_pending_seconds,
        max_consecutive_failures=limits.max_consecutive_failures,
        max_request_failures=limits.max_request_failures,
        max_in_flight_backtests=limits.max_in_flight_backtests,
        real_backtests_authorized=limits.real_backtests_authorized,
        automatic_submissions_enabled=limits.automatic_submissions_enabled,
        optimization_only=limits.optimization_only,
        status="created",
        current_cycle=0,
        consecutive_failures=0,
        request_failure_count=0,
        last_request_failure_code=None,
        last_request_status_code=None,
        last_request_failure_at=None,
        last_request_retry_after_seconds=None,
        created_at=created_at,
        started_at=None,
        retry_not_before=None,
        finished_at=None,
        stop_reason=None,
        candidate_planning_stop_diagnostic_json=None,
    )
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        load_generation_catalog(
            connection,
            FieldCatalogContext(
                instrument_type=settings_policy.instrument_type,
                region=settings_policy.region,
                universe=settings_policy.universe,
                delay=settings_policy.delay,
            ),
            account_scope=account_scope,
        )
        if limits.real_backtests_authorized:
            _require_no_unresolved_external_backtests(
                connection,
                account_scope,
            )
        result = create_automated_run(connection, record)
        create_run_allocation(connection, RunAllocationRecord(
            run_id=record.run_id,
            exploration_percent=limits.exploration_percent,
            self_correlation_percent=limits.self_correlation_percent,
            mutation_percent=limits.mutation_percent,
            direction_validation_percent=limits.direction_validation_percent,
        ))
        return result


def start_automated_run(
    database_path: str | Path,
    run_id: str,
    *,
    started_at: str,
) -> AutomatedRunRecord:
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        record = get_automated_run(connection, run_id)
        if record is None:
            raise ValueError("automated_run_missing")
        if record.status == "running":
            return record
        if record.status != "created":
            raise ValueError("automated_run_status_invalid")
        if record.real_backtests_authorized:
            linked_task_ids = frozenset(
                link.task_id
                for link in list_automated_run_backtests(
                    connection,
                    record.run_id,
                )
            )
            _require_no_unresolved_external_backtests(
                connection,
                record.account_scope,
                excluded_task_ids=linked_task_ids,
            )
        updated = replace(record, status="running", started_at=started_at)
        return replace_automated_run(
            connection,
            updated,
            expected_status="created",
        )


def complete_optimization_run(
    database_path: str | Path, run_id: str, *, completed_at: str,
) -> AutomatedRunRecord:
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        run = get_automated_run(connection, run_id)
        if run is None or not run.optimization_only:
            raise ValueError("automated_run_optimization_mode_invalid")
        if any(link.cycle_number > run.current_cycle
               for link in list_automated_run_backtests(connection, run_id)):
            raise ValueError("optimization_run_has_unsettled_cycle")
        return _complete_automated_run(
            connection, run, completed_at=completed_at,
            reason="optimization_candidates_exhausted",
        )


def complete_automated_run_after_candidate_planning_stop(
    database_path: str | Path,
    run_id: str,
    *,
    completed_at: str,
    reason: str,
    diagnostic: CandidatePlanningDiagnostic,
) -> AutomatedRunRecord:
    if reason not in CANDIDATE_PLANNING_STOP_REASONS:
        raise ValueError("candidate_planning_stop_reason_invalid")
    if not isinstance(diagnostic, CandidatePlanningDiagnostic):
        raise ValueError("candidate_planning_diagnostic_invalid")
    diagnostic_json = diagnostic.canonical_json(stop_reason=reason)
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        record = get_automated_run(connection, run_id)
        if record is None:
            raise ValueError("automated_run_missing")
        if record.status == "completed":
            if (
                record.stop_reason != reason
                or record.candidate_planning_stop_diagnostic_json != diagnostic_json
            ):
                raise ValueError("automated_run_candidate_planning_stop_conflict")
            return record
        if record.status != "running":
            raise ValueError("automated_run_status_invalid")
        if diagnostic.cycle_number != record.current_cycle + 1:
            raise ValueError("automated_run_candidate_planning_cycle_invalid")
        if diagnostic.seed_attempt_limit != (
            diagnostic.exploration_generation_target_count
            * record.exploration_seed_attempt_multiplier
        ):
            raise ValueError("automated_run_candidate_planning_budget_invalid")
        if (
            any(
                link.cycle_number == diagnostic.cycle_number
                for link in list_automated_run_backtests(connection, record.run_id)
            )
            or get_automated_cycle_settlement(
                connection,
                record.run_id,
                diagnostic.cycle_number,
            )
            is not None
        ):
            raise ValueError("automated_run_candidate_planning_cycle_not_empty")
        return _complete_automated_run(
            connection,
            record,
            completed_at=completed_at,
            reason=reason,
            candidate_planning_stop_diagnostic_json=diagnostic_json,
        )


def record_automated_cycle_settlement(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    cycle_number: int,
    outcome: str,
    frontier_advanced: bool,
    observed_at: str,
) -> tuple[AutomatedRunRecord, AutomatedCycleSettlementRecord, bool]:
    decision = AutomatedCycleSettlementRecord(
        run_id=run_id,
        cycle_number=cycle_number,
        outcome=outcome,
        frontier_advanced=frontier_advanced,
        settled_at=observed_at,
    )
    if outcome not in {
        "qualified",
        "frontier_advanced",
        "not_qualified",
        "failed",
    }:
        raise ValueError("automated_cycle_settlement_outcome_invalid")
    run = get_automated_run(connection, run_id)
    if run is None:
        raise ValueError("automated_run_missing")
    existing = get_automated_cycle_settlement(
        connection,
        run_id,
        cycle_number,
    )
    if existing is not None:
        existing = create_automated_cycle_settlement(
            connection,
            replace(decision, settled_at=existing.settled_at),
        )
        return run, existing, True
    links = list_automated_run_backtests(connection, run_id)
    cycle_links = tuple(link for link in links if link.cycle_number == cycle_number)
    if run.current_cycle + 1 != cycle_number and not cycle_links:
        raise ValueError("automated_cycle_settlement_order_invalid")
    for link in cycle_links:
        snapshot = get_backtest_task(connection, link.task_id)
        if snapshot is None:
            raise ValueError("automated_run_backtest_task_missing")
        if snapshot.task.status not in {"completed", "failed"}:
            raise ValueError("automated_run_completion_has_active_backtests")
    created = create_automated_cycle_settlement(connection, decision)
    run = _record_settled_cycle(
        connection,
        run,
        cycle_failed=outcome == "failed",
        observed_at=observed_at,
    )
    return run, created, False


def decide_automated_cycle_transition(
    connection: sqlite3.Connection,
    record: AutomatedRunRecord,
    *,
    cycle_failed: bool,
) -> AutomatedCycleTransition:
    if record.status != "running" or record.started_at is None:
        raise ValueError("automated_run_status_invalid")
    if not isinstance(cycle_failed, bool):
        raise ValueError("automated_cycle_failed_invalid")

    cycle = record.current_cycle + 1
    consecutive_failures = record.consecutive_failures + 1 if cycle_failed else 0
    backtest_limit = automated_run_backtest_limit(record)
    backtest_limit_reached = backtest_limit is not None and (
        len(list_automated_run_backtests(connection, record.run_id))
        >= backtest_limit
    )
    # A finished later batch cannot finish the run while an earlier batch is active.
    links = list_automated_run_backtests(connection, record.run_id)
    unfinished_backtests = len({link.cycle_number for link in links}) > cycle or any(
        snapshot.task.status in {"created", "pending", "submission_unknown"}
        for link in links
        if (snapshot := get_backtest_task(connection, link.task_id)) is not None
    )
    if (
        not uses_continuous_recovery(record)
        and consecutive_failures >= record.max_consecutive_failures
    ):
        return AutomatedCycleTransition(
            status="failed",
            stop_reason="consecutive_failures_reached",
            consecutive_failures=consecutive_failures,
        )
    if cycle_failed and not unfinished_backtests and (
        (record.max_cycles > 0 and cycle >= record.max_cycles) or backtest_limit_reached
    ):
        return AutomatedCycleTransition(
            status="failed",
            stop_reason="final_cycle_failed",
            consecutive_failures=consecutive_failures,
        )
    if not unfinished_backtests and record.max_cycles > 0 and cycle >= record.max_cycles:
        return AutomatedCycleTransition(
            status="completed",
            stop_reason="max_cycles_reached",
            consecutive_failures=consecutive_failures,
        )
    if not unfinished_backtests and backtest_limit_reached:
        return AutomatedCycleTransition(
            status="completed",
            stop_reason="backtest_limit_reached",
            consecutive_failures=consecutive_failures,
        )
    return AutomatedCycleTransition(
        status="running",
        stop_reason=None,
        consecutive_failures=consecutive_failures,
    )


def _record_settled_cycle(
    connection: sqlite3.Connection,
    record: AutomatedRunRecord,
    *,
    cycle_failed: bool,
    observed_at: str,
) -> AutomatedRunRecord:
    if record.status == "running":
        return _advance_automated_cycle(
            connection,
            record,
            cycle_failed=cycle_failed,
            observed_at=observed_at,
        )
    if record.status not in {"completed", "failed"} or record.started_at is None:
        raise ValueError("automated_run_status_invalid")
    observed = _timestamp(observed_at, "automated_run_observed_at_invalid")
    started = _timestamp(record.started_at, "automated_run_started_at_invalid")
    if observed < started:
        raise ValueError("automated_run_observed_at_invalid")
    updated = replace(record, current_cycle=record.current_cycle + 1)
    return replace_automated_run(
        connection,
        updated,
        expected_status=record.status,
    )


def _advance_automated_cycle(
    connection: sqlite3.Connection,
    record: AutomatedRunRecord,
    *,
    cycle_failed: bool,
    observed_at: str,
) -> AutomatedRunRecord:
    if record.status != "running" or record.started_at is None:
        raise ValueError("automated_run_status_invalid")
    observed = _timestamp(observed_at, "automated_run_observed_at_invalid")
    started = _timestamp(record.started_at, "automated_run_started_at_invalid")
    if observed < started:
        raise ValueError("automated_run_observed_at_invalid")

    cycle = record.current_cycle + 1
    decision = decide_automated_cycle_transition(
        connection,
        record,
        cycle_failed=cycle_failed,
    )
    status = decision.status
    stop_reason = decision.stop_reason
    if status != "running":
        status, stop_reason = _terminal_run_state(
            connection,
            record.run_id,
            status=status,
            reason=stop_reason,
        )
    if status == "failed" and stop_reason != "submission_reconciliation_required":
        _cancel_unsubmitted_automated_run_backtests(
            connection,
            record.run_id,
            observed_at=observed_at,
        )
    updated = replace(
        record,
        status=status,
        current_cycle=cycle,
        consecutive_failures=decision.consecutive_failures,
        retry_not_before=record.retry_not_before if status == "running" else None,
        finished_at=observed_at if status != "running" else None,
        stop_reason=stop_reason,
    )
    return replace_automated_run(
        connection,
        updated,
        expected_status="running",
    )


def _complete_automated_run(
    connection: sqlite3.Connection,
    record: AutomatedRunRecord,
    *,
    completed_at: str,
    reason: str,
    candidate_planning_stop_diagnostic_json: str | None = None,
) -> AutomatedRunRecord:
    if record.status != "running" or record.started_at is None:
        raise ValueError("automated_run_status_invalid")
    completed = _timestamp(
        completed_at,
        "automated_run_finished_at_invalid",
    )
    started = _timestamp(record.started_at, "automated_run_started_at_invalid")
    if completed < started:
        raise ValueError("automated_run_finished_at_invalid")
    status, stop_reason = _terminal_run_state(
        connection,
        record.run_id,
        status="completed",
        reason=reason,
    )
    if stop_reason != reason:
        candidate_planning_stop_diagnostic_json = None
    updated = replace(
        record,
        status=status,
        retry_not_before=None,
        finished_at=completed_at,
        stop_reason=stop_reason,
        candidate_planning_stop_diagnostic_json=(
            candidate_planning_stop_diagnostic_json
        ),
    )
    return replace_automated_run(
        connection,
        updated,
        expected_status="running",
    )


def retire_previous_automated_runs(
    database_path: str | Path, *, account_scope: str, observed_at: str,
    reason: str = "replaced_by_new_run",
) -> None:
    """Under the process lock, stop old plans without cancelling remote requests."""
    if reason not in {"replaced_by_new_run", "stopped_for_manual_submission"}:
        raise ValueError("automated_run_retirement_reason_invalid")
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        active = list_active_automated_runs(connection)
        if any(run.account_scope != account_scope for run in active):
            raise ValueError("automated_run_account_scope_mismatch")
        if account_has_active_formal_submission(connection, account_scope):
            raise ValueError("automated_run_formal_submission_reconciliation_required")
        for run in active:
            if run.status == "created":
                replace_automated_run(
                    connection,
                    replace(run, status="failed", finished_at=observed_at,
                            stop_reason=reason),
                    expected_status="created",
                )
            else:
                record_automated_run_failure(
                    connection, run.run_id, failed_at=observed_at,
                    reason=reason,
                )
        # Older reconciliation failures may still contain unsubmitted tasks.
        # They no longer have permission to extend an abandoned plan.
        for task in list_active_backtest_tasks(connection):
            if task.account_scope != account_scope or task.status != "created":
                continue
            link = get_automated_run_backtest_by_task(connection, task.task_id)
            owner = get_automated_run(connection, link.run_id) if link else None
            if owner is not None and owner.status == "failed":
                cancel_unsubmitted_backtest_task(
                    connection, task.task_id, observed_at=observed_at
                )


def _require_no_unresolved_external_backtests(
    connection: sqlite3.Connection,
    account_scope: str,
    *,
    excluded_task_ids: frozenset[str] = frozenset(),
) -> None:
    unresolved = []
    for task in list_active_backtest_tasks(connection):
        if (task.account_scope != account_scope or task.status != "pending"
                or task.task_id in excluded_task_ids):
            continue
        link = get_automated_run_backtest_by_task(connection, task.task_id)
        owner = get_automated_run(connection, link.run_id) if link else None
        if owner is None or owner.account_scope != account_scope or owner.status != "failed":
            unresolved.append(task)
    if unresolved:
        raise ValueError("automated_run_backtest_reconciliation_required")
    if account_has_active_formal_submission(
        connection,
        account_scope,
    ):
        raise ValueError("automated_run_formal_submission_reconciliation_required")


def fail_automated_run(
    database_path: str | Path,
    run_id: str,
    *,
    failed_at: str,
    reason: str,
    request_failure_code: str | None = None,
    request_status_code: int | None = None,
    request_retry_after_seconds: float | None = None,
) -> AutomatedRunRecord:
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        return record_automated_run_failure(
            connection,
            run_id,
            failed_at=failed_at,
            reason=reason,
            request_failure_code=request_failure_code,
            request_status_code=request_status_code,
            request_retry_after_seconds=request_retry_after_seconds,
        )


def record_automated_run_failure(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    failed_at: str,
    reason: str,
    request_failure_code: str | None = None,
    request_status_code: int | None = None,
    request_retry_after_seconds: float | None = None,
) -> AutomatedRunRecord:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("automated_run_failure_reason_missing")
    record = get_automated_run(connection, run_id)
    if record is None:
        raise ValueError("automated_run_missing")
    status, stop_reason = _terminal_run_state(
        connection,
        record.run_id,
        status="failed",
        reason=reason,
    )
    if record.status == "failed":
        if record.stop_reason != stop_reason:
            raise ValueError("automated_run_failure_reason_conflict")
        return record
    if record.status != "running" or record.started_at is None:
        raise ValueError("automated_run_status_invalid")
    failed = _timestamp(failed_at, "automated_run_finished_at_invalid")
    started = _timestamp(record.started_at, "automated_run_started_at_invalid")
    if failed < started:
        raise ValueError("automated_run_finished_at_invalid")
    if stop_reason != "submission_reconciliation_required":
        _cancel_unsubmitted_automated_run_backtests(
            connection,
            record.run_id,
            observed_at=failed_at,
        )
    updated = replace(
        record,
        status=status,
        retry_not_before=None,
        last_request_failure_code=(
            request_failure_code
            if request_failure_code is not None
            else record.last_request_failure_code
        ),
        last_request_status_code=(
            request_status_code
            if request_failure_code is not None
            else record.last_request_status_code
        ),
        last_request_failure_at=(
            failed_at
            if request_failure_code is not None
            else record.last_request_failure_at
        ),
        last_request_retry_after_seconds=(
            request_retry_after_seconds
            if request_failure_code is not None
            else record.last_request_retry_after_seconds
        ),
        finished_at=failed_at,
        stop_reason=stop_reason,
    )
    return replace_automated_run(
        connection,
        updated,
        expected_status="running",
    )


def resume_automated_run_after_submission_reconciliation(
    database_path: str | Path,
    run_id: str,
) -> AutomatedRunRecord:
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        record = get_automated_run(connection, run_id)
        if record is None:
            raise ValueError("automated_run_missing")
        if (
            record.status != "failed"
            or record.stop_reason != "submission_reconciliation_required"
            or record.started_at is None
        ):
            raise ValueError("automated_run_reconciliation_resume_invalid")
        if automated_run_has_hard_stop_evidence(connection, record):
            raise ValueError("automated_run_reconciliation_resume_blocked")
        settings_policy = BacktestSettingsPolicy.from_config_dict(
            json.loads(record.settings_policy_json)
        )
        load_generation_catalog(
            connection,
            FieldCatalogContext(
                instrument_type=settings_policy.instrument_type,
                region=settings_policy.region,
                universe=settings_policy.universe,
                delay=settings_policy.delay,
            ),
            account_scope=record.account_scope,
        )
        links = list_automated_run_backtests(connection, record.run_id)
        for link in links:
            snapshot = get_backtest_task(connection, link.task_id)
            if snapshot is None:
                raise ValueError("automated_run_backtest_task_missing")
            if (
                snapshot.task.status == "submission_unknown"
                and not can_resume_independent_backtests(connection, record)
            ):
                raise ValueError("automated_run_submission_reconciliation_incomplete")
        if (
            record.automatic_submissions_enabled
            and automated_run_has_unresolved_formal_submission(
                connection,
                record.run_id,
            )
        ):
            raise ValueError("automated_run_submission_reconciliation_incomplete")
        _require_no_unresolved_external_backtests(
            connection,
            record.account_scope,
            excluded_task_ids=frozenset(link.task_id for link in links),
        )
        resumed = replace(
            record,
            status="running",
            finished_at=None,
            stop_reason=None,
        )
        return replace_automated_run(
            connection,
            resumed,
            expected_status="failed",
        )


def record_backtest_submission_rate_limit(
    database_path: str | Path, run_id: str, task_id: str, *,
    observed_at: str, retry_after_seconds: float,
) -> AutomatedRunRecord:
    """Retry the frozen queue head; successful reads do not reset its POST count."""
    if (isinstance(retry_after_seconds, bool) or not isinstance(retry_after_seconds, (int, float))
            or not math.isfinite(retry_after_seconds) or retry_after_seconds <= 0):
        raise ValueError("automated_run_retry_delay_invalid")
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        run = get_automated_run(connection, run_id)
        snapshot = get_backtest_task(connection, task_id)
        link = get_automated_run_backtest_by_task(connection, task_id)
        if (run is None or run.status != "running" or snapshot is None
                or snapshot.task.status != "created" or link is None or link.run_id != run_id):
            raise ValueError("automated_run_rate_limit_identity_invalid")
        count = (run.request_failure_count if submission_rate_limited(run)
                 and run.request_failure_count <= run.max_request_failures else 0) + 1
        exhausted = count > run.max_request_failures
        if exhausted:
            replace_backtest_task(connection, replace(
                snapshot.task, status="failed", last_observed_at=observed_at,
                finished_at=observed_at, failure_code="platform_submission_rate_limit_exhausted",
                failure_message=f"429限流，{run.max_request_failures}次重试耗尽；回测请求未被接受",
            ), expected_status="created")
        return replace_automated_run(connection, replace(
            run, request_failure_count=count,
            retry_not_before=(_timestamp(observed_at, "automated_run_observed_at_invalid")
                              + timedelta(seconds=retry_after_seconds)).isoformat(),
            last_request_failure_code="worldquant_submission_http_error",
            last_request_status_code=429, last_request_failure_at=observed_at,
            last_request_retry_after_seconds=retry_after_seconds,
        ), expected_status="running")


def submission_rate_limited(run: AutomatedRunRecord) -> bool:
    return (run.last_request_status_code == 429
            and run.last_request_failure_code == "worldquant_submission_http_error"
            and run.retry_not_before is not None)


def resume_request_failed_run(database_path: str | Path, run_id: str) -> None:
    """Explicitly resume retryable request failures without reposting sent tasks."""
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        run = get_automated_run(connection, run_id)
        if (run is None or run.status != "failed"
                or run.stop_reason != "request_failure_limit_reached"):
            return
        if list_active_automated_runs(connection):
            raise ValueError("automated_run_already_active")
        links = list_automated_run_backtests(connection, run_id)
        # Authentication can exhaust retries before any candidates are planned.
        # That empty plan still needs reopening under its original authority.
        if links and not any(get_backtest_task(connection, link.task_id).task.failure_code
                             == "automated_run_stopped_before_submission" for link in links):
            return
        if not run.real_backtests_authorized or automated_run_has_unresolved_formal_submission(connection, run_id):
            raise ValueError("automated_run_request_failure_resume_blocked")
        policy = BacktestSettingsPolicy.from_config_dict(json.loads(run.settings_policy_json))
        load_generation_catalog(connection, FieldCatalogContext(
            policy.instrument_type, policy.region, policy.universe, policy.delay,
        ), account_scope=run.account_scope)
        _require_no_unresolved_external_backtests(
            connection, run.account_scope, excluded_task_ids=frozenset(link.task_id for link in links),
        )
        for link in links:
            snapshot = get_backtest_task(connection, link.task_id)
            task = snapshot.task
            if task.failure_code != "automated_run_stopped_before_submission":
                continue
            reserved = get_backtest_task_by_identity(
                connection, account_scope=task.account_scope, formula_fingerprint=task.formula_fingerprint,
                settings_json=task.settings_json,
            )
            if reserved is not None and reserved.task.task_id != task.task_id:
                raise ValueError("原候选已在后续批次重新安排，不能恢复旧批次")
            if (task.status != "failed" or task.submission_started_at is not None
                    or task.remote_id is not None or task.platform_alpha_id is not None
                    or snapshot.result is not None or get_automated_cycle_settlement(
                        connection, run_id, link.cycle_number) is not None):
                raise ValueError("automated_run_cancelled_task_not_resumable")
            replace_backtest_task(connection, replace(
                task, status="created", last_observed_at=None, retry_not_before=None,
                finished_at=None, failure_code=None, failure_message=None,
            ), expected_status="failed")
        replace_automated_run(connection, replace(
            run, status="running", finished_at=None, stop_reason=None,
            request_failure_count=0, retry_not_before=None,
        ), expected_status="failed")


def record_automated_request_failure(
    database_path: str | Path,
    run_id: str,
    *,
    observed_at: str,
    retry_after_seconds: float,
    error_code: str,
    status_code: int | None,
) -> AutomatedRunRecord:
    if (
        isinstance(retry_after_seconds, bool)
        or not isinstance(retry_after_seconds, (int, float))
        or not math.isfinite(retry_after_seconds)
        or retry_after_seconds <= 0
    ):
        raise ValueError("automated_run_retry_delay_invalid")
    if not isinstance(error_code, str) or not error_code.strip():
        raise ValueError("automated_run_request_failure_code_invalid")
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        record = get_automated_run(connection, run_id)
        if record is None:
            raise ValueError("automated_run_missing")
        if record.status != "running" or record.started_at is None:
            raise ValueError("automated_run_status_invalid")
        observed = _timestamp(observed_at, "automated_run_observed_at_invalid")
        started = _timestamp(record.started_at, "automated_run_started_at_invalid")
        if observed < started:
            raise ValueError("automated_run_observed_at_invalid")
        failure_count = record.request_failure_count + 1
        if (
            not uses_continuous_recovery(record)
            and failure_count >= record.max_request_failures
        ):
            status, stop_reason = _terminal_run_state(
                connection,
                record.run_id,
                status="failed",
                reason="request_failure_limit_reached",
            )
            if stop_reason != "submission_reconciliation_required":
                _cancel_unsubmitted_automated_run_backtests(
                    connection,
                    record.run_id,
                    observed_at=observed_at,
                )
            updated = replace(
                record,
                status=status,
                request_failure_count=failure_count,
                retry_not_before=None,
                last_request_failure_code=error_code,
                last_request_status_code=status_code,
                last_request_failure_at=observed_at,
                last_request_retry_after_seconds=retry_after_seconds,
                finished_at=observed_at,
                stop_reason=stop_reason,
            )
        else:
            retry_not_before = observed + timedelta(seconds=retry_after_seconds)
            updated = replace(
                record,
                request_failure_count=failure_count,
                retry_not_before=retry_not_before.isoformat(),
                last_request_failure_code=error_code,
                last_request_status_code=status_code,
                last_request_failure_at=observed_at,
                last_request_retry_after_seconds=retry_after_seconds,
            )
        return replace_automated_run(
            connection,
            updated,
            expected_status="running",
        )


def clear_automated_request_failures(
    database_path: str | Path,
    run_id: str,
) -> AutomatedRunRecord:
    with open_database(database_path) as connection:
        record = get_automated_run(connection, run_id)
        if record is None:
            raise ValueError("automated_run_missing")
        if record.status != "running":
            return record
        if record.request_failure_count == 0 and record.retry_not_before is None:
            return record
        updated = replace(
            record,
            request_failure_count=0,
            retry_not_before=None,
        )
        return replace_automated_run(
            connection,
            updated,
            expected_status="running",
        )


def _terminal_run_state(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    status: str,
    reason: str | None,
) -> tuple[str, str | None]:
    run = get_automated_run(connection, run_id)
    if run is None:
        raise ValueError("automated_run_missing")
    if (status == "completed" and automated_run_has_submission_unknown(connection, run_id)) or (
        run.automatic_submissions_enabled
        and automated_run_has_unresolved_formal_submission(connection, run_id)
    ):
        return "failed", "submission_reconciliation_required"
    return status, reason


def _cancel_unsubmitted_automated_run_backtests(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    observed_at: str,
) -> None:
    for link in list_automated_run_backtests(connection, run_id):
        snapshot = get_backtest_task(connection, link.task_id)
        if snapshot is None:
            raise ValueError("automated_run_backtest_task_missing")
        if snapshot.task.status == "created":
            cancel_unsubmitted_backtest_task(
                connection,
                link.task_id,
                observed_at=observed_at,
            )


def _run_id(
    *,
    account_scope: str,
    settings_policy_json: str,
    limits: AutomatedRunLimits,
    created_at: str,
) -> str:
    payload = json.dumps(
        {
            "account_scope": account_scope,
            "settingsPolicy": json.loads(settings_policy_json),
            "limits": {
                name: getattr(limits, name) for name in limits.__dataclass_fields__
            },
            "created_at": created_at,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"run_{sha256(payload.encode('utf-8')).hexdigest()}"


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
