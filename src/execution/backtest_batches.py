from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from execution.backtests import prepare_backtest_task
from execution.runs import (
    automated_run_backtest_limit,
    remaining_automated_run_backtests,
)
from execution.seeds import load_signal_frontiers
from execution.cycle_schedule import scheduled_cycle_number
from generation.candidate import FormulaCandidate
from generation.direction import DIRECTION_REVERSAL, reverse_direction_candidate
from generation.self_correlation import SELF_CORRELATION_REPAIR_FAMILIES
from generation.parser import parse_formula
from learning.direction import negative_direction_is_testable
from persistence.backtests import (
    backtest_was_cancelled_before_submission,
    get_backtest_task_by_identity,
    BacktestMutationRecord,
    BacktestSnapshot,
    create_backtest_mutation,
    get_backtest_task,
    get_backtest_mutation,
)
from persistence.database import open_database
from persistence.runs import (
    AutomatedRunBacktestRecord,
    AutomatedRunRecord,
    attach_backtest_to_automated_run,
    get_automated_run,
    get_automated_cycle_settlement,
    list_automated_run_backtests,
)
from selection.settings import BacktestSettingsPolicy
from worldquant.backtests import BacktestSettings


@dataclass(frozen=True, slots=True)
class AutomatedRunBacktestUsage:
    run_id: str
    prepared_backtests: int
    attempted_backtests: int
    remaining_backtests: int | None
    cycle_number: int
    cycle_prepared_backtests: int
    cycle_remaining_backtests: int


@dataclass(frozen=True, slots=True)
class AutomatedCandidateBacktest:
    candidate: FormulaCandidate
    settings: BacktestSettings


@dataclass(frozen=True, slots=True)
class _AutomatedBatchItem:
    formula: str
    settings: BacktestSettings
    mutation: FormulaCandidate | None = None


def prepare_automated_candidate_backtest_batch(
    database_path: str | Path,
    *,
    run_id: str,
    candidates: tuple[AutomatedCandidateBacktest, ...],
    created_at: str,
    cycle_number: int | None = None,
) -> tuple[BacktestSnapshot, ...]:
    _validate_automated_candidates(candidates)
    return _prepare_automated_items(
        database_path,
        run_id=run_id,
        items=tuple(
            _AutomatedBatchItem(
                item.candidate.formula,
                item.settings,
                (
                    item.candidate
                    if item.candidate.generation_action == "mutation"
                    else None
                ),
            )
            for item in candidates
        ),
        created_at=created_at,
        cycle_number=cycle_number,
    )


def _prepare_automated_items(
    database_path: str | Path,
    *,
    run_id: str,
    items: tuple[_AutomatedBatchItem, ...],
    created_at: str,
    cycle_number: int | None = None,
) -> tuple[BacktestSnapshot, ...]:
    prepared: list[BacktestSnapshot] = []
    request_fingerprints: set[str] = set()
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        run = _required_running_run(connection, run_id)
        if not run.real_backtests_authorized:
            raise ValueError("automated_run_backtests_not_authorized")
        policy = BacktestSettingsPolicy.from_config_dict(
            json.loads(run.settings_policy_json)
        )
        scheduled = scheduled_cycle_number(connection, run)
        if cycle_number is None:
            cycle_number = scheduled
        if cycle_number != scheduled:
            raise ValueError("automated_cycle_planning_state_changed")
        eligible_parent_ids = set(
            load_signal_frontiers(connection, optimization_only=run.optimization_only).active_branch_task_ids
        )
        for item in items:
            if run.optimization_only and (
                item.mutation is None or item.mutation.change is None
                or item.mutation.change.action in (*SELF_CORRELATION_REPAIR_FAMILIES, DIRECTION_REVERSAL)
            ):
                raise ValueError("optimization_run_candidate_not_allowed")
            settings = item.settings
            if not policy.allows(settings):
                raise ValueError("automated_run_settings_not_allowed")
            parent = None
            if item.mutation is not None:
                assert item.mutation.parent_task_id is not None
                assert item.mutation.parent_formula_fingerprint is not None
                is_direction = (
                    item.mutation.change is not None
                    and item.mutation.change.action == DIRECTION_REVERSAL
                )
                if (
                    not is_direction
                    and item.mutation.parent_task_id not in eligible_parent_ids
                ):
                    raise ValueError("backtest_mutation_parent_not_eligible")
                parent = get_backtest_task(
                    connection,
                    item.mutation.parent_task_id,
                )
                if parent is None or parent.result is None:
                    raise ValueError("backtest_mutation_parent_result_missing")
                if parent.task.status != "completed":
                    raise ValueError("backtest_mutation_parent_not_completed")
                if parent.task.account_scope != run.account_scope:
                    raise ValueError("backtest_mutation_parent_account_mismatch")
                if (
                    parent.task.formula_fingerprint
                    != item.mutation.parent_formula_fingerprint
                ):
                    raise ValueError("backtest_mutation_parent_formula_mismatch")
                parent_settings = BacktestSettings.from_platform_dict(
                    json.loads(parent.task.settings_json)
                )
                if parent_settings != settings:
                    raise ValueError("backtest_mutation_settings_mismatch")
                if is_direction:
                    expected = reverse_direction_candidate(
                        parse_formula(parent.task.formula).expression,
                        parent_task_id=parent.task.task_id,
                    )
                    if (
                        not negative_direction_is_testable(parent)
                        or get_backtest_mutation(connection, parent.task.task_id)
                        is not None
                        or item.mutation != expected
                    ):
                        raise ValueError("backtest_direction_source_invalid")
                    existing = get_backtest_task_by_identity(
                        connection, account_scope=run.account_scope,
                        formula_fingerprint=item.mutation.fingerprint, settings_json=parent.task.settings_json,
                    )
                    if existing is not None and not backtest_was_cancelled_before_submission(existing.task):
                        raise ValueError("backtest_direction_request_already_exists")
            snapshot = prepare_backtest_task(
                connection,
                account_scope=run.account_scope,
                formula=item.formula,
                settings=settings.as_platform_dict(),
                created_at=created_at,
            )
            request_fingerprint = snapshot.task.request_fingerprint
            if request_fingerprint in request_fingerprints:
                raise ValueError("backtest_batch_formula_duplicate")
            request_fingerprints.add(request_fingerprint)
            if item.mutation is not None:
                assert parent is not None
                assert item.mutation.change is not None
                create_backtest_mutation(
                    connection,
                    BacktestMutationRecord(
                        child_task_id=snapshot.task.task_id,
                        parent_task_id=parent.task.task_id,
                        action=item.mutation.change.action,
                        location=item.mutation.change.location,
                        before=item.mutation.change.before,
                        after=item.mutation.change.after,
                    ),
                )
            attach_backtest_to_automated_run(
                connection,
                AutomatedRunBacktestRecord(
                    run_id=run.run_id,
                    task_id=snapshot.task.task_id,
                    cycle_number=cycle_number,
                ),
            )
            prepared.append(snapshot)

        usage = _load_usage(connection, run)
        backtest_limit = automated_run_backtest_limit(run)
        if backtest_limit is not None and usage.prepared_backtests > backtest_limit:
            raise ValueError("automated_run_backtest_limit_reached")
        cycle_count = sum(
            link.cycle_number == cycle_number
            for link in list_automated_run_backtests(connection, run.run_id)
        )
        if cycle_count > run.backtest_count:
            raise ValueError("automated_run_cycle_backtest_limit_reached")
    return tuple(prepared)


def load_automated_run_backtest_usage(
    database_path: str | Path,
    run_id: str,
) -> AutomatedRunBacktestUsage:
    with open_database(database_path) as connection:
        run = get_automated_run(connection, run_id)
        if run is None:
            raise ValueError("automated_run_missing")
        return _load_usage(connection, run)


def _load_usage(
    connection: sqlite3.Connection,
    run: AutomatedRunRecord,
) -> AutomatedRunBacktestUsage:
    records = list_automated_run_backtests(connection, run.run_id)
    attempted = 0
    for record in records:
        snapshot = get_backtest_task(connection, record.task_id)
        if snapshot is None:
            raise ValueError("automated_run_backtest_task_missing")
        if snapshot.task.status != "created":
            attempted += 1
    cycle_number = (
        run.current_cycle + 1
        if run.status in {"created", "running"}
        else run.current_cycle
    )
    if records:
        latest = max((record.cycle_number for record in records), default=0)
        cycle_number = latest or 1
        if latest and get_automated_cycle_settlement(connection, run.run_id, latest):
            cycle_number = latest + 1
    cycle_prepared = sum(record.cycle_number == cycle_number for record in records)
    return AutomatedRunBacktestUsage(
        run_id=run.run_id,
        prepared_backtests=len(records),
        attempted_backtests=attempted,
        remaining_backtests=remaining_automated_run_backtests(run, len(records)),
        cycle_number=cycle_number,
        cycle_prepared_backtests=cycle_prepared,
        cycle_remaining_backtests=max(0, run.backtest_count - cycle_prepared),
    )


def _required_running_run(
    connection: sqlite3.Connection,
    run_id: str,
) -> AutomatedRunRecord:
    run = get_automated_run(connection, run_id)
    if run is None:
        raise ValueError("automated_run_missing")
    if run.status != "running":
        raise ValueError("automated_run_status_invalid")
    return run


def _validate_mutation_candidates(
    candidates: tuple[FormulaCandidate, ...],
) -> None:
    if not isinstance(candidates, tuple) or not candidates:
        raise ValueError("backtest_mutation_candidates_invalid")
    if any(
        not isinstance(candidate, FormulaCandidate)
        or candidate.generation_action != "mutation"
        or candidate.parent_task_id is None
        or candidate.parent_formula_fingerprint is None
        or candidate.change is None
        for candidate in candidates
    ):
        raise ValueError("backtest_mutation_candidate_invalid")


def _validate_automated_candidates(
    candidates: tuple[AutomatedCandidateBacktest, ...],
) -> None:
    if not isinstance(candidates, tuple) or not candidates:
        raise ValueError("backtest_automated_candidates_invalid")
    for item in candidates:
        if not isinstance(item, AutomatedCandidateBacktest):
            raise ValueError("backtest_automated_candidate_invalid")
        if not isinstance(item.settings, BacktestSettings):
            raise ValueError("backtest_automated_candidate_settings_invalid")
        candidate = item.candidate
        if not isinstance(candidate, FormulaCandidate):
            raise ValueError("backtest_automated_candidate_invalid")
        if candidate.generation_action == "exploration":
            if (
                candidate.parent_task_id is not None
                or candidate.parent_formula_fingerprint is not None
                or candidate.change is not None
            ):
                raise ValueError("backtest_automated_candidate_invalid")
        elif candidate.generation_action == "mutation":
            _validate_mutation_candidates((candidate,))
        else:
            raise ValueError("backtest_automated_candidate_invalid")
