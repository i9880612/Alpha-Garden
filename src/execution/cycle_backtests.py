from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from evaluation.backtests import evaluate_backtest
from execution.backtests import (
    remaining_backtest_pending_seconds,
    remaining_backtest_retry_seconds,
    record_pending_observation,
    expire_pending_backtest_task,
)
from execution.qualified_archive import synchronize_qualified_alpha_archive
from execution.seeds import (
    batch_advances_signal_frontier,
    synchronize_signal_seeds,
)
from execution.submission_queue import (
    SubmissionQueueSynchronization,
    synchronize_submission_queue,
)
from execution.real_backtests import advance_real_backtest, account_in_flight_backtest_count
from execution.cycle_schedule import unsettled_cycle_numbers
from execution.runs import (
    fail_automated_run,
    remaining_automated_run_backtests,
    record_automated_cycle_settlement,
    uses_continuous_recovery,
    record_backtest_submission_rate_limit,
    submission_rate_limited,
    clear_automated_request_failures,
)
from persistence.backtests import BacktestSnapshot, get_backtest_task
from persistence.database import open_database
from persistence.runs import (
    AutomatedRunBacktestRecord,
    AutomatedRunRecord,
    get_automated_cycle_settlement,
    get_automated_run,
    list_automated_run_backtests,
)
from worldquant.client import WorldQuantClient, WorldQuantRequestError
from execution.submission_checks import check_completed_backtest, remaining_submission_check_seconds
from execution.seed_evidence import capture_next_seed_series


@dataclass(frozen=True, slots=True)
class CycleBacktestCounts:
    created: int
    submission_unknown: int
    pending: int
    completed: int
    failed: int

    @property
    def terminal(self) -> bool:
        return self.created == self.submission_unknown == self.pending == 0


@dataclass(frozen=True, slots=True)
class AutomatedCycleBacktestAdvance:
    run_id: str
    cycle_number: int
    action: str
    snapshot: BacktestSnapshot | None
    counts: CycleBacktestCounts
    platform_request_performed: bool
    run_status: str
    retry_after_seconds: float | None = None
    request_error_code: str | None = None


@dataclass(frozen=True, slots=True)
class AutomatedCycleSettlement:
    run: AutomatedRunRecord
    cycle_number: int
    outcome: str
    completed_backtests: int
    failed_backtests: int
    passed_evaluations: int
    pending_evaluations: int
    frontier_advanced: bool
    already_settled: bool


def advance_automated_cycle_backtests(
    database_path: str | Path,
    client: WorldQuantClient,
    run_id: str,
    *,
    observed_at: str,
    cycle_number: int | None = None,
) -> AutomatedCycleBacktestAdvance:
    run, cycle_number, snapshots = _load_cycle(database_path, run_id, cycle_number)
    if run.status != "running":
        raise ValueError("automated_run_status_invalid")
    counts = _counts(snapshots)
    if counts.terminal:
        for snapshot in snapshots:
            if check_completed_backtest(database_path, client, snapshot.task.task_id, observed_at=observed_at):
                return _advance_result(
                    run=run, cycle_number=cycle_number, action="submission_check_observed",
                    snapshot=snapshot, counts=counts, platform_request_performed=True,
                )
        with open_database(database_path) as connection:
            delays = [delay for snapshot in snapshots
                      if (delay := remaining_submission_check_seconds(connection, snapshot, observed_at)) is not None]
        if delays:
            return _advance_result(
                run=run, cycle_number=cycle_number, action="submission_check_wait", snapshot=None,
                counts=counts, platform_request_performed=False, retry_after_seconds=min(delays),
            )
        seed_delay = capture_next_seed_series(
            database_path, client, account_scope=run.account_scope, observed_at=observed_at,
            candidate_task_ids=tuple(s.task.task_id for s in snapshots if s.task.status == "completed"),
            admit_seeds=not run.optimization_only,
        )
        if seed_delay is not None:
            return _advance_result(
                run=run, cycle_number=cycle_number, action="seed_evidence_captured", snapshot=None,
                counts=counts, platform_request_performed=True, retry_after_seconds=seed_delay,
            )
        return _advance_result(
            run=run,
            cycle_number=cycle_number,
            action="cycle_terminal",
            snapshot=None,
            counts=counts,
            platform_request_performed=False,
        )

    scheduling = _load_run_active_snapshots(database_path, run)
    unknown = tuple(
        snapshot
        for snapshot in scheduling
        if snapshot.task.status == "submission_unknown"
    )
    pending = tuple(
        snapshot for snapshot in scheduling if snapshot.task.status == "pending"
    )
    created = tuple(
        snapshot for snapshot in snapshots if snapshot.task.status == "created"
    )
    with open_database(database_path) as connection:
        account_in_flight = account_in_flight_backtest_count(connection, run.account_scope)
    submit_delay = _submission_delay(run, observed_at)
    overdue = tuple(item for item in pending
                    if remaining_backtest_pending_seconds(
                        item, observed_at, max_pending_seconds=run.max_pending_seconds) == 0)
    if overdue:
        return _advance_task(database_path, client, run, cycle_number,
                             _oldest_observation(overdue), observed_at=observed_at,
                             allow_submission=False)
    if created and submit_delay is None and account_in_flight < run.max_in_flight_backtests:
        return _advance_task(
            database_path,
            client,
            run,
            cycle_number,
            sorted(created, key=lambda item: item.task.task_id)[0],
            observed_at=observed_at,
            allow_submission=True,
        )
    if pending:
        ready = tuple(
            snapshot
            for snapshot in pending
            if remaining_backtest_retry_seconds(snapshot, observed_at) is None
        )
        if not ready:
            return _advance_result(
                run=run,
                cycle_number=cycle_number,
                action="retry_wait",
                snapshot=None,
                counts=counts,
                platform_request_performed=False,
                retry_after_seconds=min(
                    *(remaining_backtest_retry_seconds(snapshot, observed_at) for snapshot in pending),
                    *([submit_delay] if created and submit_delay is not None else []),
                    60.0,
                ),
            )
        return _advance_task(
            database_path,
            client,
            run,
            cycle_number,
            _oldest_observation(ready),
            observed_at=observed_at,
            allow_submission=False,
        )
    if created and submit_delay is not None:
        return _advance_result(
            run=run, cycle_number=cycle_number, action="submission_cooldown", snapshot=None,
            counts=counts, platform_request_performed=False, retry_after_seconds=submit_delay,
        )
    if created and account_in_flight >= run.max_in_flight_backtests:
        return _advance_result(
            run=run, cycle_number=cycle_number, action="capacity_wait", snapshot=None,
            counts=counts, platform_request_performed=False, retry_after_seconds=60.0,
        )
    if unknown:
        return _advance_task(
            database_path,
            client,
            run,
            cycle_number,
            _oldest_observation(unknown),
            observed_at=observed_at,
            allow_submission=False,
        )
    raise ValueError("automated_cycle_backtest_state_invalid")


def settle_automated_cycle(
    database_path: str | Path,
    run_id: str,
    *,
    cycle_number: int,
    observed_at: str,
) -> AutomatedCycleSettlement:
    if (
        isinstance(cycle_number, bool)
        or not isinstance(cycle_number, int)
        or cycle_number <= 0
    ):
        raise ValueError("automated_cycle_number_invalid")
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        run = get_automated_run(connection, run_id)
        if run is None:
            raise ValueError("automated_run_missing")
        links = list_automated_run_backtests(connection, run.run_id)
        snapshots = _required_cycle_snapshots(
            connection,
            run,
            links,
            cycle_number=cycle_number,
        )
        saved_settlement = get_automated_cycle_settlement(
            connection,
            run.run_id,
            cycle_number,
        )
        counts = _counts(snapshots)
        if not counts.terminal:
            raise ValueError("automated_cycle_tasks_not_terminal")

        evaluations = tuple(
            evaluate_backtest(snapshot)
            for snapshot in snapshots
            if snapshot.task.status == "completed"
        )
        passed = sum(item.state == "passed" for item in evaluations)
        pending = sum(item.state == "pending" for item in evaluations)
        completed_task_ids = tuple(
            snapshot.task.task_id
            for snapshot in snapshots
            if snapshot.task.status == "completed"
        )
        synchronize_submission_queue(
            connection,
            candidate_task_ids=completed_task_ids,
            enqueued_at=observed_at,
        )
        synchronize_signal_seeds(connection, candidate_task_ids=completed_task_ids)
        synchronize_qualified_alpha_archive(connection, observed_at=observed_at)
        if saved_settlement is not None:
            settled_run, decision, _ = record_automated_cycle_settlement(
                connection,
                run.run_id,
                cycle_number=cycle_number,
                outcome=saved_settlement.outcome,
                frontier_advanced=saved_settlement.frontier_advanced,
                observed_at=observed_at,
            )
            return AutomatedCycleSettlement(
                run=settled_run,
                cycle_number=cycle_number,
                outcome=decision.outcome,
                completed_backtests=counts.completed,
                failed_backtests=counts.failed,
                passed_evaluations=passed,
                pending_evaluations=pending,
                frontier_advanced=decision.frontier_advanced,
                already_settled=True,
            )

        frontier_advanced = batch_advances_signal_frontier(
            connection,
            completed_task_ids,
        )
        outcome = (
            "qualified"
            if passed
            else "frontier_advanced"
            if frontier_advanced
            else "not_qualified"
            if evaluations
            else "failed"
        )
        settled_run, decision, already_settled = record_automated_cycle_settlement(
            connection,
            run.run_id,
            cycle_number=cycle_number,
            outcome=outcome,
            frontier_advanced=frontier_advanced,
            observed_at=observed_at,
        )
        return AutomatedCycleSettlement(
            run=settled_run,
            cycle_number=cycle_number,
            outcome=decision.outcome,
            completed_backtests=counts.completed,
            failed_backtests=counts.failed,
            passed_evaluations=passed,
            pending_evaluations=pending,
            frontier_advanced=decision.frontier_advanced,
            already_settled=already_settled,
        )


def synchronize_automated_cycle_submission_queue(
    database_path: str | Path,
    run_id: str,
    *,
    cycle_number: int,
    observed_at: str,
) -> SubmissionQueueSynchronization:
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        run = get_automated_run(connection, run_id)
        if run is None:
            raise ValueError("automated_run_missing")
        links = list_automated_run_backtests(connection, run.run_id)
        snapshots = _required_cycle_snapshots(
            connection,
            run,
            links,
            cycle_number=cycle_number,
        )
        return synchronize_submission_queue(
            connection,
            candidate_task_ids=tuple(
                snapshot.task.task_id
                for snapshot in snapshots
                if snapshot.task.status == "completed"
            ),
            enqueued_at=observed_at,
        )


def _advance_task(
    database_path: str | Path,
    client: WorldQuantClient,
    run: AutomatedRunRecord,
    cycle_number: int,
    snapshot: BacktestSnapshot,
    *,
    observed_at: str,
    allow_submission: bool,
) -> AutomatedCycleBacktestAdvance:
    with open_database(database_path) as connection:
        cycle_number = next(
            link.cycle_number
            for link in list_automated_run_backtests(connection, run.run_id)
            if link.task_id == snapshot.task.task_id
        )
    try:
        advanced = advance_real_backtest(
            database_path,
            client,
            snapshot.task.task_id,
            observed_at=observed_at,
            allow_submission=allow_submission,
            max_pending_seconds=run.max_pending_seconds,
        )
    except WorldQuantRequestError as exc:
        if exc.code == "worldquant_authentication_expired" and not exc.outcome_unknown:
            with open_database(database_path) as connection:
                unclaimed = get_backtest_task(connection, snapshot.task.task_id)
            return _advance_result(
                run=run, cycle_number=cycle_number, action="authentication_required",
                snapshot=unclaimed,
                counts=_counts(_load_cycle_snapshots(database_path, run, cycle_number=cycle_number)),
                platform_request_performed=True,
            )
        if not exc.outcome_unknown and (exc.status_code == 429 or (
                snapshot.task.status == "pending" and exc.retryable
                and (uses_continuous_recovery(run) or submission_rate_limited(run)))):
            delay = exc.retry_after_seconds or 120.0
            if snapshot.task.status == "created":
                run = record_backtest_submission_rate_limit(
                    database_path, run.run_id, snapshot.task.task_id,
                    observed_at=observed_at, retry_after_seconds=delay,
                )
                with open_database(database_path) as connection:
                    deferred = get_backtest_task(connection, snapshot.task.task_id)
                action = "failed" if deferred.task.status == "failed" else "submission_rate_limited"
            elif snapshot.task.status == "pending":
                with open_database(database_path) as connection:
                    deferred = record_pending_observation(
                        connection, snapshot.task.task_id, observed_at=observed_at,
                        retry_after_seconds=delay,
                    )
                    expired = expire_pending_backtest_task(
                        connection, snapshot.task.task_id, observed_at=observed_at,
                        max_pending_seconds=run.max_pending_seconds,
                    )
                    deferred = expired or deferred
                action = ("pending_timeout" if deferred.task.status == "failed" else
                          "poll_rate_limited" if exc.status_code == 429 else "poll_retry_scheduled")
            else:
                raise
            current = _load_run_active_snapshots(database_path, run)
            peer_waits = [remaining_backtest_retry_seconds(item, observed_at) or 0.0
                          for item in current if item.task.status == "pending"]
            submit_delay = _submission_delay(run, observed_at)
            if submit_delay is not None and any(item.task.status == "created" for item in current):
                peer_waits.append(submit_delay)
            if snapshot.task.status == "pending" and _submission_delay(run, observed_at) is None:
                with open_database(database_path) as connection:
                    if (any(item.task.status == "created" for item in current)
                            and account_in_flight_backtest_count(connection, run.account_scope)
                            < run.max_in_flight_backtests):
                        peer_waits.append(0.0)
            return _advance_result(
                run=run, cycle_number=cycle_number, action=action, snapshot=deferred,
                counts=_counts(_load_cycle_snapshots(database_path, run, cycle_number=cycle_number)),
                platform_request_performed=True, retry_after_seconds=min([delay, *peer_waits]),
            )
        if not exc.outcome_unknown:
            raise
        with open_database(database_path) as connection:
            guarded = get_backtest_task(connection, snapshot.task.task_id)
        if guarded is None or guarded.task.status != "submission_unknown":
            raise ValueError("automated_cycle_unknown_guard_missing") from exc
        if submission_rate_limited(run):
            run = clear_automated_request_failures(database_path, run.run_id)
        current = _load_cycle_snapshots(
            database_path,
            run,
            cycle_number=cycle_number,
        )
        current_counts = _counts(_load_run_active_snapshots(database_path, run))
        run_status = run.status
        if (
            current_counts.pending == 0
            and current_counts.submission_unknown >= run.max_in_flight_backtests
            and not uses_continuous_recovery(run)
        ):
            run_status = fail_automated_run(
                database_path,
                run.run_id,
                failed_at=observed_at,
                reason="submission_reconciliation_required",
            ).status
        return AutomatedCycleBacktestAdvance(
            run_id=run.run_id,
            cycle_number=cycle_number,
            action="reconciliation_required",
            snapshot=guarded,
            counts=current_counts,
            platform_request_performed=True,
            run_status=run_status,
            request_error_code=exc.code,
            retry_after_seconds=60.0 if uses_continuous_recovery(run) else None,
        )

    if advanced.action == "reconciliation_required":
        current = _load_cycle_snapshots(
            database_path,
            run,
            cycle_number=cycle_number,
        )
        current_counts = _counts(_load_run_active_snapshots(database_path, run))
        run_status = run.status
        if not uses_continuous_recovery(run) and current_counts.pending == 0 and (
            current_counts.submission_unknown >= run.max_in_flight_backtests
            or (not allow_submission and current_counts.created == 0)
        ):
            run_status = fail_automated_run(
                database_path,
                run.run_id,
                failed_at=observed_at,
                reason="submission_reconciliation_required",
            ).status
    else:
        current = _load_cycle_snapshots(
            database_path,
            run,
            cycle_number=cycle_number,
        )
        current_counts = _counts(current)
        run_status = run.status
    retry_after_seconds = _bounded_pending_wait(
        current,
        observed_at=observed_at,
        max_pending_seconds=run.max_pending_seconds,
        requested_seconds=(60.0 if advanced.action == "reconciliation_required"
                           and uses_continuous_recovery(run) else advanced.retry_after_seconds),
    )
    if retry_after_seconds is not None:
        current = _load_run_active_snapshots(database_path, run)
        with open_database(database_path) as connection:
            account_in_flight = account_in_flight_backtest_count(connection, run.account_scope)
        # A task-specific wait must not put ready peers to sleep.
        if (
            any(item.task.status == "created" for item in current)
            and _submission_delay(run, observed_at) is None
            and (
                account_in_flight < run.max_in_flight_backtests
            )
        ):
            retry_after_seconds = 0.0
        else:
            peer_waits = [
                remaining_backtest_retry_seconds(item, observed_at) or 0.0
                for item in current
                if item.task.status == "pending"
            ]
            if peer_waits:
                retry_after_seconds = min(retry_after_seconds, *peer_waits)
            submit_delay = _submission_delay(run, observed_at)
            if submit_delay is not None and any(item.task.status == "created" for item in current):
                retry_after_seconds = min(retry_after_seconds, submit_delay)
    return _advance_result(
        run=run,
        cycle_number=cycle_number,
        action=advanced.action,
        snapshot=advanced.snapshot,
        counts=current_counts,
        platform_request_performed=advanced.platform_request_performed,
        run_status=run_status,
        retry_after_seconds=retry_after_seconds,
    )


def remaining_automated_pending_seconds(
    database_path: str | Path,
    run_id: str,
    *,
    observed_at: str,
) -> float | None:
    with open_database(database_path) as connection:
        run = get_automated_run(connection, run_id)
        if run is None:
            raise ValueError("automated_run_missing")
        remaining: list[float] = []
        for link in list_automated_run_backtests(connection, run.run_id):
            snapshot = get_backtest_task(connection, link.task_id)
            if snapshot is None:
                raise ValueError("automated_run_backtest_task_missing")
            if snapshot.task.status == "pending":
                remaining.append(
                    remaining_backtest_pending_seconds(
                        snapshot,
                        observed_at,
                        max_pending_seconds=run.max_pending_seconds,
                    )
                )
    return min(remaining) if remaining else None


def failed_automated_run_has_local_work(
    connection: sqlite3.Connection,
    run: AutomatedRunRecord,
) -> bool:
    if run.status != "failed":
        return False
    links = list_automated_run_backtests(connection, run.run_id)
    for link in links:
        snapshot = get_backtest_task(connection, link.task_id)
        if snapshot is None:
            raise ValueError("automated_run_backtest_task_missing")
        if snapshot.task.status == "pending":
            return True
    return _terminal_unsettled_cycle_number(connection, run, links) is not None


def settle_failed_automated_cycle_if_terminal(
    database_path: str | Path,
    run_id: str,
    *,
    observed_at: str,
) -> AutomatedCycleSettlement | None:
    with open_database(database_path) as connection:
        run = get_automated_run(connection, run_id)
        if run is None:
            raise ValueError("automated_run_missing")
        if run.status != "failed":
            return None
        links = list_automated_run_backtests(connection, run.run_id)
        cycle_number = _terminal_unsettled_cycle_number(connection, run, links, require_checked=True)
    if cycle_number is None:
        return None
    return settle_automated_cycle(
        database_path,
        run_id,
        cycle_number=cycle_number,
        observed_at=observed_at,
    )


def _terminal_unsettled_cycle_number(
    connection: sqlite3.Connection,
    run: AutomatedRunRecord,
    links: tuple[AutomatedRunBacktestRecord, ...],
    *, require_checked: bool = False,
) -> int | None:
    for cycle_number in unsettled_cycle_numbers(connection, run.run_id):
        snapshots = _required_cycle_snapshots(
            connection,
            run,
            links,
            cycle_number=cycle_number,
        )
        if _counts(snapshots).terminal and (not require_checked or not any(
            remaining_submission_check_seconds(connection, snapshot, snapshot.task.finished_at) is not None
            for snapshot in snapshots
        )):
            return cycle_number
    return None


def _bounded_pending_wait(
    snapshots: tuple[BacktestSnapshot, ...],
    *,
    observed_at: str,
    max_pending_seconds: int | None,
    requested_seconds: float | None,
) -> float | None:
    if requested_seconds is None:
        return None
    if max_pending_seconds is None:
        return requested_seconds
    remaining = tuple(
        remaining_backtest_pending_seconds(
            snapshot,
            observed_at,
            max_pending_seconds=max_pending_seconds,
        )
        for snapshot in snapshots
        if snapshot.task.status == "pending"
    )
    return min((requested_seconds, *(seconds for seconds in remaining if seconds > 0)))


def _submission_delay(run, observed_at: str) -> float | None:
    if not submission_rate_limited(run):
        return None
    remaining = (datetime.fromisoformat(run.retry_not_before)
                 - datetime.fromisoformat(observed_at)).total_seconds()
    return remaining if remaining > 0 else None


def _load_cycle(
    database_path: str | Path,
    run_id: str,
    cycle_number: int | None = None,
) -> tuple[AutomatedRunRecord, int, tuple[BacktestSnapshot, ...]]:
    with open_database(database_path) as connection:
        run = get_automated_run(connection, run_id)
        if run is None:
            raise ValueError("automated_run_missing")
        if cycle_number is None:
            cycle_number = run.current_cycle + 1
        links = list_automated_run_backtests(connection, run.run_id)
        snapshots = _required_cycle_snapshots(
            connection,
            run,
            links,
            cycle_number=cycle_number,
        )
    return run, cycle_number, snapshots


def _load_run_active_snapshots(database_path, run) -> tuple[BacktestSnapshot, ...]:
    with open_database(database_path) as connection:
        snapshots = tuple(
            get_backtest_task(connection, link.task_id)
            for link in list_automated_run_backtests(connection, run.run_id)
        )
    if any(snapshot is None for snapshot in snapshots):
        raise ValueError("automated_run_backtest_task_missing")
    return tuple(
        snapshot
        for snapshot in snapshots
        if snapshot.task.status in {"created", "pending", "submission_unknown"}
    )


def _load_cycle_snapshots(
    database_path: str | Path,
    run: AutomatedRunRecord,
    *,
    cycle_number: int,
) -> tuple[BacktestSnapshot, ...]:
    with open_database(database_path) as connection:
        links = list_automated_run_backtests(connection, run.run_id)
        return _required_cycle_snapshots(
            connection,
            run,
            links,
            cycle_number=cycle_number,
        )


def _required_cycle_snapshots(
    connection: sqlite3.Connection,
    run: AutomatedRunRecord,
    links: tuple[AutomatedRunBacktestRecord, ...],
    *,
    cycle_number: int,
) -> tuple[BacktestSnapshot, ...]:
    prior_count = sum(link.cycle_number < cycle_number for link in links)
    remaining_backtests = remaining_automated_run_backtests(run, prior_count)
    expected = (
        run.backtest_count
        if remaining_backtests is None
        else min(run.backtest_count, remaining_backtests)
    )
    cycle_links = tuple(link for link in links if link.cycle_number == cycle_number)
    if (expected <= 0 or not 0 < len(cycle_links) <= expected
            or (not run.optimization_only and len(cycle_links) != expected)):
        raise ValueError("automated_cycle_backtest_batch_incomplete")
    snapshots: list[BacktestSnapshot] = []
    for link in cycle_links:
        snapshot = get_backtest_task(connection, link.task_id)
        if snapshot is None:
            raise ValueError("automated_run_backtest_task_missing")
        snapshots.append(snapshot)
    return tuple(snapshots)


def _counts(snapshots: tuple[BacktestSnapshot, ...]) -> CycleBacktestCounts:
    statuses = [snapshot.task.status for snapshot in snapshots]
    return CycleBacktestCounts(
        created=statuses.count("created"),
        submission_unknown=statuses.count("submission_unknown"),
        pending=statuses.count("pending"),
        completed=statuses.count("completed"),
        failed=statuses.count("failed"),
    )


def _oldest_observation(
    snapshots: tuple[BacktestSnapshot, ...],
) -> BacktestSnapshot:
    return sorted(
        snapshots,
        key=lambda item: (
            item.task.last_observed_at or "",
            item.task.task_id,
        ),
    )[0]


def _advance_result(
    *,
    run: AutomatedRunRecord,
    cycle_number: int,
    action: str,
    snapshot: BacktestSnapshot | None,
    counts: CycleBacktestCounts,
    platform_request_performed: bool,
    run_status: str | None = None,
    retry_after_seconds: float | None = None,
) -> AutomatedCycleBacktestAdvance:
    return AutomatedCycleBacktestAdvance(
        run_id=run.run_id,
        cycle_number=cycle_number,
        action=action,
        snapshot=snapshot,
        counts=counts,
        platform_request_performed=platform_request_performed,
        run_status=run_status or run.status,
        retry_after_seconds=retry_after_seconds,
    )
