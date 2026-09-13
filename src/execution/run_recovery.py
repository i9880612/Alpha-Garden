from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from execution.backtests import (
    record_pending_observation, remaining_backtest_retry_seconds,
    remaining_backtest_pending_seconds,
)
from execution.cycle_backtests import settle_failed_automated_cycle_if_terminal
from execution.real_backtests import RealBacktestAdvance, advance_real_backtest
from execution.seeds import synchronize_signal_seeds
from execution.qualified_archive import synchronize_qualified_alpha_archive
from execution.submission_checks import check_completed_backtest, remaining_submission_check_seconds
from persistence.backtests import get_backtest_task, list_active_backtest_tasks
from persistence.database import open_database
from persistence.runs import (
    get_automated_run, get_automated_run_backtest_by_task,
    list_failed_automated_runs, list_automated_run_backtests, get_automated_cycle_settlement,
)
from worldquant.client import WorldQuantClient, WorldQuantRequestError


def _stopped_work(connection, account_scope, observed_at, run_id=None):
    work = []
    for owner in list_failed_automated_runs(connection, account_scope=account_scope):
        if run_id is not None and owner.run_id != run_id:
            continue
        for link in list_automated_run_backtests(connection, owner.run_id):
            snapshot = get_backtest_task(connection, link.task_id)
            if snapshot.task.account_scope != account_scope:
                continue
            if snapshot.task.status == "pending":
                delay = min(
                    remaining_backtest_retry_seconds(snapshot, observed_at) or 0.0,
                    remaining_backtest_pending_seconds(
                        snapshot, observed_at, max_pending_seconds=owner.max_pending_seconds,
                    ),
                )
            elif get_automated_cycle_settlement(connection, owner.run_id, link.cycle_number) is None:
                delay = remaining_submission_check_seconds(connection, snapshot, observed_at)
            else:
                continue
            if delay is not None:
                work.append((snapshot, owner, delay))
    return sorted(work, key=lambda item: (
        datetime.fromisoformat(item[0].task.last_observed_at or item[0].task.created_at),
        item[0].task.task_id,
    ))


def remaining_stopped_run_seconds(database_path, *, account_scope, run_id, observed_at):
    with open_database(database_path) as connection:
        work = _stopped_work(connection, account_scope, observed_at, run_id)
    return min((item[2] for item in work), default=None)


def advance_stopped_run_check(
    database_path: str | Path, client: WorldQuantClient, *, account_scope: str,
    observed_at: str, run_id: str | None = None,
) -> RealBacktestAdvance | None:
    """Drain unfinished check work only at a batch boundary, never historical settlements."""
    with open_database(database_path) as connection:
        checks = [item for item in _stopped_work(connection, account_scope, observed_at, run_id)
                  if item[0].task.status == "completed"]
    if not checks:
        return None
    selected, owner, delay = min(checks, key=lambda item: item[2])
    if delay > 0:
        return RealBacktestAdvance("submission_check_wait", selected, False, delay)
    performed = check_completed_backtest(database_path, client, selected.task.task_id,
                                       observed_at=observed_at, recovered=True)
    while settle_failed_automated_cycle_if_terminal(database_path, owner.run_id, observed_at=observed_at) is not None:
        pass
    return RealBacktestAdvance("submission_check_observed", selected, performed)


def advance_stopped_run_backtest(
    database_path: str | Path, client: WorldQuantClient, *,
    account_scope: str, observed_at: str, run_id: str | None = None,
) -> RealBacktestAdvance | None:
    """Drain one old request under its original deadline, without resending it."""
    with open_database(database_path) as connection:
        ready = [item for item in _stopped_work(connection, account_scope, observed_at, run_id)
                 if item[0].task.status == "pending" and item[2] == 0]
        if not ready:
            return None
        selected, owner, _ = ready[0]
    try:
        result = advance_real_backtest(
            database_path, client, selected.task.task_id,
            observed_at=observed_at, allow_submission=False,
            max_pending_seconds=owner.max_pending_seconds,
        )
    except WorldQuantRequestError as exc:
        if exc.status_code in {401, 403} or exc.code.startswith("worldquant_authentication_"):
            raise
        # Defer a task-specific failed read within the original wait bound.
        # Account authentication and permission errors must remain visible.
        with open_database(database_path) as connection:
            deferred = record_pending_observation(
                connection, selected.task.task_id, observed_at=observed_at,
                retry_after_seconds=max(60.0, exc.retry_after_seconds or 0.0),
            )
        logging.getLogger(__name__).warning(
            "旧回测 %s 收取结果暂未成功（%s），保留名额，稍后重查。",
            selected.task.task_id, exc.code,
        )
        return RealBacktestAdvance("recovery_deferred", deferred, True)
    if result.snapshot.task.status == "pending":
        progressed = (result.snapshot.task.platform_alpha_id != selected.task.platform_alpha_id
                      or result.snapshot.result != selected.result)
        if progressed:
            return result
        with open_database(database_path) as connection:
            deferred = record_pending_observation(
                connection, selected.task.task_id, observed_at=observed_at,
                retry_after_seconds=max(60.0, result.retry_after_seconds or 0.0),
            )
        return replace(result, snapshot=deferred)
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        synchronize_signal_seeds(connection, candidate_task_ids=(selected.task.task_id,))
        synchronize_qualified_alpha_archive(connection, observed_at=observed_at)
    while settle_failed_automated_cycle_if_terminal(
        database_path, owner.run_id, observed_at=observed_at
    ) is not None:
        pass
    return result


def settle_stopped_run_results(
    database_path: str | Path, *, account_scope: str, observed_at: str
) -> None:
    """Recover completion interrupted between result persistence and settlement."""
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        owners = list_failed_automated_runs(connection, account_scope=account_scope)
        completed = tuple(
            link.task_id
            for owner in owners
            for link in list_automated_run_backtests(connection, owner.run_id)
            if get_backtest_task(connection, link.task_id).task.status == "completed"
        )
        if completed:
            synchronize_signal_seeds(connection, candidate_task_ids=completed)
        synchronize_qualified_alpha_archive(connection, observed_at=observed_at)
    for owner in owners:
        while settle_failed_automated_cycle_if_terminal(
            database_path, owner.run_id, observed_at=observed_at
        ) is not None:
            pass


def finish_stopped_run_backtests(
    database_path: str | Path, client: WorldQuantClient, *,
    account_scope: str, clock: Callable[[], datetime],
    waiter: Callable[[float], None], poll_interval_seconds: float,
) -> int:
    """Collect old requests before manual submission, without extending research.

    The collection phase is bounded and each task keeps its original deadline.
    Local timeout releases its slot without claiming a remote failure or resending.
    """
    started = clock()
    request_count = 0
    while True:
        observed = clock()
        with open_database(database_path) as connection:
            for task in list_active_backtest_tasks(connection):
                if task.account_scope != account_scope or task.status == "submission_unknown":
                    continue
                link = get_automated_run_backtest_by_task(connection, task.task_id)
                owner = get_automated_run(connection, link.run_id) if link else None
                if (task.status != "pending" or owner is None
                        or owner.account_scope != account_scope or owner.status != "failed"):
                    raise ValueError("formal_submission_active_backtest_exists")
            work = _stopped_work(connection, account_scope, observed.isoformat())
        if not work:
            return request_count
        remaining = max(item[1].max_pending_seconds for item in work) - (observed - started).total_seconds()
        if remaining <= 0:
            raise ValueError(
                "旧回测结果仍待定，本次收取等待已结束；请求和队列均保留，请稍后再次 submit。"
            )
        delays = [item[2] for item in work]
        if all(delay > 0 for delay in delays):
            waiter(min(60.0, remaining, min(delays)))
            continue
        if not client.authenticated:
            client.authenticate()
        advanced = advance_stopped_run_backtest(
            database_path, client, account_scope=account_scope,
            observed_at=observed.isoformat(),
        )
        if advanced is None:
            advanced = advance_stopped_run_check(
                database_path, client, account_scope=account_scope, observed_at=observed.isoformat(),
            )
        if advanced is not None:
            request_count += int(advanced.platform_request_performed)
        waiter(min(60.0, remaining, poll_interval_seconds))
