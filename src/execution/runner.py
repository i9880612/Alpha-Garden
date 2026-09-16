from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from execution.cycle_backtests import (
    remaining_automated_pending_seconds,
    settle_failed_automated_cycle_if_terminal,
)
from execution.backtest_reconciliation import (
    expire_unknown_backtest_submissions,
    reconcile_run_backtest_submissions,
    reconcile_account_backtest_capacity,
)
from execution.driver import advance_automated_run
from execution.progress import RunProgress
from execution.run_recovery import advance_stopped_run_backtest, advance_stopped_run_check, remaining_stopped_run_seconds
from execution.submission_checks import advance_deferred_submission_check
from execution.request_failures import (
    handle_automated_request_failure,
    remaining_automated_request_retry_seconds,
)
from execution.runs import (
    AutomatedRunPaused,
    automated_run_has_hard_stop_evidence,
    can_resume_independent_backtests,
    clear_automated_request_failures,
    submission_rate_limited,
)
from execution.runs import resume_automated_run_after_submission_reconciliation
from execution.submissions import reconcile_failed_run_formal_submission
from persistence.database import open_database
from persistence.runs import (
    AutomatedRunRecord,
    automated_run_has_submission_unknown,
    get_automated_run,
)
from persistence.submissions import automated_run_has_unresolved_formal_submission
from persistence.submissions import (
    FORMAL_SUBMISSION_UNRESOLVED_STATUSES,
    list_formal_submission_attempts,
)
from worldquant.client import WorldQuantClient, WorldQuantRequestError


Clock = Callable[[], datetime]
Waiter = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class AutomatedRunCompletion:
    run: AutomatedRunRecord
    step_count: int
    platform_request_count: int
    authentication_performed: bool
    formal_submission_claimed_count: int
    formal_submission_confirmed_count: int
    formal_submission_unresolved_count: int


def run_automated_run(
    database_path: str | Path,
    client: WorldQuantClient,
    run_id: str,
    *,
    poll_interval_seconds: float = 1.0,
    clock: Clock | None = None,
    waiter: Waiter | None = None,
) -> AutomatedRunCompletion:
    validate_automated_run_poll_interval(poll_interval_seconds)
    current_time = clock or _utc_now
    wait = waiter or time.sleep

    initial_run = _load_run(database_path, run_id)
    if initial_run.stop_reason == "user_paused":
        raise AutomatedRunPaused()
    progress = RunProgress(database_path, run_id)
    if initial_run.status == "completed":
        return _completion(
            database_path,
            initial_run,
            step_count=0,
            platform_request_count=0,
            authentication_performed=False,
        )
    if not initial_run.real_backtests_authorized:
        raise ValueError("automated_run_backtests_not_authorized")

    authentication_performed = False
    step_count = 0
    platform_request_count = 0
    allow_independent_resume = initial_run.status == "failed"
    while True:
        observed = _aware_time(current_time())
        run = _load_run(database_path, run_id)
        if run.status == "completed":
            return _completion(
                database_path,
                run,
                step_count=step_count,
                platform_request_count=platform_request_count,
                authentication_performed=authentication_performed,
            )
        with open_database(database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            expired_unknown = expire_unknown_backtest_submissions(
                connection, account_scope=run.account_scope,
                observed_at=observed.isoformat(),
            )
        for expired in expired_unknown:
            progress.backtest(expired.task.task_id, snapshot=expired)
        if run.status == "failed":
            with open_database(database_path) as connection:
                formal_reconciliation_required = bool(
                    run.automatic_submissions_enabled
                    and automated_run_has_unresolved_formal_submission(
                        connection,
                        run.run_id,
                    )
                )
                backtest_reconciliation_required = (
                    (
                        run.stop_reason == "submission_reconciliation_required"
                        or (initial_run.status == "failed" and step_count == 0)
                    )
                    and automated_run_has_submission_unknown(connection, run.run_id)
                )
            if formal_reconciliation_required:
                if not client.authenticated:
                    client.authenticate()
                    authentication_performed = True
                reconciliation = reconcile_failed_run_formal_submission(
                    database_path,
                    client,
                    run.run_id,
                    observed_at=observed.isoformat(),
                )
                if reconciliation is None:
                    raise ValueError("formal_submission_reconciliation_missing")
                step_count += 1
                if reconciliation.platform_request_performed:
                    platform_request_count += 1
                if reconciliation.action == "formal_submission_confirmed":
                    continue
                return _completion(
                    database_path,
                    _load_run(database_path, run.run_id),
                    step_count=step_count,
                    platform_request_count=platform_request_count,
                    authentication_performed=authentication_performed,
                )
            if backtest_reconciliation_required:
                with open_database(database_path) as connection:
                    independent_work = can_resume_independent_backtests(connection, run)
                if allow_independent_resume and independent_work:
                    allow_independent_resume = False
                    resume_automated_run_after_submission_reconciliation(
                        database_path, run.run_id
                    )
                    step_count += 1
                    continue
                if not client.authenticated:
                    client.authenticate()
                    authentication_performed = True
                reconciliation = reconcile_run_backtest_submissions(
                    database_path,
                    client,
                    run.run_id,
                    observed_at=observed.isoformat(),
                )
                step_count += 1
                platform_request_count += reconciliation.platform_request_count
                if reconciliation.action in {"confirmed", "not_required"}:
                    continue
                return _completion(
                    database_path,
                    _load_run(database_path, run.run_id),
                    step_count=step_count,
                    platform_request_count=platform_request_count,
                    authentication_performed=authentication_performed,
                )
            if (
                run.stop_reason == "submission_reconciliation_required"
            ):
                with open_database(database_path) as connection:
                    hard_stop = automated_run_has_hard_stop_evidence(connection, run)
                if not hard_stop:
                    resume_automated_run_after_submission_reconciliation(
                        database_path,
                        run.run_id,
                    )
                    step_count += 1
                    continue
            if initial_run.status != "failed":
                return _completion(database_path, run, step_count=step_count,
                                   platform_request_count=platform_request_count,
                                   authentication_performed=authentication_performed)
            pending_seconds = remaining_stopped_run_seconds(
                database_path,
                account_scope=run.account_scope, run_id=run.run_id,
                observed_at=observed.isoformat(),
            )
            if pending_seconds is None:
                settlement = settle_failed_automated_cycle_if_terminal(
                    database_path,
                    run.run_id,
                    observed_at=observed.isoformat(),
                )
                if settlement is not None:
                    step_count += 1
                    continue
                return _completion(
                    database_path,
                    run,
                    step_count=step_count,
                    platform_request_count=platform_request_count,
                    authentication_performed=authentication_performed,
                )
            if pending_seconds > 0:
                wait(min(60.0, pending_seconds))
                continue
            if not client.authenticated:
                client.authenticate()
                authentication_performed = True
            recovered = advance_stopped_run_check(
                database_path, client, account_scope=run.account_scope, run_id=run.run_id,
                observed_at=observed.isoformat(),
            ) or advance_stopped_run_backtest(
                database_path, client, account_scope=run.account_scope, run_id=run.run_id,
                observed_at=observed.isoformat(),
            )
            if recovered is not None:
                progress.backtest(recovered.snapshot.task.task_id, action=recovered.action,
                                  snapshot=recovered.snapshot)
                step_count += 1
                platform_request_count += int(recovered.platform_request_performed)
                if recovered.action == "recovery_deferred":
                    return _completion(database_path, run, step_count=step_count,
                                       platform_request_count=platform_request_count,
                                       authentication_performed=authentication_performed)
            wait(poll_interval_seconds)
            continue
        retry_after = remaining_automated_request_retry_seconds(run, observed)
        if retry_after is not None and not submission_rate_limited(run):
            progress.notice(f"请求退避等待中，剩余 {retry_after:g} 秒；不会提前重发。")
            wait(
                _bounded_wait(
                    database_path,
                    run.run_id,
                    observed_at=observed.isoformat(),
                    requested_seconds=retry_after,
                )
            )
            continue

        if run.status == "running" and not client.authenticated:
            progress.notice("阶段[0] 平台认证中...")
            try:
                client.authenticate()
            except WorldQuantRequestError as exc:
                failure = handle_automated_request_failure(
                    database_path,
                    run,
                    exc,
                    observed_at=observed.isoformat(),
                    non_retryable_reason="platform_authentication_not_retryable",
                )
                progress.notice(f"阶段[0] 平台认证暂未成功（{exc.code}），按原故障边界处理")
                if failure.run.status == "failed":
                    continue
                assert failure.retry_after_seconds is not None
                _wait_until_next_step(
                    database_path,
                    run.run_id,
                    current_time=current_time,
                    wait=wait,
                    requested_seconds=failure.retry_after_seconds,
                )
                continue
            authentication_performed = True
            progress.notice("阶段[0] 平台认证完成")
            if not submission_rate_limited(run):
                clear_automated_request_failures(database_path, run.run_id)
            continue

        try:
            recovered = advance_stopped_run_backtest(
                database_path, client, account_scope=run.account_scope,
                observed_at=observed.isoformat(),
            ) if run.status == "running" else None
            observed = _aware_time(current_time())
            if run.status == "running" and advance_deferred_submission_check(
                database_path, client, account_scope=run.account_scope, observed_at=observed.isoformat(),
            ):
                step_count += 1
                platform_request_count += 1
        except WorldQuantRequestError as exc:
            handle_automated_request_failure(
                database_path, run, exc, observed_at=observed.isoformat(),
                non_retryable_reason="platform_request_not_retryable",
            )
            continue
        if recovered is not None:
            progress.backtest(recovered.snapshot.task.task_id, action=recovered.action,
                              snapshot=recovered.snapshot)
            step_count += 1
            platform_request_count += int(recovered.platform_request_performed)
        observed = _aware_time(current_time())
        advance = advance_automated_run(
            database_path,
            client,
            run.run_id,
            observed_at=observed.isoformat(),
            checkpoint=current_time,
        )
        step_count += 1
        progress.advanced(advance)
        if advance.platform_request_performed:
            platform_request_count += 1
        if advance.run.status == "completed":
            return _completion(
                database_path,
                advance.run,
                step_count=step_count,
                platform_request_count=platform_request_count,
                authentication_performed=authentication_performed,
            )
        if advance.run.status == "failed":
            continue
        if advance.backtest_action in {"capacity_wait", "reconciliation_required"}:
            observed = _aware_time(current_time())
            try:
                reconciliation = reconcile_account_backtest_capacity(
                    database_path, client, account_scope=run.account_scope,
                    observed_at=observed.isoformat(),
                )
            except WorldQuantRequestError as exc:
                failure = handle_automated_request_failure(
                    database_path, run, exc, observed_at=observed.isoformat(),
                    non_retryable_reason="platform_request_not_retryable",
                )
                if failure.run.status == "failed":
                    continue
            else:
                platform_request_count += reconciliation.platform_request_count
                if reconciliation.platform_request_count:
                    clear_automated_request_failures(database_path, run.run_id)
                if reconciliation.reconciled_task_count:
                    continue
        if advance.retry_after_seconds is not None:
            _wait_until_next_step(
                database_path,
                run.run_id,
                current_time=current_time,
                wait=wait,
                requested_seconds=advance.retry_after_seconds,
            )
        elif (
            advance.platform_request_performed
            or advance.backtest_action == "submission_in_progress"
        ):
            _wait_until_next_step(
                database_path,
                run.run_id,
                current_time=current_time,
                wait=wait,
                requested_seconds=poll_interval_seconds,
            )


def _wait_until_next_step(
    database_path: str | Path,
    run_id: str,
    *,
    current_time: Clock,
    wait: Waiter,
    requested_seconds: float,
) -> None:
    observed = _aware_time(current_time())
    delay = _bounded_wait(
        database_path,
        run_id,
        observed_at=observed.isoformat(),
        requested_seconds=requested_seconds,
    )
    if delay > 0:
        wait(delay)


def _bounded_wait(
    database_path: str | Path,
    run_id: str,
    *,
    observed_at: str,
    requested_seconds: float,
) -> float:
    pending_seconds = remaining_automated_pending_seconds(
        database_path,
        run_id,
        observed_at=observed_at,
    )
    return (
        min(requested_seconds, pending_seconds)
        if pending_seconds is not None and pending_seconds > 0
        else requested_seconds
    )


def _load_run(database_path: str | Path, run_id: str) -> AutomatedRunRecord:
    with open_database(database_path) as connection:
        run = get_automated_run(connection, run_id)
    if run is None:
        raise ValueError("automated_run_missing")
    return run


def _completion(
    database_path: str | Path,
    run: AutomatedRunRecord,
    *,
    step_count: int,
    platform_request_count: int,
    authentication_performed: bool,
) -> AutomatedRunCompletion:
    with open_database(database_path) as connection:
        attempts = tuple(
            attempt
            for attempt in list_formal_submission_attempts(
                connection,
                run_id=run.run_id,
            )
            if attempt.submission_mode == "automatic"
        )
    return AutomatedRunCompletion(
        run=run,
        step_count=step_count,
        platform_request_count=platform_request_count,
        authentication_performed=authentication_performed,
        formal_submission_claimed_count=sum(
            attempt.submission_claimed_at is not None for attempt in attempts
        ),
        formal_submission_confirmed_count=sum(
            attempt.status == "submitted" for attempt in attempts
        ),
        formal_submission_unresolved_count=sum(
            attempt.status in FORMAL_SUBMISSION_UNRESOLVED_STATUSES
            for attempt in attempts
        ),
    )


def validate_automated_run_poll_interval(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("automated_run_poll_interval_invalid")


def _aware_time(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("automated_run_clock_invalid")
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)
