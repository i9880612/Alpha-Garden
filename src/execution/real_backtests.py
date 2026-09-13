from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from execution.backtests import (
    apply_backtest_detail,
    apply_backtest_poll_observation,
    apply_backtest_submission_observation,
    expire_pending_backtest_task,
    fail_backtest_task,
    prepare_backtest_task,
    record_backtest_yearly_stats,
    record_pending_observation,
    record_submission_not_accepted,
    record_submission_unknown,
    remaining_backtest_retry_seconds,
    remaining_backtest_pending_seconds,
)
from execution.submission_queue import synchronize_submission_queue
from execution.backtest_reconciliation import expire_unknown_backtest_submissions
from execution.runs import submission_rate_limited
from persistence.backtests import BacktestSnapshot, get_backtest_task, list_active_backtest_tasks
from persistence.database import DATABASE_LOCK_TIMEOUT_SECONDS, open_database
from persistence.runs import (
    AutomatedRunRecord,
    get_automated_run,
    get_automated_run_backtest_by_task,
    list_active_automated_runs,
)
from worldquant.backtests import (
    BacktestSettings,
    WorldQuantDetailMetricUnavailable,
    WorldQuantProtocolError,
    WorldQuantZeroCapitalResult,
    build_backtest_request,
)
from worldquant.client import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    WorldQuantClient,
    WorldQuantRequestError,
)


# Claim acquisition and commit can each wait before the POST starts.
_SUBMISSION_COMMIT_GRACE_SECONDS = 2 * DATABASE_LOCK_TIMEOUT_SECONDS
MAX_BACKTEST_RESPONSE_ATTEMPTS = 3
_TASK_RESPONSE_ERRORS = frozenset({
    "worldquant_poll_status_unknown",
    "worldquant_detail_formula_mismatch",
})


@dataclass(frozen=True, slots=True)
class RealBacktestAdvance:
    action: str
    snapshot: BacktestSnapshot
    platform_request_performed: bool
    retry_after_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class _RealBacktestContext:
    snapshot: BacktestSnapshot
    formula: str
    settings: BacktestSettings


@dataclass(frozen=True, slots=True)
class _SubmissionClaim:
    claimed: bool
    action: str
    snapshot: BacktestSnapshot


def prepare_real_backtest(
    database_path: str | Path,
    *,
    account_scope: str,
    formula: str,
    settings: BacktestSettings,
    created_at: str,
) -> BacktestSnapshot:
    with open_database(database_path) as connection:
        return prepare_backtest_task(
            connection,
            account_scope=account_scope,
            formula=formula,
            settings=settings.as_platform_dict(),
            created_at=created_at,
        )


def advance_real_backtest(
    database_path: str | Path,
    client: WorldQuantClient,
    task_id: str,
    *,
    observed_at: str,
    allow_submission: bool = False,
    max_pending_seconds: int | None = None,
) -> RealBacktestAdvance:
    # Only retry reads whose invalid responses belong to this task. Never retry
    # POSTs, transport errors, or arbitrary programming/data invariant errors.
    for attempt in range(1, MAX_BACKTEST_RESPONSE_ATTEMPTS + 1):
        try:
            advanced = _advance_real_backtest_once(
                database_path, client, task_id,
                observed_at=observed_at,
                allow_submission=allow_submission,
                max_pending_seconds=max_pending_seconds,
            )
            if max_pending_seconds is not None and advanced.action in {
                "pending", "checks_pending", "yearly_stats_pending",
            }:
                expired = expire_real_backtest_if_pending_timeout(
                    database_path, task_id, observed_at=observed_at,
                    max_pending_seconds=max_pending_seconds,
                )
                if expired is not None:
                    return RealBacktestAdvance("pending_timeout", expired, True)
            return advanced
        except WorldQuantRequestError as exc:
            if (max_pending_seconds is not None and not exc.outcome_unknown
                    and (exc.retryable or exc.status_code in {404, 410})
                    and exc.status_code not in {401, 403}
                    and not exc.code.startswith("worldquant_authentication_")):
                expired = expire_real_backtest_if_pending_timeout(
                    database_path, task_id, observed_at=observed_at,
                    max_pending_seconds=max_pending_seconds,
                )
                if expired is not None:
                    return RealBacktestAdvance("pending_timeout", expired, True)
            raise
        except WorldQuantProtocolError as exc:
            if exc.code not in _TASK_RESPONSE_ERRORS:
                raise
            if attempt < MAX_BACKTEST_RESPONSE_ATTEMPTS:
                continue
            with open_database(database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                snapshot = fail_backtest_task(
                    connection, task_id,
                    failure_code="platform_response_retry_exhausted",
                    failure_message=(
                        f"结果读取连续 {attempt} 次异常：{exc.code}；"
                        "本地任务失败，远端结果未确认"
                    ),
                    observed_at=observed_at,
                )
            return RealBacktestAdvance(
                action="failed", snapshot=snapshot,
                platform_request_performed=True,
            )
    raise AssertionError("backtest_response_attempts_invalid")


def _advance_real_backtest_once(
    database_path: str | Path,
    client: WorldQuantClient,
    task_id: str,
    *,
    observed_at: str,
    allow_submission: bool,
    max_pending_seconds: int | None,
) -> RealBacktestAdvance:
    if max_pending_seconds is not None and (
        isinstance(max_pending_seconds, bool)
        or not isinstance(max_pending_seconds, int)
        or max_pending_seconds <= 0
    ):
        raise ValueError("backtest_max_pending_seconds_invalid")
    context = _load_context(database_path, task_id)
    task = context.snapshot.task
    if task.status in {"completed", "failed"}:
        return RealBacktestAdvance(
            action=f"already_{task.status}",
            snapshot=context.snapshot,
            platform_request_performed=False,
        )
    if task.status == "submission_unknown":
        return RealBacktestAdvance(
            action=(
                "submission_in_progress"
                if _submission_may_still_be_in_progress(
                    context.snapshot,
                    client,
                    observed_at=observed_at,
                )
                else "reconciliation_required"
            ),
            snapshot=context.snapshot,
            platform_request_performed=False,
        )
    if task.status == "created":
        if allow_submission is not True:
            return RealBacktestAdvance(
                action="submission_confirmation_required",
                snapshot=context.snapshot,
                platform_request_performed=False,
            )
        return _submit_once(
            database_path,
            client,
            context,
            observed_at=observed_at,
        )
    if task.status != "pending" or task.remote_id is None:
        raise ValueError("real_backtest_state_invalid")
    retry_after_seconds = remaining_backtest_retry_seconds(
        context.snapshot,
        observed_at,
    )
    pending_remaining = _pending_remaining(
        context.snapshot,
        observed_at=observed_at,
        max_pending_seconds=max_pending_seconds,
    )
    if retry_after_seconds is not None:
        if pending_remaining == 0:
            expired = expire_real_backtest_if_pending_timeout(
                database_path, task_id, observed_at=observed_at,
                max_pending_seconds=max_pending_seconds,
            )
            if expired is not None:
                return RealBacktestAdvance("pending_timeout", expired, False)
        return RealBacktestAdvance(
            action="retry_wait",
            snapshot=context.snapshot,
            platform_request_performed=False,
            retry_after_seconds=(
                min(retry_after_seconds, pending_remaining)
                if pending_remaining is not None and pending_remaining > 0
                else retry_after_seconds
            ),
        )
    if task.platform_alpha_id is None:
        observation = client.poll_backtest(task.remote_id)
        try:
            with open_database(database_path) as connection:
                snapshot = apply_backtest_poll_observation(
                    connection,
                    task_id,
                    observation,
                    observed_at=observed_at,
                )
        except ValueError:
            timeout = _timeout_after_platform_request(database_path, task_id)
            if timeout is None:
                raise
            return timeout
        action = {
            "pending": "pending",
            "completed": "detail_ready",
            "failed": "failed",
        }[observation.state]
        return RealBacktestAdvance(
            action=action,
            snapshot=snapshot,
            platform_request_performed=True,
        )

    if context.snapshot.result is None:
        try:
            detail = client.fetch_backtest_detail(
                platform_alpha_id=task.platform_alpha_id,
                expected_formula=context.formula,
                expected_settings=context.settings.as_platform_dict(),
            )
        except WorldQuantZeroCapitalResult:
            try:
                with open_database(database_path) as connection:
                    snapshot = fail_backtest_task(
                        connection,
                        task_id,
                        failure_code="platform_zero_capital",
                        failure_message="平台回测未形成有效持仓，资本与主要指标均为0",
                        observed_at=observed_at,
                        platform_alpha_id=task.platform_alpha_id,
                    )
            except ValueError:
                timeout = _timeout_after_platform_request(database_path, task_id)
                if timeout is None:
                    raise
                return timeout
            return RealBacktestAdvance(
                action="failed",
                snapshot=snapshot,
                platform_request_performed=True,
            )
        except WorldQuantDetailMetricUnavailable as exc:
            try:
                with open_database(database_path) as connection:
                    snapshot = fail_backtest_task(
                        connection,
                        task_id,
                        failure_code="platform_detail_metric_unavailable",
                        failure_message=(
                            f"平台回测详情未提供必需指标：{exc.metric_name}"
                        ),
                        observed_at=observed_at,
                        platform_alpha_id=task.platform_alpha_id,
                    )
            except ValueError:
                timeout = _timeout_after_platform_request(database_path, task_id)
                if timeout is None:
                    raise
                return timeout
            return RealBacktestAdvance(
                action="failed",
                snapshot=snapshot,
                platform_request_performed=True,
            )
        if not detail.check_set_complete and not detail.has_failed_check:
            try:
                with open_database(database_path) as connection:
                    snapshot = record_pending_observation(
                        connection,
                        task_id,
                        observed_at=observed_at,
                        platform_alpha_id=task.platform_alpha_id,
                    )
            except ValueError:
                timeout = _timeout_after_platform_request(database_path, task_id)
                if timeout is None:
                    raise
                return timeout
            return RealBacktestAdvance(
                action="checks_pending",
                snapshot=snapshot,
                platform_request_performed=True,
            )
        try:
            with open_database(database_path) as connection:
                snapshot = apply_backtest_detail(
                    connection,
                    task_id,
                    detail,
                    observed_at=observed_at,
                )
        except ValueError:
            timeout = _timeout_after_platform_request(database_path, task_id)
            if timeout is None:
                raise
            return timeout
        return RealBacktestAdvance(
            action="detail_captured",
            snapshot=snapshot,
            platform_request_performed=True,
        )

    if context.snapshot.yearly_stats is not None:
        raise ValueError("real_backtest_pending_facts_conflict")
    yearly_observation = client.fetch_backtest_yearly_stats(
        platform_alpha_id=task.platform_alpha_id,
    )
    try:
        with open_database(database_path) as connection:
            if yearly_observation.state == "pending":
                snapshot = record_pending_observation(
                    connection,
                    task_id,
                    observed_at=observed_at,
                    platform_alpha_id=task.platform_alpha_id,
                    retry_after_seconds=yearly_observation.retry_after_seconds,
                )
                action = "yearly_stats_pending"
            elif yearly_observation.state == "ready":
                snapshot = record_backtest_yearly_stats(
                    connection,
                    task_id,
                    yearly_observation.stats,
                    observed_at=observed_at,
                )
                if get_automated_run_backtest_by_task(connection, task_id) is not None:
                    synchronize_submission_queue(
                        connection,
                        candidate_task_ids=(task_id,),
                        enqueued_at=observed_at,
                    )
                action = "completed"
            else:
                raise ValueError("worldquant_yearly_stats_observation_invalid")
    except ValueError:
        timeout = _timeout_after_platform_request(database_path, task_id)
        if timeout is None:
            raise
        return timeout
    return RealBacktestAdvance(
        action=action,
        snapshot=snapshot,
        platform_request_performed=True,
        retry_after_seconds=(
            min(yearly_observation.retry_after_seconds, pending_remaining)
            if yearly_observation.retry_after_seconds is not None
            and pending_remaining is not None
            else yearly_observation.retry_after_seconds
        ),
    )


def expire_real_backtest_if_pending_timeout(
    database_path: str | Path,
    task_id: str,
    *,
    observed_at: str,
    max_pending_seconds: int,
) -> BacktestSnapshot | None:
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        return expire_pending_backtest_task(
            connection,
            task_id,
            observed_at=observed_at,
            max_pending_seconds=max_pending_seconds,
        )


def _pending_remaining(
    snapshot: BacktestSnapshot,
    *,
    observed_at: str,
    max_pending_seconds: int | None,
) -> float | None:
    if max_pending_seconds is None:
        return None
    return remaining_backtest_pending_seconds(
        snapshot,
        observed_at,
        max_pending_seconds=max_pending_seconds,
    )


def _timeout_after_platform_request(
    database_path: str | Path,
    task_id: str,
) -> RealBacktestAdvance | None:
    with open_database(database_path) as connection:
        snapshot = get_backtest_task(connection, task_id)
    if (
        snapshot is None
        or snapshot.task.status != "failed"
        or snapshot.task.failure_code != "platform_pending_timeout"
    ):
        return None
    return RealBacktestAdvance(
        action="pending_timeout",
        snapshot=snapshot,
        platform_request_performed=True,
    )


def _submit_once(
    database_path: str | Path,
    client: WorldQuantClient,
    context: _RealBacktestContext,
    *,
    observed_at: str,
) -> RealBacktestAdvance:
    if not client.authenticated:
        raise RuntimeError("worldquant_authentication_required")
    build_backtest_request(context.formula, context.settings)
    request_timeout_seconds = _submission_request_timeout_seconds(client)
    if request_timeout_seconds > DEFAULT_REQUEST_TIMEOUT_SECONDS:
        raise ValueError("real_backtest_submission_timeout_unsupported")
    task_id = context.snapshot.task.task_id
    claim = _claim_backtest_submission(
        database_path,
        task_id=task_id,
        account_scope=context.snapshot.task.account_scope,
        observed_at=observed_at,
    )
    if not claim.claimed:
        return RealBacktestAdvance(
            action=claim.action,
            snapshot=claim.snapshot,
            platform_request_performed=False,
        )
    try:
        observation = client.submit_backtest(
            formula=context.formula,
            settings=context.settings,
        )
    except WorldQuantRequestError as exc:
        if not exc.outcome_unknown:
            with open_database(database_path) as connection:
                record_submission_not_accepted(connection, task_id)
        raise
    with open_database(database_path) as connection:
        snapshot = apply_backtest_submission_observation(
            connection,
            task_id,
            observation,
            observed_at=observed_at,
        )
    return RealBacktestAdvance(
        action=(
            "submitted"
            if snapshot.task.status == "pending"
            else "reconciliation_required"
        ),
        snapshot=snapshot,
        platform_request_performed=True,
    )


def _claim_backtest_submission(
    database_path: str | Path,
    *,
    task_id: str,
    account_scope: str,
    observed_at: str,
) -> _SubmissionClaim:
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        snapshot = get_backtest_task(connection, task_id)
        if snapshot is None:
            raise ValueError("backtest_task_missing")
        if snapshot.task.status != "created":
            return _claim_for_existing_state(snapshot)
        run = _submission_run(
            connection,
            task_id=task_id,
            account_scope=account_scope,
        )
        if run is not None:
            if (submission_rate_limited(run)
                    and datetime.fromisoformat(observed_at) < datetime.fromisoformat(run.retry_not_before)):
                return _SubmissionClaim(False, "submission_deferred", snapshot)
            expire_unknown_backtest_submissions(
                connection, account_scope=account_scope, observed_at=observed_at,
            )
        if run is not None and account_in_flight_backtest_count(connection, account_scope) >= (
            run.max_in_flight_backtests
        ):
            return _SubmissionClaim(
                claimed=False,
                action="submission_deferred",
                snapshot=snapshot,
            )
        claimed = record_submission_unknown(
            connection,
            task_id,
            observed_at=observed_at,
        )
        return _SubmissionClaim(
            claimed=True,
            action="claimed",
            snapshot=claimed,
        )


def _submission_run(
    connection: sqlite3.Connection,
    *,
    task_id: str,
    account_scope: str,
) -> AutomatedRunRecord | None:
    link = get_automated_run_backtest_by_task(connection, task_id)
    if link is not None:
        run = get_automated_run(connection, link.run_id)
        if run is None:
            raise ValueError("real_backtest_automated_run_missing")
        if (
            run.account_scope != account_scope
            or run.status != "running"
            or not run.real_backtests_authorized
        ):
            raise ValueError("real_backtest_automated_run_inactive")
        return run
    running = tuple(
        run
        for run in list_active_automated_runs(connection)
        if run.status == "running" and run.account_scope == account_scope
    )
    if not running:
        return None
    raise ValueError("real_backtest_automated_run_active")


def account_in_flight_backtest_count(
    connection: sqlite3.Connection,
    account_scope: str,
) -> int:
    return sum(
        task.account_scope == account_scope
        and task.status in {"submission_unknown", "pending"}
        for task in list_active_backtest_tasks(connection)
    )


def _claim_for_existing_state(snapshot: BacktestSnapshot) -> _SubmissionClaim:
    action = {
        "submission_unknown": "submission_in_progress",
        "pending": "pending",
        "completed": "already_completed",
        "failed": "already_failed",
    }.get(snapshot.task.status)
    if action is None:
        raise ValueError("real_backtest_state_invalid")
    return _SubmissionClaim(
        claimed=False,
        action=action,
        snapshot=snapshot,
    )


def _submission_may_still_be_in_progress(
    snapshot: BacktestSnapshot,
    client: WorldQuantClient,
    *,
    observed_at: str,
) -> bool:
    started_at = snapshot.task.submission_started_at
    if snapshot.task.status != "submission_unknown" or started_at is None:
        raise ValueError("real_backtest_submission_claim_invalid")
    started = _aware_timestamp(
        started_at,
        error="backtest_submission_started_at_invalid",
    )
    observed = _aware_timestamp(
        observed_at,
        error="backtest_observed_at_invalid",
    )
    if observed < started:
        raise ValueError("backtest_observed_before_submission")
    _submission_request_timeout_seconds(client)
    claim_window = DEFAULT_REQUEST_TIMEOUT_SECONDS + _SUBMISSION_COMMIT_GRACE_SECONDS
    return (observed - started).total_seconds() <= claim_window


def _submission_request_timeout_seconds(client: WorldQuantClient) -> float:
    timeout_seconds = getattr(
        client,
        "timeout_seconds",
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    return float(timeout_seconds)


def _aware_timestamp(value: str, *, error: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if parsed.utcoffset() is None:
        raise ValueError(error)
    return parsed


def _load_context(
    database_path: str | Path,
    task_id: str,
) -> _RealBacktestContext:
    with open_database(database_path) as connection:
        snapshot = get_backtest_task(connection, task_id)
        if snapshot is None:
            raise ValueError("backtest_task_missing")
    try:
        raw_settings = json.loads(snapshot.task.settings_json)
    except json.JSONDecodeError as exc:
        raise ValueError("backtest_settings_invalid") from exc
    return _RealBacktestContext(
        snapshot=snapshot,
        formula=snapshot.task.formula,
        settings=BacktestSettings.from_platform_dict(raw_settings),
    )
