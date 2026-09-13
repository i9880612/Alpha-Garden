from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from execution.backtests import fail_backtest_task
from persistence.backtests import (
    BacktestSnapshot,
    BacktestTaskRecord,
    get_backtest_task,
    list_backtest_tasks_by_platform_alpha_id,
    list_active_backtest_tasks,
)
from persistence.database import DATABASE_LOCK_TIMEOUT_SECONDS, open_database
from persistence.runs import (
    AutomatedRunRecord,
    get_automated_run,
    get_automated_run_backtest_by_task,
    list_automated_run_backtests,
)
from worldquant.alphas import UserAlphaRecord
from worldquant.backtests import (
    BacktestSettings,
    WorldQuantProtocolError,
    backtest_settings_match,
)
from worldquant.client import DEFAULT_REQUEST_TIMEOUT_SECONDS, WorldQuantClient


_ALPHA_PAGE_LIMIT = 100
_SUBMISSION_COMMIT_GRACE_SECONDS = 2 * DATABASE_LOCK_TIMEOUT_SECONDS
_ALPHA_VISIBILITY_GRACE_SECONDS = 5.0
_REMOTE_ACCEPTED_FAILURE_CODE = "remote_accepted_without_simulation_id"
_REMOTE_ACCEPTED_FAILURE_MESSAGE = (
    "平台 Alpha 列表确认回测已被接收，但未提供可继续轮询的 simulation 编号"
)
SUBMISSION_OUTCOME_TIMEOUT = "submission_outcome_timeout"


def expire_unknown_backtest_submissions(
    connection: sqlite3.Connection,
    *,
    account_scope: str,
    observed_at: str,
) -> tuple[BacktestSnapshot, ...]:
    """End local waiting at the owner's deadline without claiming remote failure.

    The caller owns the write transaction. Failed task identities remain consumed,
    so releasing an expired slot never permits resending the original request.
    """
    observed = _timestamp(observed_at, "backtest_reconciliation_observed_at_invalid")
    expired = []
    for task in list_active_backtest_tasks(connection):
        if task.account_scope != account_scope or task.status != "submission_unknown":
            continue
        link = get_automated_run_backtest_by_task(connection, task.task_id)
        if link is None:
            continue
        owner = get_automated_run(connection, link.run_id)
        if owner is None or owner.account_scope != account_scope:
            raise ValueError("backtest_reconciliation_owner_invalid")
        started = _submission_started_at(task, observed)
        elapsed = (observed - started).total_seconds()
        if (
            elapsed < owner.max_pending_seconds
            or elapsed <= DEFAULT_REQUEST_TIMEOUT_SECONDS + _SUBMISSION_COMMIT_GRACE_SECONDS
        ):
            continue
        expired.append(fail_backtest_task(
            connection, task.task_id,
            failure_code=SUBMISSION_OUTCOME_TIMEOUT,
            failure_message=(
                f"请求接收状态超过原运行的 {owner.max_pending_seconds} 秒等待上限；"
                "本地任务超时，平台是否接受仍未知，释放名额但不重发原请求"
            ),
            observed_at=observed_at,
        ))
    return tuple(expired)


@dataclass(frozen=True, slots=True)
class BacktestReconciliation:
    action: str
    platform_request_count: int
    reconciled_task_count: int
    unresolved_task_count: int


def reconcile_account_backtest_capacity(
    database_path: str | Path,
    client: WorldQuantClient,
    *,
    account_scope: str,
    observed_at: str,
) -> BacktestReconciliation:
    """Reconcile unknown requests without resuming owners or resending requests."""
    with open_database(database_path) as connection:
        owners = set()
        for task in list_active_backtest_tasks(connection):
            if task.account_scope != account_scope or task.status != "submission_unknown":
                continue
            link = get_automated_run_backtest_by_task(connection, task.task_id)
            if link is None:
                continue
            run = get_automated_run(connection, link.run_id)
            if run is not None and run.status in {"failed", "running"}:
                owners.add(run.run_id)
    results = tuple(
        reconcile_run_backtest_submissions(database_path, client, run_id, observed_at=observed_at)
        for run_id in sorted(owners)
    )
    return BacktestReconciliation(
        action="capacity_checked",
        platform_request_count=sum(item.platform_request_count for item in results),
        reconciled_task_count=sum(item.reconciled_task_count for item in results),
        unresolved_task_count=sum(item.unresolved_task_count for item in results),
    )


def reconcile_run_backtest_submissions(
    database_path: str | Path,
    client: WorldQuantClient,
    run_id: str,
    *,
    observed_at: str,
) -> BacktestReconciliation:
    observed = _timestamp(observed_at, "backtest_reconciliation_observed_at_invalid")
    run, unknown_tasks = _load_reconciliation_context(database_path, run_id)
    if not unknown_tasks:
        return BacktestReconciliation(
            action="not_required",
            platform_request_count=0,
            reconciled_task_count=0,
            unresolved_task_count=0,
        )
    if run.status == "failed" and run.finished_at is None:
        raise ValueError("backtest_reconciliation_run_finished_at_missing")
    finished = _timestamp(
        run.finished_at if run.status == "failed" else observed_at,
        "backtest_reconciliation_run_finished_at_invalid",
    )
    if observed < finished:
        raise ValueError("backtest_reconciliation_observed_before_run_finished")
    starts = {task.task_id: _submission_started_at(task, finished) for task in unknown_tasks}
    timeout_seconds = _submission_timeout_seconds(client)
    ends = {
        task.task_id: _submission_window_end(
            starts[task.task_id],
            finished_at=finished,
            observed_at=observed,
            timeout_seconds=timeout_seconds,
        )
        for task in unknown_tasks
    }
    # Platform visibility settling is independent of local database contention.
    settle_seconds = timeout_seconds + _ALPHA_VISIBILITY_GRACE_SECONDS
    if observed < max(ends.values()) + timedelta(seconds=settle_seconds):
        return BacktestReconciliation(
            action="unresolved",
            platform_request_count=0,
            reconciled_task_count=0,
            unresolved_task_count=len(unknown_tasks),
        )
    records, request_count = _discover_window_alphas(
        client,
        earliest_started=min(starts.values()),
    )

    candidates: dict[str, tuple[UserAlphaRecord, ...]] = {}
    for task in unknown_tasks:
        candidates[task.task_id] = tuple(
            record
            for record in records
            if starts[task.task_id] < record.created_at < ends[task.task_id]
            and _matches_task(record, task)
        )
    assignment_counts: dict[str, int] = {}
    for task_matches in candidates.values():
        for match in task_matches:
            assignment_counts[match.platform_alpha_id] = (
                assignment_counts.get(match.platform_alpha_id, 0) + 1
            )
    matches = {
        task_id: task_matches[0]
        for task_id, task_matches in candidates.items()
        if len(task_matches) == 1
        and assignment_counts[task_matches[0].platform_alpha_id] == 1
    }

    if matches:
        _record_reconciled_tasks(
            database_path,
            run,
            unknown_tasks,
            matches,
            observed_at=observed_at,
        )
    unresolved_count = len(unknown_tasks) - len(matches)
    return BacktestReconciliation(
        action="confirmed" if unresolved_count == 0 else "unresolved",
        platform_request_count=request_count,
        reconciled_task_count=len(matches),
        unresolved_task_count=unresolved_count,
    )


def _load_reconciliation_context(
    database_path: str | Path,
    run_id: str,
) -> tuple[AutomatedRunRecord, tuple[BacktestTaskRecord, ...]]:
    with open_database(database_path) as connection:
        run = get_automated_run(connection, run_id)
        if run is None:
            raise ValueError("automated_run_missing")
        if run.status not in {"failed", "running"}:
            raise ValueError("backtest_reconciliation_run_invalid")
        tasks: list[BacktestTaskRecord] = []
        for link in list_automated_run_backtests(connection, run_id):
            snapshot = get_backtest_task(connection, link.task_id)
            if snapshot is None:
                raise ValueError("automated_run_backtest_task_missing")
            if snapshot.task.status == "submission_unknown":
                tasks.append(snapshot.task)
        return run, tuple(tasks)


def _discover_window_alphas(
    client: WorldQuantClient,
    *,
    earliest_started: datetime,
) -> tuple[tuple[UserAlphaRecord, ...], int]:
    first, first_request_count = _scan_window_alphas(
        client,
        earliest_started=earliest_started,
    )
    second, second_request_count = _scan_window_alphas(
        client,
        earliest_started=earliest_started,
    )
    if first != second:
        raise WorldQuantProtocolError("worldquant_user_alphas_snapshot_changed")
    return second, first_request_count + second_request_count


def _scan_window_alphas(
    client: WorldQuantClient,
    *,
    earliest_started: datetime,
) -> tuple[tuple[UserAlphaRecord, ...], int]:
    records: list[UserAlphaRecord] = []
    identities: set[str] = set()
    request_count = 0
    for hidden in (False, True):
        offset = 0
        expected_count: int | None = None
        previous_created_at: datetime | None = None
        while True:
            page = client.fetch_user_alpha_page(
                limit=_ALPHA_PAGE_LIMIT,
                offset=offset,
                hidden=hidden,
            )
            request_count += 1
            if expected_count is None:
                expected_count = page.total_count
            elif page.total_count != expected_count:
                raise WorldQuantProtocolError(
                    "worldquant_user_alphas_count_changed"
                )
            if offset + len(page.records) > page.total_count:
                raise WorldQuantProtocolError(
                    "worldquant_user_alphas_pagination_invalid"
                )
            for record in page.records:
                if (
                    previous_created_at is not None
                    and record.created_at > previous_created_at
                ):
                    raise WorldQuantProtocolError(
                        "worldquant_user_alphas_order_invalid"
                    )
                previous_created_at = record.created_at
                if record.platform_alpha_id in identities:
                    raise WorldQuantProtocolError(
                        "worldquant_user_alpha_duplicate"
                    )
                identities.add(record.platform_alpha_id)
                if record.created_at > earliest_started:
                    records.append(record)

            consumed = offset + len(page.records)
            if consumed == page.total_count:
                if page.has_next:
                    raise WorldQuantProtocolError(
                        "worldquant_user_alphas_pagination_invalid"
                    )
                break
            if not page.records or not page.has_next:
                raise WorldQuantProtocolError(
                    "worldquant_user_alphas_pagination_invalid"
                )
            if page.records[-1].created_at <= earliest_started:
                break
            offset = consumed
    return tuple(records), request_count


def _record_reconciled_tasks(
    database_path: str | Path,
    expected_run: AutomatedRunRecord,
    expected_tasks: tuple[BacktestTaskRecord, ...],
    matches: dict[str, UserAlphaRecord],
    *,
    observed_at: str,
) -> None:
    expected_by_id = {task.task_id: task for task in expected_tasks}
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current_run = get_automated_run(connection, expected_run.run_id)
        if current_run != expected_run:
            raise ValueError("backtest_reconciliation_run_changed")
        for task_id, record in matches.items():
            snapshot = get_backtest_task(connection, task_id)
            if snapshot is None or snapshot.task != expected_by_id[task_id]:
                raise ValueError("backtest_reconciliation_task_changed")
            conflicts = list_backtest_tasks_by_platform_alpha_id(
                connection,
                account_scope=expected_run.account_scope,
                platform_alpha_id=record.platform_alpha_id,
            )
            if any(conflict.task_id != task_id for conflict in conflicts):
                raise ValueError("backtest_reconciliation_alpha_identity_conflict")
            fail_backtest_task(
                connection,
                task_id,
                failure_code=_REMOTE_ACCEPTED_FAILURE_CODE,
                failure_message=_REMOTE_ACCEPTED_FAILURE_MESSAGE,
                observed_at=observed_at,
                platform_alpha_id=record.platform_alpha_id,
            )


def _matches_task(record: UserAlphaRecord, task: BacktestTaskRecord) -> bool:
    if (
        record.alpha_type != "REGULAR"
        or record.formula != task.formula
        or record.settings is None
    ):
        return False
    try:
        expected = json.loads(task.settings_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("backtest_reconciliation_settings_invalid") from exc
    if not isinstance(expected, dict) or not expected:
        raise ValueError("backtest_reconciliation_settings_invalid")
    try:
        BacktestSettings.from_platform_dict(expected)
    except (TypeError, ValueError) as exc:
        raise ValueError("backtest_reconciliation_settings_invalid") from exc
    return backtest_settings_match(record.settings, expected)


def _submission_started_at(
    task: BacktestTaskRecord,
    finished_at: datetime,
) -> datetime:
    if task.submission_started_at is None:
        raise ValueError("backtest_reconciliation_submission_started_at_missing")
    started = _timestamp(
        task.submission_started_at,
        "backtest_reconciliation_submission_started_at_invalid",
    )
    if started > finished_at:
        raise ValueError("backtest_reconciliation_time_window_invalid")
    return started


def _submission_window_end(
    started_at: datetime,
    *,
    finished_at: datetime,
    observed_at: datetime,
    timeout_seconds: float,
) -> datetime:
    request_deadline = started_at + timedelta(
        seconds=timeout_seconds + _SUBMISSION_COMMIT_GRACE_SECONDS
    )
    if finished_at == started_at:
        return min(request_deadline, observed_at)
    return min(request_deadline, finished_at, observed_at)


def _submission_timeout_seconds(client: WorldQuantClient) -> float:
    value = getattr(client, "timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    return float(value)


def _timestamp(value: str, error: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if parsed.utcoffset() is None:
        raise ValueError(error)
    return parsed
