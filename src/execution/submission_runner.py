from __future__ import annotations

import math
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from execution.submissions import (
    advance_submission_queue,
    require_standalone_submission_scope,
)
from execution.process_lock import exclusive_run_process
from execution.submission_progress import SubmissionProgress
from execution.run_recovery import finish_stopped_run_backtests, settle_stopped_run_results
from execution.runs import retire_previous_automated_runs
from persistence.backtests import get_backtest_task, list_active_backtest_tasks
from persistence.database import open_database
from persistence.runs import get_automated_run, list_active_automated_runs
from execution.submission_queue import list_submission_candidates, require_selected_submission
from persistence.submissions import (
    FORMAL_SUBMISSION_ACTIVE_STATUSES,
    FORMAL_SUBMISSION_UNRESOLVED_STATUSES,
    FormalSubmissionAttemptRecord,
    list_formal_submission_attempts,
    replace_formal_submission_attempt,
)
from worldquant.client import WorldQuantClient, WorldQuantRequestError
from worldquant.config import (
    WorldQuantConnectionSettings,
    load_worldquant_connection_settings,
)


Clock = Callable[[], datetime]
Waiter = Callable[[float], None]
ClientFactory = Callable[[WorldQuantConnectionSettings], WorldQuantClient]


@dataclass(frozen=True, slots=True)
class SubmissionQueueCompletion:
    account_scope: str
    initial_queue_count: int
    step_count: int
    platform_request_count: int
    authentication_performed: bool
    submission_claimed_count: int
    submitted_count: int
    already_active_count: int
    ineligible_count: int
    failed_count: int
    unresolved_count: int
    remaining_queue_count: int
    max_submissions: int | None = None

    @property
    def submission_limit_reached(self) -> bool:
        return (
            self.max_submissions is not None
            and self.submitted_count - self.already_active_count >= self.max_submissions
        )

    @property
    def completed(self) -> bool:
        return self.unresolved_count == 0 and (
            self.remaining_queue_count == 0 or self.submission_limit_reached
        )


def submit_queued_alphas(
    database_path: str | Path,
    environment_path: str | Path,
    *,
    max_submissions: int | None = None,
    poll_interval_seconds: float = 1.0,
    clock: Clock | None = None,
    waiter: Waiter | None = None,
    client_factory: ClientFactory | None = None,
    source: str = "queue",
    grade: str | None = None,
    selected_task_id: str | None = None,
) -> SubmissionQueueCompletion:
    _validate_max_submissions(max_submissions)
    _validate_selection(source, selected_task_id, max_submissions, grade)
    _validate_poll_interval(poll_interval_seconds)
    current_time = clock or _utc_now
    def checked_clock() -> datetime:
        return _aware_time(current_time())
    wait = waiter or time.sleep
    settings = load_worldquant_connection_settings(environment_path)
    make_client = client_factory or _default_client_factory
    client = make_client(settings)
    with exclusive_run_process(database_path):
        observed = checked_clock()
        with open_database(database_path) as connection:
            if selected_task_id is not None:
                require_selected_submission(connection, account_scope=settings.account_scope, task_id=selected_task_id, source=source)
            queued_task_ids = frozenset(
                item.task_id for item in list_submission_candidates(
                    connection, account_scope=settings.account_scope, source=source, grade=grade,
                )
                if selected_task_id is None or item.task_id == selected_task_id
            )
            needs_recovery = bool(list_active_automated_runs(connection)) or any(
                task.account_scope == settings.account_scope
                and task.status in {"created", "pending"}
                for task in list_active_backtest_tasks(connection)
            )
        initially_authenticated = client.authenticated
        recovered_requests = 0
        if needs_recovery:
            retire_previous_automated_runs(
                database_path, account_scope=settings.account_scope,
                observed_at=observed.isoformat(), reason="stopped_for_manual_submission",
            )
            logging.getLogger("execution.progress").info(
                "正在收尾已中断的研究批次：取消未发送任务，收取已发送回测；本次只提交已选定的公式。"
            )
            settle_stopped_run_results(
                database_path, account_scope=settings.account_scope,
                observed_at=observed.isoformat(),
            )
            recovered_requests = finish_stopped_run_backtests(
                database_path, client, account_scope=settings.account_scope,
                clock=checked_clock, waiter=wait,
                poll_interval_seconds=poll_interval_seconds,
            )
        completion = run_submission_queue(
            database_path,
            client,
            account_scope=settings.account_scope,
            max_submissions=max_submissions,
            poll_interval_seconds=poll_interval_seconds,
            clock=checked_clock,
            waiter=wait,
            queued_task_ids=queued_task_ids,
            source=source,
            grade=grade,
            selected_task_id=selected_task_id,
        )
        return replace(
            completion,
            platform_request_count=completion.platform_request_count + recovered_requests,
            authentication_performed=(
                completion.authentication_performed
                or (not initially_authenticated and client.authenticated)
            ),
        )


def run_submission_queue(
    database_path: str | Path,
    client: WorldQuantClient,
    *,
    account_scope: str,
    max_submissions: int | None = None,
    poll_interval_seconds: float = 1.0,
    clock: Clock | None = None,
    waiter: Waiter | None = None,
    queued_task_ids: frozenset[str] | None = None,
    source: str = "queue",
    grade: str | None = None,
    selected_task_id: str | None = None,
) -> SubmissionQueueCompletion:
    _validate_max_submissions(max_submissions)
    _validate_selection(source, selected_task_id, max_submissions, grade)
    _validate_poll_interval(poll_interval_seconds)
    current_time = clock or _utc_now
    wait = waiter or time.sleep
    with open_database(database_path) as connection:
        require_standalone_submission_scope(connection, account_scope)
        if selected_task_id is not None:
            require_selected_submission(connection, account_scope=account_scope, task_id=selected_task_id, source=source)
        queue = list_submission_candidates(
            connection,
            account_scope=account_scope,
            source=source,
            grade=grade,
        )
        if queued_task_ids is not None:
            queue = tuple(item for item in queue if item.task_id in queued_task_ids)
        if selected_task_id is not None:
            queue = tuple(item for item in queue if item.task_id == selected_task_id)
        active_attempts = tuple(
            attempt
            for attempt in list_formal_submission_attempts(
                connection,
                account_scope=account_scope,
            )
            if attempt.status in FORMAL_SUBMISSION_ACTIVE_STATUSES
        )
        active_grade = None
        if len(active_attempts) == 1 and active_attempts[0].source == "qualified_archive":
            snapshot = get_backtest_task(connection, active_attempts[0].task_id)
            if snapshot is None or snapshot.result is None:
                raise ValueError("formal_submission_source_invalid")
            active_grade = snapshot.result.grade
    if any(attempt.submission_mode == "automatic" for attempt in active_attempts):
        raise ValueError("formal_submission_automated_attempt_exists")
    active = tuple(
        attempt for attempt in active_attempts if attempt.submission_mode == "manual"
    )
    if len(active) > 1:
        raise ValueError("formal_submission_active_attempt_conflict")
    if active and (active[0].source != source or grade is not None and active_grade != grade):
        if active[0].source == "optimization":
            raise ValueError("formal_submission_other_attempt_active")
        command = "submit " + active_grade.lower() if active_grade is not None else "submit"
        raise ValueError("存在另一来源的未完成提交，请先运行 " + command + " 收尾")
    authorized_task_ids = {record.task_id for record in queue}
    authorized_task_ids.update(attempt.task_id for attempt in active)
    initial_queue_count = len(queue)
    progress = SubmissionProgress(
        database_path, account_scope, frozenset(authorized_task_ids),
        max_submissions=max_submissions,
    )
    progress.started()
    if not authorized_task_ids:
        return _completion(
            database_path,
            account_scope=account_scope,
            authorized_task_ids=frozenset(),
            initial_queue_count=0,
            step_count=0,
            platform_request_count=0,
            authentication_performed=False,
            max_submissions=max_submissions,
            source=source,
        )

    authentication_performed = False
    if not client.authenticated:
        logging.getLogger("execution.progress").info("正式提交：平台认证中")
        client.authenticate()
        authentication_performed = True
        logging.getLogger("execution.progress").info("正式提交：平台认证完成")

    next_observed = _aware_time(current_time())
    expired_reconciliation_task_ids = _expired_reconciliation_task_ids(
        database_path,
        attempts=active,
        observed=next_observed,
    )
    step_count = 0
    platform_request_count = 0
    consecutive_read_failures = 0
    while True:
        if not client.authenticated:
            client.authenticate()
            authentication_performed = True
            logging.getLogger("execution.progress").info("正式提交：会话已重新认证")
        observed = next_observed or _aware_time(current_time())
        next_observed = None
        try:
            advanced = advance_submission_queue(
                database_path,
                client,
                account_scope=account_scope,
                observed_at=observed.isoformat(),
                authorized_task_ids=frozenset(authorized_task_ids),
                allow_expired_confirmation=bool(expired_reconciliation_task_ids),
                source=source,
            )
        except WorldQuantRequestError as exc:
            consecutive_read_failures += 1
            step_count += 1
            platform_request_count += 1
            handled = _handle_manual_read_failure(
                database_path, account_scope=account_scope,
                authorized_task_ids=frozenset(authorized_task_ids), error=exc,
                failure_count=consecutive_read_failures,
                observed_at=_aware_time(current_time()),
            )
            if handled is None:
                raise
            task_id, delay = handled
            if delay is None:
                progress.observe(task_id)
                consecutive_read_failures = 0
                continue
            logging.getLogger("execution.progress").info(
                "提交前读取暂未成功（%s），%g 秒后重试；尚未发送正式提交。",
                exc.code, delay,
            )
            while delay > 0:
                interval = min(60.0, delay)
                wait(interval)
                delay -= interval
            continue
        step_count += 1
        if advanced.task_id is not None:
            if advanced.task_id not in authorized_task_ids:
                raise ValueError("formal_submission_authority_scope_changed")
        if advanced.platform_request_performed:
            consecutive_read_failures = 0
            platform_request_count += 1
            if advanced.task_id is not None:
                expired_reconciliation_task_ids.discard(advanced.task_id)
        progress.observe(advanced.task_id)
        if advanced.phase_terminal or max_submissions is not None:
            completion = _completion(
                database_path,
                account_scope=account_scope,
                authorized_task_ids=frozenset(authorized_task_ids),
                initial_queue_count=initial_queue_count,
                step_count=step_count,
                platform_request_count=platform_request_count,
                authentication_performed=authentication_performed,
                max_submissions=max_submissions,
                source=source,
            )
            if advanced.phase_terminal or completion.submission_limit_reached:
                return completion
        delay = advanced.retry_after_seconds
        if delay is None and (
            advanced.platform_request_performed
            or advanced.action == "formal_submission_in_progress"
        ):
            delay = poll_interval_seconds
        if delay is not None and delay > 0:
            wait(delay)


def _handle_manual_read_failure(
    database_path: str | Path, *, account_scope: str,
    authorized_task_ids: frozenset[str], error: WorldQuantRequestError,
    failure_count: int,
    observed_at: datetime,
) -> tuple[str, float | None] | None:
    if error.outcome_unknown or error.code.startswith("worldquant_authentication_"):
        return None
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        active = tuple(
            item for item in list_formal_submission_attempts(
                connection, account_scope=account_scope
            ) if item.status in FORMAL_SUBMISSION_ACTIVE_STATUSES
        )
        if len(active) != 1:
            return None
        attempt = active[0]
        if (attempt.task_id not in authorized_task_ids
                or attempt.submission_mode != "manual"
                or attempt.status not in {"detail_pending", "check_pending"}
                or attempt.submission_claimed_at is not None):
            return None
        run = get_automated_run(connection, attempt.run_id)
        if run is None:
            return None
        delay = max(min(60.0, float(2 ** min(failure_count - 1, 6))),
                    error.retry_after_seconds or 0.0)
        started = datetime.fromisoformat(attempt.created_at)
        remaining = run.max_pending_seconds - (observed_at - started).total_seconds()
        if error.retryable and failure_count < run.max_request_failures and delay <= remaining:
            return attempt.task_id, delay
        # Account-wide restrictions and unknown errors must not consume a candidate.
        if error.status_code in {401, 403, 429} or not error.retryable:
            return None
        replace_formal_submission_attempt(
            connection,
            replace(attempt, status="failed", retry_not_before=None,
                    failure_code="formal_submission_read_failed:" + error.code,
                    updated_at=observed_at.isoformat()),
            expected_status=attempt.status, expected_updated_at=attempt.updated_at,
        )
        return attempt.task_id, None


def _completion(
    database_path: str | Path,
    *,
    account_scope: str,
    authorized_task_ids: frozenset[str],
    initial_queue_count: int,
    step_count: int,
    platform_request_count: int,
    authentication_performed: bool,
    max_submissions: int | None,
    source: str = "queue",
) -> SubmissionQueueCompletion:
    with open_database(database_path) as connection:
        attempts = tuple(
            attempt
            for attempt in list_formal_submission_attempts(
                connection,
                account_scope=account_scope,
            )
            if attempt.task_id in authorized_task_ids
        )
        remaining_queue_count = sum(
            record.task_id in authorized_task_ids
            for record in list_submission_candidates(
                connection,
                account_scope=account_scope,
                source=source,
            )
        )
    statuses = [attempt.status for attempt in attempts]
    return SubmissionQueueCompletion(
        account_scope=account_scope,
        initial_queue_count=initial_queue_count,
        step_count=step_count,
        platform_request_count=platform_request_count,
        authentication_performed=authentication_performed,
        submission_claimed_count=sum(
            attempt.submission_claimed_at is not None for attempt in attempts
        ),
        submitted_count=statuses.count("submitted"),
        already_active_count=sum(
            attempt.status == "submitted" and attempt.submission_claimed_at is None
            for attempt in attempts
        ),
        ineligible_count=statuses.count("ineligible"),
        failed_count=statuses.count("failed"),
        unresolved_count=sum(
            status in FORMAL_SUBMISSION_ACTIVE_STATUSES for status in statuses
        ),
        remaining_queue_count=remaining_queue_count,
        max_submissions=max_submissions,
    )


def _expired_reconciliation_task_ids(
    database_path: str | Path,
    *,
    attempts: Sequence[FormalSubmissionAttemptRecord],
    observed: datetime,
) -> set[str]:
    expired: set[str] = set()
    with open_database(database_path) as connection:
        for attempt in attempts:
            if (
                attempt.status not in FORMAL_SUBMISSION_UNRESOLVED_STATUSES
                or attempt.submission_claimed_at is None
            ):
                continue
            run = get_automated_run(connection, attempt.run_id)
            if run is None:
                raise ValueError("automated_run_missing")
            claimed_at = _aware_time(
                datetime.fromisoformat(attempt.submission_claimed_at)
            )
            elapsed = (
                observed.astimezone(UTC) - claimed_at.astimezone(UTC)
            ).total_seconds()
            if elapsed < 0:
                raise ValueError("formal_submission_observed_before_claim")
            if elapsed >= run.max_pending_seconds:
                expired.add(attempt.task_id)
    return expired


def _validate_max_submissions(value: int | None) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("提交数量必须为正整数。")


def _validate_selection(source: str, task_id: str | None, limit: int | None, grade: str | None) -> None:
    if source == "optimization":
        if not isinstance(task_id, str) or not task_id.strip() or limit != 1 or grade is not None:
            raise ValueError("formal_submission_selection_required")
    elif task_id is not None and (source != "qualified_archive" or not isinstance(task_id, str)
                                 or not task_id.strip() or limit != 1 or grade is not None):
        raise ValueError("formal_submission_selection_required")


def _validate_poll_interval(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("formal_submission_poll_interval_invalid")


def _default_client_factory(
    settings: WorldQuantConnectionSettings,
) -> WorldQuantClient:
    return WorldQuantClient(
        base_url=settings.base_url,
        credentials=settings.credentials,
    )


def _aware_time(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("formal_submission_clock_invalid")
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)
