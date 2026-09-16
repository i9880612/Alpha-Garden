from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from execution.runs import record_automated_run_failure
from execution.qualified_archive import consume_qualified_alpha_archive
from execution.qualified_candidates import load_submission_opportunity_ids
from execution.submission_queue import claim_next_submission_queue_item
from execution.submitted_formulas import refresh_submitted_formulas
from persistence.backtests import (
    BacktestSnapshot,
    get_backtest_task,
    list_active_backtest_tasks,
)
from persistence.database import DATABASE_LOCK_TIMEOUT_SECONDS, open_database
from persistence.qualified_archive import archive_qualified_alpha
from persistence.submission_queue import FormalSubmissionQueueRecord, create_formal_submission_queue_item
from persistence.submission_checks import SubmissionCheckRecord, get_submission_check, save_submission_check
from persistence.runs import (
    AutomatedRunRecord,
    get_automated_run,
    get_automated_cycle_settlement,
    list_automated_run_backtests,
    list_active_automated_runs,
)
from persistence.submissions import (
    FORMAL_SUBMISSION_ACTIVE_STATUSES,
    FORMAL_SUBMISSION_UNRESOLVED_STATUSES,
    FormalSubmissionAttemptRecord,
    PlatformSubmittedAlphaRecord,
    canonical_submission_json,
    get_formal_submission_attempt,
    list_formal_submission_attempts,
    list_platform_submitted_alphas,
    normalize_submitted_formula,
    record_platform_submitted_alphas,
    replace_formal_submission_attempt,
    release_unposted_submission,
)
from submission.formal import (
    assess_formal_check_payload,
    confirm_formal_submission,
    local_formal_submission_eligible,
    formal_submission_grade_rejection,
)
from worldquant.backtests import parse_alpha_grade
from worldquant.client import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    WorldQuantClient,
    WorldQuantRequestError,
)


_DEFAULT_RETRY_SECONDS = 1.0
_SUBMISSION_COMMIT_GRACE_SECONDS = 2 * DATABASE_LOCK_TIMEOUT_SECONDS
_FORMAL_CHECK_FRESHNESS_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class FormalSubmissionAdvance:
    run_id: str | None
    source_cycle_number: int | None
    action: str
    task_id: str | None
    platform_request_performed: bool
    phase_terminal: bool
    run_status: str | None
    retry_after_seconds: float | None = None


def advance_automated_run_formal_submissions(
    database_path: str | Path,
    client: WorldQuantClient,
    run_id: str,
    *,
    finalization_cycle_number: int,
    observed_at: str,
) -> FormalSubmissionAdvance:
    return _advance_formal_submission_scope(
        database_path,
        client,
        observed_at=observed_at,
        run_id=run_id,
        account_scope=None,
        finalization_cycle_number=finalization_cycle_number,
        automated=True,
    )


def advance_submission_queue(
    database_path: str | Path,
    client: WorldQuantClient,
    *,
    account_scope: str,
    observed_at: str,
    authorized_task_ids: frozenset[str],
    allow_expired_confirmation: bool = False,
    source: str = "queue",
) -> FormalSubmissionAdvance:
    return _advance_formal_submission_scope(
        database_path,
        client,
        observed_at=observed_at,
        run_id=None,
        account_scope=account_scope,
        finalization_cycle_number=None,
        automated=False,
        authorized_task_ids=authorized_task_ids,
        allow_expired_confirmation=allow_expired_confirmation,
        source=source,
    )


def fail_unclaimed_automated_submission(
    database_path: str | Path,
    run_id: str,
    *,
    observed_at: str,
    failure_code: str,
) -> FormalSubmissionAttemptRecord | None:
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        active = tuple(
            attempt
            for attempt in list_formal_submission_attempts(
                connection,
                run_id=run_id,
            )
            if attempt.status in FORMAL_SUBMISSION_ACTIVE_STATUSES
            and attempt.submission_mode == "automatic"
        )
        if not active:
            return None
        if len(active) != 1 or active[0].status not in {
            "detail_pending",
            "check_pending",
            "ready",
        }:
            raise ValueError("formal_submission_active_attempt_conflict")
        current = active[0]
        failed = replace(
            current,
            status="failed",
            retry_not_before=None,
            failure_code=failure_code,
            updated_at=observed_at,
        )
        return replace_formal_submission_attempt(
            connection,
            failed,
            expected_status=current.status,
            expected_updated_at=current.updated_at,
        )


def _advance_formal_submission_scope(
    database_path: str | Path,
    client: WorldQuantClient,
    *,
    observed_at: str,
    run_id: str | None,
    account_scope: str | None,
    finalization_cycle_number: int | None,
    automated: bool,
    authorized_task_ids: frozenset[str] | None = None,
    allow_expired_confirmation: bool = False,
    source: str = "queue",
) -> FormalSubmissionAdvance:
    if source not in {"queue", "qualified_archive", "optimization"} or (automated and source != "queue"):
        raise ValueError("formal_submission_source_invalid")
    observed = _timestamp(observed_at)
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if automated:
            if run_id is None or account_scope is not None:
                raise ValueError("formal_submission_scope_invalid")
            run = _required_running_run(
                connection,
                run_id,
                finalization_cycle_number,
            )
            account_scope = run.account_scope
        else:
            if run_id is not None or account_scope is None:
                raise ValueError("formal_submission_scope_invalid")
            require_standalone_submission_scope(connection, account_scope)
            run = None
        assert account_scope is not None
        if automated and run is not None and not run.automatic_submissions_enabled:
            return _advance_result(
                run,
                None,
                action="formal_submission_disabled",
                phase_terminal=True,
            )
        active = tuple(
            attempt
            for attempt in list_formal_submission_attempts(
                connection,
                run_id=run.run_id if run is not None else None,
                account_scope=account_scope,
            )
            if attempt.status in FORMAL_SUBMISSION_ACTIVE_STATUSES
            and attempt.submission_mode == ("automatic" if automated else "manual")
        )
        if len(active) > 1:
            raise ValueError("formal_submission_active_attempt_conflict")
        if active:
            attempt = active[0]
            if attempt.source != source:
                raise ValueError("formal_submission_active_source_conflict:" + attempt.source)
            if authorized_task_ids is not None and attempt.task_id not in authorized_task_ids:
                raise ValueError("formal_submission_authority_scope_changed")
        else:
            attempt = claim_next_submission_queue_item(
                connection,
                account_scope=account_scope,
                run_id=run.run_id if run is not None else None,
                observed_at=observed_at,
                submission_mode="automatic" if automated else "manual",
                allowed_task_ids=authorized_task_ids,
                source=source,
            )
            if attempt is None:
                return _advance_result(
                    run,
                    None,
                    action="formal_submission_candidates_exhausted",
                    phase_terminal=True,
                )
            source_run = _required_run(connection, attempt.run_id)
            return _advance_result(
                run or source_run,
                attempt.cycle_number,
                action="formal_submission_prepared",
                task_id=attempt.task_id,
            )

        source_run = run or _required_run(connection, attempt.run_id)

    remaining = None
    if attempt.status in {"detail_pending", "check_pending", "ready"}:
        snapshot = _required_snapshot(database_path, attempt.task_id)
        assert snapshot.result is not None
        rejection = formal_submission_grade_rejection(snapshot.result.grade, source=attempt.source)
        if rejection is not None:
            return _reject_submission_grade(
                database_path, source_run, attempt, rejection, observed_at=observed_at,
            )
        remaining = source_run.max_pending_seconds - _elapsed_seconds(attempt.created_at, observed)
        if remaining <= 0:
            _finish_attempt(database_path, attempt, status="failed", observed_at=observed_at,
                            failure_code="formal_submission_check_timeout")
            return _advance_result(
                source_run, attempt.cycle_number, action="formal_submission_check_timeout",
                task_id=attempt.task_id,
            )
    if attempt.retry_not_before is not None:
        retry_at = _timestamp(attempt.retry_not_before)
        if retry_at > observed:
            return _advance_result(
                source_run,
                attempt.cycle_number,
                action="formal_submission_retry_wait",
                task_id=attempt.task_id,
                retry_after_seconds=min((retry_at - observed).total_seconds(),
                                        remaining if remaining is not None else math.inf),
            )
    if attempt.status in {"detail_pending", "check_pending"}:
        read = _advance_formal_detail if attempt.status == "detail_pending" else _advance_formal_check
        try:
            result = read(database_path, client, source_run, attempt, observed_at=observed_at)
        except WorldQuantRequestError as exc:
            if (exc.status_code in {404, 410} and not exc.outcome_unknown
                    and not exc.code.startswith("worldquant_authentication_")):
                _finish_attempt(database_path, attempt, status="failed", observed_at=observed_at,
                                failure_code="formal_submission_read_failed:" + exc.code)
                return _advance_result(
                    source_run, attempt.cycle_number, action="formal_submission_read_failed",
                    task_id=attempt.task_id, platform_request_performed=True,
                )
            if (not automated or not exc.retryable or exc.outcome_unknown
                    or exc.code.startswith("worldquant_authentication_")):
                raise
            delay = max(60.0, exc.retry_after_seconds or 0.0)
            with open_database(database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = _required_attempt(connection, attempt.task_id)
                replace_formal_submission_attempt(
                    connection, replace(current,
                        retry_not_before=(observed + timedelta(seconds=delay)).isoformat(),
                        updated_at=observed_at),
                    expected_status=attempt.status, expected_updated_at=attempt.updated_at,
                )
            logging.getLogger(__name__).warning(
                "提交前读取 %s 暂未成功（%s），按原等待上限重试。", attempt.task_id, exc.code,
            )
            result = _advance_result(
                source_run, attempt.cycle_number, action="formal_submission_read_retry",
                task_id=attempt.task_id, platform_request_performed=True, retry_after_seconds=delay,
            )
        if remaining is not None and result.retry_after_seconds is not None:
            result = replace(result, retry_after_seconds=min(result.retry_after_seconds, remaining))
        return result
    if attempt.status == "ready":
        guarded = _guard_ready_attempt(
            database_path,
            source_run,
            attempt,
            observed_at=observed_at,
            automated=automated,
        )
        if guarded is not None:
            return guarded
        return _advance_formal_post(
            database_path,
            client,
            source_run,
            attempt,
            observed_at=observed_at,
            automated=automated,
        )
    if attempt.status == "submitting" and _submission_may_still_be_in_progress(
        attempt,
        client,
        observed=observed,
    ):
        return _advance_result(
            source_run,
            attempt.cycle_number,
            action="formal_submission_in_progress",
            task_id=attempt.task_id,
            retry_after_seconds=_DEFAULT_RETRY_SECONDS,
        )
    if attempt.status in FORMAL_SUBMISSION_UNRESOLVED_STATUSES:
        return _advance_formal_confirmation(
            database_path,
            client,
            source_run,
            attempt,
            observed_at=observed_at,
            allow_failed_run=False,
            fail_run_on_timeout=automated,
            allow_expired_confirmation=allow_expired_confirmation,
        )
    raise ValueError("formal_submission_active_status_invalid")


def reconcile_failed_run_formal_submission(
    database_path: str | Path,
    client: WorldQuantClient,
    run_id: str,
    *,
    observed_at: str,
) -> FormalSubmissionAdvance | None:
    with open_database(database_path) as connection:
        run = get_automated_run(connection, run_id)
        if run is None:
            raise ValueError("automated_run_missing")
        if (
            run.status != "failed"
            or run.stop_reason != "submission_reconciliation_required"
        ):
            return None
        unresolved = tuple(
            attempt
            for attempt in list_formal_submission_attempts(
                connection,
                run_id=run.run_id,
            )
            if attempt.status in FORMAL_SUBMISSION_UNRESOLVED_STATUSES
            and attempt.submission_mode == "automatic"
        )
    if not unresolved:
        return None
    if len(unresolved) != 1:
        raise ValueError("formal_submission_unresolved_attempt_conflict")
    return _advance_formal_confirmation(
        database_path,
        client,
        run,
        unresolved[0],
        observed_at=observed_at,
        allow_failed_run=True,
        fail_run_on_timeout=True,
    )


def _advance_formal_detail(
    database_path: str | Path,
    client: WorldQuantClient,
    run: AutomatedRunRecord,
    attempt: FormalSubmissionAttemptRecord,
    *,
    observed_at: str,
) -> FormalSubmissionAdvance:
    snapshot = _required_snapshot(database_path, attempt.task_id)
    assert snapshot.task.platform_alpha_id is not None
    detail = client.fetch_alpha_detail(
        platform_alpha_id=snapshot.task.platform_alpha_id,
    )
    try:
        confirmed = confirm_formal_submission(
            detail.payload,
            expected_alpha_id=snapshot.task.platform_alpha_id,
            expected_normalized_formula=normalize_submitted_formula(
                snapshot.task.formula
            ),
            normalize_formula=normalize_submitted_formula,
        )
    except ValueError as exc:
        _finish_attempt(
            database_path,
            attempt,
            status="failed",
            observed_at=observed_at,
            failure_code=f"formal_submission_detail_invalid:{exc}",
        )
        return _advance_result(
            run,
            attempt.cycle_number,
            action="formal_submission_detail_invalid",
            task_id=attempt.task_id,
            platform_request_performed=True,
        )

    if confirmed is None:
        rejection = formal_submission_grade_rejection(
            parse_alpha_grade(detail.payload), source=attempt.source,
            expected_grade=snapshot.result.grade if attempt.source != "queue" else None,
        )
        if rejection is not None:
            return _reject_submission_grade(
                database_path, run, attempt, rejection, observed_at=observed_at,
                platform_request_performed=True,
            )

    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current = _required_attempt(connection, attempt.task_id)
        if current.status != "detail_pending":
            raise ValueError("formal_submission_attempt_transition_conflict")
        if confirmed is None:
            updated = replace(
                current,
                status="check_pending",
                updated_at=observed_at,
            )
            action = "formal_submission_detail_checked"
        else:
            record_platform_submitted_alphas(
                connection,
                (
                    PlatformSubmittedAlphaRecord(
                        account_scope=snapshot.task.account_scope,
                        platform_alpha_id=confirmed.platform_alpha_id,
                        formula=confirmed.formula,
                        status=confirmed.status,
                        date_submitted=confirmed.date_submitted,
                        hidden=confirmed.hidden,
                        raw_payload=confirmed.raw_payload,
                        observed_at=observed_at,
                    ),
                ),
            )
            updated = replace(
                current,
                status="submitted",
                confirmation_observed_at=observed_at,
                updated_at=observed_at,
            )
            action = "formal_submission_already_active"
            consume_qualified_alpha_archive(connection, task_id=current.task_id)
        replace_formal_submission_attempt(
            connection,
            updated,
            expected_status=current.status,
            expected_updated_at=current.updated_at,
        )
    if confirmed is not None:
        refresh_submitted_formulas(database_path)
    return _advance_result(
        run,
        attempt.cycle_number,
        action=action,
        task_id=attempt.task_id,
        platform_request_performed=True,
    )


def _advance_formal_check(
    database_path: str | Path,
    client: WorldQuantClient,
    run: AutomatedRunRecord,
    attempt: FormalSubmissionAttemptRecord,
    *,
    observed_at: str,
) -> FormalSubmissionAdvance:
    observed = _timestamp(observed_at)
    snapshot = _required_snapshot(database_path, attempt.task_id)
    assert snapshot.task.platform_alpha_id is not None
    observation = client.fetch_formal_submission_check(
        platform_alpha_id=snapshot.task.platform_alpha_id,
    )
    assessment = assess_formal_check_payload(observation.payload)
    check_payload_json = canonical_submission_json(observation.payload)
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current = _required_attempt(connection, attempt.task_id)
        if current.status != "check_pending":
            raise ValueError("formal_submission_attempt_transition_conflict")
        if assessment.state == "passed":
            updated = replace(
                current,
                status="ready",
                check_attempt_count=current.check_attempt_count + 1,
                check_payload_json=check_payload_json,
                check_observed_at=observed_at,
                retry_not_before=None,
                updated_at=observed_at,
            )
            action = "formal_submission_check_passed"
            retry_after = None
        elif assessment.state == "failed":
            updated = replace(
                current,
                status="ineligible",
                check_attempt_count=current.check_attempt_count + 1,
                check_payload_json=check_payload_json,
                check_observed_at=observed_at,
                retry_not_before=None,
                failure_code=(
                    "formal_submission_checks_failed:"
                    + ",".join(assessment.failed_checks)
                ),
                updated_at=observed_at,
            )
            action = "formal_submission_check_failed"
            retry_after = None
        else:
            retry_after = max(
                observation.retry_after_seconds or _DEFAULT_RETRY_SECONDS,
                _DEFAULT_RETRY_SECONDS,
            )
            updated = replace(
                current,
                check_attempt_count=current.check_attempt_count + 1,
                check_payload_json=check_payload_json,
                check_observed_at=observed_at,
                retry_not_before=(
                    observed + timedelta(seconds=retry_after)
                ).isoformat(),
                updated_at=observed_at,
            )
            action = "formal_submission_check_pending"
        replace_formal_submission_attempt(
            connection,
            updated,
            expected_status=current.status,
            expected_updated_at=current.updated_at,
        )
    return _advance_result(
        run,
        attempt.cycle_number,
        action=action,
        task_id=attempt.task_id,
        platform_request_performed=True,
        retry_after_seconds=retry_after,
    )


def _advance_formal_post(
    database_path: str | Path,
    client: WorldQuantClient,
    run: AutomatedRunRecord,
    attempt: FormalSubmissionAttemptRecord,
    *,
    observed_at: str,
    automated: bool,
) -> FormalSubmissionAdvance:
    observed = _timestamp(observed_at)
    request_timeout_seconds = _submission_request_timeout_seconds(client)
    if request_timeout_seconds > DEFAULT_REQUEST_TIMEOUT_SECONDS:
        raise ValueError("formal_submission_timeout_unsupported")
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current_run = (
            _required_running_run(connection, run.run_id)
            if automated
            else _required_run(connection, run.run_id)
        )
        if not automated:
            require_standalone_submission_scope(
                connection,
                current_run.account_scope,
            )
        current = _required_attempt(connection, attempt.task_id)
        if current.status != "ready":
            return _advance_result(
                current_run,
                attempt.cycle_number,
                action="formal_submission_already_advanced",
                task_id=attempt.task_id,
                retry_after_seconds=(
                    _DEFAULT_RETRY_SECONDS
                    if current.status in FORMAL_SUBMISSION_ACTIVE_STATUSES
                    else None
                ),
            )
        assert current.check_observed_at is not None
        previous_check = get_submission_check(connection, current.task_id)
        newer_check_blocks = (
            previous_check is not None
            and _timestamp(previous_check.observed_at) > _timestamp(current.check_observed_at)
            and assess_formal_check_payload(
                json.loads(previous_check.payload_json) if previous_check.payload_json else None
            ).state != "passed"
        )
        if newer_check_blocks or current.task_id not in load_submission_opportunity_ids(
            connection, account_scope=current_run.account_scope,
            include_optimization=current.source == "optimization", reserved_task_id=current.task_id,
        ):
            # No POST has happened. Keep the fresh check as evidence, return the
            # candidate to its source and release the reservation atomically.
            # Replanning excludes it while pending, so independent work can finish.
            if current.check_payload_json is not None and (
                previous_check is None or _timestamp(current.check_observed_at)
                >= _timestamp(previous_check.retry_not_before or previous_check.observed_at)
            ):
                save_submission_check(connection, SubmissionCheckRecord(
                    current.task_id, current.check_observed_at, current.check_payload_json, None,
                    previous_check.attempt_count + 1 if previous_check else 1))
            release_unposted_submission(connection, current)
            if current.source == "qualified_archive":
                archive_qualified_alpha(connection, task_id=current.task_id, archived_at=observed_at)
            elif current.source == "queue":
                snapshot = get_backtest_task(connection, current.task_id)
                create_formal_submission_queue_item(connection, FormalSubmissionQueueRecord(
                    current.task_id, current_run.account_scope, current.run_id, current.cycle_number,
                    current.family_root_task_id, normalize_submitted_formula(snapshot.task.formula), observed_at))
            logging.getLogger("execution.progress").info(
                "正式提交：%s 本地相关性或改善机会待重新确认，暂缓提交；未发送提交请求。", current.task_id)
            return _advance_result(current_run, current.cycle_number,
                action="formal_submission_deferred", task_id=current.task_id)
        claimed = replace(
            current,
            status="submitting",
            retry_not_before=None,
            submission_claimed_at=observed_at,
            updated_at=observed_at,
        )
        replace_formal_submission_attempt(
            connection,
            claimed,
            expected_status=current.status,
            expected_updated_at=current.updated_at,
        )

    snapshot = _required_snapshot(database_path, attempt.task_id)
    assert snapshot.task.platform_alpha_id is not None
    try:
        observation = client.submit_formal_alpha(
            platform_alpha_id=snapshot.task.platform_alpha_id,
        )
    except WorldQuantRequestError as exc:
        with open_database(database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = _required_attempt(connection, attempt.task_id)
            if current.status != "submitting":
                raise ValueError("formal_submission_attempt_transition_conflict")
            if exc.code == "worldquant_authentication_expired" and not exc.outcome_unknown:
                replace_formal_submission_attempt(
                    connection, replace(current, status="ready", submission_claimed_at=None,
                                        updated_at=observed_at),
                    expected_status=current.status, expected_updated_at=current.updated_at,
                )
                return _advance_result(
                    run, attempt.cycle_number, action="authentication_required",
                    task_id=attempt.task_id, platform_request_performed=True,
                )
            updated = replace(
                current,
                status="submission_unknown" if exc.outcome_unknown else "failed",
                retry_not_before=(
                    observed + timedelta(seconds=_request_retry_seconds(exc))
                ).isoformat()
                if exc.outcome_unknown
                else None,
                failure_code=(
                    "formal_submission_outcome_unknown:"
                    if exc.outcome_unknown
                    else "formal_submission_rejected:"
                )
                + exc.code,
                updated_at=observed_at,
            )
            replace_formal_submission_attempt(
                connection,
                updated,
                expected_status=current.status,
                expected_updated_at=current.updated_at,
            )
        return _advance_result(
            run,
            attempt.cycle_number,
            action=(
                "formal_submission_outcome_unknown"
                if exc.outcome_unknown
                else "formal_submission_rejected"
            ),
            task_id=attempt.task_id,
            platform_request_performed=True,
            retry_after_seconds=(
                _request_retry_seconds(exc) if exc.outcome_unknown else None
            ),
        )

    response_json = canonical_submission_json(
        {
            "status_code": observation.status_code,
            "payload": observation.payload,
        }
    )
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current = _required_attempt(connection, attempt.task_id)
        if current.status != "submitting":
            raise ValueError("formal_submission_attempt_transition_conflict")
        updated = replace(
            current,
            status="confirmation_pending",
            submit_http_status=observation.status_code,
            submit_response_json=response_json,
            retry_not_before=None,
            updated_at=observed_at,
        )
        replace_formal_submission_attempt(
            connection,
            updated,
            expected_status=current.status,
            expected_updated_at=current.updated_at,
        )
    return _advance_result(
        run,
        attempt.cycle_number,
        action="formal_submission_confirmation_pending",
        task_id=attempt.task_id,
        platform_request_performed=True,
    )


def _guard_ready_attempt(
    database_path: str | Path,
    run: AutomatedRunRecord,
    attempt: FormalSubmissionAttemptRecord,
    *,
    observed_at: str,
    automated: bool,
) -> FormalSubmissionAdvance | None:
    observed = _timestamp(observed_at)
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current_run = (
            _required_running_run(connection, run.run_id)
            if automated
            else _required_run(connection, run.run_id)
        )
        if not automated:
            require_standalone_submission_scope(
                connection,
                current_run.account_scope,
            )
        current = _required_attempt(connection, attempt.task_id)
        if current.status != "ready":
            return _advance_result(
                current_run,
                attempt.cycle_number,
                action="formal_submission_already_advanced",
                task_id=attempt.task_id,
                retry_after_seconds=(
                    _DEFAULT_RETRY_SECONDS
                    if current.status in FORMAL_SUBMISSION_ACTIVE_STATUSES
                    else None
                ),
            )
        snapshot = get_backtest_task(connection, current.task_id)
        if snapshot is None:
            raise ValueError("formal_submission_source_invalid")
        normalized_formula = normalize_submitted_formula(snapshot.task.formula)
        submitted_formulas = {
            record.normalized_formula
            for record in list_platform_submitted_alphas(
                connection,
                account_scope=current_run.account_scope,
            )
        }
        if normalized_formula in submitted_formulas:
            skipped = replace(
                current,
                status="ineligible",
                retry_not_before=None,
                failure_code="formal_submission_formula_already_submitted",
                updated_at=observed_at,
            )
            replace_formal_submission_attempt(
                connection,
                skipped,
                expected_status=current.status,
                expected_updated_at=current.updated_at,
            )
            return _advance_result(
                current_run,
                attempt.cycle_number,
                action="formal_submission_formula_already_submitted",
                task_id=attempt.task_id,
            )
        assert current.check_observed_at is not None
        check_age = _elapsed_seconds(current.check_observed_at, observed)
        if check_age < 0:
            raise ValueError("formal_submission_observed_before_check")
        if check_age <= _FORMAL_CHECK_FRESHNESS_SECONDS:
            return None
        refresh = replace(
            current,
            status="check_pending",
            retry_not_before=None,
            updated_at=observed_at,
        )
        replace_formal_submission_attempt(
            connection,
            refresh,
            expected_status=current.status,
            expected_updated_at=current.updated_at,
        )
        return _advance_result(
            current_run,
            attempt.cycle_number,
            action="formal_submission_check_expired",
            task_id=attempt.task_id,
        )


def _advance_formal_confirmation(
    database_path: str | Path,
    client: WorldQuantClient,
    run: AutomatedRunRecord,
    attempt: FormalSubmissionAttemptRecord,
    *,
    observed_at: str,
    allow_failed_run: bool,
    fail_run_on_timeout: bool,
    allow_expired_confirmation: bool = False,
) -> FormalSubmissionAdvance:
    observed = _timestamp(observed_at)
    assert attempt.submission_claimed_at is not None
    if (
        not allow_failed_run
        and not allow_expired_confirmation
        and _elapsed_seconds(attempt.submission_claimed_at, observed)
        >= run.max_pending_seconds
    ):
        with open_database(database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = _required_attempt(connection, attempt.task_id)
            unknown = replace(
                current,
                status="submission_unknown",
                retry_not_before=None,
                failure_code=(
                    current.failure_code or "formal_submission_confirmation_timeout"
                ),
                updated_at=observed_at,
            )
            replace_formal_submission_attempt(
                connection,
                unknown,
                expected_status=current.status,
                expected_updated_at=current.updated_at,
            )
            result_run = run
            if fail_run_on_timeout:
                result_run = record_automated_run_failure(
                    connection,
                    run.run_id,
                    failed_at=observed_at,
                    reason="submission_reconciliation_required",
                )
        return _advance_result(
            result_run,
            attempt.cycle_number,
            action="formal_submission_reconciliation_required",
            task_id=attempt.task_id,
            phase_terminal=not fail_run_on_timeout,
        )

    snapshot = _required_snapshot(database_path, attempt.task_id)
    assert snapshot.task.platform_alpha_id is not None
    try:
        detail = client.fetch_alpha_detail(
            platform_alpha_id=snapshot.task.platform_alpha_id,
        )
        confirmed = confirm_formal_submission(
            detail.payload,
            expected_alpha_id=snapshot.task.platform_alpha_id,
            expected_normalized_formula=normalize_submitted_formula(
                snapshot.task.formula
            ),
            normalize_formula=normalize_submitted_formula,
        )
        result = None
        if confirmed is None and attempt.submit_http_status in {200, 201, 202}:
            result = client.fetch_formal_submission_result(
                platform_alpha_id=snapshot.task.platform_alpha_id,
            )
    except (WorldQuantRequestError, ValueError) as exc:
        return _record_confirmation_unknown(
            database_path,
            run,
            attempt,
            observed_at=observed_at,
            failure_code=(
                f"formal_submission_confirmation_unknown:{type(exc).__name__}:{exc}"
            ),
            retry_after_seconds=(
                _request_retry_seconds(exc)
                if isinstance(exc, WorldQuantRequestError)
                else _DEFAULT_RETRY_SECONDS
            ),
        )
    if confirmed is None:
        assessment = assess_formal_check_payload(result.payload) if result else None
        if (result is not None and result.status_code == 403
                and result.retry_after_seconds is None
                and assessment is not None and assessment.state == "failed"):
            with open_database(database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = _required_attempt(connection, attempt.task_id)
                if current.status not in FORMAL_SUBMISSION_UNRESOLVED_STATUSES:
                    raise ValueError("formal_submission_attempt_transition_conflict")
                assert current.submit_response_json is not None
                response = json.loads(current.submit_response_json)
                response["confirmation"] = {
                    "status_code": result.status_code,
                    "payload": result.payload,
                    "observed_at": observed_at,
                }
                replace_formal_submission_attempt(
                    connection,
                    replace(
                        current, status="failed", retry_not_before=None,
                        submit_response_json=canonical_submission_json(response),
                        failure_code="formal_submission_rejected:"
                        + ",".join(assessment.failed_checks),
                        updated_at=observed_at,
                    ),
                    expected_status=current.status,
                    expected_updated_at=current.updated_at,
                )
            return _advance_result(
                run, attempt.cycle_number, action="formal_submission_rejected",
                task_id=attempt.task_id, platform_request_performed=True,
            )
        return _record_confirmation_unknown(
            database_path,
            run,
            attempt,
            observed_at=observed_at,
            failure_code="formal_submission_confirmation_pending",
            retry_after_seconds=(result.retry_after_seconds or _DEFAULT_RETRY_SECONDS)
            if result else _DEFAULT_RETRY_SECONDS,
        )

    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current = _required_attempt(connection, attempt.task_id)
        if current.status not in FORMAL_SUBMISSION_UNRESOLVED_STATUSES:
            raise ValueError("formal_submission_attempt_transition_conflict")
        record_platform_submitted_alphas(
            connection,
            (
                PlatformSubmittedAlphaRecord(
                    account_scope=snapshot.task.account_scope,
                    platform_alpha_id=confirmed.platform_alpha_id,
                    formula=confirmed.formula,
                    status=confirmed.status,
                    date_submitted=confirmed.date_submitted,
                    hidden=confirmed.hidden,
                    raw_payload=confirmed.raw_payload,
                    observed_at=observed_at,
                ),
            ),
        )
        submitted = replace(
            current,
            status="submitted",
            retry_not_before=None,
            confirmation_observed_at=observed_at,
            failure_code=None,
            updated_at=observed_at,
        )
        consume_qualified_alpha_archive(connection, task_id=current.task_id)
        replace_formal_submission_attempt(
            connection,
            submitted,
            expected_status=current.status,
            expected_updated_at=current.updated_at,
        )
    refresh_submitted_formulas(database_path)
    return _advance_result(
        run,
        attempt.cycle_number,
        action="formal_submission_confirmed",
        task_id=attempt.task_id,
        platform_request_performed=True,
    )


def _record_confirmation_unknown(
    database_path: str | Path,
    run: AutomatedRunRecord,
    attempt: FormalSubmissionAttemptRecord,
    *,
    observed_at: str,
    failure_code: str,
    retry_after_seconds: float,
) -> FormalSubmissionAdvance:
    observed = _timestamp(observed_at)
    retry_after = max(retry_after_seconds, _DEFAULT_RETRY_SECONDS)
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current = _required_attempt(connection, attempt.task_id)
        if current.status not in FORMAL_SUBMISSION_UNRESOLVED_STATUSES:
            raise ValueError("formal_submission_attempt_transition_conflict")
        unknown = replace(
            current,
            status="submission_unknown",
            retry_not_before=(observed + timedelta(seconds=retry_after)).isoformat(),
            failure_code=failure_code,
            updated_at=observed_at,
        )
        replace_formal_submission_attempt(
            connection,
            unknown,
            expected_status=current.status,
            expected_updated_at=current.updated_at,
        )
    return _advance_result(
        run,
        attempt.cycle_number,
        action="formal_submission_confirmation_pending",
        task_id=attempt.task_id,
        platform_request_performed=True,
        retry_after_seconds=retry_after,
    )


def _required_running_run(
    connection,
    run_id: str,
    finalization_cycle_number: int | None = None,
) -> AutomatedRunRecord:
    run = get_automated_run(connection, run_id)
    if run is None:
        raise ValueError("automated_run_missing")
    if run.status != "running":
        raise ValueError("automated_run_status_invalid")
    if finalization_cycle_number is not None:
        links = list_automated_run_backtests(connection, run.run_id)
        if get_automated_cycle_settlement(
            connection, run.run_id, finalization_cycle_number
        ):
            raise ValueError("automated_run_status_invalid")
        if not any(link.cycle_number == finalization_cycle_number for link in links):
            if finalization_cycle_number != max(
                (link.cycle_number for link in links), default=0
            ) + 1:
                raise ValueError("automated_run_status_invalid")
    return run


def _required_run(connection, run_id: str) -> AutomatedRunRecord:
    run = get_automated_run(connection, run_id)
    if run is None:
        raise ValueError("automated_run_missing")
    return run


def require_standalone_submission_scope(
    connection,
    account_scope: str,
) -> None:
    if list_active_automated_runs(connection):
        raise ValueError("formal_submission_active_run_exists")
    if any(
        task.account_scope == account_scope and task.status != "submission_unknown"
        for task in list_active_backtest_tasks(connection)
    ):
        raise ValueError("formal_submission_active_backtest_exists")


def _required_attempt(connection, task_id: str) -> FormalSubmissionAttemptRecord:
    attempt = get_formal_submission_attempt(connection, task_id)
    if attempt is None:
        raise ValueError("formal_submission_attempt_missing")
    return attempt


def _required_snapshot(
    database_path: str | Path,
    task_id: str,
) -> BacktestSnapshot:
    with open_database(database_path) as connection:
        snapshot = get_backtest_task(connection, task_id)
    if snapshot is None or not local_formal_submission_eligible(snapshot):
        raise ValueError("formal_submission_source_invalid")
    return snapshot


def _reject_submission_grade(
    database_path: str | Path,
    run: AutomatedRunRecord,
    attempt: FormalSubmissionAttemptRecord,
    rejection: str,
    *,
    observed_at: str,
    platform_request_performed: bool = False,
) -> FormalSubmissionAdvance:
    _finish_attempt(
        database_path, attempt,
        status="failed" if rejection == "formal_submission_grade_unavailable" else "ineligible",
        observed_at=observed_at, failure_code=rejection,
    )
    return _advance_result(
        run, attempt.cycle_number, action="formal_submission_grade_rejected",
        task_id=attempt.task_id, platform_request_performed=platform_request_performed,
    )


def _finish_attempt(
    database_path: str | Path,
    attempt: FormalSubmissionAttemptRecord,
    *,
    status: str,
    observed_at: str,
    failure_code: str,
) -> None:
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current = _required_attempt(connection, attempt.task_id)
        updated = replace(
            current,
            status=status,
            retry_not_before=None,
            failure_code=failure_code,
            updated_at=observed_at,
        )
        replace_formal_submission_attempt(
            connection,
            updated,
            expected_status=attempt.status,
            expected_updated_at=attempt.updated_at,
        )


def _request_retry_seconds(error: WorldQuantRequestError) -> float:
    return max(
        error.retry_after_seconds or _DEFAULT_RETRY_SECONDS,
        _DEFAULT_RETRY_SECONDS,
    )


def _elapsed_seconds(started_at: str, observed: datetime) -> float:
    return (observed - _timestamp(started_at)).total_seconds()


def _submission_may_still_be_in_progress(
    attempt: FormalSubmissionAttemptRecord,
    client: WorldQuantClient,
    *,
    observed: datetime,
) -> bool:
    if attempt.status != "submitting" or attempt.submission_claimed_at is None:
        raise ValueError("formal_submission_claim_invalid")
    started = _timestamp(attempt.submission_claimed_at)
    if observed < started:
        raise ValueError("formal_submission_observed_before_submission")
    timeout_seconds = _submission_request_timeout_seconds(client)
    if timeout_seconds > DEFAULT_REQUEST_TIMEOUT_SECONDS:
        raise ValueError("formal_submission_timeout_unsupported")
    return (observed - started).total_seconds() <= (
        timeout_seconds + _SUBMISSION_COMMIT_GRACE_SECONDS
    )


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


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("formal_submission_timestamp_invalid") from exc
    if parsed.utcoffset() is None:
        raise ValueError("formal_submission_timestamp_invalid")
    return parsed


def _advance_result(
    run: AutomatedRunRecord | None,
    source_cycle_number: int | None,
    *,
    action: str,
    task_id: str | None = None,
    platform_request_performed: bool = False,
    phase_terminal: bool = False,
    retry_after_seconds: float | None = None,
) -> FormalSubmissionAdvance:
    return FormalSubmissionAdvance(
        run_id=run.run_id if run is not None else None,
        source_cycle_number=source_cycle_number,
        action=action,
        task_id=task_id,
        platform_request_performed=platform_request_performed,
        phase_terminal=phase_terminal,
        run_status=run.status if run is not None else None,
        retry_after_seconds=retry_after_seconds,
    )
