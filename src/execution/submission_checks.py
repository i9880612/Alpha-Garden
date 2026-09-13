from __future__ import annotations

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

from persistence.backtests import get_backtest_task
from persistence.database import open_database
from persistence.submission_checks import SubmissionCheckRecord, get_submission_check, save_submission_check
from persistence.submissions import (
    canonical_submission_json, get_formal_submission_attempt,
    list_platform_submitted_alphas, normalize_submitted_formula,
)
from execution.submission_queue import synchronize_submission_queue
from execution.qualified_archive import synchronize_qualified_alpha_archive
from submission.formal import (assess_formal_check_payload, local_formal_submission_eligible,
                               formal_submission_grade_rejection)
from worldquant.backtests import WorldQuantProtocolError
from worldquant.client import WorldQuantClient, WorldQuantRequestError


MAX_SUBMISSION_CHECK_ATTEMPTS = 3
DEFERRED_SUBMISSION_CHECK_SECONDS = 600


def remaining_submission_check_seconds(connection, snapshot, observed_at: str, *, include_deferred: bool = False) -> float | None:
    if not local_formal_submission_eligible(snapshot):
        return None
    if get_formal_submission_attempt(connection, snapshot.task.task_id) is not None:
        return None
    formula = normalize_submitted_formula(snapshot.task.formula)
    if any(item.normalized_formula == formula or item.platform_alpha_id == snapshot.task.platform_alpha_id
           for item in list_platform_submitted_alphas(connection, account_scope=snapshot.task.account_scope)):
        return None
    saved = get_submission_check(connection, snapshot.task.task_id)
    if saved is None:
        return 0.0
    if saved.attempt_count >= MAX_SUBMISSION_CHECK_ATTEMPTS and not include_deferred:
        return None
    retry_at = saved.retry_not_before
    if retry_at is None:
        # Older runs stopped polling after three empty observations. Their raw
        # pending result remains usable without rewriting history on reads.
        if (saved.attempt_count < MAX_SUBMISSION_CHECK_ATTEMPTS
                or saved.payload_json is None
                or assess_formal_check_payload(json.loads(saved.payload_json)).state != "pending"):
            return None
        retry_at = (datetime.fromisoformat(saved.observed_at)
                    + timedelta(seconds=DEFERRED_SUBMISSION_CHECK_SECONDS)).isoformat()
    return max(0.0, (datetime.fromisoformat(retry_at)
                     - datetime.fromisoformat(observed_at)).total_seconds())


def check_completed_backtest(
    database_path: str | Path, client: WorldQuantClient, task_id: str, *, observed_at: str,
    recovered: bool = False, include_deferred: bool = False,
) -> bool:
    """Run one due check read; terminal checks are not repeated, transient reads are bounded."""
    with open_database(database_path) as connection:
        snapshot = get_backtest_task(connection, task_id)
        if snapshot is None or not local_formal_submission_eligible(snapshot):
            return False
        delay = remaining_submission_check_seconds(connection, snapshot, observed_at, include_deferred=include_deferred)
        if delay is None or delay > 0:
            return False
        previous = get_submission_check(connection, task_id)
        attempts = previous.attempt_count + 1 if previous else 1
    assert snapshot.task.platform_alpha_id is not None
    try:
        observation = client.fetch_formal_submission_check(platform_alpha_id=snapshot.task.platform_alpha_id)
        payload = canonical_submission_json(observation.payload)
        error_code = None
        state = assess_formal_check_payload(observation.payload).state
        retryable = state == "pending"
        retry_after = observation.retry_after_seconds
    except (WorldQuantRequestError, WorldQuantProtocolError) as exc:
        if isinstance(exc, WorldQuantRequestError) and exc.status_code in {401, 403}:
            raise
        payload, error_code = None, exc.code
        state = "error"
        retryable = isinstance(exc, WorldQuantProtocolError) or exc.retryable
        retry_after = getattr(exc, "retry_after_seconds", None)
    retry_at = None
    if retryable:
        interval = (30.0 * attempts if attempts < MAX_SUBMISSION_CHECK_ATTEMPTS
                    else DEFERRED_SUBMISSION_CHECK_SECONDS)
        retry_at = (datetime.fromisoformat(observed_at)
                    + timedelta(seconds=max(interval, retry_after or 0.0))).isoformat()
    record = SubmissionCheckRecord(task_id, observed_at, payload, error_code, attempts, retry_at)
    with open_database(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        save_submission_check(connection, record)
        synchronize_submission_queue(connection, candidate_task_ids=(task_id,), enqueued_at=observed_at)
        synchronize_qualified_alpha_archive(connection, observed_at=observed_at)
    label = {"passed": "通过，已核对入队资格", "failed": "未通过，不入队",
             "pending": "待定，不入队", "error": f"请求异常（{record.error_code}），不入队"}[state]
    if state == "passed" and snapshot.result is not None:
        rejection = formal_submission_grade_rejection(snapshot.result.grade)
        if rejection == "formal_submission_grade_unavailable":
            label = "检查通过；评级未确认，不入队"
        elif rejection is not None:
            label = f"检查通过；评级 {snapshot.result.grade} 未达 SPECTACULAR，不入队"
    deferred = retry_at is not None and attempts >= MAX_SUBMISSION_CHECK_ATTEMPTS
    if deferred:
        label += "；已安排延后自动复查，不阻塞回测"
    logging.getLogger("execution.progress").info(
        "%s入队检查 %s：%s", "旧" if recovered else "本批", snapshot.task.platform_alpha_id, label,
        extra={"transient": retry_at is not None and not deferred},
    )
    return True


def advance_deferred_submission_check(database_path, client, *, account_scope: str, observed_at: str) -> bool:
    """Read at most one due deferred check; never wait or reopen a backtest."""
    with open_database(database_path) as connection:
        rows = connection.execute(
            "SELECT c.task_id FROM submission_checks c JOIN backtest_tasks t ON t.task_id=c.task_id "
            "WHERE t.account_scope=? AND t.status='completed' AND c.attempt_count>=? "
            "ORDER BY c.observed_at, c.task_id", (account_scope, MAX_SUBMISSION_CHECK_ATTEMPTS),
        ).fetchall()
        selected = None
        for row in rows:
            snapshot = get_backtest_task(connection, row["task_id"])
            delay = remaining_submission_check_seconds(connection, snapshot, observed_at, include_deferred=True)
            if delay == 0:
                selected = row["task_id"]
                break
    return selected is not None and check_completed_backtest(
        database_path, client, selected, observed_at=observed_at, recovered=True, include_deferred=True,
    )
