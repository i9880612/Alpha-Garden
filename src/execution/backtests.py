from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from hashlib import sha256

from generation.formula import render_formula
from generation.logic import prepare_formula_logic
from generation.parser import FormulaSyntaxError, parse_formula
from persistence.backtests import (
    BacktestCheckRecord,
    BacktestResultRecord,
    BacktestSnapshot,
    BacktestTaskRecord,
    BacktestYearlyStatRecord,
    create_backtest_task,
    get_backtest_task,
    get_backtest_task_by_identity,
    backtest_was_cancelled_before_submission,
    replace_backtest_task,
    save_backtest_result,
    save_backtest_yearly_stats,
)
from worldquant.backtests import (
    BacktestCheck,
    BacktestDetail,
    BacktestPollObservation,
    BacktestSubmissionObservation,
    BacktestYearlyStat,
)


def prepare_backtest_task(
    connection: sqlite3.Connection,
    *,
    account_scope: str,
    formula: str,
    settings: Mapping[str, object],
    created_at: str,
) -> BacktestSnapshot:
    try:
        parsed = parse_formula(formula)
    except FormulaSyntaxError as exc:
        raise ValueError(f"backtest_formula_invalid:{exc.code}") from exc
    logic = prepare_formula_logic(parsed.expression)
    if logic.issue is not None:
        raise ValueError(f"backtest_formula_logic_invalid:{logic.issue.code}")
    normalized_formula = render_formula(logic.expression)
    normalized_identity = parse_formula(normalized_formula).fingerprint
    settings_json = canonical_backtest_settings(settings)
    fingerprint = backtest_request_fingerprint(
        account_scope=account_scope,
        formula=normalized_formula,
        settings_json=settings_json,
    )
    task = BacktestTaskRecord(
        task_id=f"backtest_{fingerprint}",
        account_scope=account_scope,
        formula=normalized_formula,
        formula_fingerprint=normalized_identity,
        settings_json=settings_json,
        request_fingerprint=fingerprint,
        status="created",
        remote_id=None,
        platform_alpha_id=None,
        created_at=created_at,
        submission_started_at=None,
        last_observed_at=None,
        retry_not_before=None,
        finished_at=None,
        failure_code=None,
        failure_message=None,
    )
    previous = get_backtest_task_by_identity(
        connection, account_scope=account_scope, formula_fingerprint=normalized_identity,
        settings_json=settings_json,
    )
    if previous is not None and not backtest_was_cancelled_before_submission(previous.task):
        return previous
    if previous is not None and backtest_was_cancelled_before_submission(previous.task):
        if (previous.result is not None or previous.yearly_stats is not None
                or datetime.fromisoformat(created_at) < datetime.fromisoformat(previous.task.finished_at)):
            raise ValueError("backtest_cancelled_retry_evidence_invalid")
        identity = sha256(f"{previous.task.task_id}|{created_at}".encode()).hexdigest()
        task = replace(task, task_id=f"backtest_{identity}")
    persisted = create_backtest_task(connection, task)
    return _required_snapshot(connection, persisted.task_id)


def record_submission_unknown(
    connection: sqlite3.Connection,
    task_id: str,
    *,
    observed_at: str,
) -> BacktestSnapshot:
    snapshot = _required_snapshot(connection, task_id)
    if snapshot.task.status == "submission_unknown":
        return snapshot
    _require_status(snapshot, "created")
    updated = replace(
        snapshot.task,
        status="submission_unknown",
        submission_started_at=observed_at,
        last_observed_at=observed_at,
    )
    replace_backtest_task(connection, updated, expected_status="created")
    return _required_snapshot(connection, task_id)


def record_submission_not_accepted(
    connection: sqlite3.Connection,
    task_id: str,
) -> BacktestSnapshot:
    snapshot = _required_snapshot(connection, task_id)
    if snapshot.task.status == "created":
        return snapshot
    _require_status(snapshot, "submission_unknown")
    updated = replace(
        snapshot.task,
        status="created",
        submission_started_at=None,
        last_observed_at=None,
    )
    replace_backtest_task(
        connection,
        updated,
        expected_status="submission_unknown",
    )
    return _required_snapshot(connection, task_id)


def record_submission_accepted(
    connection: sqlite3.Connection,
    task_id: str,
    *,
    remote_id: str,
    observed_at: str,
    platform_alpha_id: str | None = None,
) -> BacktestSnapshot:
    _require_text(remote_id, "backtest_remote_id_missing")
    if platform_alpha_id is not None:
        _require_text(platform_alpha_id, "backtest_platform_alpha_id_invalid")
    snapshot = _required_snapshot(connection, task_id)
    if snapshot.task.status == "pending":
        if snapshot.task.remote_id != remote_id:
            raise ValueError("backtest_remote_identity_conflict")
        if (
            snapshot.task.platform_alpha_id is not None
            and platform_alpha_id is not None
            and snapshot.task.platform_alpha_id != platform_alpha_id
        ):
            raise ValueError("backtest_alpha_identity_conflict")
        if platform_alpha_id is None or (
            snapshot.task.platform_alpha_id == platform_alpha_id
        ):
            return snapshot
        return record_pending_observation(
            connection,
            task_id,
            observed_at=observed_at,
            platform_alpha_id=platform_alpha_id,
        )
    if snapshot.task.status not in {"created", "submission_unknown"}:
        raise ValueError("backtest_status_invalid")
    started_at = snapshot.task.submission_started_at or observed_at
    updated = replace(
        snapshot.task,
        status="pending",
        remote_id=remote_id,
        platform_alpha_id=platform_alpha_id,
        submission_started_at=started_at,
        last_observed_at=observed_at,
    )
    replace_backtest_task(
        connection,
        updated,
        expected_status=snapshot.task.status,
    )
    return _required_snapshot(connection, task_id)


def record_pending_observation(
    connection: sqlite3.Connection,
    task_id: str,
    *,
    observed_at: str,
    platform_alpha_id: str | None = None,
    retry_after_seconds: float | None = None,
) -> BacktestSnapshot:
    snapshot = _required_snapshot(connection, task_id)
    _require_status(snapshot, "pending")
    _require_observation_order(snapshot, observed_at)
    alpha_id = snapshot.task.platform_alpha_id
    if platform_alpha_id is not None:
        _require_text(platform_alpha_id, "backtest_platform_alpha_id_invalid")
        if alpha_id is not None and alpha_id != platform_alpha_id:
            raise ValueError("backtest_alpha_identity_conflict")
        alpha_id = platform_alpha_id
    retry_not_before = _retry_deadline(observed_at, retry_after_seconds)
    if (
        snapshot.task.last_observed_at == observed_at
        and alpha_id == snapshot.task.platform_alpha_id
        and snapshot.task.retry_not_before == retry_not_before
    ):
        return snapshot
    updated = replace(
        snapshot.task,
        platform_alpha_id=alpha_id,
        last_observed_at=observed_at,
        retry_not_before=retry_not_before,
    )
    replace_backtest_task(connection, updated, expected_status="pending")
    return _required_snapshot(connection, task_id)


def record_backtest_yearly_stats(
    connection: sqlite3.Connection,
    task_id: str,
    yearly_stats: Sequence[BacktestYearlyStat],
    *,
    observed_at: str,
) -> BacktestSnapshot:
    records = _yearly_stat_records(task_id, yearly_stats)
    snapshot = _required_snapshot(connection, task_id)
    if snapshot.task.status == "completed":
        if snapshot.yearly_stats != records:
            raise ValueError("backtest_yearly_stats_conflict")
        return snapshot
    _require_status(snapshot, "pending")
    _require_observation_order(snapshot, observed_at)
    if snapshot.task.platform_alpha_id is None:
        raise ValueError("backtest_platform_alpha_id_missing")
    if snapshot.result is None:
        raise ValueError("backtest_detail_not_captured")
    save_backtest_yearly_stats(connection, task_id, records)
    updated = replace(
        snapshot.task,
        status="completed",
        last_observed_at=observed_at,
        retry_not_before=None,
        finished_at=observed_at,
    )
    replace_backtest_task(connection, updated, expected_status="pending")
    return _required_snapshot(connection, task_id)


def fail_backtest_task(
    connection: sqlite3.Connection,
    task_id: str,
    *,
    failure_code: str,
    failure_message: str,
    observed_at: str,
    platform_alpha_id: str | None = None,
) -> BacktestSnapshot:
    _require_text(failure_code, "backtest_failure_code_missing")
    _require_text(failure_message, "backtest_failure_message_missing")
    if platform_alpha_id is not None:
        _require_text(platform_alpha_id, "backtest_platform_alpha_id_invalid")
    snapshot = _required_snapshot(connection, task_id)
    if snapshot.task.status == "failed":
        if (
            snapshot.task.failure_code != failure_code
            or snapshot.task.failure_message != failure_message
            or (
                platform_alpha_id is not None
                and snapshot.task.platform_alpha_id != platform_alpha_id
            )
        ):
            raise ValueError("backtest_failure_conflict")
        return snapshot
    if snapshot.task.status not in {"submission_unknown", "pending"}:
        raise ValueError("backtest_status_invalid")
    if snapshot.result is not None:
        raise ValueError("backtest_result_already_captured")
    _require_observation_order(snapshot, observed_at)
    updated = replace(
        snapshot.task,
        status="failed",
        platform_alpha_id=platform_alpha_id or snapshot.task.platform_alpha_id,
        last_observed_at=observed_at,
        retry_not_before=None,
        finished_at=observed_at,
        failure_code=failure_code,
        failure_message=failure_message,
    )
    replace_backtest_task(
        connection,
        updated,
        expected_status=snapshot.task.status,
    )
    return _required_snapshot(connection, task_id)


def expire_pending_backtest_task(
    connection: sqlite3.Connection,
    task_id: str,
    *,
    observed_at: str,
    max_pending_seconds: int,
) -> BacktestSnapshot | None:
    snapshot = _required_snapshot(connection, task_id)
    if snapshot.task.status != "pending":
        return None
    remaining = remaining_backtest_pending_seconds(
        snapshot,
        observed_at,
        max_pending_seconds=max_pending_seconds,
    )
    if remaining > 0:
        return None
    updated = replace(
        snapshot.task,
        status="failed",
        last_observed_at=observed_at,
        retry_not_before=None,
        finished_at=observed_at,
        failure_code="platform_pending_timeout",
        failure_message=(
            f"本地等待超过运行计划上限 {max_pending_seconds} 秒，释放本地名额；"
            "远端结果未确认，保留请求身份，不重发"
        ),
    )
    replace_backtest_task(
        connection,
        updated,
        expected_status="pending",
    )
    return _required_snapshot(connection, task_id)


def cancel_unsubmitted_backtest_task(
    connection: sqlite3.Connection,
    task_id: str,
    *,
    observed_at: str,
) -> BacktestSnapshot:
    failure_code = "automated_run_stopped_before_submission"
    failure_message = "自动运行已经停止，任务未提交到平台"
    snapshot = _required_snapshot(connection, task_id)
    if snapshot.task.status == "failed":
        if (
            snapshot.task.failure_code != failure_code
            or snapshot.task.failure_message != failure_message
        ):
            raise ValueError("backtest_failure_conflict")
        return snapshot
    _require_status(snapshot, "created")
    _require_observation_order(snapshot, observed_at)
    updated = replace(
        snapshot.task,
        status="failed",
        last_observed_at=observed_at,
        retry_not_before=None,
        finished_at=observed_at,
        failure_code=failure_code,
        failure_message=failure_message,
    )
    replace_backtest_task(
        connection,
        updated,
        expected_status="created",
    )
    return _required_snapshot(connection, task_id)


def apply_backtest_submission_observation(
    connection: sqlite3.Connection,
    task_id: str,
    observation: BacktestSubmissionObservation,
    *,
    observed_at: str,
) -> BacktestSnapshot:
    if observation.state == "accepted":
        if observation.remote_id is None:
            raise ValueError("backtest_submission_remote_id_missing")
        return record_submission_accepted(
            connection,
            task_id,
            remote_id=observation.remote_id,
            observed_at=observed_at,
        )
    if observation.state == "unknown":
        if observation.remote_id is not None:
            raise ValueError("backtest_submission_unknown_remote_id_invalid")
        return record_submission_unknown(
            connection,
            task_id,
            observed_at=observed_at,
        )
    raise ValueError("backtest_submission_observation_invalid")


def apply_backtest_poll_observation(
    connection: sqlite3.Connection,
    task_id: str,
    observation: BacktestPollObservation,
    *,
    observed_at: str,
) -> BacktestSnapshot:
    if observation.state == "pending":
        return record_pending_observation(
            connection,
            task_id,
            observed_at=observed_at,
        )
    if observation.state == "completed":
        if observation.platform_alpha_id is None:
            raise ValueError("backtest_poll_alpha_id_missing")
        return record_pending_observation(
            connection,
            task_id,
            observed_at=observed_at,
            platform_alpha_id=observation.platform_alpha_id,
        )
    if observation.state == "failed":
        status = observation.platform_status.strip().lower()
        if not status:
            raise ValueError("backtest_poll_failure_status_missing")
        return fail_backtest_task(
            connection,
            task_id,
            failure_code=f"platform_{status}",
            failure_message=(
                observation.failure_message
                or f"平台回测终态：{observation.platform_status}"
            ),
            observed_at=observed_at,
            platform_alpha_id=observation.platform_alpha_id,
        )
    raise ValueError("backtest_poll_observation_invalid")


def apply_backtest_detail(
    connection: sqlite3.Connection,
    task_id: str,
    detail: BacktestDetail,
    *,
    observed_at: str,
) -> BacktestSnapshot:
    _require_text(detail.platform_alpha_id, "backtest_platform_alpha_id_missing")
    result = BacktestResultRecord(
        task_id=task_id,
        sharpe=detail.sharpe,
        fitness=detail.fitness,
        turnover=detail.turnover,
        returns=detail.returns,
        drawdown=detail.drawdown,
        margin=detail.margin,
        book_size=detail.book_size,
        pnl=detail.pnl,
        long_count=detail.long_count,
        short_count=detail.short_count,
        check_details_captured=True,
        checks=_result_checks(detail.checks),
        grade=detail.grade,
    )
    snapshot = _required_snapshot(connection, task_id)
    if snapshot.task.status == "completed":
        if snapshot.result != result:
            raise ValueError("backtest_result_conflict")
        return snapshot
    _require_status(snapshot, "pending")
    _require_observation_order(snapshot, observed_at)
    if (
        snapshot.task.platform_alpha_id is not None
        and snapshot.task.platform_alpha_id != detail.platform_alpha_id
    ):
        raise ValueError("backtest_alpha_identity_conflict")
    if snapshot.yearly_stats is not None:
        raise ValueError("backtest_yearly_stats_already_captured")
    if snapshot.result is not None:
        if snapshot.result != result:
            raise ValueError("backtest_result_conflict")
        return snapshot
    updated = replace(
        snapshot.task,
        platform_alpha_id=detail.platform_alpha_id,
        last_observed_at=observed_at,
        retry_not_before=None,
    )
    replace_backtest_task(connection, updated, expected_status="pending")
    save_backtest_result(connection, result)
    return _required_snapshot(connection, task_id)


def canonical_backtest_settings(settings: Mapping[str, object]) -> str:
    if not isinstance(settings, Mapping) or not settings:
        raise ValueError("backtest_settings_invalid")
    normalized: dict[str, object] = {}
    for name, value in settings.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("backtest_settings_invalid")
        if isinstance(value, (dict, list, tuple, set)) or value is None:
            raise ValueError("backtest_settings_invalid")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError("backtest_settings_invalid")
        normalized[name] = value
    return _canonical_json_object(normalized, "backtest_settings_invalid")


def backtest_request_fingerprint(
    *,
    account_scope: str,
    formula: str,
    settings_json: str,
) -> str:
    _require_text(account_scope, "backtest_account_scope_missing")
    _require_text(formula, "backtest_formula_missing")
    payload = json.dumps(
        {
            "account_scope": account_scope,
            "formula": formula,
            "settings": json.loads(settings_json),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json_object(value: Mapping[str, object], error: str) -> str:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc


def _result_checks(checks: Sequence[BacktestCheck]) -> tuple[BacktestCheckRecord, ...]:
    if isinstance(checks, (str, bytes)):
        raise ValueError("backtest_checks_invalid")
    records: list[BacktestCheckRecord] = []
    for check in checks:
        if not isinstance(check, BacktestCheck):
            raise ValueError("backtest_checks_invalid")
        records.append(
            BacktestCheckRecord(
                name=check.name,
                status=check.status,
                threshold=check.threshold,
                actual=check.actual,
                platform_date=check.platform_date,
            )
        )
    return tuple(sorted(records, key=lambda record: record.name))


def _yearly_stat_records(
    task_id: str,
    yearly_stats: Sequence[BacktestYearlyStat],
) -> tuple[BacktestYearlyStatRecord, ...]:
    if isinstance(yearly_stats, (str, bytes)):
        raise ValueError("backtest_yearly_stats_invalid")
    records: list[BacktestYearlyStatRecord] = []
    for stat in yearly_stats:
        if not isinstance(stat, BacktestYearlyStat):
            raise ValueError("backtest_yearly_stats_invalid")
        records.append(
            BacktestYearlyStatRecord(
                task_id=task_id,
                year=stat.year,
                pnl=stat.pnl,
                book_size=stat.book_size,
                long_count=stat.long_count,
                short_count=stat.short_count,
                turnover=stat.turnover,
                sharpe=stat.sharpe,
                returns=stat.returns,
                drawdown=stat.drawdown,
                margin=stat.margin,
                fitness=stat.fitness,
                stage=stat.stage,
            )
        )
    return tuple(sorted(records, key=lambda record: (record.stage, record.year)))


def _required_snapshot(
    connection: sqlite3.Connection,
    task_id: str,
) -> BacktestSnapshot:
    snapshot = get_backtest_task(connection, task_id)
    if snapshot is None:
        raise ValueError("backtest_task_missing")
    return snapshot


def remaining_backtest_retry_seconds(
    snapshot: BacktestSnapshot,
    observed_at: str,
) -> float | None:
    _require_observation_order(snapshot, observed_at)
    if snapshot.task.retry_not_before is None:
        return None
    retry_at = datetime.fromisoformat(snapshot.task.retry_not_before)
    observed = datetime.fromisoformat(observed_at)
    remaining = (retry_at - observed).total_seconds()
    return remaining if remaining > 0 else None


def remaining_backtest_pending_seconds(
    snapshot: BacktestSnapshot,
    observed_at: str,
    *,
    max_pending_seconds: int,
) -> float:
    if (
        isinstance(max_pending_seconds, bool)
        or not isinstance(max_pending_seconds, int)
        or max_pending_seconds <= 0
    ):
        raise ValueError("backtest_max_pending_seconds_invalid")
    if snapshot.task.status != "pending":
        raise ValueError("backtest_status_invalid")
    started_at = snapshot.task.submission_started_at
    if started_at is None:
        raise ValueError("backtest_submission_started_at_invalid")
    _require_observation_order(snapshot, observed_at)
    started = datetime.fromisoformat(started_at)
    observed = datetime.fromisoformat(observed_at)
    if observed < started:
        raise ValueError("backtest_observed_before_submission")
    return max(0.0, max_pending_seconds - (observed - started).total_seconds())


def _require_status(snapshot: BacktestSnapshot, expected: str) -> None:
    if snapshot.task.status != expected:
        raise ValueError("backtest_status_invalid")


def _require_observation_order(
    snapshot: BacktestSnapshot,
    observed_at: str,
) -> None:
    try:
        observed = datetime.fromisoformat(observed_at)
    except (TypeError, ValueError) as exc:
        raise ValueError("backtest_last_observed_at_invalid") from exc
    if observed.utcoffset() is None:
        raise ValueError("backtest_last_observed_at_invalid")
    previous_value = snapshot.task.last_observed_at
    if previous_value is None:
        return
    previous = datetime.fromisoformat(previous_value)
    if observed < previous:
        raise ValueError("backtest_observation_out_of_order")


def _retry_deadline(
    observed_at: str,
    retry_after_seconds: float | None,
) -> str | None:
    if retry_after_seconds is None:
        return None
    if (
        isinstance(retry_after_seconds, bool)
        or not isinstance(retry_after_seconds, (int, float))
        or not math.isfinite(retry_after_seconds)
        or retry_after_seconds <= 0
    ):
        raise ValueError("backtest_retry_after_invalid")
    try:
        observed = datetime.fromisoformat(observed_at)
    except (TypeError, ValueError) as exc:
        raise ValueError("backtest_last_observed_at_invalid") from exc
    if observed.utcoffset() is None:
        raise ValueError("backtest_last_observed_at_invalid")
    return (observed + timedelta(seconds=retry_after_seconds)).isoformat()


def _require_text(value: object, error: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)
