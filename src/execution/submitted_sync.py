"""Read all submitted account alphas and PnL without importing research seeds."""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from execution.process_lock import exclusive_run_process
from persistence.database import open_database
from persistence.pnl import PnlSeriesRecord, pnl_capture_states, save_pnl_series
from persistence.schema import add_submitted_sync_storage
from persistence.submissions import PlatformSubmittedAlphaRecord, record_platform_submitted_alphas
from persistence.submitted_sync import complete_submitted_sync, invalidate_submitted_sync
from worldquant.backtests import WorldQuantProtocolError
from worldquant.client import WorldQuantClient, WorldQuantRequestError
from worldquant.config import load_worldquant_connection_settings


@dataclass(frozen=True, slots=True)
class SubmittedSyncPolicy:
    page_size: int = 100
    request_interval_seconds: float = 1.2
    max_attempts: int = 3
    max_retry_wait_seconds: float = 30.0

    def __post_init__(self):
        if type(self.page_size) is not int or not 1 <= self.page_size <= 100:
            raise ValueError("submitted_sync_page_size_invalid")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 10:
            raise ValueError("submitted_sync_attempts_invalid")
        for value in (self.request_interval_seconds, self.max_retry_wait_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("submitted_sync_wait_invalid")


@dataclass(frozen=True, slots=True)
class SubmittedSyncResult:
    account_scope: str
    submitted_count: int
    hidden_count: int
    pnl_captured_count: int
    pnl_pending_count: int
    completed: bool


def sync_submitted_alphas(database_path, environment_path, *, policy=None,
                          client_factory=None, clock=None, waiter=None) -> SubmittedSyncResult:
    """A failed scan never marks the baseline complete; captured facts survive retry.

    The explicit command can add only its own missing metadata table to the
    immediately preceding schema. It never creates seeds or backtest history.
    """
    path = Path(database_path)
    if not path.is_file():
        raise ValueError("submitted_sync_database_missing:run init first")
    policy = policy or SubmittedSyncPolicy()
    if not isinstance(policy, SubmittedSyncPolicy):
        raise ValueError("submitted_sync_policy_invalid")
    settings = load_worldquant_connection_settings(environment_path)
    now = clock or (lambda: datetime.now(UTC))
    wait = waiter or time.sleep
    def timestamp():
        value = now()
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ValueError("submitted_sync_clock_invalid")
        return value
    logger = logging.getLogger("execution.progress")

    with exclusive_run_process(path):
        with open_database(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            add_submitted_sync_storage(connection)
            invalidate_submitted_sync(connection, account_scope=settings.account_scope)
        client = (client_factory(settings) if client_factory else
                  WorldQuantClient(base_url=settings.base_url, credentials=settings.credentials))
        _read(client.authenticate, policy, wait)
        listed = _scan(client, policy, wait)
        logger.info("已提交基线：完整列表 %s 条，正在核对详情", len(listed))
        records = []
        for item in listed:
            if item.alpha_type != "REGULAR" or item.formula is None:
                raise ValueError("submitted_sync_alpha_type_unsupported:" + item.alpha_type)
            payload = _read(lambda: client.fetch_alpha_detail(
                platform_alpha_id=item.platform_alpha_id), policy, wait).payload
            if (not isinstance(payload, Mapping) or not isinstance(payload.get("regular"), Mapping)
                    or payload.get("id") != item.platform_alpha_id or payload.get("status") != item.status
                    or payload.get("hidden") is not item.hidden
                    or payload.get("regular", {}).get("code") != item.formula
                    or payload.get("settings") != item.settings
                    or _submitted_time(payload.get("dateSubmitted")) != item.submitted_at):
                raise ValueError("submitted_sync_detail_changed")
            records.append(PlatformSubmittedAlphaRecord(
                settings.account_scope, item.platform_alpha_id, item.formula, item.status,
                payload["dateSubmitted"], item.hidden, payload, timestamp().isoformat()))
        with open_database(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            record_platform_submitted_alphas(connection, tuple(records))
            existing = pnl_capture_states(connection, account_scope=settings.account_scope,
                                          platform_alpha_ids=frozenset(item.platform_alpha_id for item in listed))
        captured = 0
        for item in listed:
            cached = existing.get(item.platform_alpha_id)
            if cached is not None and cached[0]:
                captured += 1
                continue
            observed = timestamp()
            if (cached is not None and cached[1] is not None
                    and datetime.fromisoformat(cached[1]) > observed):
                continue
            error_code = None
            try:
                observation = _read(lambda: client.fetch_pnl(
                    platform_alpha_id=item.platform_alpha_id), policy, wait)
                points, retry = observation.points, observation.retry_after_seconds
            except (WorldQuantRequestError, WorldQuantProtocolError) as exc:
                # An exhausted account throttle must stop all later reads, not
                # turn into a per-alpha cooldown followed by another request.
                if isinstance(exc, WorldQuantRequestError) and exc.status_code in {401, 403, 429}:
                    raise
                points, retry, error_code = None, getattr(exc, "retry_after_seconds", None), exc.code
            retry_at = (observed + timedelta(seconds=max(600, retry or 0))).isoformat() if points is None else None
            with open_database(path) as connection:
                save_pnl_series(connection, PnlSeriesRecord(settings.account_scope,
                    item.platform_alpha_id, observed.isoformat(), points, retry_at))
            if points is not None:
                captured += 1
            else:
                logger.warning("已提交基线：PnL 暂缺（%s），保留待定并继续其他记录", error_code or "pending")
            logger.info("已提交基线：PnL 已保存 %s/%s", captured, len(listed))
        # Offset pagination is not a snapshot. Recheck identities, statuses and
        # submission times after the detail/PnL reads before claiming completeness.
        if _scan(client, policy, wait) != listed:
            raise ValueError("submitted_sync_snapshot_changed")
        complete = captured == len(listed)
        if complete:
            with open_database(path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                complete_submitted_sync(connection, account_scope=settings.account_scope,
                    alpha_ids=tuple(item.platform_alpha_id for item in listed),
                    completed_at=timestamp().isoformat())
        return SubmittedSyncResult(settings.account_scope, len(listed),
            sum(item.hidden for item in listed), captured, len(listed) - captured, complete)


def _scan(client, policy, wait):
    records, identities = [], set()
    for hidden in (False, True):
        offset, expected, previous_time = 0, None, None
        while True:
            page = _read(lambda: client.fetch_user_alpha_page(limit=policy.page_size,
                offset=offset, hidden=hidden, exclude_status="UNSUBMITTED", order="-dateSubmitted"), policy, wait)
            if expected is None:
                expected = page.total_count
            if page.total_count != expected:
                raise ValueError("submitted_sync_count_changed")
            consumed = offset + len(page.records)
            if consumed > expected or page.has_next != (consumed < expected):
                raise ValueError("submitted_sync_pagination_invalid")
            for item in page.records:
                if item.hidden is not hidden or item.status.upper() == "UNSUBMITTED":
                    raise ValueError("submitted_sync_filter_mismatch")
                if item.submitted_at is None:
                    raise ValueError("submitted_sync_submission_time_missing")
                if previous_time is not None and item.submitted_at > previous_time:
                    raise ValueError("submitted_sync_order_invalid")
                previous_time = item.submitted_at
                if item.platform_alpha_id in identities:
                    raise ValueError("submitted_sync_alpha_duplicate")
                identities.add(item.platform_alpha_id)
                records.append(item)
            if consumed == expected:
                break
            if not page.records:
                raise ValueError("submitted_sync_page_empty")
            offset = consumed
    # Equal timestamps may have a different server tie order on the second scan.
    return tuple(sorted(records, key=lambda item: item.platform_alpha_id))


def _read(operation, policy, wait):
    for attempt in range(policy.max_attempts):
        if policy.request_interval_seconds:
            wait(policy.request_interval_seconds)
        try:
            return operation()
        except WorldQuantRequestError as exc:
            delay = exc.retry_after_seconds if exc.retry_after_seconds is not None else 2 ** attempt
            if (not exc.retryable or attempt + 1 == policy.max_attempts
                    or delay > policy.max_retry_wait_seconds or exc.status_code in {401, 403}):
                raise
            wait(delay)
    raise AssertionError("submitted_sync_retry_unreachable")


def _submitted_time(value):
    try:
        result = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("submitted_sync_submission_time_invalid") from exc
    if result.utcoffset() is None:
        raise ValueError("submitted_sync_submission_time_invalid")
    return result
