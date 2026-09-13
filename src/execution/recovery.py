from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3

from execution.self_correlation import load_self_correlation_references
from execution.catalog import load_generation_catalog
from generation.self_correlation import (
    SELF_CORRELATION_REPAIR_FAMILIES, SELF_CORRELATION_HALF_FAMILIES,
    SELF_CORRELATION_LIGHT_FAMILIES,
)
from learning.recovery import RecoveryComparison, recovery_comparisons
from persistence.backtests import (
    BacktestSnapshot,
    list_completed_backtests,
    list_backtest_mutations,
)
from persistence.database import open_database
from persistence.pnl import PnlSeriesRecord, list_pnl_series, save_pnl_series
from persistence.submissions import list_platform_submitted_alphas
from persistence.catalog import get_platform_catalog_sync
from worldquant.client import WorldQuantClient, WorldQuantRequestError
from worldquant.pnl import PnlObservation


def load_recovery_comparisons(
    connection: sqlite3.Connection,
    *,
    snapshots: tuple[BacktestSnapshot, ...] | None = None,
    observed_at: datetime | None = None,
) -> tuple[RecoveryComparison, ...]:
    completed = list_completed_backtests(connection) if snapshots is None else snapshots
    mutations = list_backtest_mutations(connection)
    parent_ids = {
        item.parent_task_id
        for item in mutations
        if item.action in SELF_CORRELATION_REPAIR_FAMILIES
    }
    if not parent_ids:
        return ()
    comparisons = []
    for account in sorted(
        {
            item.task.account_scope
            for item in completed
            if item.task.task_id in parent_ids
        }
    ):
        parents = tuple(
            item
            for item in completed
            if item.task.task_id in parent_ids and item.task.account_scope == account
        )
        references = load_self_correlation_references(
            connection,
            parents=parents,
            submitted_alphas=list_platform_submitted_alphas(connection, account_scope=account),
            account_scope=account,
            observed_at=observed_at or datetime.now().astimezone(),
        )
        catalog = None
        account_parents = {p.task.task_id for p in parents}
        if any(m.action in (*SELF_CORRELATION_HALF_FAMILIES, *SELF_CORRELATION_LIGHT_FAMILIES)
               and m.parent_task_id in account_parents
               for m in mutations):
            sync = get_platform_catalog_sync(connection)
            if sync is not None and sync.account_scope == account:
                catalog = load_generation_catalog(connection, sync.context, account_scope=account)
        comparisons.extend(recovery_comparisons(
            tuple(s for s in completed if s.task.account_scope == account),
            mutations, tuple(references), catalog=catalog,
        ))
    return tuple(comparisons)


def capture_next_recovery_series(
    database_path: str | Path,
    client: WorldQuantClient,
    *,
    account_scope: str,
    observed_at: str,
    retry_interval_seconds: int = 600,
) -> float | None:
    """Capture one series per advance; planning itself stays read-only.

    The client may follow one short Retry-After with a second GET.

    None means no missing series. A numeric delay means a request was made.
    Empty HTTP bodies and object-level 404s remain pending, never zero/failed
    facts. Other request errors keep their existing run-level boundary.
    """
    with open_database(database_path) as connection:
        comparisons = load_recovery_comparisons(
            connection, observed_at=datetime.fromisoformat(observed_at)
        )
        existing = {
            (item.account_scope, item.platform_alpha_id)
            for item in list_pnl_series(connection)
            if item.points is not None
            or (
                item.retry_not_before is not None
                and datetime.fromisoformat(item.retry_not_before)
                > datetime.fromisoformat(observed_at)
            )
        }
    required = dict.fromkeys(
        alpha
        for item in comparisons
        if item.account_scope == account_scope
        for alpha in (
            item.parent_alpha_id,
            item.child_alpha_id,
            item.reference_alpha_id,
        )
    )
    alpha = next(
        (alpha for alpha in required if (account_scope, alpha) not in existing), None
    )
    if alpha is None:
        return None
    try:
        observation = client.fetch_pnl(platform_alpha_id=alpha)
    except WorldQuantRequestError as exc:
        if exc.status_code != 404:
            raise
        logging.getLogger("execution.progress").warning(
            "恢复相关性取数 %s：HTTP 404，证据待定，稍后重试", alpha,
        )
        observation = PnlObservation(None, exc.retry_after_seconds)
    if observation.points is None:
        retry = max(retry_interval_seconds, observation.retry_after_seconds or 0.0)
        with open_database(database_path) as connection:
            save_pnl_series(
                connection,
                PnlSeriesRecord(
                    account_scope,
                    alpha,
                    observed_at,
                    None,
                    (
                        datetime.fromisoformat(observed_at) + timedelta(seconds=retry)
                    ).isoformat(),
                ),
            )
        # Unknown research evidence does not block unrelated backtests. Respect
        # explicit platform Retry-After before any further platform request.
        return observation.retry_after_seconds or 0.0
    with open_database(database_path) as connection:
        save_pnl_series(
            connection,
            PnlSeriesRecord(account_scope, alpha, observed_at, observation.points),
        )
    return 0.0
