"""Read ordinary research duplicate evidence at the batch planning boundary."""
from __future__ import annotations

import sqlite3
from datetime import datetime

from learning.action_effects import ExplicitSelfCorrelationEvidence
from learning.seed_correlation import submitted_seed
from learning.signal_duplicates import SignalDuplicate, find_signal_duplicates
from persistence.backtests import BacktestSnapshot
from persistence.pnl import list_pnl_series
from persistence.submissions import PlatformSubmittedAlphaRecord


def load_ordinary_signal_duplicates(
    connection: sqlite3.Connection,
    candidates: tuple[BacktestSnapshot, ...],
    *,
    account_scope: str,
    observed_at: datetime,
    explicit_checks: tuple[ExplicitSelfCorrelationEvidence, ...],
    submitted_alphas: tuple[PlatformSubmittedAlphaRecord, ...],
) -> tuple[SignalDuplicate, ...]:
    candidates = tuple(snapshot for snapshot in candidates
        if snapshot.task.account_scope == account_scope
        and datetime.fromisoformat(snapshot.task.finished_at) <= observed_at)
    if len(candidates) < 2:
        return ()
    checks = {check.task_id: check for check in explicit_checks
              if datetime.fromisoformat(check.observed_at) <= observed_at}
    submitted = tuple(record for record in submitted_alphas
                      if datetime.fromisoformat(record.observed_at) <= observed_at)
    groups = {}
    for snapshot in candidates:
        task = snapshot.task
        statuses = {check.name: check.status for check in snapshot.result.checks}
        latest = checks.get(task.task_id)
        if latest is not None and datetime.fromisoformat(latest.observed_at) < datetime.fromisoformat(task.finished_at):
            latest = None
        if latest is not None:
            statuses["SELF_CORRELATION"] = latest.status
        # Distinct repair needs stay independent. Submitted parents must not
        # suppress the existing priority of unsubmitted research opportunities.
        groups[task.task_id] = (submitted_seed(snapshot, submitted), tuple(sorted(statuses.items())),
                               latest.other_checks_passed if latest is not None else None)
    series = tuple(record for record in list_pnl_series(
        connection, account_scope=account_scope,
        platform_alpha_ids=frozenset(snapshot.task.platform_alpha_id for snapshot in candidates
                                    if snapshot.task.platform_alpha_id is not None),
    ) if datetime.fromisoformat(record.observed_at) <= observed_at)
    return find_signal_duplicates(candidates, series, comparison_groups=groups)
