from __future__ import annotations

import logging
from datetime import datetime, timedelta

from execution.seeds import signal_seed_evidence_task_ids, synchronize_signal_seeds
from execution.progress import live_progress
from generation.direction import DIRECTION_REVERSAL
from learning.seed_correlation import assess_seed_correlation
from learning.seeds import assess_signal_seed
from persistence.backtests import list_backtest_mutations, list_completed_backtests
from persistence.database import open_database
from persistence.pnl import PnlSeriesRecord, list_pnl_series, save_pnl_series
from persistence.submissions import list_platform_submitted_alphas
from worldquant.client import WorldQuantRequestError
from worldquant.backtests import WorldQuantProtocolError


def capture_next_seed_series(
    database_path, client, *, account_scope: str, observed_at: str,
    candidate_task_ids: tuple[str, ...] | None = None,
) -> float | None:
    """One due PnL read at a batch boundary; missing evidence never admits seeds.

    Retry metadata uses the existing PnL fact store. A deferred seed does not
    block other research or make a failed request into a correlation failure.
    """
    now = datetime.fromisoformat(observed_at)
    with open_database(database_path) as connection:
        completed = tuple(s for s in list_completed_backtests(connection)
                          if s.task.account_scope == account_scope)
        if candidate_task_ids is None:
            parents = set(signal_seed_evidence_task_ids(connection))
            mutations = {m.child_task_id: m for m in list_backtest_mutations(connection)}
            # Include roots whose admission was deferred at an earlier boundary.
            parents.update(s.task.task_id for s in completed
                           if s.task.task_id not in mutations
                           or mutations[s.task.task_id].action == DIRECTION_REVERSAL)
        else:
            parents = set(candidate_task_ids)
        candidates = tuple(s for s in completed if s.task.task_id in parents
                           and assess_signal_seed(s).eligible)
        references = list_platform_submitted_alphas(connection, account_scope=account_scope)
        series = list_pnl_series(connection)
        synchronize_signal_seeds(connection, candidate_task_ids=tuple(s.task.task_id for s in candidates))
        required_candidates = tuple(s for s in candidates
                                    if assess_seed_correlation(s, references, series).state == "pending")
        if not required_candidates:
            return None
        required = tuple(dict.fromkeys((
            *(r.platform_alpha_id for r in references),
            *(s.task.platform_alpha_id for s in required_candidates),
        )))
        existing = {s.platform_alpha_id for s in series if s.account_scope == account_scope
                    and (s.points is not None or (s.retry_not_before is not None
                         and datetime.fromisoformat(s.retry_not_before) > now))}
        alpha = next((a for a in required if a and a not in existing), None)
    if alpha is None:
        return None
    ready_count = sum(s.account_scope == account_scope and s.platform_alpha_id in required
                      and s.points is not None for s in series)
    live_progress(f"种子相关性取数 {ready_count}/{len(required)}，当前 {alpha}；不新增回测")
    try:
        observation = client.fetch_pnl(platform_alpha_id=alpha)
        points, retry_after = observation.points, observation.retry_after_seconds
    except (WorldQuantRequestError, WorldQuantProtocolError) as exc:
        points, retry_after = None, getattr(exc, "retry_after_seconds", None)
        logging.getLogger("execution.progress").warning(
            "种子相关性取数 %s：%s，资格待定", alpha, exc.code,
        )
    retry_at = (now + timedelta(seconds=max(600, retry_after or 0))).isoformat() if points is None else None
    with open_database(database_path) as connection:
        save_pnl_series(connection, PnlSeriesRecord(account_scope, alpha, observed_at, points, retry_at))
        synchronize_signal_seeds(connection, candidate_task_ids=tuple(s.task.task_id for s in candidates))
    return retry_after or 0.0
