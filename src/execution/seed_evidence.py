from __future__ import annotations

from datetime import datetime

from execution.seeds import signal_seed_evidence_task_ids, synchronize_signal_seeds
from execution.pnl import capture_pnl_series
from generation.direction import DIRECTION_REVERSAL
from learning.seeds import assess_signal_seed
from persistence.backtests import list_backtest_mutations, list_completed_backtests
from persistence.database import open_database
from persistence.pnl import pnl_capture_states
from persistence.submissions import list_platform_submitted_alphas


def capture_next_seed_series(
    database_path, client, *, account_scope: str, observed_at: str,
    candidate_task_ids: tuple[str, ...] | None = None,
    admit_seeds: bool = True,
    research_task_ids: tuple[str, ...] | None = None,
) -> float | None:
    """Backfill one due historical/deferred curve; missing evidence never admits seeds.

    Retry metadata uses the existing PnL fact store. A deferred seed does not
    block other research or make a failed request into a correlation failure.
    """
    now = datetime.fromisoformat(observed_at)
    with open_database(database_path) as connection:
        completed = tuple(s for s in list_completed_backtests(connection)
                          if s.task.account_scope == account_scope
                          and (research_task_ids is None or s.task.task_id in research_task_ids))
        references = list_platform_submitted_alphas(connection, account_scope=account_scope)
        states = pnl_capture_states(connection, account_scope=account_scope,
                                    platform_alpha_ids=frozenset(
                                        [r.platform_alpha_id for r in references]
                                        + [s.task.platform_alpha_id for s in completed if s.task.platform_alpha_id]))
        existing = {alpha for alpha, (captured, retry_at) in states.items()
                    if captured or (retry_at is not None and datetime.fromisoformat(retry_at) > now)}
        # Every completed result needs a curve, including failed-quality signals.
        # Eligibility decides research admission, never whether to collect evidence.
        submitted_ids = {r.platform_alpha_id for r in references}
        checked = tuple(s for s in completed if s.task.platform_alpha_id not in submitted_ids)
        required = tuple(dict.fromkeys((
            *(r.platform_alpha_id for r in references),
            *(s.task.platform_alpha_id for s in checked),
        )))
        alpha = next((a for a in required if a and a not in existing), None)
        if alpha is not None:
            # A candidate curve can affect only its own admission. A newly read
            # reference is accounted for by the final seed pass before research resumes.
            candidates = tuple(s for s in checked if admit_seeds and s.task.platform_alpha_id == alpha
                               and (candidate_task_ids is None or s.task.task_id in candidate_task_ids))
        else:
            if not admit_seeds:
                parents = set()
            elif candidate_task_ids is None:
                parents = set(signal_seed_evidence_task_ids(connection))
                mutations = {m.child_task_id: m for m in list_backtest_mutations(connection)}
                # Include roots whose admission was deferred at an earlier boundary.
                parents.update(s.task.task_id for s in completed
                               if s.task.task_id not in mutations
                               or mutations[s.task.task_id].action == DIRECTION_REVERSAL)
            else:
                parents = set(candidate_task_ids)
            candidates = tuple(s for s in completed if admit_seeds and s.task.task_id in parents
                               and assess_signal_seed(s).eligible)
            if admit_seeds:
                synchronize_signal_seeds(connection, observed_at=observed_at, candidate_task_ids=tuple(s.task.task_id for s in candidates))
    if alpha is None:
        return None
    observation = capture_pnl_series(database_path, client, account_scope=account_scope,
        platform_alpha_id=alpha, observed_at=observed_at)
    if observation is None:
        return None
    if observation.points is not None and candidates:
        with open_database(database_path) as connection:
            synchronize_signal_seeds(connection, observed_at=observed_at, candidate_task_ids=tuple(s.task.task_id for s in candidates))
    return observation.retry_after_seconds or 0.0
