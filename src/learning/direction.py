from __future__ import annotations

from learning.seeds import SIGNAL_MIN_FITNESS, SIGNAL_MIN_SHARPE, assess_signal_seed
from persistence.backtests import BacktestSnapshot


_POSITIVE_FLOOR_REJECTIONS = frozenset(
    {
        "signal_seed_sharpe_below_positive_threshold",
        "signal_seed_fitness_below_positive_threshold",
    }
)


def negative_direction_is_testable(snapshot: BacktestSnapshot) -> bool:
    """Select a direction question, not a positive seed or a predicted result."""
    if (
        snapshot.task.status != "completed"
        or snapshot.result is None
        or snapshot.task.platform_alpha_id is None
        or snapshot.task.submission_started_at is None
        or snapshot.task.finished_at is None
    ):
        return False
    if not (
        snapshot.result.sharpe <= -SIGNAL_MIN_SHARPE
        and snapshot.result.fitness <= -SIGNAL_MIN_FITNESS
    ):
        return False
    # Reuse every existing non-numeric safety rejection without altering facts.
    return not (
        set(assess_signal_seed(snapshot).rejection_reasons) - _POSITIVE_FLOOR_REJECTIONS
    )
