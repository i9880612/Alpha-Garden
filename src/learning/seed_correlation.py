from __future__ import annotations

from dataclasses import dataclass

from learning.pnl import daily_pnl_correlation

from persistence.backtests import BacktestSnapshot
from persistence.pnl import PnlSeriesRecord
from persistence.submissions import PlatformSubmittedAlphaRecord, normalize_submitted_formula


# Seed admission, not a substitute for the platform's submission check.
MAX_SEED_CORRELATION = 0.7
MIN_SEED_CORRELATION_INTERVALS = 252


@dataclass(frozen=True, slots=True)
class SeedCorrelationAssessment:
    state: str
    maximum: float | None
    reference_id: str | None
    compared: int
    required: int


def submitted_seed(
    snapshot: BacktestSnapshot,
    references: tuple[PlatformSubmittedAlphaRecord, ...],
) -> bool:
    formula = normalize_submitted_formula(snapshot.task.formula)
    return any(
        ref.account_scope == snapshot.task.account_scope
        and (ref.platform_alpha_id == snapshot.task.platform_alpha_id
             or ref.normalized_formula == formula)
        for ref in references
    )


def assess_seed_correlation(
    snapshot: BacktestSnapshot,
    references: tuple[PlatformSubmittedAlphaRecord, ...],
    series: tuple[PnlSeriesRecord, ...],
) -> SeedCorrelationAssessment:
    """Compare with every recorded submitted alpha of the same account.

    A known high pair rejects even with incomplete coverage. A pass requires
    all pairs. Missing/flat/insufficient-overlap series are unknown, never zero correlation.
    Submitted parents retain their separate SC-repair role, including self=1.
    """
    refs = tuple(ref for ref in references if ref.account_scope == snapshot.task.account_scope)
    if submitted_seed(snapshot, refs):
        return SeedCorrelationAssessment("submitted", None, None, 0, len(refs))
    by_id = {
        item.platform_alpha_id: item.points for item in series
        if item.account_scope == snapshot.task.account_scope and item.points is not None
    }
    points = by_id.get(snapshot.task.platform_alpha_id)
    measured = []
    if points is not None:
        for ref in refs:
            other = by_id.get(ref.platform_alpha_id)
            value = daily_pnl_correlation(points, other, minimum_intervals=MIN_SEED_CORRELATION_INTERVALS) if other is not None else None
            if value is not None:
                measured.append((value, ref.platform_alpha_id))
    maximum, reference = max(measured, default=(None, None))
    if maximum is not None and maximum >= MAX_SEED_CORRELATION:
        state = "failed"
    elif len(measured) == len(refs):
        state = "passed"
    else:
        state = "pending"
    return SeedCorrelationAssessment(state, maximum, reference, len(measured), len(refs))
