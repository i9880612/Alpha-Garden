from __future__ import annotations

from dataclasses import dataclass

from evaluation.correlation import (
    CORRELATION_CUTOFF, MIN_CORRELATION_INTERVALS, correlation_check, submitted_sharpe,
)

from learning.pnl import PnlCorrelations

from persistence.backtests import BacktestSnapshot
from persistence.pnl import PnlSeriesRecord
from persistence.submissions import PlatformSubmittedAlphaRecord, normalize_submitted_formula


# Seed admission, not a substitute for the platform's submission check.
MAX_SEED_CORRELATION = CORRELATION_CUTOFF
MIN_SEED_CORRELATION_INTERVALS = MIN_CORRELATION_INTERVALS


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
    *, correlations: PnlCorrelations | None = None,
) -> SeedCorrelationAssessment:
    """Compare with every recorded submitted alpha of the same account.

    A known blocking pair rejects even with incomplete coverage. A pass requires
    all pairs. Missing/flat/insufficient-overlap series are unknown, never zero correlation.
    Submitted parents retain their separate SC-repair role, including self=1.
    """
    refs = tuple(ref for ref in references if ref.account_scope == snapshot.task.account_scope)
    if submitted_seed(snapshot, refs):
        return SeedCorrelationAssessment("submitted", None, None, 0, len(refs))
    correlations = correlations or PnlCorrelations(series)
    measured = []
    states = []
    for ref in refs:
        value = correlations.correlation(
            (snapshot.task.account_scope, snapshot.task.platform_alpha_id),
            (ref.account_scope, ref.platform_alpha_id), minimum_intervals=MIN_SEED_CORRELATION_INTERVALS)
        if value is not None:
            measured.append((value, ref.platform_alpha_id))
        states.append(correlation_check(value,
            snapshot.result.sharpe if snapshot.result is not None else None,
            submitted_sharpe(ref.raw_payload)))
    maximum, reference = max(measured, default=(None, None))
    if "failed" in states:
        state = "failed"
    elif len(states) == len(refs) and all(state == "passed" for state in states):
        state = "passed"
    else:
        state = "pending"
    return SeedCorrelationAssessment(state, maximum, reference, len(measured), len(refs))
