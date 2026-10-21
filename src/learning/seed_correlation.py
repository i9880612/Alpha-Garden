from __future__ import annotations

from dataclasses import dataclass

from evaluation.correlation import (
    CORRELATION_CALENDAR_YEARS, CORRELATION_CUTOFF, MIN_CORRELATION_INTERVALS, correlation_check,
    sharpe_improves, submitted_sharpe,
)

from learning.pnl import PnlCorrelations

from persistence.backtests import BacktestSnapshot
from persistence.pnl import PnlSeriesRecord
from persistence.submissions import PlatformSubmittedAlphaRecord, normalize_submitted_formula


# Seed admission, not a substitute for the platform's submission check.
MAX_SEED_CORRELATION = CORRELATION_CUTOFF
MIN_SEED_CORRELATION_INTERVALS = MIN_CORRELATION_INTERVALS


def assess_seed_redundancy(
    snapshot: BacktestSnapshot,
    seeds: tuple[BacktestSnapshot, ...],
    *, correlations: PnlCorrelations,
) -> str:
    """Require usable daily evidence before granting an independent seed root.

    Unlike the platform's submission check, a higher Sharpe does not excuse a
    correlation at or above MAX_SEED_CORRELATION: every seed root gets its own
    mutation budget, and near-copies of one idea would split it over children that
    block each other at submission. A stronger version of an existing idea can still
    take over inside that seed's lineage.
    """
    task = snapshot.task
    key = (task.account_scope, task.platform_alpha_id)
    if correlations.correlation(key, key, minimum_intervals=MIN_SEED_CORRELATION_INTERVALS,
                                calendar_years=CORRELATION_CALENDAR_YEARS) is None:
        return "pending"
    states = []
    for seed in seeds:
        if seed.task.task_id == task.task_id or seed.task.account_scope != task.account_scope:
            continue
        value = correlations.correlation(
            key, (seed.task.account_scope, seed.task.platform_alpha_id),
            minimum_intervals=MIN_SEED_CORRELATION_INTERVALS,
            calendar_years=CORRELATION_CALENDAR_YEARS,
        )
        states.append(seed_overlap_check(value))
    if "failed" in states:
        return "failed"
    return "pending" if "pending" in states else "passed"


def blocking_seeds(
    snapshot: BacktestSnapshot,
    seeds: tuple[BacktestSnapshot, ...],
    *, correlations: PnlCorrelations,
) -> tuple[BacktestSnapshot, ...]:
    """The seeds this candidate overlaps at or above MAX_SEED_CORRELATION."""
    key = (snapshot.task.account_scope, snapshot.task.platform_alpha_id)
    return tuple(
        seed for seed in seeds
        if seed.task.task_id != snapshot.task.task_id
        and seed.task.account_scope == snapshot.task.account_scope
        and seed_overlap_check(correlations.correlation(
            key, (seed.task.account_scope, seed.task.platform_alpha_id),
            minimum_intervals=MIN_SEED_CORRELATION_INTERVALS,
            calendar_years=CORRELATION_CALENDAR_YEARS)) == "failed"
    )


def clearly_stronger(candidate: BacktestSnapshot, seed: BacktestSnapshot) -> bool:
    """At least 10% more Sharpe (the platform's own bar) and no worse Fitness."""
    if candidate.result is None or seed.result is None:
        return False
    return bool(sharpe_improves(abs(candidate.result.sharpe), abs(seed.result.sharpe))) and (
        abs(candidate.result.fitness) >= abs(seed.result.fitness))


def seed_overlap_check(correlation: float | None) -> str:
    if correlation is None:
        return "pending"
    return "failed" if correlation >= MAX_SEED_CORRELATION else "passed"


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
            (ref.account_scope, ref.platform_alpha_id), minimum_intervals=MIN_SEED_CORRELATION_INTERVALS,
            calendar_years=CORRELATION_CALENDAR_YEARS)
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
