"""Choose locally safe next submissions without discarding improvement steps."""
from evaluation.correlation import CORRELATION_CUTOFF, MIN_CORRELATION_INTERVALS, correlation_check
from learning.pnl import PnlCorrelations
from learning.seed_correlation import assess_seed_correlation
from persistence.backtests import BacktestSnapshot
from persistence.pnl import PnlSeriesRecord
from persistence.submissions import PlatformSubmittedAlphaRecord


def select_submission_opportunities(
    candidates: tuple[BacktestSnapshot, ...],
    protected: tuple[BacktestSnapshot, ...],
    references: tuple[PlatformSubmittedAlphaRecord, ...],
    series: tuple[PnlSeriesRecord, ...],
    *, correlations: PnlCorrelations | None = None,
) -> tuple[str, ...]:
    """Return independent next steps; recompute after each confirmed submission.

    Prefer the step that blocks the fewest currently eligible formulas, breaking
    ties by quality. This is a deterministic local ordering, not a claim of global
    portfolio optimality. Correlation is pairwise, never a transitive family group.
    The caller must partition accounts and apply UI/source filters afterwards.
    """
    correlations = correlations or PnlCorrelations(series)
    by_id = {s.task.task_id: s for s in (*candidates, *protected)}

    def correlation(first, second):
        a, b = by_id[first].task, by_id[second].task
        return correlations.correlation((a.account_scope, a.platform_alpha_id),
            (b.account_scope, b.platform_alpha_id), minimum_intervals=MIN_CORRELATION_INTERVALS)

    def after(candidate, submitted):
        return correlation_check(correlation(candidate.task.task_id, submitted.task.task_id),
                                 candidate.result.sharpe, submitted.result.sharpe)

    protected_states = {}

    def preserves(candidate, other):
        if other.task.task_id == candidate.task.task_id or after(other, candidate) == "passed":
            return True
        # Only inspect the other's submitted-reference eligibility when this
        # candidate would block it. Unknown evidence remains protected.
        task_id = other.task.task_id
        if task_id not in protected_states:
            protected_states[task_id] = assess_seed_correlation(
                other, references, series, correlations=correlations).state
        return protected_states[task_id] == "failed"

    eligible = [s for s in candidates if all(preserves(s, other) for other in protected)]
    eligible = [s for s in eligible if assess_seed_correlation(
        s, references, series, correlations=correlations).state == "passed"]
    # A missing pair cannot establish that either order preserves the other.
    eligible = [s for s in eligible if all(other.task.task_id == s.task.task_id
                or correlation(s.task.task_id, other.task.task_id) is not None for other in eligible)]
    result = []
    grades = ("INFERIOR", "AVERAGE", "GOOD", "EXCELLENT", "SPECTACULAR")
    while eligible:
        def order(s):
            blocked = sum(other.task.task_id != s.task.task_id and after(other, s) != "passed"
                          for other in eligible)
            return (blocked, -grades.index(s.result.grade), -s.result.sharpe, -s.result.fitness,
                    s.task.finished_at, s.task.task_id)
        chosen = min(eligible, key=order)
        result.append(chosen.task.task_id)
        # Related successors wait for this step to be confirmed. Incompatible
        # peers remain history; independent formulas can be offered together.
        eligible = [other for other in eligible if other.task.task_id != chosen.task.task_id
                    and correlation(other.task.task_id, chosen.task.task_id) < CORRELATION_CUTOFF]
    return tuple(result)
