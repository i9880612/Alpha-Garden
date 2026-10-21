"""Choose locally safe next submissions; each research lineage offers only its peak."""
from evaluation.correlation import (
    CORRELATION_CALENDAR_YEARS, CORRELATION_CUTOFF, MIN_CORRELATION_INTERVALS, correlation_check,
)
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
    lineage_roots: dict[str, str] | None = None,
    submitted: tuple[BacktestSnapshot, ...] = (),
) -> tuple[str, ...]:
    """Return independent next steps; recompute after each confirmed submission.

    Quality comes first: higher grade, then Fitness, then Sharpe, and only then the
    step that blocks the fewest other eligible formulas. Submitting weaker alphas
    first would bank more of them through the platform's 10% rule, but weak
    submissions lower quality-weighted scores and can block a stronger alpha for
    good. For the same reason a protected formula (still being optimized) holds a
    candidate back only if it is at least as good. This is a deterministic local
    ordering, not a claim of global portfolio optimality. Correlation is pairwise,
    never a transitive family group. The caller must partition accounts and apply
    UI/source filters afterwards.

    When lineage_roots maps task ids to their research lineage root, only the peak
    formula of each lineage is ever submitted. Weaker rungs stay in the qualified
    archive even while the peak is still being optimized, and stay suppressed
    after the peak is submitted: research keeps climbing the ladder, but
    submission no longer banks every rung, so weak submissions neither dilute
    quality-weighted scores nor consume self-correlation headroom needed by
    future peaks. A strictly better descendant still supersedes a submitted peak.
    """
    correlations = correlations or PnlCorrelations(series)
    by_id = {s.task.task_id: s for s in (*candidates, *protected)}

    def correlation(first, second):
        a, b = by_id[first].task, by_id[second].task
        return correlations.correlation((a.account_scope, a.platform_alpha_id),
            (b.account_scope, b.platform_alpha_id), minimum_intervals=MIN_CORRELATION_INTERVALS,
            calendar_years=CORRELATION_CALENDAR_YEARS)

    def after(candidate, submitted):
        return correlation_check(correlation(candidate.task.task_id, submitted.task.task_id),
                                 candidate.result.sharpe, submitted.result.sharpe)

    protected_states = {}
    grades = ("INFERIOR", "AVERAGE", "GOOD", "EXCELLENT", "SPECTACULAR")

    def quality(snapshot):
        result = snapshot.result
        grade = grades.index(result.grade) if result.grade in grades else -1
        return (grade, result.fitness, result.sharpe)

    def preserves(candidate, other):
        if other.task.task_id == candidate.task.task_id or after(other, candidate) == "passed":
            return True
        if quality(other) < quality(candidate):
            return True  # A weaker formula never holds back a stronger one.
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
    if lineage_roots:
        # Peak over candidates, protected, and already submitted: a weaker rung
        # waits in the archive while a better formula of the same lineage is
        # still being optimized, and stays suppressed after the peak is
        # submitted, instead of being banked through the ten-percent rule.
        # A strictly better descendant still supersedes a submitted peak.
        pool = {snapshot.task.task_id: snapshot
                for snapshot in (*eligible, *protected, *submitted)}
        peaks: dict[str, tuple[tuple, tuple, BacktestSnapshot]] = {}
        for task_id, snapshot in pool.items():
            if snapshot.result is None or snapshot.result.grade not in grades:
                continue  # Unknown quality never suppresses a qualified candidate.
            root = lineage_roots.get(task_id, task_id)
            peak_key = (grades.index(snapshot.result.grade), snapshot.result.sharpe,
                        snapshot.result.fitness)
            # Earlier finish wins ties, matching the standing order preference.
            tiebreak = (snapshot.task.finished_at or "", task_id)
            incumbent = peaks.get(root)
            if (incumbent is None or peak_key > incumbent[0]
                    or (peak_key == incumbent[0] and tiebreak < incumbent[1])):
                peaks[root] = (peak_key, tiebreak, snapshot)
        peak_ids = {snapshot.task.task_id for _, _, snapshot in peaks.values()}
        eligible = [snapshot for snapshot in eligible if snapshot.task.task_id in peak_ids]
    while eligible:
        def order(s):
            blocked = sum(other.task.task_id != s.task.task_id and after(other, s) != "passed"
                          for other in eligible)
            grade, fitness, sharpe = quality(s)
            return (-grade, -fitness, -sharpe, blocked, s.task.finished_at, s.task.task_id)
        chosen = min(eligible, key=order)
        result.append(chosen.task.task_id)
        # Related successors wait for this step to be confirmed. Incompatible
        # peers remain history; independent formulas can be offered together.
        eligible = [other for other in eligible if other.task.task_id != chosen.task.task_id
                    and correlation(other.task.task_id, chosen.task.task_id) < CORRELATION_CUTOFF]
    return tuple(result)
