from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from learning.frontiers import PARENT_ATTEMPT_BUDGET, count_parent_attempts
from learning.pnl import PnlCorrelations
from persistence.backtests import BacktestMutationRecord, BacktestSnapshot
from persistence.pnl import PnlSeriesRecord


MIN_DUPLICATE_CORRELATION = 0.9999
MIN_DUPLICATE_INTERVALS = 252
_GRADES = ("INFERIOR", "AVERAGE", "GOOD", "EXCELLENT", "SPECTACULAR")


@dataclass(frozen=True, slots=True)
class QualifiedEvolutionRecord:
    task_id: str
    remaining_attempts: int
    replacement_task_id: str | None

    @property
    def retired(self) -> bool:
        return self.remaining_attempts == 0 or self.replacement_task_id is not None


def assess_qualified_evolution(
    checked: tuple[BacktestSnapshot, ...],
    mutations: tuple[BacktestMutationRecord, ...],
    attempted_child_task_ids: frozenset[str],
    series: tuple[PnlSeriesRecord, ...] = (),
    *, correlations: PnlCorrelations | None = None,
) -> tuple[QualifiedEvolutionRecord, ...]:
    """Keep individual budgets while retiring improvements and measured duplicates."""
    by_task = {snapshot.task.task_id: snapshot for snapshot in checked}
    parents = {mutation.child_task_id: mutation.parent_task_id for mutation in mutations}
    attempts = count_parent_attempts(mutations, attempted_child_task_ids)
    replacements: dict[str, str] = {}
    improvements: list[tuple[str, str]] = []
    roots: dict[str, str] = {}
    for child in sorted(checked, key=lambda s: (-s.result.sharpe, s.task.finished_at, s.task.task_id)):
        task_id = child.task.task_id
        visited = {task_id}
        while task_id in parents:
            task_id = parents[task_id]
            if task_id in visited:
                raise ValueError("qualified_evolution_cycle_detected")
            visited.add(task_id)
            parent = by_task.get(task_id)
            if parent is not None and (
                parent.task.account_scope == child.task.account_scope
                and parent.task.settings_json == child.task.settings_json
                and child.result.sharpe > parent.result.sharpe
            ):
                improvements.append((task_id, child.task.task_id))
        roots[child.task.task_id] = task_id

    points = {(record.account_scope, record.platform_alpha_id): record.points
              for record in series if record.points is not None}
    correlations = correlations or PnlCorrelations(series)
    representatives: dict[tuple[str, str, str], list[BacktestSnapshot]] = {}
    for candidate in sorted(checked, key=_representative_order):
        task = candidate.task
        candidate_points = points.get((task.account_scope, task.platform_alpha_id))
        if candidate_points is None:
            continue
        family = (roots[task.task_id], task.account_scope, task.settings_json)
        kept = representatives.setdefault(family, [])
        for representative in kept:
            # Missing historical grades cannot establish a cross-grade preference.
            if ((candidate.result.grade in _GRADES) != (representative.result.grade in _GRADES)):
                continue
            other = representative.task
            correlation = correlations.correlation(
                (task.account_scope, task.platform_alpha_id), (other.account_scope, other.platform_alpha_id),
                minimum_intervals=MIN_DUPLICATE_INTERVALS,
            )
            if correlation is not None and correlation >= MIN_DUPLICATE_CORRELATION:
                replacements[task.task_id] = other.task_id
                break
        else:
            # Exhausted/submitted representatives still suppress equivalent fresh budgets.
            # Compare directly with a retained representative, never by transitive similarity.
            kept.append(candidate)

    duplicates = frozenset(replacements)
    for parent_id, child_id in improvements:
        # A redundant child must not retire its preferred representative or revive
        # a signal through an improvement chain. Distinct descendants keep the prior rule.
        if parent_id not in duplicates and child_id not in duplicates:
            replacements.setdefault(parent_id, child_id)
    return tuple(
        QualifiedEvolutionRecord(
            task_id=task_id,
            remaining_attempts=max(0, PARENT_ATTEMPT_BUDGET - attempts.get(task_id, 0)),
            replacement_task_id=replacements.get(task_id),
        )
        for task_id in sorted(by_task)
    )


def _representative_order(snapshot: BacktestSnapshot) -> tuple:
    result = snapshot.result
    return (
        -_GRADES.index(result.grade) if result.grade in _GRADES else 1,
        -result.sharpe, -result.fitness, datetime.fromisoformat(snapshot.task.finished_at), snapshot.task.task_id,
    )
