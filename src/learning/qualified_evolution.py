from __future__ import annotations

from dataclasses import dataclass

from learning.frontiers import PARENT_ATTEMPT_BUDGET, count_parent_attempts
from learning.pnl import PnlCorrelations
from learning.signal_duplicates import find_signal_duplicates
from persistence.backtests import BacktestMutationRecord, BacktestSnapshot
from persistence.pnl import PnlSeriesRecord


@dataclass(frozen=True, slots=True)
class QualifiedEvolutionRecord:
    task_id: str
    remaining_attempts: int
    replacement_task_id: str | None
    parent_attempt_count: int = 0

    @property
    def retired(self) -> bool:
        return self.remaining_attempts == 0 or self.replacement_task_id is not None


def assess_qualified_evolution(
    checked: tuple[BacktestSnapshot, ...],
    mutations: tuple[BacktestMutationRecord, ...],
    attempted_child_task_ids: frozenset[str],
    series: tuple[PnlSeriesRecord, ...] = (),
    *, correlations: PnlCorrelations | None = None,
    quality_remaining: dict[str, int] | None = None,
) -> tuple[QualifiedEvolutionRecord, ...]:
    """Keep individual budgets while retiring improvements and measured duplicates."""
    by_task = {snapshot.task.task_id: snapshot for snapshot in checked}
    parents = {mutation.child_task_id: mutation.parent_task_id for mutation in mutations}
    attempts = count_parent_attempts(mutations, attempted_child_task_ids)
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

    replacements = {record.task_id: record.representative_task_id for record in find_signal_duplicates(
        checked, series, comparison_groups=roots, correlations=correlations,
    )}

    duplicates = frozenset(replacements)
    for parent_id, child_id in improvements:
        # A redundant child must not retire its preferred representative or revive
        # a signal through an improvement chain. Distinct descendants keep the prior rule.
        if parent_id not in duplicates and child_id not in duplicates:
            replacements.setdefault(parent_id, child_id)
    return tuple(
        QualifiedEvolutionRecord(
            task_id=task_id,
            remaining_attempts=min(max(0, PARENT_ATTEMPT_BUDGET - attempts.get(task_id, 0)),
                                   quality_remaining[task_id] if quality_remaining is not None else PARENT_ATTEMPT_BUDGET),
            replacement_task_id=replacements.get(task_id),
            parent_attempt_count=attempts.get(task_id, 0),
        )
        for task_id in sorted(by_task)
    )
