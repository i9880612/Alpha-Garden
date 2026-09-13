from __future__ import annotations

from dataclasses import dataclass

from learning.frontiers import PARENT_ATTEMPT_BUDGET, count_parent_attempts
from persistence.backtests import BacktestMutationRecord, BacktestSnapshot


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
) -> tuple[QualifiedEvolutionRecord, ...]:
    """Each checked formula owns its budget; only its better descendants replace it."""
    by_task = {snapshot.task.task_id: snapshot for snapshot in checked}
    parents = {mutation.child_task_id: mutation.parent_task_id for mutation in mutations}
    attempts = count_parent_attempts(mutations, attempted_child_task_ids)
    replacements: dict[str, str] = {}
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
                replacements.setdefault(task_id, child.task.task_id)
    return tuple(
        QualifiedEvolutionRecord(
            task_id=task_id,
            remaining_attempts=max(0, PARENT_ATTEMPT_BUDGET - attempts.get(task_id, 0)),
            replacement_task_id=replacements.get(task_id),
        )
        for task_id in sorted(by_task)
    )
