"""One quality allowance per account, settings and seed lineage across runs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from learning.quality import QUALITY_LINEAGE_ATTEMPT_BUDGET
from persistence.backtests import BacktestMutationRecord, BacktestSnapshot
from persistence.quality_research import QualityResearchTask


@dataclass(frozen=True, slots=True)
class QualityBudget:
    root_task_id: str
    attempted: int
    reserved: int

    @property
    def available(self) -> int:
        return max(0, self.unspent - self.reserved)

    @property
    def unspent(self) -> int:
        return max(0, QUALITY_LINEAGE_ATTEMPT_BUDGET - self.attempted)


def quality_budgets(
    candidates: tuple[BacktestSnapshot, ...], mutations: tuple[BacktestMutationRecord, ...],
    seed_roots: frozenset[str], tasks: tuple[QualityResearchTask, ...],
) -> dict[str, QualityBudget]:
    parents = {item.child_task_id: item.parent_task_id for item in mutations}
    roots: dict[str, str] = {}
    def root(task_id):
        if task_id in roots:
            return roots[task_id]
        visited = set()
        current = task_id
        while current not in seed_roots and current in parents:
            if current in visited:
                raise ValueError("quality_budget_lineage_cycle")
            visited.add(current)
            current = parents[current]
        roots[task_id] = current
        return current
    attempted, reserved = Counter(), Counter()
    for task in tasks:
        identity = (task.account_scope, task.settings_json, task.root_task_id or root(task.task_id))
        (attempted if task.attempted else reserved)[identity] += 1
    result = {}
    for candidate in candidates:
        task = candidate.task
        lineage_root = root(task.task_id)
        key = (task.account_scope, task.settings_json, lineage_root)
        result[task.task_id] = QualityBudget(lineage_root, attempted[key], reserved[key])
    return result
