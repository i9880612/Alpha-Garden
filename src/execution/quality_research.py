"""Read and reserve quality allowances from task provenance, never mutable counters."""
from __future__ import annotations

import sqlite3

from learning.quality_budget import QualityBudget, quality_budgets
from persistence.backtests import BacktestMutationRecord, BacktestSnapshot, list_backtest_mutations
from persistence.quality_research import list_quality_research_tasks, record_quality_research_task
from persistence.seeds import list_signal_seeds


def load_quality_budgets(
    connection: sqlite3.Connection, candidates: tuple[BacktestSnapshot, ...], *,
    mutations: tuple[BacktestMutationRecord, ...] | None = None,
) -> dict[str, QualityBudget]:
    return quality_budgets(
        candidates, list_backtest_mutations(connection) if mutations is None else mutations,
        frozenset(seed.root_task_id for seed in list_signal_seeds(connection)),
        list_quality_research_tasks(connection),
    )


def reserve_quality_task(
    connection: sqlite3.Connection, *, parent: BacktestSnapshot, child_task_id: str, root_task_id: str,
) -> None:
    budget = load_quality_budgets(connection, (parent,))[parent.task.task_id]
    if budget.root_task_id != root_task_id:
        raise ValueError("quality_research_lineage_mismatch")
    existing = connection.execute("SELECT root_task_id FROM quality_research_tasks WHERE task_id=?", (child_task_id,)).fetchone()
    if existing is None and budget.available <= 0:
        raise ValueError("quality_lineage_budget_exhausted")
    record_quality_research_task(connection, task_id=child_task_id, root_task_id=root_task_id)
