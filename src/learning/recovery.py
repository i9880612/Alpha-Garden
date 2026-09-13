from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from generation.parser import parse_formula
from generation.catalog import GenerationCatalog
from generation.self_correlation import (
    SELF_CORRELATION_REPAIR_FAMILIES, matches_self_correlation_repair,
)
from learning.seeds import assess_signal_seed
from learning.pnl import daily_pnl_correlation
from learning.self_correlation import SelfCorrelationReference
from persistence.backtests import BacktestMutationRecord, BacktestSnapshot
from persistence.pnl import PnlSeriesRecord


# A research-evidence floor, not the platform's SC calculation or threshold.
MIN_RECOVERY_INTERVALS = 252


@dataclass(frozen=True, slots=True)
class RecoveryComparison:
    task_id: str
    repair_task_id: str
    original_parent_task_id: str
    account_scope: str
    child_alpha_id: str
    parent_alpha_id: str
    reference_alpha_id: str


def recovery_comparisons(
    snapshots: tuple[BacktestSnapshot, ...],
    mutations: tuple[BacktestMutationRecord, ...],
    references: tuple[SelfCorrelationReference, ...],
    *, catalog: GenerationCatalog | None = None,
) -> tuple[RecoveryComparison, ...]:
    """Only descendants of a verifiable repair against the original conflict."""
    by_task = {item.task.task_id: item for item in snapshots}
    by_child = {item.child_task_id: item for item in mutations}
    by_parent = {item.parent_task_id: item for item in references}
    comparisons = []
    for child in snapshots:
        if not child.task.platform_alpha_id or not assess_signal_seed(child).eligible:
            continue
        current = child
        seen = set()
        while current.task.task_id in by_child:
            task = current.task
            if task.task_id in seen:
                raise ValueError("recovery_lineage_cycle")
            seen.add(task.task_id)
            mutation = by_child[task.task_id]
            parent = by_task.get(mutation.parent_task_id)
            if (
                parent is None
                or parent.task.account_scope != child.task.account_scope
                or parent.task.settings_json != child.task.settings_json
                or not parent.task.finished_at
                or not task.finished_at
                or datetime.fromisoformat(parent.task.finished_at)
                > datetime.fromisoformat(task.finished_at)
                or not assess_signal_seed(parent).eligible
            ):
                break
            if mutation.action in SELF_CORRELATION_REPAIR_FAMILIES:
                reference = by_parent.get(parent.task.task_id)
                if (
                    reference is None
                    or not reference.platform_alpha_id
                    or not parent.task.platform_alpha_id
                ):
                    break
                settings = json.loads(parent.task.settings_json)
                matching_catalog = catalog
                if catalog is not None and tuple(settings.get(k) for k in (
                    "instrumentType", "region", "universe", "delay",
                )) != (catalog.context.instrument_type, catalog.context.region,
                       catalog.context.universe, catalog.context.delay):
                    matching_catalog = None
                if (
                    mutation.before != parent.task.formula
                    or mutation.after != task.formula
                    or not matches_self_correlation_repair(
                        parse_formula(parent.task.formula).expression,
                        parse_formula(reference.formula).expression,
                        parse_formula(task.formula).expression,
                        action=mutation.action, location=mutation.location,
                        catalog=matching_catalog,
                    )
                ):
                    break
                comparisons.append(
                    RecoveryComparison(
                        child.task.task_id,
                        task.task_id,
                        parent.task.task_id,
                        task.account_scope,
                        child.task.platform_alpha_id,
                        parent.task.platform_alpha_id,
                        reference.platform_alpha_id,
                    )
                )
                break
            current = parent
    return tuple(comparisons)


def recovery_correlations(
    comparison: RecoveryComparison,
    series: tuple[PnlSeriesRecord, ...],
) -> tuple[float, float] | None:
    """Compare aligned daily increments against the same original conflict.

    This is a pairwise research observation, never an estimate of full-library SC.
    No zero filling, unequal interval lengths, or cumulative-level correlation.
    """
    by_id = {
        item.platform_alpha_id: item.points
        for item in series
        if item.account_scope == comparison.account_scope and item.points is not None
    }
    ids = (
        comparison.parent_alpha_id,
        comparison.child_alpha_id,
        comparison.reference_alpha_id,
    )
    if any(alpha not in by_id for alpha in ids):
        return None
    points = [by_id[alpha] for alpha in ids]
    dates = [tuple(day for day, _ in item) for item in points]
    if (
        len(dates[0]) - 1 < MIN_RECOVERY_INTERVALS
        or not dates[0] == dates[1] == dates[2]
    ):
        return None
    parent = daily_pnl_correlation(points[0], points[2], minimum_intervals=MIN_RECOVERY_INTERVALS)
    child = daily_pnl_correlation(points[1], points[2], minimum_intervals=MIN_RECOVERY_INTERVALS)
    return (parent, child) if parent is not None and child is not None else None


def recovery_task_ids(
    comparisons: tuple[RecoveryComparison, ...], series: tuple[PnlSeriesRecord, ...]
) -> frozenset[str]:
    return frozenset(
        item.task_id
        for item in comparisons
        if (correlations := recovery_correlations(item, series)) is not None
        and abs(correlations[1]) < abs(correlations[0]) - 1e-12
    )
