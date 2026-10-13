"""Identify measured near-duplicate signals without changing lineage or budgets."""
from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from datetime import datetime

from learning.pnl import PnlCorrelations
from learning.quality import QUALITY_GRADES
from persistence.backtests import BacktestSnapshot
from persistence.pnl import PnlSeriesRecord


MIN_DUPLICATE_CORRELATION = 0.9999
MIN_DUPLICATE_INTERVALS = 252


@dataclass(frozen=True, slots=True)
class SignalDuplicate:
    task_id: str
    representative_task_id: str
    correlation: float


def find_signal_duplicates(
    candidates: tuple[BacktestSnapshot, ...],
    series: tuple[PnlSeriesRecord, ...],
    *,
    comparison_groups: Mapping[str, Hashable],
    correlations: PnlCorrelations | None = None,
) -> tuple[SignalDuplicate, ...]:
    """Compare directly to retained representatives, never transitively.

    Callers supply their research scope and include exhausted representatives.
    Missing grades cannot establish a preference against known grades.
    """
    available = {(record.account_scope, record.platform_alpha_id)
                 for record in series if record.points is not None}
    correlations = correlations or PnlCorrelations(series)
    representatives: dict[tuple, list[BacktestSnapshot]] = {}
    duplicates = []
    for candidate in sorted(candidates, key=_representative_order):
        task = candidate.task
        if (task.account_scope, task.platform_alpha_id) not in available:
            continue
        group = (task.account_scope, task.settings_json, comparison_groups[task.task_id],
                 candidate.result.grade in QUALITY_GRADES)
        kept = representatives.setdefault(group, [])
        for representative in kept:
            other = representative.task
            correlation = correlations.correlation(
                (task.account_scope, task.platform_alpha_id), (other.account_scope, other.platform_alpha_id),
                minimum_intervals=MIN_DUPLICATE_INTERVALS,
            )
            if correlation is not None and correlation >= MIN_DUPLICATE_CORRELATION:
                duplicates.append(SignalDuplicate(task.task_id, other.task_id, correlation))
                break
        else:
            kept.append(candidate)
    return tuple(duplicates)


def _representative_order(snapshot: BacktestSnapshot) -> tuple:
    result = snapshot.result
    return (
        -QUALITY_GRADES.index(result.grade) if result.grade in QUALITY_GRADES else 1,
        -result.sharpe, -result.fitness, datetime.fromisoformat(snapshot.task.finished_at), snapshot.task.task_id,
    )
