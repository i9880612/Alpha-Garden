"""Pick companions for combo enhancement from saved backtests, within fixed bounds.

Companions are single-template signals (the raw material combos are made of) with
a daily PnL curve. Near-copies of one idea count once, and each companion joins at
most MAX_LINEAGES_PER_COMPANION parents over all history, so a handful of strong
companions cannot end up inside every lineage and make the lineages resemble each other.
"""
from __future__ import annotations

import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping

from generation.catalog import GenerationCatalog
from generation.combo_enhancement import (
    COMBO_ENHANCEMENT,
    BacktestMetrics,
    MIN_COMPANION_ABS_SHARPE,
    ComboEnhancementLeaf,
    EnhancementSignal,
    companion_marker,
    iter_combo_enhancement_leaves,
)
from generation.ideas import COMBO_TEMPLATE_ID
from generation.parser import FormulaSyntaxError, parse_formula
from persistence.backtests import BacktestMutationRecord, BacktestSnapshot
from persistence.pnl import list_pnl_series

MAX_COMPANIONS = 60
MAX_LINEAGES_PER_COMPANION = 3
# Companions this correlated are one idea written differently (the platform's line).
SAME_IDEA_CORRELATION = 0.7
_MIN_OVERLAP_DAYS = 250


def load_combo_enhancement_leaves(
    connection: sqlite3.Connection,
    catalog: GenerationCatalog,
    *,
    account_scope: str,
    parents: tuple[BacktestSnapshot, ...],
    mutations: tuple[BacktestMutationRecord, ...],
) -> dict[str, tuple[ComboEnhancementLeaf, ...]]:
    """Leaves per parent task id; parents nearest a higher grade (highest Fitness) pick first."""
    parents = tuple(p for p in parents if p.result is not None and p.task.platform_alpha_id)
    if not parents:
        return {}
    companions = _companions(connection, account_scope)
    if not companions:
        return {}
    parent_pnl = _daily_series(connection, account_scope,
                               frozenset(p.task.platform_alpha_id for p in parents))
    used: dict[str, set[str]] = defaultdict(set)
    markers = {companion.alpha_id: companion_marker(companion.expression) for companion in companions}
    for mutation in mutations:
        if mutation.action != COMBO_ENHANCEMENT:
            continue
        for alpha_id, marker in markers.items():
            if marker in mutation.after:
                used[alpha_id].add(mutation.parent_task_id)
    offered: Counter[str] = Counter()
    leaves: dict[str, tuple[ComboEnhancementLeaf, ...]] = {}
    for parent in sorted(parents, key=lambda p: (-abs(p.result.fitness), p.task.task_id)):
        daily = parent_pnl.get(parent.task.platform_alpha_id)
        if not daily:
            continue
        try:
            expression = parse_formula(parent.task.formula).expression
        except FormulaSyntaxError:
            continue
        available = tuple(
            companion for companion in companions
            # A pair this parent already tried would only repeat its saved result.
            if parent.task.task_id not in used[companion.alpha_id]
            and len(used[companion.alpha_id]) + offered[companion.alpha_id] < MAX_LINEAGES_PER_COMPANION
        )
        result = parent.result
        found = tuple(iter_combo_enhancement_leaves(
            expression, EnhancementSignal(parent.task.platform_alpha_id, expression, daily, BacktestMetrics(
                result.sharpe, result.fitness, result.returns, result.turnover)),
            available, catalog))
        offered.update(leaf.companion_alpha_id for leaf in found)
        if found:
            leaves[parent.task.task_id] = found
    return leaves


def _companions(connection: sqlite3.Connection, account_scope: str) -> tuple[EnhancementSignal, ...]:
    rows = connection.execute(
        """SELECT t.formula, t.platform_alpha_id, r.sharpe, r.fitness, r.returns, r.turnover
           FROM formula_ideas i
           JOIN backtest_tasks t ON t.task_id = i.task_id
           JOIN backtest_results r ON r.task_id = i.task_id
           WHERE t.account_scope = ? AND t.status = 'completed'
             AND t.platform_alpha_id IS NOT NULL AND i.template_id != ?
             AND abs(r.sharpe) >= ?""",
        (account_scope, COMBO_TEMPLATE_ID, MIN_COMPANION_ABS_SHARPE),
    ).fetchall()
    rows = sorted(rows, key=lambda row: (-abs(row[2]), row[1]))
    series = _daily_series(connection, account_scope, frozenset(row[1] for row in rows))
    leaders: list[EnhancementSignal] = []
    for formula, alpha_id, sharpe, fitness, returns, turnover in rows:
        daily = series.get(alpha_id)
        if not daily:
            continue
        try:
            expression = parse_formula(formula).expression
        except FormulaSyntaxError:
            continue
        if any((value := _correlation(daily, leader.daily_pnl)) is not None
               and abs(value) > SAME_IDEA_CORRELATION for leader in leaders):
            continue
        metrics = (BacktestMetrics(sharpe, fitness, returns, turnover)
                   if None not in (fitness, returns, turnover) else None)
        leaders.append(EnhancementSignal(alpha_id, expression, daily, metrics))
        if len(leaders) == MAX_COMPANIONS:
            break
    return tuple(leaders)


def _daily_series(connection, account_scope, alpha_ids) -> dict[str, dict[str, float]]:
    if not alpha_ids:
        return {}
    return {
        record.platform_alpha_id: {
            record.points[i][0]: record.points[i][1] - record.points[i - 1][1]
            for i in range(1, len(record.points))
        }
        for record in list_pnl_series(connection, platform_alpha_ids=alpha_ids, account_scope=account_scope)
        if record.points
    }


def _correlation(first: Mapping[str, float], second: Mapping[str, float]) -> float | None:
    days = sorted(set(first) & set(second))
    if len(days) < _MIN_OVERLAP_DAYS:
        return None
    x = [first[day] for day in days]
    y = [second[day] for day in days]
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((v - mean_x) ** 2 for v in x)
    syy = sum((v - mean_y) ** 2 for v in y)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / math.sqrt(sxx * syy)
