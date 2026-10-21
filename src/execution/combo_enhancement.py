"""Pick companions for combo enhancement from saved backtests, within fixed bounds.

Companions are single-template signals (the raw material combos are made of) with
a daily PnL curve. Near-copies of one idea count once, and each companion joins at
most MAX_LINEAGES_PER_COMPANION parents over all history, so a handful of strong
companions cannot end up inside every lineage.

That cap alone does not stop two lineages from meeting in the middle: a price signal
plus a fundamental companion and that fundamental signal plus a price companion are
the same alpha (the first real run made two pairs at correlation 1.0). A child's PnL
is close to parent + weight * companion (expected and real Fitness agreed within
about 0.1), so before any backtest the predicted curve is compared with every checked
alpha of other lineages, every submitted alpha and the children already chosen this
cycle; at the platform's self-correlation line or above, the pair is skipped and the
next best one takes its place.
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from generation.catalog import GenerationCatalog
from generation.combo_enhancement import (
    COMBO_ENHANCEMENT,
    BacktestMetrics,
    MIN_COMPANION_ABS_SHARPE,
    ComboEnhancementLeaf,
    EnhancementSignal,
    companion_markers,
    iter_combo_enhancement_leaves,
)
from generation.ideas import COMBO_TEMPLATE_ID
from generation.parser import FormulaSyntaxError, parse_formula
from persistence.backtests import BacktestMutationRecord, BacktestSnapshot
from persistence.pnl import list_pnl_series
from persistence.submission_checks import list_submission_checks
from persistence.submissions import list_platform_submitted_alphas
from submission.formal import assess_formal_check_payload

MAX_COMPANIONS = 60
MAX_LINEAGES_PER_COMPANION = 3
# The platform's self-correlation line: above it two alphas block each other at submission.
MAX_CROSS_LINEAGE_CORRELATION = 0.7
# Companions this correlated are one idea written differently (the platform's line).
SAME_IDEA_CORRELATION = 0.7
_MIN_OVERLAP_DAYS = 250


@dataclass(frozen=True, slots=True)
class ComboEnhancementPlan:
    leaves: dict[str, tuple[ComboEnhancementLeaf, ...]]
    # Pairs skipped because their predicted curve copies another lineage.
    skipped_as_duplicates: int = 0


def load_combo_enhancement_leaves(
    connection: sqlite3.Connection,
    catalog: GenerationCatalog,
    *,
    account_scope: str,
    parents: tuple[BacktestSnapshot, ...],
    mutations: tuple[BacktestMutationRecord, ...],
) -> ComboEnhancementPlan:
    """Leaves per parent task id; parents nearest a higher grade (highest Fitness) pick first."""
    parents = tuple(p for p in parents if p.result is not None and p.task.platform_alpha_id)
    if not parents:
        return ComboEnhancementPlan({})
    companions = _companions(connection, account_scope)
    if not companions:
        return ComboEnhancementPlan({})
    parent_pnl = _daily_series(connection, account_scope,
                               frozenset(p.task.platform_alpha_id for p in parents))
    used: dict[str, set[str]] = defaultdict(set)
    markers = {companion.alpha_id: companion_markers(companion.expression) for companion in companions}
    for mutation in mutations:
        if mutation.action != COMBO_ENHANCEMENT:
            continue
        for alpha_id, texts in markers.items():
            # Added by this step: in the child but not already in the parent.
            if any(text in mutation.after and text not in mutation.before for text in texts):
                used[alpha_id].add(mutation.parent_task_id)
    lineage = _lineage_roots(mutations)
    references = _reference_curves(connection, account_scope, lineage)
    companion_by_id = {companion.alpha_id: companion for companion in companions}
    offered: Counter[str] = Counter()
    skipped = 0
    leaves: dict[str, tuple[ComboEnhancementLeaf, ...]] = {}
    for parent in sorted(parents, key=lambda p: (-abs(p.result.fitness), p.task.task_id)):
        daily = parent_pnl.get(parent.task.platform_alpha_id)
        if not daily:
            continue
        root = lineage(parent.task.task_id)

        def distinct(plan, companion, daily=daily, root=root) -> bool:
            nonlocal skipped
            if references.distinct(_predicted(daily, companion.daily_pnl, plan.sign, plan.weight), root):
                return True
            skipped += 1
            return False

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
            available, catalog, accept=distinct))
        offered.update(leaf.companion_alpha_id for leaf in found)
        for leaf in found:
            references.add(_predicted(daily, companion_by_id[leaf.companion_alpha_id].daily_pnl,
                                      leaf.sign, leaf.weight), root)
        if found:
            leaves[parent.task.task_id] = found
    return ComboEnhancementPlan(leaves, skipped)


def _predicted(parent: Mapping[str, float], companion: Mapping[str, float], sign: int,
               weight: float) -> dict[str, float]:
    return {day: value + sign * weight * companion.get(day, 0.0) for day, value in parent.items()}


class _ReferenceCurves:
    """Daily PnL curves tagged with their lineage, for a bounded correlation scan."""

    def __init__(self) -> None:
        self._curves: list[tuple[str, dict[str, float], float]] = []

    def add(self, daily: Mapping[str, float], root: str) -> None:
        if len(daily) < _MIN_OVERLAP_DAYS:
            return
        mean = sum(daily.values()) / len(daily)
        centered = {day: value - mean for day, value in daily.items()}
        norm = math.sqrt(sum(v * v for v in centered.values()))
        if norm > 0:
            self._curves.append((root, centered, norm))

    def distinct(self, daily: Mapping[str, float], root: str) -> bool:
        """False if any curve of another lineage correlates at the line or above."""
        if len(daily) < _MIN_OVERLAP_DAYS:
            return True
        mean = sum(daily.values()) / len(daily)
        centered = {day: value - mean for day, value in daily.items()}
        norm = math.sqrt(sum(v * v for v in centered.values()))
        if norm == 0:
            return True
        for other_root, other, other_norm in self._curves:
            if other_root == root:
                continue
            dot = sum(value * other.get(day, 0.0) for day, value in centered.items())
            if dot / (norm * other_norm) >= MAX_CROSS_LINEAGE_CORRELATION:
                return False
        return True


def _lineage_roots(mutations: tuple[BacktestMutationRecord, ...]):
    parents = {mutation.child_task_id: mutation.parent_task_id for mutation in mutations}
    cache: dict[str, str] = {}

    def root(task_id: str) -> str:
        if task_id in cache:
            return cache[task_id]
        seen, current = set(), task_id
        while current in parents and current not in seen:
            seen.add(current)
            current = parents[current]
        cache[task_id] = current
        return current

    return root


def _reference_curves(connection: sqlite3.Connection, account_scope: str, lineage) -> _ReferenceCurves:
    """Every alpha whose submission check passed, and every submitted alpha."""
    checked = {}
    for record in list_submission_checks(connection, account_scope=account_scope):
        payload = json.loads(record.payload_json) if record.payload_json else None
        if assess_formal_check_payload(payload).state == "passed":
            row = connection.execute("SELECT platform_alpha_id FROM backtest_tasks WHERE task_id=?",
                                     (record.task_id,)).fetchone()
            if row is not None and row[0]:
                checked[row[0]] = lineage(record.task_id)
    for reference in list_platform_submitted_alphas(connection, account_scope=account_scope):
        if reference.platform_alpha_id not in checked:
            row = connection.execute(
                "SELECT task_id FROM backtest_tasks WHERE account_scope=? AND platform_alpha_id=?",
                (account_scope, reference.platform_alpha_id)).fetchone()
            checked[reference.platform_alpha_id] = (lineage(row[0]) if row is not None
                                                     else f"submitted:{reference.platform_alpha_id}")
    curves = _ReferenceCurves()
    for alpha_id, daily in _daily_series(connection, account_scope, frozenset(checked)).items():
        curves.add(daily, checked[alpha_id])
    return curves


def _companions(connection: sqlite3.Connection, account_scope: str) -> tuple[EnhancementSignal, ...]:
    rows = connection.execute(
        """SELECT t.formula, t.platform_alpha_id, r.sharpe, r.fitness, r.returns, r.turnover
           FROM formula_ideas i
           JOIN backtest_tasks t ON t.task_id = i.task_id
           JOIN backtest_results r ON r.task_id = i.task_id
           WHERE t.account_scope = ? AND t.status = 'completed'
             AND t.platform_alpha_id IS NOT NULL AND i.template_id != ?
             AND abs(r.sharpe) >= ?
             -- A signal piling its book into a few names passes that flaw to every mix.
             AND NOT EXISTS (SELECT 1 FROM backtest_checks k WHERE k.task_id = t.task_id
                             AND k.check_name = 'CONCENTRATED_WEIGHT' AND k.status = 'FAIL')""",
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
