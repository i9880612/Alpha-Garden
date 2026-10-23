"""Combo enhancement: add a proven, independent signal to a qualified parent.

Small edits move a qualified parent's Fitness by about 0.1 either way, while every GOOD
alpha so far came from adding two signals whose daily PnL moves independently: their
noise partly cancels while their edge adds up. Here the companion is a single-template
signal that has already been backtested, and the weights are the ones that maximise the
combined Sharpe given both daily PnL curves.

Each side is wrapped in zscore. Alone, an alpha's PnL does not depend on its scale (the
platform neutralizes and rescales the book), so zscore(parent) trades exactly like the
parent; in a sum, zscore puts both sides on the same scale, so the weights computed from
the PnL curves are the weights the platform sees.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass

from generation.catalog import GenerationCatalog
from generation.formula import Call, Expression, formula_fingerprint, render_formula
from generation.logic import prepare_formula_logic
from generation.parser import FormulaSyntaxError, parse_formula
from generation.structure_limits import fits_formula_structure_limits
from generation.transformations import TransformationChange
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import validate_formula

COMBO_ENHANCEMENT = "combo_enhancement"
COMBO_ENHANCEMENT_FAMILIES = (COMBO_ENHANCEMENT,)
# The same independence bar as two-signal combos.
MAX_COMPANION_CORRELATION = 0.5
MIN_COMPANION_ABS_SHARPE = 0.7
# Below this the expected gain is within the noise of one backtest.
MIN_EXPECTED_GAIN = 1.05
MAX_LEAVES_PER_PARENT = 3
MIN_OVERLAP_DAYS = 250
MIN_WEIGHT = 0.1
MAX_WEIGHT = 4.0


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    sharpe: float
    fitness: float
    returns: float
    turnover: float


@dataclass(frozen=True, slots=True)
class EnhancementSignal:
    """A backtested formula, its day-to-day PnL changes and (if known) its platform metrics."""
    alpha_id: str
    expression: Expression
    daily_pnl: Mapping[str, float]
    metrics: BacktestMetrics | None = None


@dataclass(frozen=True, slots=True)
class EnhancementPlan:
    sign: int
    weight: float
    # Expected Fitness over the parent's when both metrics are known, else expected Sharpe.
    expected_gain: float
    correlation: float
    sharpe_gain: float


@dataclass(frozen=True, slots=True)
class ComboEnhancementLeaf:
    leaf_id: str
    expression: Expression
    change: TransformationChange
    companion_alpha_id: str
    expected_gain: float
    sign: int = 1
    weight: float = 1.0
    family: str = COMBO_ENHANCEMENT
    # The companion is a qualified alpha of another lineage (execution/combo_enhancement).
    lineage_merge: bool = False


# rank output is uniform on [0, 1] every day, so its cross-sectional spread is the constant
# 1/sqrt(12): zscore(rank(x)) is sqrt(12) * rank(x) plus a constant the platform's
# neutralization removes. Writing the rank term with that weight saves a nesting level.
RANK_ZSCORE_SCALE = math.sqrt(12.0)


def _term(expression: Expression) -> tuple[str, float]:
    """A side of the sum on the common zscore scale: (text, weight that puts it there)."""
    text = render_formula(expression)
    if isinstance(expression, Call) and expression.operator == "rank" and len(expression.arguments) == 1:
        return text, RANK_ZSCORE_SCALE
    if isinstance(expression, Call) and expression.operator == "zscore" and len(expression.arguments) == 1:
        return text, 1.0  # already on the zscore scale
    return f"zscore({text})", 1.0


def companion_markers(expression: Expression) -> tuple[str, ...]:
    """Texts a companion leaves in a child formula built with it (current and older forms)."""
    text = render_formula(expression)
    return (f"zscore({text})", f"*{text}", f"+{text}", f"-{text}")


def companion_marker(expression: Expression) -> str:
    """The zscore form older children were written in."""
    return companion_markers(expression)[0]


def plan_enhancement(parent: EnhancementSignal, companion: EnhancementSignal) -> EnhancementPlan | None:
    """Best weights for parent + companion, or None if the pair is not worth a backtest.

    With daily Sharpe s1, s2 (companion flipped if it loses money) and correlation rho,
    the best mix puts u1 = s1 - rho*s2 and u2 = s2 - rho*s1 units of volatility on each
    side and earns sqrt((s1^2 + s2^2 - 2*rho*s1*s2) / (1 - rho^2)). A companion that
    needs a negative weight, or that adds less than MIN_EXPECTED_GAIN, is skipped.
    """
    days = sorted(set(parent.daily_pnl) & set(companion.daily_pnl))
    if len(days) < MIN_OVERLAP_DAYS:
        return None
    x = [parent.daily_pnl[day] for day in days]
    y = [companion.daily_pnl[day] for day in days]
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    sd_x = math.sqrt(sum((v - mean_x) ** 2 for v in x) / len(x))
    sd_y = math.sqrt(sum((v - mean_y) ** 2 for v in y) / len(y))
    if sd_x == 0 or sd_y == 0 or mean_x <= 0 or mean_y == 0:
        return None
    sign = 1 if mean_y > 0 else -1
    s1, s2 = mean_x / sd_x, abs(mean_y) / sd_y
    rho = sign * sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / (len(x) * sd_x * sd_y)
    if rho > MAX_COMPANION_CORRELATION or rho <= -1:
        return None
    u1, u2 = s1 - rho * s2, s2 - rho * s1
    if u1 <= 0 or u2 <= 0:
        return None
    sharpe_gain = math.sqrt((s1 * s1 + s2 * s2 - 2 * rho * s1 * s2) / (1 - rho * rho)) / s1
    weight = min(MAX_WEIGHT, max(MIN_WEIGHT, float(f"{(u2 / sd_y) / (u1 / sd_x):.2g}")))
    gain = sharpe_gain
    if parent.metrics is not None and companion.metrics is not None:
        gain = _fitness_gain(parent.metrics, companion.metrics, weight, sharpe_gain)
    if gain < MIN_EXPECTED_GAIN:
        return None
    return EnhancementPlan(sign=sign, weight=weight, expected_gain=gain, correlation=rho,
                           sharpe_gain=sharpe_gain)


def _fitness_gain(parent: BacktestMetrics, companion: BacktestMetrics, weight: float,
                  sharpe_gain: float) -> float:
    """Expected Fitness over the parent's: Sharpe * sqrt(Returns / max(Turnover, 0.125)).

    Each zscore side puts book in proportion to its weight, so the mix earns and trades
    about the weighted average of the two (netting between them shrinks both alike). The
    Sharpe gain comes from the noise that cancels. A strong but fast companion can
    therefore raise Sharpe and still lower Fitness through turnover.
    """
    if parent.fitness == 0:
        return 0.0
    returns = (abs(parent.returns) + weight * abs(companion.returns)) / (1 + weight)
    turnover = (parent.turnover + weight * companion.turnover) / (1 + weight)
    fitness = abs(parent.sharpe) * sharpe_gain * math.sqrt(returns / max(turnover, 0.125))
    return fitness / abs(parent.fitness)


def iter_combo_enhancement_leaves(
    parent_expression: Expression,
    parent: EnhancementSignal,
    companions: Iterable[EnhancementSignal],
    catalog: GenerationCatalog,
    *,
    limit: int = MAX_LEAVES_PER_PARENT,
    accept: Callable[[EnhancementPlan, EnhancementSignal], bool] | None = None,
) -> Iterator[ComboEnhancementLeaf]:
    """Up to ``limit`` locally valid children, largest expected gain first.

    ``accept`` may veto a pair before it counts toward the limit, so a vetoed pair hands
    its place to the next best one instead of leaving the parent with fewer children.
    """
    plans = []
    for companion in companions:
        if companion.alpha_id == parent.alpha_id:
            continue
        plan = plan_enhancement(parent, companion)
        if plan is not None:
            plans.append((plan, companion))
    plans.sort(key=lambda item: (-item[0].expected_gain, item[1].alpha_id))
    before = render_formula(parent_expression)
    parent_fingerprint = formula_fingerprint(parent_expression)
    produced = 0
    for plan, companion in plans:
        if produced >= limit:
            return
        if accept is not None and not accept(plan, companion):
            continue
        parent_term, parent_scale = _term(parent_expression)
        companion_term, companion_scale = _term(companion.expression)
        # Both sides on the zscore scale, then the parent's weight set to 1.
        weight = float(f"{plan.weight * companion_scale / parent_scale:.2g}")
        weighted = companion_term if weight == 1 else f"{weight:g}*{companion_term}"
        formula = f"{parent_term}{'+' if plan.sign > 0 else '-'}{weighted}"
        expression = _prepare(formula, catalog)
        if expression is None:
            continue
        produced += 1
        yield ComboEnhancementLeaf(
            leaf_id=f"{COMBO_ENHANCEMENT}:{parent_fingerprint}:{companion.alpha_id}",
            expression=expression,
            change=TransformationChange(location="formula", before=before,
                                        after=render_formula(expression)),
            companion_alpha_id=companion.alpha_id,
            expected_gain=plan.expected_gain,
            sign=plan.sign,
            weight=plan.weight,
        )


def _prepare(formula: str, catalog: GenerationCatalog) -> Expression | None:
    try:
        expression = parse_formula(formula).expression
    except FormulaSyntaxError:
        return None
    if not validate_formula(expression, catalog).is_valid or find_coarse_unit_issues(expression, catalog):
        return None
    logic = prepare_formula_logic(expression)
    if logic.issue is not None or not fits_formula_structure_limits(logic.expression):
        return None
    if not validate_formula(logic.expression, catalog).is_valid:
        return None
    return logic.expression
