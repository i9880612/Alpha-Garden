from __future__ import annotations

import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generation.catalog import (
    CatalogContext,
    FieldDefinition,
    GenerationCatalog,
    OperatorDefinition,
    OperatorParameter,
    WindowDefinition,
)
from generation.combo_enhancement import (
    COMBO_ENHANCEMENT,
    MIN_EXPECTED_GAIN,
    EnhancementSignal,
    companion_marker,
    iter_combo_enhancement_leaves,
    plan_enhancement,
)
from generation.parser import parse_formula
from generation.validation import validate_formula


def _catalog():
    x, d = OperatorParameter("x", "expr"), OperatorParameter("d", "window")
    operators = (
        OperatorDefinition("rank", "t", ("REGULAR",), (x,), ("cross_sectional_normalization",), "signal"),
        OperatorDefinition("zscore", "t", ("REGULAR",), (x,), ("cross_sectional_normalization",), "signal"),
        OperatorDefinition("ts_delta", "t", ("REGULAR",), (x, d), ("time_series_change",), "signal"),
    )
    fields = tuple(FieldDefinition(name, "pv1", "Price Volume", None, "MATRIX", 1.0)
                   for name in ("close", "open", "volume"))
    return GenerationCatalog(CatalogContext("EQUITY", "USA", "TOP3000", 1), fields, operators,
                             (WindowDefinition(5, "week"),))


def _daily(edge, noise, scale=1.0):
    start = date(2020, 1, 1)
    return {(start + timedelta(days=i)).isoformat(): scale * (edge + n) for i, n in enumerate(noise)}


def _signal(alpha, formula, edge, noise, scale=1.0):
    return EnhancementSignal(alpha, parse_formula(formula).expression, _daily(edge, noise, scale))


def _noise(seed, days=500):
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(days)]


def test_independent_companion_gets_sharpe_weights_and_expected_gain():
    parent = _signal("p", "rank(close)", 0.12, _noise(1))
    companion = _signal("c", "rank(open)", 0.09, _noise(7), scale=3.0)
    plan = plan_enhancement(parent, companion)
    assert plan is not None and plan.sign == 1
    # Nearly independent: gain close to sqrt(1 + (s2/s1)^2), weight about (s2/s1) * (sd1/sd2).
    assert abs(plan.correlation) < 0.15
    assert 1.05 < plan.expected_gain < 1.25
    assert 0.1 < plan.weight < 0.4


def test_losing_companion_is_flipped_and_correlated_or_weak_ones_skipped():
    parent = _signal("p", "rank(close)", 0.12, _noise(1))
    loser = _signal("c", "rank(open)", -0.08, _noise(3))
    assert plan_enhancement(parent, loser).sign == -1
    twin = EnhancementSignal("t", parse_formula("rank(open)").expression,
                             {day: value * 0.9 + 0.01 for day, value in parent.daily_pnl.items()})
    assert plan_enhancement(parent, twin) is None
    weak = _signal("w", "rank(open)", 0.001, _noise(4))
    assert plan_enhancement(parent, weak) is None
    short = EnhancementSignal("s", parse_formula("rank(open)").expression,
                              dict(list(_daily(0.1, _noise(5)).items())[:100]))
    assert plan_enhancement(parent, short) is None


def test_leaves_are_valid_ordered_and_keep_both_sides_on_one_scale():
    catalog = _catalog()
    parent_expression = parse_formula("rank(close)+1.3*rank(volume)").expression
    parent = EnhancementSignal("p", parent_expression, _daily(0.12, _noise(1)))
    companions = (
        _signal("a", "rank(open)", 0.03, _noise(6)),
        _signal("b", "rank(-ts_delta(open,5))", 0.09, _noise(7)),
        _signal("p", "rank(close)", 0.2, _noise(8)),  # the parent itself is never its companion
    )
    leaves = list(iter_combo_enhancement_leaves(parent_expression, parent, companions, catalog))
    assert [leaf.companion_alpha_id for leaf in leaves][0] == "b"
    assert all(leaf.family == COMBO_ENHANCEMENT for leaf in leaves)
    assert all(leaf.expected_gain >= MIN_EXPECTED_GAIN for leaf in leaves)
    first = leaves[0].change.after
    assert first.startswith("zscore(rank(close)+1.3*rank(volume))+")
    assert companion_marker(companions[1].expression) in first
    assert all(validate_formula(leaf.expression, catalog).is_valid for leaf in leaves)
    assert len({leaf.leaf_id for leaf in leaves}) == len(leaves)
    assert list(iter_combo_enhancement_leaves(parent_expression, parent, companions, catalog, limit=1))[0] \
        .leaf_id == leaves[0].leaf_id


def test_a_fast_companion_that_costs_fitness_through_turnover_is_skipped():
    from generation.combo_enhancement import BacktestMetrics
    parent = EnhancementSignal("p", parse_formula("rank(close)").expression, _daily(0.12, _noise(1)),
                               BacktestMetrics(sharpe=1.8, fitness=1.5, returns=0.12, turnover=0.18))
    slow = EnhancementSignal("s", parse_formula("rank(open)").expression, _daily(0.09, _noise(7)),
                             BacktestMetrics(sharpe=1.1, fitness=0.9, returns=0.08, turnover=0.10))
    fast = EnhancementSignal("f", parse_formula("rank(open)").expression, _daily(0.09, _noise(7)),
                             BacktestMetrics(sharpe=1.1, fitness=0.5, returns=0.08, turnover=0.80))
    kept = plan_enhancement(parent, slow)
    assert kept is not None and kept.expected_gain >= MIN_EXPECTED_GAIN
    # Same PnL curve, so the same Sharpe gain; only the turnover differs.
    assert plan_enhancement(parent, fast) is None


def test_a_vetoed_pair_hands_its_place_to_the_next_best():
    catalog = _catalog()
    parent_expression = parse_formula("rank(close)").expression
    parent = EnhancementSignal("p", parent_expression, _daily(0.12, _noise(1)))
    companions = (_signal("a", "rank(open)", 0.06, _noise(6)), _signal("b", "rank(-ts_delta(open,5))", 0.09, _noise(7)))
    best = list(iter_combo_enhancement_leaves(parent_expression, parent, companions, catalog, limit=1))
    assert [leaf.companion_alpha_id for leaf in best] == ["b"]
    vetoed = list(iter_combo_enhancement_leaves(parent_expression, parent, companions, catalog, limit=1,
                                                accept=lambda plan, companion: companion.alpha_id != "b"))
    assert [leaf.companion_alpha_id for leaf in vetoed] == ["a"]
