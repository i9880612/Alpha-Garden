from __future__ import annotations

import sys
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
from generation.decay_ladder import DECAY_LADDER_RUNGS, SIGNAL_DECAY_LADDER, iter_decay_ladder_leaves
from generation.parser import parse_formula
from generation.validation import validate_formula


def _catalog(with_decay=True):
    x, d = OperatorParameter("x", "expr"), OperatorParameter("d", "window")
    operators = [
        OperatorDefinition("rank", "t", ("REGULAR",), (x,), ("cross_sectional_normalization",), "signal"),
        OperatorDefinition("ts_delta", "t", ("REGULAR",), (x, d), ("time_series_change",), "signal"),
    ]
    if with_decay:
        operators.append(OperatorDefinition("ts_decay_linear", "t", ("REGULAR",), (x, d),
                                            ("time_series_smoothing",), "signal"))
    return GenerationCatalog(
        CatalogContext("EQUITY", "USA", "TOP3000", 1),
        (FieldDefinition("close", "pv1", "Price Volume", None, "MATRIX", 1.0),),
        tuple(operators), (WindowDefinition(5, "week"),))


def test_ladder_offers_each_rung_smallest_first():
    catalog = _catalog()
    leaves = list(iter_decay_ladder_leaves(parse_formula("rank(-ts_delta(close,5))").expression, catalog))
    assert [leaf.rung for leaf in leaves] == list(DECAY_LADDER_RUNGS)
    assert leaves[0].change.after == f"ts_decay_linear(rank(-ts_delta(close,5)),{DECAY_LADDER_RUNGS[0]})"
    assert all(leaf.family == SIGNAL_DECAY_LADDER for leaf in leaves)
    assert all(validate_formula(leaf.expression, catalog).is_valid for leaf in leaves)
    assert len({leaf.leaf_id for leaf in leaves}) == len(leaves)


def test_ladder_does_not_stack_decay_or_run_without_the_operator():
    decayed = parse_formula("ts_decay_linear(rank(close),6)").expression
    assert list(iter_decay_ladder_leaves(decayed, _catalog())) == []
    plain = parse_formula("rank(close)").expression
    assert list(iter_decay_ladder_leaves(plain, _catalog(with_decay=False))) == []
