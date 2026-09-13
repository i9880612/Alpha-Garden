from __future__ import annotations

import sys
import unittest
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
from generation.formula import render_formula
from generation.parser import parse_formula
from generation.polishing import (
    SINGLE_WINDOW_MUTATION,
    iter_window_mutation_leaves,
)


class SingleWindowMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            fields=(
                FieldDefinition(
                    field_id="close",
                    dataset_id="prices",
                    category="Price Volume",
                    subcategory="Price",
                    field_type="MATRIX",
                    coverage=1.0,
                ),
            ),
            operators=(
                self._operator(
                    "rank",
                    OperatorParameter("x", "expr"),
                ),
                self._operator(
                    "ts_mean",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("d", "window"),
                ),
                self._operator(
                    "sample_int",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("k", "int"),
                ),
                self._operator(
                    "multiply",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("y", "expr"),
                ),
            ),
            windows=tuple(
                WindowDefinition(value, f"window-{value}") for value in (5, 22, 66, 120)
            ),
        )

    def test_enumerates_each_single_window_change_once_and_stably(self) -> None:
        expression = parse_formula("rank(ts_mean(ts_mean(close,22),66))").expression

        first = tuple(iter_window_mutation_leaves(expression, self.catalog))
        second = tuple(iter_window_mutation_leaves(expression, self.catalog))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(len({leaf.leaf_id for leaf in first}), 6)
        self.assertTrue(all(leaf.family == SINGLE_WINDOW_MUTATION for leaf in first))
        self.assertEqual(
            {(leaf.change.before, leaf.change.after) for leaf in first},
            {
                ("22", "5"),
                ("22", "66"),
                ("22", "120"),
                ("66", "5"),
                ("66", "22"),
                ("66", "120"),
            },
        )
        self.assertEqual(
            {leaf.change.location for leaf in first},
            {
                "formula.arguments[0].arguments[1]",
                "formula.arguments[0].arguments[0].arguments[1]",
            },
        )
        self.assertEqual(
            {render_formula(leaf.expression) for leaf in first},
            {
                "rank(ts_mean(ts_mean(close,5),66))",
                "rank(ts_mean(ts_mean(close,66),66))",
                "rank(ts_mean(ts_mean(close,120),66))",
                "rank(ts_mean(ts_mean(close,22),5))",
                "rank(ts_mean(ts_mean(close,22),22))",
                "rank(ts_mean(ts_mean(close,22),120))",
            },
        )

    def test_named_window_argument_keeps_its_name(self) -> None:
        expression = parse_formula("rank(ts_mean(close,d=22))").expression

        leaves = tuple(iter_window_mutation_leaves(expression, self.catalog))

        self.assertEqual(len(leaves), 3)
        self.assertEqual(
            {leaf.change.after for leaf in leaves},
            {"5", "66", "120"},
        )
        self.assertEqual(
            {leaf.expression.arguments[0].value.arguments[1].name for leaf in leaves},
            {"d"},
        )
        self.assertEqual(
            {leaf.change.location for leaf in leaves},
            {"formula.arguments[0].arguments[1]"},
        )

    def test_integer_parameter_equal_to_window_value_is_not_modified(self) -> None:
        expression = parse_formula("rank(sample_int(close,22))").expression

        leaves = tuple(iter_window_mutation_leaves(expression, self.catalog))

        self.assertEqual(leaves, ())

    def test_nonstandard_window_can_mutate_to_each_standard_window(self) -> None:
        leaves = tuple(
            iter_window_mutation_leaves(
                parse_formula("rank(ts_mean(close,10))").expression,
                self.catalog,
            )
        )

        self.assertEqual(len(leaves), 4)
        self.assertEqual({leaf.change.before for leaf in leaves}, {"10"})
        self.assertEqual(
            {leaf.change.after for leaf in leaves},
            {"5", "22", "66", "120"},
        )

    def test_formula_without_window_has_no_mutation_leaf(self) -> None:
        leaves = tuple(
            iter_window_mutation_leaves(
                parse_formula("rank(close)").expression,
                self.catalog,
            )
        )

        self.assertEqual(leaves, ())

    def test_parent_requiring_other_logic_rewrite_has_no_polishing_leaves(
        self,
    ) -> None:
        expression = parse_formula("rank(multiply(ts_mean(close,22),1))").expression

        leaves = tuple(iter_window_mutation_leaves(expression, self.catalog))

        self.assertEqual(leaves, ())

    @staticmethod
    def _operator(
        name: str,
        *parameters: OperatorParameter,
    ) -> OperatorDefinition:
        return OperatorDefinition(
            name=name,
            category="sample",
            scope=("REGULAR",),
            parameters=tuple(parameters),
            roles=(),
            output_kind="signal",
        )


if __name__ == "__main__":
    unittest.main()
