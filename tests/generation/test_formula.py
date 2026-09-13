from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generation.formula import (
    Binary,
    Call,
    CallArgument,
    Literal,
    Name,
    Prefix,
    analyze_formula,
    formula_fingerprint,
    formula_structure_fingerprint,
    render_formula,
)
from generation.catalog import (
    CatalogContext,
    FieldDefinition,
    GenerationCatalog,
    OperatorDefinition,
    OperatorParameter,
)


class FormulaTests(unittest.TestCase):
    def test_render_preserves_right_nested_subtraction(self) -> None:
        expression = Binary(
            "-",
            Name("a"),
            Binary("-", Name("b"), Name("c")),
        )

        self.assertEqual(render_formula(expression), "a-(b-c)")

    def test_render_preserves_grouping_for_nested_ratio(self) -> None:
        expression = Binary(
            "/",
            Binary("+", Name("a"), Name("b")),
            Binary("+", Literal("1", "integer"), Binary("-", Name("c"), Name("d"))),
        )

        self.assertEqual(render_formula(expression), "(a+b)/(1+(c-d))")

    def test_render_supports_named_arguments_and_prefixes(self) -> None:
        expression = Call(
            "normalize",
            (
                CallArgument(Name("close")),
                CallArgument(Literal("true", "boolean"), name="useStd"),
                CallArgument(Prefix("-", Literal("0.5", "float")), name="limit"),
            ),
        )

        self.assertEqual(
            render_formula(expression),
            "normalize(close,useStd=true,limit=-0.5)",
        )

    def test_analysis_is_derived_from_the_expression(self) -> None:
        expression = Call(
            "rank",
            (
                CallArgument(
                    Binary(
                        "+",
                        Name("close"),
                        Call(
                            "ts_delta",
                            (
                                CallArgument(Name("volume")),
                                CallArgument(Literal("20", "integer")),
                            ),
                        ),
                    )
                ),
            ),
        )

        facts = analyze_formula(expression)

        self.assertEqual(facts.referenced_names, ("close", "volume"))
        self.assertEqual(facts.operator_names, ("rank", "+", "ts_delta"))
        self.assertEqual(facts.depth, 4)
        self.assertEqual(facts.complexity, 6)

    def test_fingerprint_uses_the_rendered_formula(self) -> None:
        left = Call("rank", (CallArgument(Name("close")),))
        right = Call("rank", (CallArgument(Name("close")),))

        self.assertEqual(formula_fingerprint(left), formula_fingerprint(right))

    def test_structure_fingerprint_ignores_only_window_parameters(self) -> None:
        catalog = self._catalog()
        short = Call(
            "ts_mean",
            (
                CallArgument(Name("close")),
                CallArgument(Literal("5", "integer")),
            ),
        )
        long = Call(
            "ts_mean",
            (
                CallArgument(Name("close")),
                CallArgument(Literal("22", "integer")),
            ),
        )
        different_field = Call(
            "ts_mean",
            (
                CallArgument(Name("open")),
                CallArgument(Literal("5", "integer")),
            ),
        )

        self.assertEqual(
            formula_structure_fingerprint(short, catalog),
            formula_structure_fingerprint(long, catalog),
        )
        self.assertNotEqual(
            formula_structure_fingerprint(short, catalog),
            formula_structure_fingerprint(different_field, catalog),
        )

    def test_structure_fingerprint_preserves_non_window_numbers_and_tree(self) -> None:
        catalog = self._catalog()
        high_threshold = Binary(">", Name("close"), Literal("0.8", "float"))
        low_threshold = Binary(">", Name("close"), Literal("0.2", "float"))
        reversed_tree = Binary(">", Literal("0.8", "float"), Name("close"))

        self.assertNotEqual(
            formula_structure_fingerprint(high_threshold, catalog),
            formula_structure_fingerprint(low_threshold, catalog),
        )
        self.assertNotEqual(
            formula_structure_fingerprint(high_threshold, catalog),
            formula_structure_fingerprint(reversed_tree, catalog),
        )

    def test_structure_fingerprint_uses_semantic_argument_binding(self) -> None:
        catalog = self._catalog()
        first = Call(
            "ts_mean",
            (
                CallArgument(Literal("5", "integer"), name="d"),
                CallArgument(Literal("1", "integer")),
            ),
        )
        window_variant = Call(
            "ts_mean",
            (
                CallArgument(Literal("22", "integer"), name="d"),
                CallArgument(Literal("1", "integer")),
            ),
        )
        signal_variant = Call(
            "ts_mean",
            (
                CallArgument(Literal("5", "integer"), name="d"),
                CallArgument(Literal("2", "integer")),
            ),
        )

        self.assertEqual(
            formula_structure_fingerprint(first, catalog),
            formula_structure_fingerprint(window_variant, catalog),
        )
        self.assertNotEqual(
            formula_structure_fingerprint(first, catalog),
            formula_structure_fingerprint(signal_variant, catalog),
        )

    def test_structure_fingerprint_ignores_signed_and_named_window_syntax(
        self,
    ) -> None:
        catalog = self._catalog()
        positional = Call(
            "ts_mean",
            (
                CallArgument(Name("close")),
                CallArgument(Prefix("-", Literal("5", "integer"))),
            ),
        )
        named = Call(
            "ts_mean",
            (
                CallArgument(Literal("22", "integer"), name="d"),
                CallArgument(Name("close"), name="x"),
            ),
        )

        self.assertEqual(
            formula_structure_fingerprint(positional, catalog),
            formula_structure_fingerprint(named, catalog),
        )

    @staticmethod
    def _catalog() -> GenerationCatalog:
        return GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            fields=(
                FieldDefinition("close", None, None, None, "MATRIX", 1.0),
                FieldDefinition("open", None, None, None, "MATRIX", 1.0),
            ),
            operators=(
                OperatorDefinition(
                    name="ts_mean",
                    category=None,
                    scope=("REGULAR",),
                    parameters=(
                        OperatorParameter("x", "expr"),
                        OperatorParameter("d", "window"),
                    ),
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
