from __future__ import annotations

import inspect
import re
import sys
import unittest
from collections import Counter
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
from generation.exploration import (
    ExplorationFailure,
    ExplorationResult,
    explore_formula,
)
from generation.formula import Call, CallArgument, Literal, Name, analyze_formula
from generation.formula import render_formula
from generation.structure_builder import build_structure_expression
from generation.unit_validation import is_proven_dimensionless
from generation.validation import validate_formula


class FormulaExplorationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            fields=(
                self._field("close", "MATRIX"),
                self._field("open", "MATRIX"),
                self._field("volume", "MATRIX"),
                self._field("news", "VECTOR"),
                self._field("sector", "GROUP"),
                self._field("industry", "GROUP"),
            ),
            operators=(
                self._operator("abs", "Arithmetic", (("x", "expr"),)),
                self._operator(
                    "add",
                    "Arithmetic",
                    (("x", "expr"), ("y", "expr")),
                ),
                self._operator(
                    "divide",
                    "Arithmetic",
                    (("x", "expr"), ("y", "expr")),
                ),
                self._operator("reverse", "Arithmetic", (("x", "expr"),)),
                self._operator("sqrt", "Arithmetic", (("x", "expr"),)),
                self._operator("rank", "Cross Sectional", (("x", "expr"),)),
                self._operator(
                    "vector_neut",
                    "Cross Sectional",
                    (("x", "expr"), ("y", "expr")),
                ),
                self._operator(
                    "group_rank",
                    "Group",
                    (("x", "expr"), ("group", "group")),
                ),
                self._operator(
                    "group_backfill",
                    "Group",
                    (("x", "expr"), ("group", "group"), ("d", "window")),
                ),
                self._operator(
                    "if_else",
                    "Logical",
                    (
                        ("input1", "expr"),
                        ("input2", "expr"),
                        ("input3", "expr"),
                    ),
                ),
                self._operator(
                    "greater",
                    "Logical",
                    (("input1", "expr"), ("input2", "expr")),
                    output="condition",
                ),
                self._operator(
                    "not",
                    "Logical",
                    (("x", "expr"),),
                    output="condition",
                ),
                self._operator(
                    "trade_when",
                    "Transformational",
                    (("x", "expr"), ("y", "expr"), ("z", "expr")),
                ),
                self._operator("hump", "Transformational", (("x", "expr"),)),
                OperatorDefinition(
                    name="bucket",
                    category="Transformational",
                    scope=("REGULAR",),
                    parameters=(
                        OperatorParameter("x", "expr"),
                        OperatorParameter("range", "string", optional=True),
                        OperatorParameter("skipBoth", "bool", optional=True),
                        OperatorParameter("NaNGroup", "bool", optional=True),
                    ),
                    roles=(),
                    output_kind="group",
                ),
                self._operator(
                    "densify",
                    "Arithmetic",
                    (("x", "expr"),),
                    output="group",
                ),
                self._operator(
                    "ts_corr",
                    "Time Series",
                    (("x", "expr"), ("y", "expr"), ("d", "window")),
                ),
                OperatorDefinition(
                    name="ts_backfill",
                    category="Time Series",
                    scope=("REGULAR",),
                    parameters=(
                        OperatorParameter("x", "expr"),
                        OperatorParameter("lookback", "window", optional=True),
                        OperatorParameter("k", "int", optional=True),
                    ),
                    roles=(),
                    output_kind="signal",
                ),
                self._operator(
                    "ts_mean",
                    "Time Series",
                    (("x", "expr"), ("d", "window")),
                ),
                self._operator(
                    "ts_product",
                    "Time Series",
                    (("x", "expr"), ("d", "window")),
                ),
                self._operator(
                    "ts_regression",
                    "Time Series",
                    (("y", "expr"), ("x", "expr"), ("d", "window")),
                ),
                OperatorDefinition(
                    name="kth_element",
                    category="Time Series",
                    scope=("REGULAR",),
                    parameters=(
                        OperatorParameter("x", "expr"),
                        OperatorParameter("d", "window"),
                        OperatorParameter("k", "int"),
                        OperatorParameter("ignore", "string", optional=True),
                    ),
                    roles=(),
                    output_kind="signal",
                ),
                self._operator(
                    "ts_step",
                    "Time Series",
                    (("step", "int"),),
                ),
                self._operator("vec_avg", "Vector", (("x", "expr"),)),
                self._operator("vec_sum", "Vector", (("x", "expr"),)),
            ),
            windows=(
                WindowDefinition(5, "week"),
                WindowDefinition(22, "month"),
                WindowDefinition(66, "quarter"),
            ),
        )

    def test_generates_repeatable_valid_open_structures(self) -> None:
        formulas: list[str] = []
        operator_sets: set[tuple[str, ...]] = set()
        failures = 0

        for seed in range(100):
            first = self._generate(seed)
            repeated = self._generate(seed)
            self.assertEqual(first, repeated)
            if isinstance(first, ExplorationFailure):
                self.assertIn(
                    first.code,
                    {"structure_budget_exceeded", "structure_signal_field_missing"},
                )
                failures += 1
                continue
            assert isinstance(first, ExplorationResult)
            self.assertTrue(validate_formula(first.expression, self.catalog).is_valid)
            facts = analyze_formula(first.expression)
            formulas.append(render_formula(first.expression))
            operator_sets.add(tuple(sorted(set(facts.operator_names))))

        self.assertGreaterEqual(len(formulas), 80)
        self.assertLessEqual(failures, 20)
        self.assertGreater(len(set(formulas)), 70)
        self.assertGreater(len(operator_sets), 20)

    def test_generator_has_no_family_input(self) -> None:
        parameters = inspect.signature(explore_formula).parameters

        self.assertNotIn("family", parameters)
        self.assertNotIn("family_quota", parameters)

    def test_explicit_operator_candidates_only_limit_diagnostic_generation(
        self,
    ) -> None:
        for seed in range(20):
            outcome = self._generate(
                seed,
                operator_candidates=("rank", "ts_mean"),
            )
            self.assertIsInstance(outcome, ExplorationResult)
            assert isinstance(outcome, ExplorationResult)
            self.assertTrue(
                set(analyze_formula(outcome.expression).operator_names)
                <= {"rank", "ts_mean"}
            )

    def test_condition_thresholds_do_not_use_negative_boundaries(self) -> None:
        pattern = re.compile(r"(?:>=|<=|==|!=|>|<)-")

        for seed in range(100):
            outcome = self._generate(seed)
            if isinstance(outcome, ExplorationResult):
                self.assertIsNone(pattern.search(render_formula(outcome.expression)))

    def test_kth_element_uses_supported_explicit_parameters(self) -> None:
        calls = self._generated_calls("kth_element")

        self.assertGreater(len(calls), 0)
        for call in calls:
            self.assertEqual(
                tuple(argument.name for argument in call.arguments),
                (None, None, "k", "ignore"),
            )
            self.assertEqual(call.arguments[3].value, Literal('"NaN"', "string"))

    def test_ts_step_uses_official_fixed_argument(self) -> None:
        calls = self._generated_calls("ts_step")

        self.assertGreater(len(calls), 0)
        for call in calls:
            self.assertEqual(
                call.arguments,
                (CallArgument(Literal("1", "integer")),),
            )

    def test_ts_backfill_uses_explicit_lookback_window(self) -> None:
        calls = self._generated_calls("ts_backfill")

        self.assertGreater(len(calls), 0)
        for call in calls:
            self.assertEqual(len(call.arguments), 2)
            self.assertEqual(call.arguments[1].name, "lookback")
            self.assertIsInstance(call.arguments[1].value, Literal)
            self.assertIn(
                int(call.arguments[1].value.value),
                self.catalog.window_values(),
            )

    def test_bucket_uses_ranked_input_and_explicit_range(self) -> None:
        calls = self._generated_calls("bucket")

        self.assertGreater(len(calls), 0)
        for call in calls:
            self.assertEqual(len(call.arguments), 2)
            self.assertEqual(call.arguments[1].name, "range")
            self.assertEqual(
                call.arguments[1].value,
                Literal('"0,1,0.1"', "string"),
            )
            self.assertIsInstance(call.arguments[0].value, Call)
            self.assertEqual(call.arguments[0].value.operator, "rank")

    def test_platform_unitless_inputs_are_normalized_before_generation(self) -> None:
        for operator_name, input_index in (
            ("group_backfill", 0),
            ("hump", 0),
            ("ts_product", 0),
            ("ts_regression", 0),
            ("ts_regression", 1),
        ):
            calls = self._generated_calls(operator_name)
            self.assertGreater(len(calls), 0)
            for call in calls:
                self.assertTrue(
                    is_proven_dimensionless(
                        call.arguments[input_index].value,
                        self.catalog,
                    )
                )

    def test_vector_fields_are_only_used_through_vector_reducers(self) -> None:
        seen_vector = False

        def visit(expression, parent=None):
            nonlocal seen_vector
            if isinstance(expression, Name) and expression.value == "news":
                seen_vector = True
                self.assertIsInstance(parent, Call)
                assert isinstance(parent, Call)
                self.assertIn(parent.operator, {"vec_avg", "vec_sum"})
            if isinstance(expression, Call):
                for argument in expression.arguments:
                    visit(argument.value, expression)
                return
            for child_name in ("left", "right", "operand"):
                child = getattr(expression, child_name, None)
                if child is not None:
                    visit(child, expression)

        for seed in range(100):
            outcome = self._generate(seed)
            if isinstance(outcome, ExplorationResult):
                visit(outcome.expression)

        self.assertTrue(seen_vector)

    def test_signal_field_choice_balances_category_then_dataset(self) -> None:
        fields = self._imbalanced_signal_fields()
        catalog = GenerationCatalog(
            context=self.catalog.context,
            fields=fields,
            operators=(
                self._operator("rank", "Cross Sectional", (("x", "expr"),)),
            ),
            windows=self.catalog.windows,
        )
        category_counts: dict[str, int] = {}
        dataset_counts: dict[str, int] = {}

        for seed in range(300):
            expression = build_structure_expression(
                catalog,
                seed=seed,
                fields=fields,
                groups=(),
                max_depth=2,
                operator_candidates=("rank",),
            )
            self.assertIsNotNone(expression)
            assert expression is not None
            field_id = analyze_formula(expression).referenced_names[0]
            field = catalog.field(field_id)
            assert field is not None
            category_counts[field.category or ""] = (
                category_counts.get(field.category or "", 0) + 1
            )
            dataset_counts[field.dataset_id or ""] = (
                dataset_counts.get(field.dataset_id or "", 0) + 1
            )

        self.assertGreaterEqual(category_counts["category_a"], 120)
        self.assertLessEqual(category_counts["category_a"], 180)
        self.assertGreaterEqual(dataset_counts["large_dataset"], 45)
        self.assertGreaterEqual(dataset_counts["small_dataset"], 45)

    def test_distinct_binary_inputs_keep_balanced_field_choice(self) -> None:
        fields = self._imbalanced_signal_fields()
        catalog = GenerationCatalog(
            context=self.catalog.context,
            fields=fields,
            operators=(
                self._operator("rank", "Cross Sectional", (("x", "expr"),)),
                self._operator(
                    "subtract",
                    "Arithmetic",
                    (("x", "expr"), ("y", "expr")),
                ),
            ),
            windows=self.catalog.windows,
        )
        right_large_fields: Counter[str] = Counter()
        subtract_count = 0

        for seed in range(5_000):
            expression = build_structure_expression(
                catalog,
                seed=seed,
                fields=fields,
                groups=(),
                max_depth=2,
                operator_candidates=("rank", "subtract"),
            )
            if not isinstance(expression, Call) or expression.operator != "subtract":
                continue
            subtract_count += 1
            right_field = analyze_formula(
                expression.arguments[1].value
            ).referenced_names[0]
            if right_field.startswith("large_"):
                right_large_fields[right_field] += 1

        self.assertGreater(subtract_count, 1_000)
        self.assertGreater(len(right_large_fields), 50)
        peer_max = max(
            count
            for field_id, count in right_large_fields.items()
            if field_id != "large_0"
        )
        self.assertLessEqual(right_large_fields["large_0"], peer_max * 3)

    def test_missing_signal_fields_returns_explicit_failure(self) -> None:
        outcome = explore_formula(
            self.catalog,
            seed=0,
            field_candidates=("sector",),
            group_candidates=("sector",),
        )

        self.assertEqual(outcome, ExplorationFailure("signal_field_candidates_empty"))

    def _generate(self, seed: int, **overrides):
        arguments = {
            "field_candidates": ("close", "open", "volume", "news"),
            "group_candidates": ("sector", "industry"),
        }
        arguments.update(overrides)
        return explore_formula(self.catalog, seed=seed, **arguments)

    def _generated_calls(self, operator: str) -> list[Call]:
        calls: list[Call] = []

        def visit(expression) -> None:
            if isinstance(expression, Call):
                if expression.operator == operator:
                    calls.append(expression)
                for argument in expression.arguments:
                    visit(argument.value)
                return
            for child_name in ("left", "right", "operand"):
                child = getattr(expression, child_name, None)
                if child is not None:
                    visit(child)

        for seed in range(1_000):
            outcome = self._generate(seed)
            if isinstance(outcome, ExplorationResult):
                visit(outcome.expression)
        return calls

    @staticmethod
    def _field(field_id: str, field_type: str) -> FieldDefinition:
        return FieldDefinition(
            field_id=field_id,
            dataset_id="sample",
            category="sample",
            subcategory=None,
            field_type=field_type,
            coverage=1.0,
        )

    @staticmethod
    def _imbalanced_signal_fields() -> tuple[FieldDefinition, ...]:
        return tuple(
            FieldDefinition(
                field_id=f"large_{index}",
                dataset_id="large_dataset",
                category="category_a",
                subcategory=None,
                field_type="MATRIX",
                coverage=1.0,
            )
            for index in range(100)
        ) + (
            FieldDefinition(
                field_id="small_dataset_field",
                dataset_id="small_dataset",
                category="category_a",
                subcategory=None,
                field_type="MATRIX",
                coverage=1.0,
            ),
            FieldDefinition(
                field_id="other_category_field",
                dataset_id="other_dataset",
                category="category_b",
                subcategory=None,
                field_type="MATRIX",
                coverage=1.0,
            ),
        )

    @staticmethod
    def _operator(
        name: str,
        category: str,
        parameters: tuple[tuple[str, str], ...],
        *,
        output: str = "signal",
    ) -> OperatorDefinition:
        return OperatorDefinition(
            name=name,
            category=category,
            scope=("REGULAR",),
            parameters=tuple(
                OperatorParameter(parameter_name, kind)
                for parameter_name, kind in parameters
            ),
            roles=(("cross_sectional_normalization",) if name == "rank" else ()),
            output_kind=output,
        )


if __name__ == "__main__":
    unittest.main()
