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
)
from generation.formula import analyze_formula
from generation.parser import parse_formula
from generation.structure_limits import (
    FORMULA_MAX_COMPLEXITY,
    FORMULA_MAX_DEPTH,
)
from selection.risks import assess_candidate_risk, has_tail_risk_structure


class CandidateRiskTests(unittest.TestCase):
    def test_zero_coverage_field_is_not_eligible(self) -> None:
        assessed = self._assess("rank(empty)")

        self.assertEqual(assessed.rejection_reasons, ("field_coverage_zero",))

    def test_two_unfilled_sparse_fields_are_not_eligible(self) -> None:
        assessed = self._assess("rank(sparse_a+sparse_b)")

        self.assertEqual(
            assessed.rejection_reasons,
            ("multiple_sparse_inputs_without_fill",),
        )

    def test_explicit_backfill_keeps_sparse_field_exploration_available(self) -> None:
        assessed = self._assess(
            "rank(ts_backfill(sparse_a,22)+ts_backfill(sparse_b,22))"
        )

        self.assertTrue(assessed.eligible)

    def test_exact_comparison_of_a_continuous_transform_is_not_eligible(self) -> None:
        assessed = self._assess("rank(close)==0.5")

        self.assertEqual(
            assessed.rejection_reasons,
            ("continuous_exact_comparison",),
        )

    def test_outer_rank_controls_tail_risk(self) -> None:
        controlled = self._assess("rank(divide(close,open))")
        uncontrolled = self._assess("divide(rank(close),rank(open))")

        self.assertTrue(controlled.eligible)
        self.assertEqual(
            uncontrolled.rejection_reasons,
            ("uncontrolled_tail",),
        )

    def test_formula_over_the_absolute_structure_limit_is_not_eligible(self) -> None:
        formula = "close"
        for _ in range(10):
            formula = f"rank({formula})"

        assessed = self._assess(formula)

        self.assertEqual(
            assessed.rejection_reasons,
            ("formula_structure_budget_exceeded",),
        )

    def test_wide_formula_over_the_complexity_limit_is_not_eligible(self) -> None:
        formula = self._balanced_sum(("close", "open") * 16)
        facts = analyze_formula(parse_formula(formula).expression)

        assessed = self._assess(formula)

        self.assertLessEqual(facts.depth, FORMULA_MAX_DEPTH)
        self.assertGreater(facts.complexity, FORMULA_MAX_COMPLEXITY)
        self.assertEqual(
            assessed.rejection_reasons,
            ("formula_structure_budget_exceeded",),
        )

    def test_tail_structure_is_reported_even_when_an_outer_operator_controls_it(
        self,
    ) -> None:
        controlled_tail = parse_formula("rank(divide(close,open))").expression
        ordinary = parse_formula("rank(close)").expression

        self.assertTrue(has_tail_risk_structure(controlled_tail))
        self.assertFalse(has_tail_risk_structure(ordinary))

    def _assess(self, formula: str):
        return assess_candidate_risk(
            parse_formula(formula).expression,
            GenerationCatalog(
                context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
                fields=(
                    self._field("close", 1.0),
                    self._field("open", 1.0),
                    self._field("sparse_a", 0.4),
                    self._field("sparse_b", 0.6),
                    self._field("empty", 0.0),
                ),
                operators=(
                    OperatorDefinition(
                        name="rank",
                        category="Cross Sectional",
                        scope=("REGULAR",),
                        parameters=(OperatorParameter("x", "expr"),),
                    ),
                ),
            ),
        )

    @staticmethod
    def _balanced_sum(terms: tuple[str, ...]) -> str:
        level = terms
        while len(level) > 1:
            level = tuple(
                f"({level[index]}+{level[index + 1]})"
                for index in range(0, len(level), 2)
            )
        return level[0]

    @staticmethod
    def _field(field_id: str, coverage: float) -> FieldDefinition:
        return FieldDefinition(
            field_id=field_id,
            dataset_id="sample",
            category="Model",
            subcategory=None,
            field_type="MATRIX",
            coverage=coverage,
        )


if __name__ == "__main__":
    unittest.main()
