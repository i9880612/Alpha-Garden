from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generation.formula import render_formula
from generation.logic import prepare_formula_logic
from generation.parser import parse_formula


class FormulaLogicTests(unittest.TestCase):
    def test_rejects_direct_and_simplified_zero_denominators(self) -> None:
        cases = (
            ("close/0", "formula.right"),
            ("close/(1-1)", "formula.right"),
            ("divide(close,0)", "formula.arguments[1]"),
            ("divide(x=close,y=0)", "formula.arguments[1]"),
            ("inverse(0)", "formula.arguments[0]"),
        )

        for formula, location in cases:
            with self.subTest(formula=formula):
                result = self._prepare(formula)
                self.assertEqual(result.issue.code, "division_by_zero")
                self.assertEqual(result.issue.location, location)

    def test_rejects_self_division_before_platform_backtest(self) -> None:
        for formula in ("close/close", "divide(close,close)"):
            with self.subTest(formula=formula):
                result = self._prepare(formula)
                self.assertEqual(result.issue.code, "self_division")

    def test_rejects_complete_cancellation_at_any_depth(self) -> None:
        formulas = (
            "close-close",
            "close+(-close)",
            "subtract(close,close)",
            "subtract(y=close,x=close)",
            "rank(volume+(close-close))",
        )

        for formula in formulas:
            with self.subTest(formula=formula):
                result = self._prepare(formula)
                self.assertEqual(result.issue.code, "complete_cancellation")

    def test_rejects_formula_that_simplifies_to_constant(self) -> None:
        for formula in ("1+2", "close*0", "multiply(close,0)"):
            with self.subTest(formula=formula):
                result = self._prepare(formula)
                self.assertEqual(result.issue.code, "constant_expression")

    def test_applies_only_safe_identity_simplifications(self) -> None:
        cases = {
            "(close+0)*1": "close",
            "divide(close,1)": "close",
            "reverse(reverse(close))": "close",
            "-(-close)": "close",
        }

        for formula, expected in cases.items():
            with self.subTest(formula=formula):
                result = self._prepare(formula)
                self.assertTrue(result.is_valid)
                self.assertEqual(render_formula(result.expression), expected)

    def test_keeps_platform_calls_opaque_when_time_behavior_is_unknown(self) -> None:
        for formula in ("ts_step()", "ts_mean(close,22)"):
            with self.subTest(formula=formula):
                result = self._prepare(formula)
                self.assertTrue(result.is_valid)
                self.assertEqual(render_formula(result.expression), formula)

    @staticmethod
    def _prepare(formula: str):
        return prepare_formula_logic(parse_formula(formula).expression)


if __name__ == "__main__":
    unittest.main()
