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
from generation.expression_types import infer_formula_type
from generation.parser import parse_formula
from generation.validation import validate_formula


class ExpressionTypeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            fields=(
                self._field("close", "MATRIX"),
                self._field("vector_a", "VECTOR"),
                self._field("sector", "GROUP"),
            ),
            operators=(
                self._operator(
                    "rank",
                    "signal",
                    OperatorParameter("x", "expr"),
                ),
                self._operator(
                    "ts_mean",
                    "signal",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("d", "window"),
                ),
                self._operator(
                    "vec_avg",
                    "signal",
                    OperatorParameter("x", "expr"),
                ),
                self._operator(
                    "bucket",
                    "group",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("range", "string", optional=True),
                ),
                self._operator(
                    "densify",
                    "group",
                    OperatorParameter("x", "expr"),
                ),
                self._operator(
                    "group_rank",
                    "signal",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("group", "group"),
                ),
                self._operator(
                    "if_else",
                    "signal",
                    OperatorParameter("input1", "expr"),
                    OperatorParameter("input2", "expr"),
                    OperatorParameter("input3", "expr"),
                ),
                self._operator(
                    "trade_when",
                    "signal",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("y", "expr"),
                    OperatorParameter("z", "expr"),
                ),
            ),
        )

    def test_matrix_and_vector_reduction_produce_signals(self) -> None:
        for formula in ("close", "vec_avg(vector_a)", "ts_mean(close,22)"):
            with self.subTest(formula=formula):
                result = self._infer(formula)
                self.assertTrue(result.is_valid, result.issues)
                self.assertEqual(result.kind, "signal")

    def test_vector_cannot_enter_ordinary_signal_operator_directly(self) -> None:
        result = self._infer("ts_mean(vector_a,22)")
        validation = self._validate("ts_mean(vector_a,22)")

        self.assertEqual(
            result.issues[0].code,
            "operator_argument_type_mismatch:ts_mean:x:vector",
        )
        self.assertFalse(validation.is_valid)
        self.assertEqual(
            validation.issues[0].code,
            "operator_argument_type_mismatch:ts_mean:x:vector",
        )

    def test_group_output_can_only_enter_group_position(self) -> None:
        valid = self._infer('group_rank(close,bucket(rank(close),range="0,1,0.1"))')
        invalid = self._infer('rank(bucket(close,range="0,1,0.1"))')
        densified = self._infer("group_rank(close,densify(sector))")

        self.assertTrue(valid.is_valid, valid.issues)
        self.assertTrue(densified.is_valid, densified.issues)
        self.assertTrue(
            self._validate("group_rank(close,densify(sector))").is_valid
        )
        self.assertEqual(
            invalid.issues[0].code,
            "operator_argument_type_mismatch:rank:x:group",
        )

    def test_condition_is_required_in_control_positions(self) -> None:
        valid = self._infer("if_else(close>0,close,-close)")
        invalid = self._infer("if_else(close,close,-close)")
        trade = self._infer("trade_when(close>0,close,close<0)")

        self.assertTrue(valid.is_valid, valid.issues)
        self.assertTrue(trade.is_valid, trade.issues)
        self.assertEqual(
            invalid.issues[0].code,
            "condition_argument_required:if_else:input1:signal",
        )

    def test_vector_group_condition_and_scalar_are_not_root_alphas(self) -> None:
        cases = {
            "vector_a": "vector",
            "sector": "group",
            "close>0": "condition",
            "1": "scalar",
        }

        for formula, kind in cases.items():
            with self.subTest(formula=formula):
                result = self._infer(formula)
                self.assertEqual(result.kind, kind)
                self.assertEqual(
                    result.issues[-1].code,
                    f"formula_output_type_mismatch:{kind}",
                )

    def _infer(self, formula: str):
        return infer_formula_type(parse_formula(formula).expression, self.catalog)

    def _validate(self, formula: str):
        return validate_formula(parse_formula(formula).expression, self.catalog)

    @staticmethod
    def _field(field_id: str, field_type: str) -> FieldDefinition:
        return FieldDefinition(
            field_id=field_id,
            dataset_id="dataset",
            category="sample",
            subcategory=None,
            field_type=field_type,
            coverage=1.0,
        )

    @staticmethod
    def _operator(
        name: str,
        output_kind: str,
        *parameters: OperatorParameter,
    ) -> OperatorDefinition:
        return OperatorDefinition(
            name=name,
            category="sample",
            scope=("REGULAR",),
            parameters=tuple(parameters),
            output_kind=output_kind,
        )


if __name__ == "__main__":
    unittest.main()
