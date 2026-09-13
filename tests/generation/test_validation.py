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
from generation.parser import parse_formula
from generation.validation import validate_formula


class FormulaValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            fields=tuple(self._field(name) for name in ("close", "open", "sector", "volume")),
            operators=(
                self._operator("rank", OperatorParameter("x", "expr")),
                self._operator(
                    "ts_mean",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("d", "window"),
                ),
                self._operator(
                    "ts_backfill",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("lookback", "window", optional=True),
                    OperatorParameter("k", "int", optional=True),
                ),
                self._operator(
                    "normalize",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("useStd", "bool", optional=True),
                    OperatorParameter("limit", "float", optional=True),
                ),
                self._operator(
                    "group_neutralize",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("group", "group"),
                ),
                self._operator(
                    "max",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("y", "expr", variadic=True),
                ),
            ),
        )

    def test_accepts_database_described_operator_shapes(self) -> None:
        formulas = (
            "rank(ts_mean(close,20))",
            "ts_backfill(close,lookback=20)",
            "normalize(close,useStd=true,limit=0.5)",
            "group_neutralize(close,sector)",
            "max(close,open,volume)",
        )

        for formula in formulas:
            with self.subTest(formula=formula):
                validation = validate_formula(
                    parse_formula(formula).expression,
                    self.catalog,
                )
                self.assertTrue(validation.is_valid, validation.issues)

    def test_reports_unknown_field_and_operator(self) -> None:
        field_validation = validate_formula(
            parse_formula("rank(missing)").expression,
            self.catalog,
        )
        operator_validation = validate_formula(
            parse_formula("unknown(close)").expression,
            self.catalog,
        )

        self.assertEqual(
            [item.code for item in field_validation.issues],
            ["unknown_field:missing"],
        )
        self.assertEqual(
            [item.code for item in operator_validation.issues],
            ["unknown_operator:unknown"],
        )

    def test_reports_missing_extra_and_unknown_named_arguments(self) -> None:
        cases = {
            "ts_mean(close)": "missing_argument:ts_mean:d",
            "rank(close,open)": "too_many_positional_arguments:rank",
            "normalize(close,unknown=true)": (
                "unknown_named_argument:normalize:unknown"
            ),
            "normalize(close,x=open)": "duplicate_argument:normalize:x",
        }

        for formula, expected in cases.items():
            with self.subTest(formula=formula):
                validation = validate_formula(
                    parse_formula(formula).expression,
                    self.catalog,
                )
                self.assertIn(expected, [item.code for item in validation.issues])

    def test_requires_platform_confirmed_ts_backfill_lookback(self) -> None:
        validation = validate_formula(
            parse_formula("ts_backfill(close)").expression,
            self.catalog,
        )

        self.assertEqual(
            tuple(item.code for item in validation.issues),
            ("missing_explicit_argument:ts_backfill:lookback",),
        )

    def test_reports_parameter_type_mismatch(self) -> None:
        cases = (
            "ts_mean(close,open)",
            "normalize(close,useStd=1)",
            "normalize(close,limit=volume)",
            "group_neutralize(close,1)",
        )

        for formula in cases:
            with self.subTest(formula=formula):
                validation = validate_formula(
                    parse_formula(formula).expression,
                    self.catalog,
                )
                self.assertTrue(
                    any(
                        item.code.startswith("argument_type_mismatch:")
                        for item in validation.issues
                    ),
                    validation.issues,
                )

    def test_uses_database_field_type_for_expression_and_group_names(self) -> None:
        expression_validation = validate_formula(
            parse_formula("rank(sector)").expression,
            self.catalog,
        )
        group_validation = validate_formula(
            parse_formula("group_neutralize(close,close)").expression,
            self.catalog,
        )

        self.assertEqual(
            [item.code for item in expression_validation.issues],
            ["field_type_mismatch:expr:GROUP"],
        )
        self.assertEqual(
            [item.code for item in group_validation.issues],
            ["field_type_mismatch:group:MATRIX"],
        )

    @staticmethod
    def _field(field_id: str) -> FieldDefinition:
        return FieldDefinition(
            field_id=field_id,
            dataset_id="dataset",
            category="market",
            subcategory="sample",
            field_type="GROUP" if field_id == "sector" else "MATRIX",
            coverage=1.0,
        )

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
        )


if __name__ == "__main__":
    unittest.main()
