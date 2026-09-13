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
from generation.unit_validation import find_coarse_unit_issues


class CoarseUnitValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            fields=(self._field("close"), self._field("volume")),
            operators=(
                self._operator("abs", (("x", "expr"),)),
                self._operator("add", (("x", "expr"), ("y", "expr"))),
                self._operator(
                    "if_else",
                    (
                        ("input1", "expr"),
                        ("input2", "expr"),
                        ("input3", "expr"),
                    ),
                ),
                self._operator("rank", (("x", "expr"),), normalized=True),
                self._operator(
                    "group_backfill",
                    (("x", "expr"), ("group", "group"), ("d", "window")),
                ),
                self._operator("hump", (("x", "expr"),)),
                self._operator(
                    "ts_product",
                    (("x", "expr"), ("d", "window")),
                ),
                self._operator(
                    "ts_regression",
                    (("y", "expr"), ("x", "expr"), ("d", "window")),
                ),
                self._operator(
                    "vector_neut",
                    (("x", "expr"), ("y", "expr")),
                ),
            ),
        )

    def test_detects_known_incompatible_unit_shapes(self) -> None:
        formulas = (
            "close>0.25",
            "add(abs(close),0.0001)",
            "if_else(rank(close)>0.25,close,volume)",
            "group_backfill(close,sector,120)",
            "hump(close)",
            "ts_product(close,5)",
            "ts_regression(close,rank(volume),5)",
            "ts_regression(x=rank(volume),y=close,d=5)",
            "ts_regression(rank(close),volume,5)",
            "ts_regression(x=volume,y=rank(close),d=5)",
            "vector_neut(close,volume)",
        )

        for formula in formulas:
            with self.subTest(formula=formula):
                expression = parse_formula(formula).expression
                self.assertTrue(find_coarse_unit_issues(expression, self.catalog))

    def test_accepts_dimensionless_branches(self) -> None:
        formulas = (
            "rank(close)>0.25",
            "add(rank(close),0.0001)",
            "if_else(rank(close)>0.25,rank(close),rank(volume))",
            "group_backfill(rank(close),sector,120)",
            "hump(rank(close))",
            "ts_product(rank(close),5)",
            "ts_regression(rank(close),rank(volume),5)",
            "ts_regression(x=rank(volume),y=rank(close),d=5)",
            "vector_neut(rank(close),rank(volume))",
        )

        for formula in formulas:
            with self.subTest(formula=formula):
                expression = parse_formula(formula).expression
                self.assertEqual(
                    find_coarse_unit_issues(expression, self.catalog),
                    (),
                )

    @staticmethod
    def _field(field_id: str) -> FieldDefinition:
        return FieldDefinition(
            field_id=field_id,
            dataset_id="sample",
            category="sample",
            subcategory=None,
            field_type="MATRIX",
            coverage=1.0,
        )

    @staticmethod
    def _operator(
        name: str,
        parameters: tuple[tuple[str, str], ...],
        *,
        normalized: bool = False,
    ) -> OperatorDefinition:
        return OperatorDefinition(
            name=name,
            category="sample",
            scope=("REGULAR",),
            parameters=tuple(
                OperatorParameter(parameter_name, kind)
                for parameter_name, kind in parameters
            ),
            roles=("cross_sectional_normalization",) if normalized else (),
            output_kind="signal",
        )


if __name__ == "__main__":
    unittest.main()
