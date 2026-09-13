from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from worldquant.catalog import (
    CatalogContext,
    WorldQuantCatalogProtocolError,
    parse_data_field_page,
    parse_operators,
)


class WorldQuantCatalogProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = CatalogContext("EQUITY", "USA", "TOP3000", 1)

    def test_parses_field_page_and_preserves_platform_payload(self) -> None:
        payload = {
            "count": 2,
            "results": [
                self._field("close", coverage=1.0),
                self._field("volume", coverage=0.98),
            ],
        }

        page = parse_data_field_page(payload, expected_context=self.context)

        self.assertEqual(page.total_count, 2)
        self.assertEqual([field.field_id for field in page.fields], ["close", "volume"])
        self.assertEqual(page.fields[0].dataset_id, "dataset-1")
        self.assertEqual(page.fields[0].category, "Price Volume")
        self.assertEqual(page.fields[0].coverage, 1.0)
        self.assertEqual(page.fields[0].raw_payload["id"], "close")

    def test_rejects_field_total_duplicates_and_context_drift(self) -> None:
        cases = (
            ({"count": 0, "results": []}, "worldquant_data_fields_total_invalid"),
            (
                {
                    "count": 2,
                    "results": [self._field("close"), self._field("close")],
                },
                "worldquant_data_fields_duplicate_id",
            ),
            (
                {
                    "count": 1,
                    "results": [self._field("close", region="EUR")],
                },
                "worldquant_data_field_context_mismatch",
            ),
        )
        for payload, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(WorldQuantCatalogProtocolError, error):
                    parse_data_field_page(payload, expected_context=self.context)

    def test_parses_function_variadic_and_expression_operator_parameters(self) -> None:
        operators = parse_operators(
            [
                self._operator("ts_rank", "ts_rank(x, d, constant = 0)"),
                self._operator(
                    "multiply",
                    "multiply(x, y, ..., filter=false), x * y",
                ),
                self._operator("greater", "input1 > input2"),
            ]
        )

        by_name = {operator.name: operator for operator in operators}
        self.assertEqual(
            by_name["ts_rank"].parameters,
            (
                {"name": "x", "kind": "expr"},
                {"name": "d", "kind": "window"},
                {"name": "constant", "kind": "float", "optional": True},
            ),
        )
        self.assertEqual(
            by_name["multiply"].parameters,
            (
                {"name": "x", "kind": "expr"},
                {"name": "y", "kind": "expr", "variadic": True},
                {"name": "filter", "kind": "bool", "optional": True},
            ),
        )
        self.assertEqual(
            by_name["greater"].parameters,
            (
                {"name": "input1", "kind": "expr"},
                {"name": "input2", "kind": "expr"},
            ),
        )

    def test_accepts_results_wrapper_and_rejects_duplicate_operator_names(self) -> None:
        wrapped = parse_operators(
            {"results": [self._operator("rank", "rank(x, rate=2)")]}
        )
        self.assertEqual(wrapped[0].name, "rank")

        with self.assertRaisesRegex(
            WorldQuantCatalogProtocolError,
            "worldquant_operators_duplicate_name",
        ):
            parse_operators(
                [
                    self._operator("rank", "rank(x)"),
                    self._operator("rank", "rank(x)"),
                ]
            )

    def _field(
        self,
        field_id: str,
        *,
        coverage: float = 1.0,
        region: str = "USA",
    ) -> dict[str, object]:
        return {
            "id": field_id,
            "dataset": {"id": "dataset-1", "name": "Dataset"},
            "category": {"id": "pv", "name": "Price Volume"},
            "subcategory": {"id": "price", "name": "Price"},
            "type": "MATRIX",
            "coverage": coverage,
            "description": field_id,
            "region": region,
            "universe": "TOP3000",
            "delay": 1,
        }

    @staticmethod
    def _operator(name: str, definition: str) -> dict[str, object]:
        return {
            "name": name,
            "category": "sample",
            "definition": definition,
            "description": name,
            "documentation": None,
            "level": "ALL",
            "scope": ["REGULAR"],
        }


if __name__ == "__main__":
    unittest.main()
