from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.catalog import load_generation_catalog
from generation.catalog import OperatorParameter
from persistence.catalog import (
    FieldCatalogContext,
    FieldCatalogRecord,
    OperatorCatalogRecord,
    OperatorOutputRecord,
    OperatorRoleRecord,
    PlatformCatalogSyncRecord,
    WindowCatalogRecord,
    initialize_catalog_schema,
    replace_operator_outputs,
    replace_operator_roles,
    replace_platform_catalog,
    replace_window_catalog,
)
from persistence.database import open_database


class LoadGenerationCatalogTests(unittest.TestCase):
    def test_loads_an_immutable_generation_snapshot_from_database(self) -> None:
        context = FieldCatalogContext("EQUITY", "USA", "TOP3000", 1)
        field = FieldCatalogRecord(
            context=context,
            field_id="close",
            dataset_id="dataset",
            category="market",
            subcategory="price",
            field_type="MATRIX",
            coverage=1.0,
            description="close price",
            dataset_name="sample",
            category_id="market",
            subcategory_id="price",
            raw_payload={"id": "close"},
            synced_at="2026-08-28T00:00:00+00:00",
        )
        operator = OperatorCatalogRecord(
            operator_name="ts_mean",
            category="Time Series",
            definition="ts_mean(x, d)",
            description="mean",
            documentation=None,
            level="ALL",
            scope=("REGULAR",),
            parameters=(
                {"name": "x", "kind": "expr"},
                {"name": "d", "kind": "window"},
            ),
            raw_payload={"name": "ts_mean"},
            synced_at="2026-08-28T00:00:00+00:00",
        )
        ts_step = OperatorCatalogRecord(
            operator_name="ts_step",
            category="Time Series",
            definition="ts_step(1)",
            description="counter",
            documentation="/operators/ts_step",
            level="ALL",
            scope=("REGULAR",),
            parameters=(),
            raw_payload={"name": "ts_step", "definition": "ts_step(1)"},
            synced_at="2026-08-28T00:00:00+00:00",
        )
        combo_operator = OperatorCatalogRecord(
            operator_name="combo_only",
            category="Composite",
            definition="combo_only(x)",
            description="not available to regular alphas",
            documentation=None,
            level="ALL",
            scope=("COMBO",),
            parameters=({"name": "x", "kind": "expr"},),
            raw_payload={"name": "combo_only"},
            synced_at="2026-08-28T00:00:00+00:00",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.sqlite3"
            with open_database(path) as connection:
                initialize_catalog_schema(connection)
                replace_operator_roles(
                    connection,
                    (
                        OperatorRoleRecord(
                            "ts_mean",
                            "time_series_smoothing",
                        ),
                    ),
                )
                replace_operator_outputs(
                    connection,
                    (
                        OperatorOutputRecord("ts_mean", "signal"),
                        OperatorOutputRecord("ts_step", "signal"),
                        OperatorOutputRecord("combo_only", "signal"),
                    ),
                )
                replace_window_catalog(
                    connection,
                    (
                        WindowCatalogRecord(60, "long"),
                        WindowCatalogRecord(5, "short"),
                    ),
                )
                replace_platform_catalog(
                    connection,
                    PlatformCatalogSyncRecord(
                        account_scope="group-account",
                        context=context,
                        field_count=1,
                        operator_count=3,
                        synced_at="2026-08-28T00:00:00+00:00",
                    ),
                    (field,),
                    (operator, ts_step, combo_operator),
                )
            with open_database(path) as connection:
                catalog = load_generation_catalog(
                    connection,
                    context,
                    account_scope="group-account",
                )

        self.assertEqual(catalog.context.universe, "TOP3000")
        self.assertEqual(catalog.field("close").field_type, "MATRIX")
        self.assertEqual(catalog.operator("ts_mean").parameters[1].kind, "window")
        self.assertTrue(catalog.operator("ts_mean").uses_window)
        self.assertEqual(
            catalog.operator("ts_mean").roles,
            ("time_series_smoothing",),
        )
        self.assertEqual(catalog.operator("ts_mean").output_kind, "signal")
        self.assertEqual(
            catalog.operator("ts_step").parameters,
            (OperatorParameter("step", "int"),),
        )
        self.assertIsNone(catalog.operator("combo_only"))
        self.assertEqual(catalog.window_values(), (5, 60))
        self.assertEqual(catalog.window_values("short"), (5,))
        self.assertEqual(len(catalog.fingerprint), 64)

    def test_rejects_missing_or_mismatched_sync_identity(self) -> None:
        context = FieldCatalogContext("EQUITY", "USA", "TOP3000", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.sqlite3"
            with open_database(path) as connection:
                with self.assertRaisesRegex(
                    ValueError,
                    "generation_catalog_sync_missing",
                ):
                    load_generation_catalog(
                        connection,
                        context,
                        account_scope="group-account",
                    )

    def test_rejects_account_and_count_drift_before_generation(self) -> None:
        context = FieldCatalogContext("EQUITY", "USA", "TOP3000", 1)
        synced_at = "2026-09-02T00:00:00+00:00"
        field = FieldCatalogRecord(
            context=context,
            field_id="close",
            dataset_id="dataset",
            category="market",
            subcategory="price",
            field_type="MATRIX",
            coverage=1.0,
            description="close",
            dataset_name="sample",
            category_id="market",
            subcategory_id="price",
            raw_payload={"id": "close"},
            synced_at=synced_at,
        )
        operator = OperatorCatalogRecord(
            operator_name="rank",
            category="Cross Sectional",
            definition="rank(x)",
            description="rank",
            documentation=None,
            level="ALL",
            scope=("REGULAR",),
            parameters=({"name": "x", "kind": "expr"},),
            raw_payload={"name": "rank"},
            synced_at=synced_at,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.sqlite3"
            with open_database(path) as connection:
                initialize_catalog_schema(connection)
                replace_operator_outputs(
                    connection,
                    (OperatorOutputRecord("rank", "signal"),),
                )
                replace_window_catalog(
                    connection,
                    (WindowCatalogRecord(22, "month"),),
                )
                replace_platform_catalog(
                    connection,
                    PlatformCatalogSyncRecord(
                        account_scope="group-account",
                        context=context,
                        field_count=1,
                        operator_count=1,
                        synced_at=synced_at,
                    ),
                    (field,),
                    (operator,),
                )

            with open_database(path) as connection:
                with self.assertRaisesRegex(
                    ValueError,
                    "generation_catalog_account_scope_mismatch",
                ):
                    load_generation_catalog(
                        connection,
                        context,
                        account_scope="other-account",
                    )
                connection.execute(
                    "UPDATE platform_catalog_syncs SET field_count = 2"
                )
            with open_database(path) as connection:
                with self.assertRaisesRegex(
                    ValueError,
                    "generation_catalog_field_count_mismatch",
                ):
                    load_generation_catalog(
                        connection,
                        context,
                        account_scope="group-account",
                    )


if __name__ == "__main__":
    unittest.main()
