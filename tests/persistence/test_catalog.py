from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persistence.catalog import (
    FieldCatalogContext,
    FieldCatalogRecord,
    OperatorCatalogRecord,
    OperatorOutputRecord,
    OperatorRoleRecord,
    PlatformCatalogSyncRecord,
    WindowCatalogRecord,
    get_platform_catalog_sync,
    initialize_catalog_schema,
    list_field_catalog,
    list_operator_catalog,
    list_operator_outputs,
    list_operator_roles,
    list_window_catalog,
    replace_field_catalog,
    replace_operator_catalog,
    replace_operator_outputs,
    replace_operator_roles,
    replace_platform_catalog,
    replace_window_catalog,
)
from persistence.database import open_database


class CatalogPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "catalog.sqlite3"
        self.context = FieldCatalogContext("EQUITY", "USA", "TOP3000", 1)
        with open_database(self.database_path) as connection:
            initialize_catalog_schema(connection)

    def test_replaces_and_reads_one_field_context(self) -> None:
        records = (
            self._field("volume", coverage=0.97),
            self._field("close", coverage=1.0),
        )

        with open_database(self.database_path) as connection:
            replace_field_catalog(connection, self.context, records)
        with open_database(self.database_path) as connection:
            loaded = list_field_catalog(connection, self.context)

        self.assertEqual([item.field_id for item in loaded], ["close", "volume"])
        self.assertEqual(loaded[0].raw_payload, {"id": "close"})

    def test_field_replacement_does_not_delete_another_context(self) -> None:
        other_context = FieldCatalogContext("EQUITY", "USA", "TOP500", 1)
        with open_database(self.database_path) as connection:
            replace_field_catalog(
                connection,
                self.context,
                (self._field("close"),),
            )
            replace_field_catalog(
                connection,
                other_context,
                (self._field("volume", context=other_context),),
            )
            replace_field_catalog(
                connection,
                self.context,
                (self._field("returns"),),
            )

        with open_database(self.database_path) as connection:
            current = list_field_catalog(connection, self.context)
            other = list_field_catalog(connection, other_context)

        self.assertEqual([item.field_id for item in current], ["returns"])
        self.assertEqual([item.field_id for item in other], ["volume"])

    def test_replaces_and_reads_operator_parameters(self) -> None:
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

        with open_database(self.database_path) as connection:
            replace_operator_catalog(connection, (operator,))
        with open_database(self.database_path) as connection:
            loaded = list_operator_catalog(connection)

        self.assertEqual(loaded, (operator,))

    def test_invalid_replacement_keeps_existing_catalog(self) -> None:
        with open_database(self.database_path) as connection:
            replace_field_catalog(
                connection,
                self.context,
                (self._field("close"),),
            )

        invalid = self._field("")
        with self.assertRaisesRegex(ValueError, "field_catalog_id_missing"):
            with open_database(self.database_path) as connection:
                replace_field_catalog(connection, self.context, (invalid,))

        with open_database(self.database_path) as connection:
            loaded = list_field_catalog(connection, self.context)

        self.assertEqual([item.field_id for item in loaded], ["close"])

    def test_replaces_one_complete_platform_catalog_and_sync_identity(self) -> None:
        synced_at = "2026-09-02T00:00:00+00:00"
        field = self._field("close", synced_at=synced_at)
        operator = self._operator("rank", synced_at=synced_at)
        sync = PlatformCatalogSyncRecord(
            account_scope="group-account",
            context=self.context,
            field_count=1,
            operator_count=1,
            synced_at=synced_at,
        )

        with open_database(self.database_path) as connection:
            replace_platform_catalog(connection, sync, (field,), (operator,))
        with open_database(self.database_path) as connection:
            loaded_sync = get_platform_catalog_sync(connection)
            fields = list_field_catalog(connection, self.context)
            operators = list_operator_catalog(connection)

        self.assertEqual(loaded_sync, sync)
        self.assertEqual(fields, (field,))
        self.assertEqual(operators, (operator,))

    def test_complete_platform_replacement_rolls_back_all_tables(self) -> None:
        old_time = "2026-09-02T00:00:00+00:00"
        old_sync = PlatformCatalogSyncRecord(
            account_scope="group-account",
            context=self.context,
            field_count=1,
            operator_count=1,
            synced_at=old_time,
        )
        with open_database(self.database_path) as connection:
            replace_platform_catalog(
                connection,
                old_sync,
                (self._field("close", synced_at=old_time),),
                (self._operator("rank", synced_at=old_time),),
            )
            connection.execute(
                """
                CREATE TRIGGER reject_broken_operator
                BEFORE INSERT ON platform_operators
                WHEN NEW.operator_name = 'broken'
                BEGIN
                    SELECT RAISE(ABORT, 'injected operator failure');
                END
                """
            )

        new_time = "2026-09-02T01:00:00+00:00"
        new_sync = PlatformCatalogSyncRecord(
            account_scope="group-account",
            context=self.context,
            field_count=1,
            operator_count=1,
            synced_at=new_time,
        )
        with self.assertRaises(sqlite3.IntegrityError):
            with open_database(self.database_path) as connection:
                replace_platform_catalog(
                    connection,
                    new_sync,
                    (self._field("volume", synced_at=new_time),),
                    (self._operator("broken", synced_at=new_time),),
                )

        with open_database(self.database_path) as connection:
            self.assertEqual(get_platform_catalog_sync(connection), old_sync)
            self.assertEqual(
                [item.field_id for item in list_field_catalog(connection, self.context)],
                ["close"],
            )
            self.assertEqual(
                [item.operator_name for item in list_operator_catalog(connection)],
                ["rank"],
            )

    def test_replaces_and_reads_generation_windows(self) -> None:
        windows = (
            WindowCatalogRecord(60, "long"),
            WindowCatalogRecord(5, "short"),
            WindowCatalogRecord(20, "medium"),
        )

        with open_database(self.database_path) as connection:
            replace_window_catalog(connection, windows)
        with open_database(self.database_path) as connection:
            loaded = list_window_catalog(connection)

        self.assertEqual(
            loaded,
            (
                WindowCatalogRecord(5, "short"),
                WindowCatalogRecord(20, "medium"),
                WindowCatalogRecord(60, "long"),
            ),
        )

    def test_invalid_window_replacement_keeps_existing_catalog(self) -> None:
        with open_database(self.database_path) as connection:
            replace_window_catalog(
                connection,
                (WindowCatalogRecord(5, "short"),),
            )

        with self.assertRaisesRegex(ValueError, "window_catalog_value_invalid"):
            with open_database(self.database_path) as connection:
                replace_window_catalog(
                    connection,
                    (WindowCatalogRecord(0, "short"),),
                )

        with open_database(self.database_path) as connection:
            loaded = list_window_catalog(connection)

        self.assertEqual(loaded, (WindowCatalogRecord(5, "short"),))

    def test_replaces_and_reads_generation_operator_roles(self) -> None:
        roles = (
            OperatorRoleRecord("ts_mean", "time_series_smoothing"),
            OperatorRoleRecord("rank", "cross_sectional_normalization"),
        )

        with open_database(self.database_path) as connection:
            replace_operator_roles(connection, roles)
        with open_database(self.database_path) as connection:
            loaded = list_operator_roles(connection)

        self.assertEqual(
            loaded,
            (
                OperatorRoleRecord("rank", "cross_sectional_normalization"),
                OperatorRoleRecord("ts_mean", "time_series_smoothing"),
            ),
        )

    def test_replaces_and_reads_generation_operator_outputs(self) -> None:
        outputs = (
            OperatorOutputRecord("rank", "signal"),
            OperatorOutputRecord("bucket", "group"),
        )

        with open_database(self.database_path) as connection:
            replace_operator_outputs(connection, outputs)
        with open_database(self.database_path) as connection:
            loaded = list_operator_outputs(connection)

        self.assertEqual(
            loaded,
            (
                OperatorOutputRecord("bucket", "group"),
                OperatorOutputRecord("rank", "signal"),
            ),
        )

    def test_schema_constraints_reject_duplicate_field_identity(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with open_database(self.database_path) as connection:
                connection.execute(
                    """
                    INSERT INTO platform_fields (
                        instrument_type, region, universe, delay, field_id,
                        raw_payload_json, synced_at
                    ) VALUES ('EQUITY', 'USA', 'TOP3000', 1, 'close', '{}', 'now')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO platform_fields (
                        instrument_type, region, universe, delay, field_id,
                        raw_payload_json, synced_at
                    ) VALUES ('EQUITY', 'USA', 'TOP3000', 1, 'close', '{}', 'now')
                    """
                )

    def _field(
        self,
        field_id: str,
        *,
        context: FieldCatalogContext | None = None,
        coverage: float = 1.0,
        synced_at: str = "2026-08-28T00:00:00+00:00",
    ) -> FieldCatalogRecord:
        return FieldCatalogRecord(
            context=context or self.context,
            field_id=field_id,
            dataset_id="dataset",
            category="market",
            subcategory="price",
            field_type="MATRIX",
            coverage=coverage,
            description="sample",
            dataset_name="sample dataset",
            category_id="market",
            subcategory_id="price",
            raw_payload={"id": field_id},
            synced_at=synced_at,
        )

    @staticmethod
    def _operator(
        name: str,
        *,
        synced_at: str,
    ) -> OperatorCatalogRecord:
        return OperatorCatalogRecord(
            operator_name=name,
            category="Cross Sectional",
            definition=f"{name}(x)",
            description=name,
            documentation=None,
            level="ALL",
            scope=("REGULAR",),
            parameters=({"name": "x", "kind": "expr"},),
            raw_payload={"name": name},
            synced_at=synced_at,
        )


if __name__ == "__main__":
    unittest.main()
