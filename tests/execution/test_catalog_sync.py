from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import Mock
from datetime import datetime
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.catalog import load_generation_catalog
from execution.backtest_batches import AutomatedCandidateBacktest, prepare_automated_candidate_backtest_batch
from execution.backtests import record_submission_unknown, record_submission_accepted, fail_backtest_task
from execution.catalog_sync import (
    CatalogRefreshPolicy,
    refresh_generation_catalog,
    refresh_operator_catalog,
)
from execution.runs import (
    AutomatedRunLimits,
    fail_automated_run,
    prepare_automated_run,
    start_automated_run,
)
from persistence.backtests import initialize_backtest_schema
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
    list_operator_roles,
    list_window_catalog,
    replace_operator_outputs,
    replace_operator_roles,
    replace_platform_catalog,
    replace_window_catalog,
)
from persistence.database import open_database
from persistence.catalog_downloads import initialize_catalog_download_schema
from persistence.runs import initialize_run_schema
from persistence.runs import get_automated_run
from persistence.submissions import initialize_submission_schema
from generation.candidate import exploration_candidate
from generation.parser import parse_formula
from worldquant.backtests import BacktestSettings
from worldquant.catalog import DataField, DataFieldPage, Operator
from worldquant.client import WorldQuantRequestError


class FakeCatalogClient:
    def __init__(
        self,
        pages: dict[int, DataFieldPage | Exception],
        operator_results: list[tuple[Operator, ...] | Exception],
        operator_observer=None,
    ) -> None:
        self.pages = pages
        self.operator_results = list(operator_results)
        self.authenticated = False
        self.field_calls: list[tuple[int, int]] = []
        self.operator_calls = 0
        self.operator_observer = operator_observer

    def authenticate(self) -> None:
        self.authenticated = True

    def fetch_data_field_page(self, *, context, limit, offset):
        self.field_calls.append((limit, offset))
        result = self.pages[offset]
        if isinstance(result, Exception):
            raise result
        return result

    def fetch_operators(self) -> tuple[Operator, ...]:
        self.operator_calls += 1
        if self.operator_observer is not None:
            self.operator_observer()
        result = self.operator_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class CatalogRefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.database_path = root / "catalog.sqlite3"
        self.settings_path = root / "backtest.json"
        self.environment_path = root / ".env"
        self.context = FieldCatalogContext("EQUITY", "USA", "TOP3000", 1)
        self.settings_path.write_text(
            json.dumps(
                {
                    "catalogContext": {
                        "instrumentType": "EQUITY",
                        "region": "USA",
                        "universe": "TOP3000",
                        "delay": 1,
                    },
                    "decay": 4,
                    "neutralization": {
                        "default": "SECTOR",
                        "rootGroupNeutralize": "NONE",
                        "byFieldCategory": {"sample": "SECTOR"},
                    },
                    "truncation": {"default": 0.08, "tailRisk": 0.05},
                    "pasteurization": "ON",
                    "unitHandling": "VERIFY",
                    "nanHandling": "OFF",
                    "language": "FASTEXPR",
                    "visualization": False,
                    "maxTrade": "OFF",
                    "maxPosition": "OFF",
                }
            ),
            encoding="utf-8",
        )
        self.environment_path.write_text(
            "\n".join(
                (
                    "WQB_ACCOUNT_SCOPE=group-account",
                    "WQB_BASE_URL=https://api.worldquantbrain.com",
                    "WQB_EMAIL=user@example.com",
                    "WQB_PASSWORD=secret",
                    "WQB_SESSION_TOKEN=",
                )
            ),
            encoding="utf-8",
        )
        with open_database(self.database_path) as connection:
            initialize_catalog_schema(connection)
            initialize_catalog_download_schema(connection)
            initialize_backtest_schema(connection)
            initialize_run_schema(connection)
            initialize_submission_schema(connection)
            replace_window_catalog(
                connection,
                (
                    WindowCatalogRecord(5, "week"),
                    WindowCatalogRecord(22, "month"),
                ),
            )
            replace_operator_roles(
                connection,
                (OperatorRoleRecord("rank", "cross_sectional_normalization"),),
            )
            replace_operator_outputs(
                connection,
                (OperatorOutputRecord("rank", "signal"),),
            )
            self._write_old_catalog(connection)

    def test_operator_refresh_preserves_fields_their_age_and_research_records(self):
        with open_database(self.database_path) as connection:
            fields = list_field_catalog(connection, self.context)
            previous = load_generation_catalog(connection, self.context, account_scope="group-account")
            table_counts = {r[0]: connection.execute('SELECT count(*) FROM "'+r[0]+'"').fetchone()[0]
                            for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
                            if r[0] not in {"platform_operators", "platform_catalog_syncs"}}
        client = FakeCatalogClient({}, [(self._operator("zscore"), self._operator("new_operator"))])
        result = refresh_operator_catalog(self.database_path, self.environment_path,
            clock=lambda: datetime.fromisoformat("2026-09-02T10:00:00+08:00"),
            client_factory=lambda _: client)
        self.assertEqual(client.field_calls, [])
        self.assertEqual((result.added, result.removed), (("new_operator", "zscore"), ("rank",)))
        with open_database(self.database_path) as connection:
            self.assertEqual(list_field_catalog(connection, self.context), fields)
            self.assertEqual(get_platform_catalog_sync(connection).synced_at, fields[0].synced_at)
            current = load_generation_catalog(connection, self.context, account_scope="group-account")
            self.assertNotEqual(current.fingerprint, previous.fingerprint)
            self.assertIsNone(current.operator("rank"))
            self.assertEqual({r.synced_at for r in list_operator_catalog(connection)}, {result.observed_at})
            for table, count in table_counts.items():
                self.assertEqual(connection.execute('SELECT count(*) FROM "'+table+'"').fetchone()[0], count)

    def test_operator_refresh_failure_keeps_previous_catalog_and_retry_is_bounded(self):
        error = WorldQuantRequestError("temporary", status_code=503, retryable=True, outcome_unknown=False)
        invalid = replace(self._operator("rank"), scope=("COMBO",))
        for results, expected, calls in (([error, error], WorldQuantRequestError, 2),
                                         ([(invalid,)], ValueError, 1)):
            with self.subTest(results=expected):
                client = FakeCatalogClient({}, results)
                with self.assertRaises(expected):
                    refresh_operator_catalog(self.database_path, self.environment_path,
                        refresh_policy=CatalogRefreshPolicy(max_attempts=2), waiter=lambda _: None,
                        client_factory=lambda _: client)
                self.assertEqual(client.operator_calls, calls)
                self.assertEqual(client.field_calls, [])
                self._assert_old_catalog_preserved()

    def test_operator_refresh_account_mismatch_stops_before_authentication(self):
        self.environment_path.write_text("WQB_ACCOUNT_SCOPE=another-account\n"
            "WQB_BASE_URL=https://api.worldquantbrain.com\nWQB_SESSION_TOKEN=synthetic\n", encoding="utf-8")
        factory = Mock()
        with self.assertRaisesRegex(ValueError, "account_scope_mismatch"):
            refresh_operator_catalog(self.database_path, self.environment_path, client_factory=factory)
        factory.assert_not_called()
        self._assert_old_catalog_preserved()

    def test_refreshes_all_pages_then_atomically_switches_platform_catalog(self) -> None:
        client = FakeCatalogClient(
            {
                0: DataFieldPage(5, (self._field("a"), self._field("b"))),
                2: DataFieldPage(5, (self._field("c"), self._field("d"))),
                4: DataFieldPage(5, (self._field("e"),)),
            },
            [(self._operator("rank"), self._operator("new_operator"))],
        )
        waits: list[float] = []

        result = refresh_generation_catalog(
            self.database_path,
            self.settings_path,
            self.environment_path,
            refresh_policy=CatalogRefreshPolicy(
                page_size=2,
                page_interval_seconds=0.25,
            ),
            clock=lambda: datetime.fromisoformat("2026-09-02T10:00:00+08:00"),
            waiter=waits.append,
            client_factory=lambda settings: client,
        )

        self.assertTrue(client.authenticated)
        self.assertEqual(client.field_calls, [(2, 0), (2, 2), (2, 4)])
        self.assertEqual(waits, [0.25, 0.25])
        self.assertEqual((result.field_count, result.operator_count), (5, 2))
        with open_database(self.database_path) as connection:
            sync = get_platform_catalog_sync(connection)
            fields = list_field_catalog(connection, self.context)
            operators = list_operator_catalog(connection)
            windows = list_window_catalog(connection)
            roles = list_operator_roles(connection)
            catalog = load_generation_catalog(
                connection,
                self.context,
                account_scope="group-account",
            )
        self.assertEqual(sync.field_count, 5)
        self.assertEqual([field.field_id for field in fields], list("abcde"))
        self.assertEqual(
            [operator.operator_name for operator in operators],
            ["new_operator", "rank"],
        )
        self.assertEqual(windows, (WindowCatalogRecord(5, "week"), WindowCatalogRecord(22, "month")))
        self.assertEqual(
            roles,
            (OperatorRoleRecord("rank", "cross_sectional_normalization"),),
        )
        self.assertEqual(catalog.operator("new_operator").output_kind, None)
        self.assertEqual(result.catalog_fingerprint, catalog.fingerprint)

    def test_retries_bounded_read_with_retry_after(self) -> None:
        limited = WorldQuantRequestError(
            "worldquant_operators_http_error",
            status_code=429,
            retryable=True,
            outcome_unknown=False,
            retry_after_seconds=45.0,
        )
        client = FakeCatalogClient(
            {0: DataFieldPage(1, (self._field("close"),))},
            [limited, (self._operator("rank"),)],
        )
        waits: list[float] = []

        refresh_generation_catalog(
            self.database_path,
            self.settings_path,
            self.environment_path,
            refresh_policy=CatalogRefreshPolicy(
                page_size=2,
                page_interval_seconds=0,
                max_attempts=2,
                retry_backoff_max_seconds=30,
            ),
            clock=lambda: datetime.fromisoformat("2026-09-02T10:00:00+08:00"),
            waiter=waits.append,
            client_factory=lambda settings: client,
        )

        self.assertEqual(client.operator_calls, 2)
        self.assertEqual(waits, [45.0])

    def test_invalid_pagination_keeps_previous_catalog(self) -> None:
        cases = (
            {
                0: DataFieldPage(3, (self._field("a"), self._field("b"))),
                2: DataFieldPage(4, (self._field("c"),)),
            },
            {
                0: DataFieldPage(3, (self._field("a"), self._field("b"))),
                2: DataFieldPage(3, ()),
            },
            {
                0: DataFieldPage(3, (self._field("a"), self._field("b"))),
                2: DataFieldPage(3, (self._field("b"),)),
            },
        )
        for pages in cases:
            with self.subTest(pages=pages):
                client = FakeCatalogClient(pages, [(self._operator("rank"),)])
                with self.assertRaises(ValueError):
                    refresh_generation_catalog(
                        self.database_path,
                        self.settings_path,
                        self.environment_path,
                        refresh_policy=CatalogRefreshPolicy(
                            page_size=2,
                            page_interval_seconds=0,
                        ),
                        waiter=lambda seconds: None,
                        client_factory=lambda settings: client,
                    )
                self._assert_old_catalog_preserved()

    def test_retry_exhaustion_keeps_previous_catalog(self) -> None:
        failures = [
            WorldQuantRequestError(
                "worldquant_data_fields_request_failed",
                retryable=True,
                outcome_unknown=False,
            )
            for _ in range(2)
        ]
        client = FakeCatalogClient(
            {0: failures.pop(0)},
            [(self._operator("rank"),)],
        )

        with self.assertRaises(WorldQuantRequestError):
            refresh_generation_catalog(
                self.database_path,
                self.settings_path,
                self.environment_path,
                refresh_policy=CatalogRefreshPolicy(
                    page_size=2,
                    page_interval_seconds=0,
                    max_attempts=1,
                ),
                waiter=lambda seconds: None,
                client_factory=lambda settings: client,
            )

        self._assert_old_catalog_preserved()

    def test_active_run_blocks_refresh_before_platform_client_creation(self) -> None:
        prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=2,
                backtest_count=1,
                max_cycles=1,
                max_backtests=1,
                max_pending_seconds=60,
                max_consecutive_failures=1,
                max_request_failures=1,
                max_in_flight_backtests=1,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ),
            created_at="2026-09-02T09:00:00+08:00",
        )

        with self.assertRaisesRegex(ValueError, "catalog_refresh_bound_run_exists"):
            refresh_generation_catalog(
                self.database_path,
                self.settings_path,
                self.environment_path,
                client_factory=lambda settings: self.fail(
                    "活动运行存在时不应创建平台客户端"
                ),
            )

    def test_invalid_refresh_policy_stops_before_platform_client_creation(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "catalog_refresh_policy_invalid"):
            refresh_generation_catalog(
                self.database_path,
                self.settings_path,
                self.environment_path,
                refresh_policy=object(),
                client_factory=lambda settings: self.fail(
                    "刷新策略无效时不应创建平台客户端"
                ),
            )

    def test_missing_required_local_semantics_stops_before_client_creation(
        self,
    ) -> None:
        cases = (
            (
                "generation_windows",
                "catalog_refresh_window_catalog_empty",
                lambda connection: replace_window_catalog(
                    connection,
                    (
                        WindowCatalogRecord(5, "week"),
                        WindowCatalogRecord(22, "month"),
                    ),
                ),
            ),
            (
                "generation_operator_outputs",
                "catalog_refresh_operator_outputs_empty",
                lambda connection: replace_operator_outputs(
                    connection,
                    (OperatorOutputRecord("rank", "signal"),),
                ),
            ),
        )
        for table, error, restore in cases:
            with self.subTest(table=table):
                with open_database(self.database_path) as connection:
                    connection.execute(f"DELETE FROM {table}")
                with self.assertRaisesRegex(ValueError, error):
                    refresh_generation_catalog(
                        self.database_path,
                        self.settings_path,
                        self.environment_path,
                        client_factory=lambda settings: self.fail(
                            "本地生成语义不完整时不应创建平台客户端"
                        ),
                    )
                self._assert_old_catalog_preserved()
                with open_database(self.database_path) as connection:
                    restore(connection)

    def test_run_created_during_fetch_blocks_final_catalog_switch(self) -> None:
        def create_run() -> None:
            prepare_automated_run(
                self.database_path,
                self.settings_path,
                account_scope="group-account",
                limits=AutomatedRunLimits(
                    exploration_percent=30,
                    self_correlation_percent=30,
                    mutation_percent=40,
                    direction_validation_percent=3,
                    generation_count=2,
                    backtest_count=1,
                    max_cycles=1,
                    max_backtests=1,
                    max_pending_seconds=60,
                    max_consecutive_failures=1,
                    max_request_failures=1,
                    max_in_flight_backtests=1,
                    exploration_seed_attempt_multiplier=4,
                    real_backtests_authorized=True,
                ),
                created_at="2026-09-02T09:00:00+08:00",
            )

        client = FakeCatalogClient(
            {0: DataFieldPage(1, (self._field("new_field"),))},
            [(self._operator("rank"),)],
            operator_observer=create_run,
        )

        with self.assertRaisesRegex(ValueError, "catalog_refresh_bound_run_exists"):
            refresh_generation_catalog(
                self.database_path,
                self.settings_path,
                self.environment_path,
                refresh_policy=CatalogRefreshPolicy(
                    page_interval_seconds=0,
                ),
                client_factory=lambda settings: client,
            )

        self._assert_old_catalog_preserved()

    def test_reconcilable_failed_run_blocks_refresh_before_client_creation(
        self,
    ) -> None:
        prepared = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=2,
                backtest_count=1,
                max_cycles=1,
                max_backtests=1,
                max_pending_seconds=60,
                max_consecutive_failures=1,
                max_request_failures=1,
                max_in_flight_backtests=1,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ),
            created_at="2026-09-02T09:00:00+08:00",
        )
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-09-02T09:01:00+08:00",
        )
        tasks = prepare_automated_candidate_backtest_batch(
            self.database_path, run_id=prepared.run_id,
            candidates=(AutomatedCandidateBacktest(
                exploration_candidate(parse_formula("rank(old_field)").expression),
                BacktestSettings("EQUITY", "USA", "TOP3000", 1, 4, "SECTOR", .08,
                                 "ON", "VERIFY", "OFF", "FASTEXPR", False, "OFF", "OFF"),
            ),), created_at="2026-09-02T09:01:10+08:00",
        )
        task_id = tasks[0].task.task_id
        with open_database(self.database_path) as connection:
            record_submission_unknown(connection, task_id, observed_at="2026-09-02T09:01:20+08:00")
        fail_automated_run(
            self.database_path,
            prepared.run_id,
            failed_at="2026-09-02T09:02:00+08:00",
            reason="submission_reconciliation_required",
        )

        with self.assertRaisesRegex(ValueError, "catalog_refresh_bound_run_exists"):
            refresh_generation_catalog(
                self.database_path,
                self.settings_path,
                self.environment_path,
                client_factory=lambda settings: self.fail(
                    "等待提交对账的运行存在时不应创建平台客户端"
                ),
            )

        self._assert_old_catalog_preserved()

        with open_database(self.database_path) as connection:
            record_submission_accepted(connection, task_id,
                remote_id="https://api.worldquantbrain.com/simulations/synthetic",
                observed_at="2026-09-02T09:03:00+08:00")
        with self.assertRaisesRegex(ValueError, "catalog_refresh_bound_run_exists"):
            refresh_operator_catalog(self.database_path, self.environment_path,
                client_factory=lambda settings: self.fail("pending tasks must still block refresh"))

        with open_database(self.database_path) as connection:
            fail_backtest_task(connection, task_id, failure_code="platform_error",
                failure_message="synthetic failure", observed_at="2026-09-02T09:04:00+08:00")
            historical_run = get_automated_run(connection, prepared.run_id)
        client = FakeCatalogClient({}, [(self._operator("rank"),)])
        refresh_operator_catalog(self.database_path, self.environment_path,
            client_factory=lambda settings: client)
        self.assertEqual(client.operator_calls, 1)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_automated_run(connection, prepared.run_id), historical_run)

    def _write_old_catalog(self, connection) -> None:
        synced_at = "2026-08-29T00:00:00+08:00"
        field = FieldCatalogRecord(
            context=self.context,
            field_id="old_field",
            dataset_id="old",
            category="sample",
            subcategory=None,
            field_type="MATRIX",
            coverage=1.0,
            description="old",
            dataset_name="old",
            category_id="sample",
            subcategory_id=None,
            raw_payload={"id": "old_field"},
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
        replace_platform_catalog(
            connection,
            PlatformCatalogSyncRecord(
                account_scope="group-account",
                context=self.context,
                field_count=1,
                operator_count=1,
                synced_at=synced_at,
            ),
            (field,),
            (operator,),
        )

    def _assert_old_catalog_preserved(self) -> None:
        with open_database(self.database_path) as connection:
            sync = get_platform_catalog_sync(connection)
            fields = list_field_catalog(connection, self.context)
            operators = list_operator_catalog(connection)
        self.assertEqual(sync.synced_at, "2026-08-29T00:00:00+08:00")
        self.assertEqual([field.field_id for field in fields], ["old_field"])
        self.assertEqual([operator.operator_name for operator in operators], ["rank"])

    @staticmethod
    def _field(field_id: str) -> DataField:
        return DataField(
            field_id=field_id,
            dataset_id="dataset",
            dataset_name="Dataset",
            category="sample",
            category_id="sample",
            subcategory=None,
            subcategory_id=None,
            field_type="MATRIX",
            coverage=1.0,
            description=field_id,
            raw_payload={
                "id": field_id,
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
            },
        )

    @staticmethod
    def _operator(name: str) -> Operator:
        return Operator(
            name=name,
            category="Cross Sectional",
            definition=f"{name}(x)",
            description=name,
            documentation=None,
            level="ALL",
            scope=("REGULAR",),
            parameters=({"name": "x", "kind": "expr"},),
            raw_payload={"name": name},
        )


if __name__ == "__main__":
    unittest.main()
