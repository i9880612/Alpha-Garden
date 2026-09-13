from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.catalog_sync import CatalogRefreshPolicy
from execution.backtests import prepare_backtest_task, record_submission_unknown
from execution.initialization import initialize_project
from execution.runs import (
    AutomatedRunLimits,
    fail_automated_run,
    prepare_automated_run,
    start_automated_run,
)
from generation.parser import parse_formula
from generation.project_catalog import (
    OPERATOR_OUTPUTS,
    OPERATOR_ROLES,
    STANDARD_WINDOWS,
)
from persistence.backtests import (
    BacktestTaskRecord,
    create_backtest_task,
)
from persistence.catalog import (
    OperatorOutputRecord,
    OperatorRoleRecord,
    WindowCatalogRecord,
    get_platform_catalog_sync,
    list_operator_outputs,
    list_operator_roles,
    list_window_catalog,
    replace_window_catalog,
)
from persistence.database import open_database
from persistence.runs import AutomatedRunBacktestRecord, attach_backtest_to_automated_run
from persistence.schema import PROJECT_TABLE_NAMES, initialize_database_schema
from worldquant.catalog import DataField, DataFieldPage, Operator
from worldquant.client import WorldQuantRequestError


class FakeCatalogClient:
    def __init__(self, *, authentication_error: Exception | None = None) -> None:
        self.authentication_error = authentication_error
        self.authenticated = False
        self.field_calls = 0
        self.operator_calls = 0

    def authenticate(self) -> None:
        if self.authentication_error is not None:
            raise self.authentication_error
        self.authenticated = True

    def fetch_data_field_page(self, *, context, limit, offset) -> DataFieldPage:
        self.field_calls += 1
        if offset != 0:
            raise AssertionError("unexpected field page")
        return DataFieldPage(
            total_count=1,
            fields=(
                DataField(
                    field_id="close",
                    dataset_id="dataset",
                    dataset_name="Dataset",
                    category="sample",
                    category_id="sample",
                    subcategory=None,
                    subcategory_id=None,
                    field_type="MATRIX",
                    coverage=1.0,
                    description="close",
                    raw_payload={
                        "id": "close",
                        "region": context.region,
                        "universe": context.universe,
                        "delay": context.delay,
                    },
                ),
            ),
        )

    def fetch_operators(self) -> tuple[Operator, ...]:
        self.operator_calls += 1
        return (
            Operator(
                name="rank",
                category="Cross Sectional",
                definition="rank(x)",
                description="rank",
                documentation=None,
                level="ALL",
                scope=("REGULAR",),
                parameters=({"name": "x", "kind": "expr"},),
                raw_payload={"name": "rank"},
            ),
        )


class ProjectInitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.database_path = root / "alpha_garden.sqlite3"
        self.settings_path = root / "backtest.json"
        self.environment_path = root / ".env"
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

    def test_initializes_fresh_database_then_synchronizes_platform_catalog(
        self,
    ) -> None:
        client = FakeCatalogClient()

        result = self._initialize(client)

        self.assertTrue(result.database_created)
        self.assertTrue(result.semantics_initialized)
        self.assertTrue(result.catalog_refreshed)
        self.assertEqual(result.project_table_count, len(PROJECT_TABLE_NAMES))
        self.assertEqual((result.window_count, result.operator_role_count), (5, 28))
        self.assertEqual(result.operator_output_count, 67)
        self.assertTrue(client.authenticated)
        self.assertEqual((client.field_calls, client.operator_calls), (1, 1))
        with open_database(self.database_path) as connection:
            table_names = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
                if not row["name"].startswith("sqlite_")
            }
            history_counts = {
                table_name: connection.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                ).fetchone()[0]
                for table_name in (
                    "backtest_tasks",
                    "backtest_results",
                    "backtest_checks",
                    "backtest_yearly_stats_captures",
                    "backtest_yearly_stats",
                    "backtest_mutations",
                    "signal_seeds",
                    "platform_submitted_alphas",
                    "automated_runs",
                    "automated_run_backtests",
                    "automated_cycle_settlements",
                )
            }
            sync = get_platform_catalog_sync(connection)
            self._assert_project_semantics(connection)
        self.assertEqual(table_names, PROJECT_TABLE_NAMES)
        self.assertEqual(set(history_counts.values()), {0})
        self.assertIsNotNone(sync)
        self.assertEqual((sync.field_count, sync.operator_count), (1, 1))

    def test_reuses_complete_catalog_and_preserves_existing_history(self) -> None:
        self._initialize(FakeCatalogClient())
        parsed = parse_formula("rank(close)")
        task = BacktestTaskRecord(
            task_id="existing-task",
            account_scope="group-account",
            formula=parsed.normalized,
            formula_fingerprint=parsed.fingerprint,
            settings_json="{}",
            request_fingerprint="existing-request",
            status="created",
            remote_id=None,
            platform_alpha_id=None,
            created_at="2026-09-02T10:00:00+08:00",
            submission_started_at=None,
            last_observed_at=None,
            retry_not_before=None,
            finished_at=None,
            failure_code=None,
            failure_message=None,
        )
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, task)

        unused_client = FakeCatalogClient(
            authentication_error=AssertionError("existing_catalog_should_be_reused")
        )
        result = self._initialize(unused_client)

        self.assertFalse(result.database_created)
        self.assertFalse(result.semantics_initialized)
        self.assertFalse(result.catalog_refreshed)
        self.assertFalse(unused_client.authenticated)
        self.assertEqual((unused_client.field_calls, unused_client.operator_calls), (0, 0))
        with open_database(self.database_path) as connection:
            stored_task_id = connection.execute(
                "SELECT task_id FROM backtest_tasks"
            ).fetchone()[0]
            self._assert_project_semantics(connection)
        self.assertEqual(stored_task_id, "existing-task")

    def test_incomplete_platform_catalog_is_refreshed(self) -> None:
        self._initialize(FakeCatalogClient())
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM platform_fields")
        client = FakeCatalogClient()

        result = self._initialize(client)

        self.assertTrue(result.catalog_refreshed)
        self.assertTrue(client.authenticated)
        self.assertEqual((client.field_calls, client.operator_calls), (1, 1))
        with open_database(self.database_path) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM platform_fields").fetchone()[0],
                1,
            )

    def test_conflicting_project_semantics_fail_before_platform_access(self) -> None:
        with open_database(self.database_path) as connection:
            initialize_database_schema(connection)
            replace_window_catalog(connection, (WindowCatalogRecord(7, "week"),))
        client_created = False

        def client_factory(settings):
            nonlocal client_created
            client_created = True
            return FakeCatalogClient()

        with self.assertRaisesRegex(
            ValueError,
            "project_generation_semantics_conflict",
        ):
            initialize_project(
                self.database_path,
                self.settings_path,
                self.environment_path,
                client_factory=client_factory,
            )

        self.assertFalse(client_created)
        with open_database(self.database_path) as connection:
            self.assertEqual(
                list_window_catalog(connection),
                (WindowCatalogRecord(7, "week"),),
            )

    def test_old_schema_fails_without_changing_history_or_accessing_platform(
        self,
    ) -> None:
        with open_database(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE backtest_tasks (
                    task_id TEXT PRIMARY KEY,
                    formula TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO backtest_tasks (task_id, formula) VALUES (?, ?)",
                ("old-task", "rank(close)"),
            )
        client_created = False

        def client_factory(settings):
            nonlocal client_created
            client_created = True
            return FakeCatalogClient()

        with self.assertRaisesRegex(ValueError, "database_schema_incompatible"):
            initialize_project(
                self.database_path,
                self.settings_path,
                self.environment_path,
                client_factory=client_factory,
            )

        self.assertFalse(client_created)
        with open_database(self.database_path) as connection:
            table_names = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            old_row = connection.execute(
                "SELECT task_id, formula FROM backtest_tasks"
            ).fetchone()
        self.assertEqual(table_names, {"backtest_tasks"})
        self.assertEqual(tuple(old_row), ("old-task", "rank(close)"))

    def test_platform_failure_keeps_completed_local_initialization(self) -> None:
        error = WorldQuantRequestError(
            "worldquant_authentication_failed",
            retryable=False,
            outcome_unknown=False,
        )

        with self.assertRaisesRegex(
            WorldQuantRequestError,
            "worldquant_authentication_failed",
        ):
            self._initialize(FakeCatalogClient(authentication_error=error))

        with open_database(self.database_path) as connection:
            self._assert_project_semantics(connection)
            self.assertIsNone(get_platform_catalog_sync(connection))
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM platform_fields").fetchone()[0],
                0,
            )

    def test_complete_catalog_avoids_platform_access(self) -> None:
        self._initialize(FakeCatalogClient())
        with open_database(self.database_path) as connection:
            original_sync = get_platform_catalog_sync(connection)
            original_field_count = connection.execute(
                "SELECT COUNT(*) FROM platform_fields"
            ).fetchone()[0]
            original_operator_count = connection.execute(
                "SELECT COUNT(*) FROM platform_operators"
            ).fetchone()[0]
        error = WorldQuantRequestError(
            "worldquant_request_failed",
            retryable=False,
            outcome_unknown=False,
        )

        client = FakeCatalogClient(authentication_error=error)
        result = self._initialize(client)

        self.assertFalse(result.catalog_refreshed)
        self.assertFalse(client.authenticated)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_platform_catalog_sync(connection), original_sync)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM platform_fields").fetchone()[0],
                original_field_count,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM platform_operators").fetchone()[0],
                original_operator_count,
            )
            self._assert_project_semantics(connection)

    def test_active_run_can_reuse_complete_catalog_without_platform_access(self) -> None:
        self._initialize(FakeCatalogClient())
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
            created_at="2026-09-02T11:00:00+08:00",
        )
        client_created = False

        def client_factory(settings):
            nonlocal client_created
            client_created = True
            return FakeCatalogClient()

        result = initialize_project(
            self.database_path,
            self.settings_path,
            self.environment_path,
            client_factory=client_factory,
        )

        self.assertFalse(result.catalog_refreshed)
        self.assertFalse(client_created)

        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM generation_windows")
            connection.execute("DELETE FROM generation_operator_roles")
            connection.execute("DELETE FROM generation_operator_outputs")

        with self.assertRaisesRegex(ValueError, "catalog_refresh_bound_run_exists"):
            initialize_project(
                self.database_path,
                self.settings_path,
                self.environment_path,
                client_factory=client_factory,
            )

        self.assertFalse(client_created)
        with open_database(self.database_path) as connection:
            self.assertEqual(list_window_catalog(connection), ())
            self.assertEqual(list_operator_roles(connection), ())
            self.assertEqual(list_operator_outputs(connection), ())

    def test_unknown_submission_survives_terminal_run_and_still_blocks_init(self) -> None:
        self._initialize(FakeCatalogClient())
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
                max_pending_seconds=1,
                max_consecutive_failures=1,
                max_request_failures=1,
                max_in_flight_backtests=1,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ),
            created_at="2026-09-02T11:00:00+08:00",
        )
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-09-02T11:00:01+08:00",
        )
        with open_database(self.database_path) as connection:
            task = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula="rank(close)",
                settings=self._backtest_settings(),
                created_at="2026-09-02T11:00:01+08:00",
            )
            attach_backtest_to_automated_run(
                connection,
                AutomatedRunBacktestRecord(
                    run_id=prepared.run_id,
                    task_id=task.task.task_id,
                    cycle_number=1,
                ),
            )
            record_submission_unknown(
                connection,
                task.task.task_id,
                observed_at="2026-09-02T11:00:02+08:00",
            )

        stopped = fail_automated_run(
            self.database_path,
            prepared.run_id,
            failed_at="2026-09-02T11:00:03+08:00",
            reason="test_failure",
        )

        self.assertEqual(stopped.status, "failed")
        self.assertEqual(
            stopped.stop_reason,
            "test_failure",
        )
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM platform_catalog_syncs")
        client_created = False

        def client_factory(settings):
            nonlocal client_created
            client_created = True
            return FakeCatalogClient()

        with self.assertRaisesRegex(ValueError, "catalog_refresh_bound_run_exists"):
            initialize_project(
                self.database_path,
                self.settings_path,
                self.environment_path,
                client_factory=client_factory,
            )

        self.assertFalse(client_created)

    def _initialize(self, client: FakeCatalogClient):
        return initialize_project(
            self.database_path,
            self.settings_path,
            self.environment_path,
            refresh_policy=CatalogRefreshPolicy(page_interval_seconds=0),
            clock=lambda: datetime.fromisoformat("2026-09-02T12:00:00+08:00"),
            waiter=lambda seconds: None,
            client_factory=lambda settings: client,
        )

    def _assert_project_semantics(self, connection) -> None:
        self.assertEqual(
            list_window_catalog(connection),
            tuple(WindowCatalogRecord(*record) for record in STANDARD_WINDOWS),
        )
        self.assertEqual(
            list_operator_roles(connection),
            tuple(OperatorRoleRecord(*record) for record in OPERATOR_ROLES),
        )
        self.assertEqual(
            list_operator_outputs(connection),
            tuple(OperatorOutputRecord(*record) for record in OPERATOR_OUTPUTS),
        )

    @staticmethod
    def _backtest_settings() -> dict[str, object]:
        return {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 4,
            "neutralization": "SECTOR",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "OFF",
            "language": "FASTEXPR",
            "visualization": False,
            "maxTrade": "OFF",
            "maxPosition": "OFF",
        }


if __name__ == "__main__":
    unittest.main()
