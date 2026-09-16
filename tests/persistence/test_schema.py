from __future__ import annotations

import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persistence.database import open_database
from persistence.schema import (
    PROJECT_TABLE_NAMES,
    initialize_database_schema,
    add_submission_check_storage,
    migrate_cancelled_backtest_reservations,
    add_submission_research_storage,
    add_optimization_run_storage,
    add_submission_source_storage,
    add_optimization_submission_source,
)


class DatabaseSchemaTests(unittest.TestCase):
    def test_optimization_submission_migration_preserves_attempts_and_rolls_back(self):
        from tests.execution.test_submission_runner import SubmissionQueueRunnerTests
        from execution.submission_queue import claim_next_submission_queue_item
        fixture = SubmissionQueueRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture._completed_run(("rank(close)",))
        with open_database(fixture.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            claim_next_submission_queue_item(connection, account_scope="group-account",
                observed_at="2026-09-04T01:00:00+00:00", submission_mode="manual")
            sql = connection.execute("SELECT sql FROM sqlite_master WHERE name='formal_submission_attempts'").fetchone()[0]
            indexes = [row[0] for row in connection.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='formal_submission_attempts' AND sql IS NOT NULL")]
            old_sql = sql.replace("source IN ('qualified_archive', 'optimization')", "source = 'qualified_archive'")
            self.assertNotEqual(old_sql, sql)
            connection.execute(old_sql.replace("formal_submission_attempts (", "previous_attempts (", 1))
            connection.execute("INSERT INTO previous_attempts SELECT * FROM formal_submission_attempts")
            connection.execute("DROP TABLE formal_submission_attempts")
            connection.execute("ALTER TABLE previous_attempts RENAME TO formal_submission_attempts")
            for index in indexes:
                connection.execute(index)
            connection.commit()
            before = tuple(tuple(row) for row in connection.execute("SELECT * FROM formal_submission_attempts"))
            old_schema = connection.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
            with self.assertRaisesRegex(ValueError, "requires_transaction"):
                add_optimization_submission_source(connection)
            connection.execute("BEGIN IMMEDIATE")
            add_optimization_submission_source(connection)
            self.assertEqual(tuple(tuple(row) for row in connection.execute("SELECT * FROM formal_submission_attempts")), before)
            connection.rollback()
            self.assertEqual(connection.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall(), old_schema)
            self.assertEqual(tuple(tuple(row) for row in connection.execute("SELECT * FROM formal_submission_attempts")), before)
            connection.execute("CREATE TABLE unrelated (value TEXT)")
            connection.commit()
            connection.execute("BEGIN IMMEDIATE")
            with self.assertRaisesRegex(ValueError, "schema_mismatch"):
                add_optimization_submission_source(connection)
            connection.rollback()
            connection.execute("DROP TABLE unrelated")
            connection.execute("BEGIN IMMEDIATE")
            add_optimization_submission_source(connection)
            add_optimization_submission_source(connection)
            initialize_database_schema(connection)
            self.assertEqual(tuple(tuple(row) for row in connection.execute("SELECT * FROM formal_submission_attempts")), before)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            connection.execute("UPDATE formal_submission_attempts SET source='optimization'")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE formal_submission_attempts SET submission_mode='automatic'")

    def test_submission_source_migration_preserves_attempts_is_atomic_and_rejects_unrelated_schema(self):
        from tests.execution.test_submission_runner import SubmissionQueueRunnerTests
        from execution.submission_queue import claim_next_submission_queue_item
        from persistence.submissions import get_formal_submission_attempt
        fixture = SubmissionQueueRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        task_ids, _ = fixture._completed_run(("rank(close)",))
        with open_database(fixture.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            original = claim_next_submission_queue_item(connection, account_scope="group-account",
                observed_at="2026-09-04T01:00:00+00:00", submission_mode="manual")
            connection.execute("ALTER TABLE formal_submission_attempts DROP COLUMN source")
            connection.commit()
            before = tuple(connection.execute("SELECT * FROM formal_submission_attempts").fetchone())
            with self.assertRaisesRegex(ValueError, "requires_transaction"):
                add_submission_source_storage(connection)
            connection.execute("BEGIN IMMEDIATE")
            add_submission_source_storage(connection)
            self.assertEqual(get_formal_submission_attempt(connection, task_ids[0]), original)
            connection.rollback()
            self.assertEqual(tuple(connection.execute("SELECT * FROM formal_submission_attempts").fetchone()), before)
            connection.execute("CREATE TABLE unrelated (value TEXT)")
            connection.commit()
            connection.execute("BEGIN IMMEDIATE")
            with self.assertRaisesRegex(ValueError, "schema_mismatch"):
                add_submission_source_storage(connection)
            connection.rollback()
            connection.execute("DROP TABLE unrelated")
            connection.execute("BEGIN IMMEDIATE")
            add_submission_source_storage(connection)
            add_submission_source_storage(connection)
            self.assertEqual(tuple(connection.execute("SELECT * FROM formal_submission_attempts").fetchone()), before + ("queue",))
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            initialize_database_schema(connection)

    def test_optimization_migration_requires_transaction_and_rejects_unrelated_schema(self):
        with open_database(":memory:") as connection:
            initialize_database_schema(connection)
            connection.execute("ALTER TABLE automated_runs DROP COLUMN optimization_only")
            connection.commit()
            with self.assertRaisesRegex(ValueError, "optimization_run_migration_requires_transaction"):
                add_optimization_run_storage(connection)
            connection.execute("CREATE TABLE unrelated (value TEXT)")
            connection.commit()
            before = connection.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
            connection.execute("BEGIN IMMEDIATE")
            with self.assertRaisesRegex(ValueError, "optimization_run_migration_schema_mismatch"):
                add_optimization_run_storage(connection)
            self.assertEqual(connection.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall(), before)

    def test_optimization_mode_migration_preserves_runs_and_rolls_back(self):
        from tests.persistence.test_runs import AutomatedRunPersistenceTests
        from persistence.runs import create_automated_run, get_automated_run
        with open_database(":memory:") as connection:
            initialize_database_schema(connection)
            record = AutomatedRunPersistenceTests._record("run-original")
            create_automated_run(connection, record)
            connection.execute("ALTER TABLE automated_runs DROP COLUMN optimization_only")
            connection.commit()
            before = tuple(connection.execute("SELECT * FROM automated_runs").fetchone())
            connection.execute("BEGIN IMMEDIATE")
            add_optimization_run_storage(connection)
            self.assertEqual(get_automated_run(connection, record.run_id), record)
            connection.rollback()
            self.assertEqual(tuple(connection.execute("SELECT * FROM automated_runs").fetchone()), before)
            connection.execute("BEGIN IMMEDIATE")
            add_optimization_run_storage(connection)
            add_optimization_run_storage(connection)
            self.assertEqual(tuple(connection.execute("SELECT * FROM automated_runs").fetchone()), before + (0,))
            initialize_database_schema(connection)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_archive_summary_migration_uses_source_facts_and_rolls_back(self):
        from tests.execution import test_qualified_archive as fixture
        with open_database(":memory:") as connection:
            initialize_database_schema(connection)
            parent = fixture.exhausted_parent(connection, grade="EXCELLENT")
            fixture.synchronize_qualified_alpha_archive(connection, observed_at=fixture.ARCHIVED)
            connection.commit()
            for column in ("grade", "platform_alpha_id", "sharpe", "fitness", "turnover"):
                connection.execute(f"ALTER TABLE qualified_alpha_archive DROP COLUMN {column}")
            before = tuple(tuple(row) for row in connection.execute("SELECT * FROM qualified_alpha_archive"))
            connection.execute("BEGIN IMMEDIATE")
            add_submission_research_storage(connection)
            self.assertEqual(connection.execute("SELECT grade FROM qualified_alpha_archive").fetchone()[0], "EXCELLENT")
            connection.rollback()
            self.assertEqual(tuple(tuple(row) for row in connection.execute("SELECT * FROM qualified_alpha_archive")), before)
            connection.execute("BEGIN IMMEDIATE")
            add_submission_research_storage(connection)
            add_submission_research_storage(connection)
            self.assertEqual(tuple(connection.execute("SELECT task_id, archived_at, grade FROM qualified_alpha_archive").fetchone()),
                             (parent.task.task_id, fixture.ARCHIVED, "EXCELLENT"))
            self.assertEqual(tuple(connection.execute("SELECT platform_alpha_id, sharpe, fitness, turnover FROM qualified_alpha_archive").fetchone()),
                             (parent.task.platform_alpha_id, 1.5, 1.6, 0.12))
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            initialize_database_schema(connection)

    def test_migrates_schema_missing_both_grade_and_archive_in_one_transaction(self):
        from tests.execution import test_qualified_archive as fixture
        with open_database(":memory:") as connection:
            initialize_database_schema(connection)
            parent = fixture.exhausted_parent(connection)
            connection.commit()
            connection.execute("DROP TABLE qualified_alpha_archive")
            connection.execute("ALTER TABLE backtest_results DROP COLUMN grade")
            before = tuple(tuple(row) for row in connection.execute("SELECT * FROM backtest_results ORDER BY task_id"))
            connection.execute("BEGIN IMMEDIATE")
            add_submission_research_storage(connection)
            after = tuple(tuple(row) for row in connection.execute("SELECT * FROM backtest_results ORDER BY task_id"))
            self.assertEqual(after, tuple(row + (None,) for row in before))
            created = fixture.synchronize_qualified_alpha_archive(connection, observed_at=fixture.ARCHIVED)
            self.assertEqual(tuple(row.task_id for row in created), (parent.task.task_id,))
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            initialize_database_schema(connection)

    def test_archive_migration_is_additive_idempotent_and_transactional(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "migration.sqlite3"
            with open_database(path) as connection:
                initialize_database_schema(connection)
                connection.execute("DROP TABLE qualified_alpha_archive")
                connection.execute("INSERT INTO generation_windows (value, horizon) VALUES (22, 'month')")
            with open_database(path) as connection:
                with self.assertRaisesRegex(ValueError, "submission_research_migration_requires_transaction"):
                    add_submission_research_storage(connection)
            with self.assertRaisesRegex(RuntimeError, "rollback"):
                with open_database(path) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    add_submission_research_storage(connection)
                    raise RuntimeError("rollback")
            with open_database(path) as connection:
                self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name='qualified_alpha_archive'").fetchone())
                connection.execute("BEGIN IMMEDIATE")
                add_submission_research_storage(connection)
                add_submission_research_storage(connection)
                self.assertEqual(tuple(connection.execute("SELECT value, horizon FROM generation_windows").fetchone()), (22, "month"))
                self.assertEqual(connection.execute("SELECT count(*) FROM qualified_alpha_archive").fetchone()[0], 0)
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
                initialize_database_schema(connection)

    def test_archive_migration_rejects_unrelated_schema_without_mutation(self):
        with open_database(":memory:") as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
            connection.execute("BEGIN IMMEDIATE")
            before = connection.execute("SELECT sql FROM sqlite_master").fetchall()
            with self.assertRaisesRegex(ValueError, "submission_research_migration_schema_mismatch"):
                add_submission_research_storage(connection)
            self.assertEqual(connection.execute("SELECT sql FROM sqlite_master").fetchall(), before)

    def test_grade_migration_preserves_history_leaves_unknown_and_rolls_back(self):
        from tests.execution import test_submission_runner as fixture
        source = fixture.SubmissionQueueRunnerTests()
        source.setUp()
        self.addCleanup(source.doCleanups)
        source._completed_run(("rank(close)", "rank(open)"))
        with open_database(source.database_path) as connection:
            connection.execute("ALTER TABLE backtest_results DROP COLUMN grade")
            before = tuple(tuple(row) for row in connection.execute("SELECT * FROM backtest_results ORDER BY task_id"))
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            with open_database(source.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                add_submission_research_storage(connection)
                raise RuntimeError("rollback")
        with open_database(source.database_path) as connection:
            self.assertNotIn("grade", [row[1] for row in connection.execute("PRAGMA table_info(backtest_results)")])
            connection.execute("BEGIN IMMEDIATE")
            add_submission_research_storage(connection)
            add_submission_research_storage(connection)
            after = tuple(tuple(row) for row in connection.execute("SELECT * FROM backtest_results ORDER BY task_id"))
            self.assertEqual(after, tuple(row + (None,) for row in before))
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            initialize_database_schema(connection)

    def test_grade_migration_rejects_unrelated_schema_without_mutation(self):
        with open_database(":memory:") as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
            connection.execute("BEGIN IMMEDIATE")
            before = connection.execute("SELECT sql FROM sqlite_master").fetchall()
            with self.assertRaisesRegex(ValueError, "submission_research_migration_schema_mismatch"):
                add_submission_research_storage(connection)
            self.assertEqual(connection.execute("SELECT sql FROM sqlite_master").fetchall(), before)

    def test_reservation_migration_preserves_populated_history_and_rolls_back(self):
        from tests.execution import test_submission_runner as fixture
        source = fixture.SubmissionQueueRunnerTests()
        source.setUp()
        self.addCleanup(source.doCleanups)
        _, formulas = source._completed_run(("rank(close)", "rank(open)"), lineage=(1, 0))
        source._submit(fixture.SubmissionClient(formulas), fixture.FakeTime("2026-09-04T01:00:00+00:00"))
        with sqlite3.connect(source.database_path) as connection:
            sql = connection.execute("SELECT sql FROM sqlite_master WHERE name='backtest_tasks'").fetchone()[0]
            old_sql = sql.replace("request_fingerprint TEXT NOT NULL,", "request_fingerprint TEXT NOT NULL UNIQUE,").replace(
                "UNIQUE (account_scope, remote_id),", "UNIQUE (account_scope, formula_fingerprint, settings_json), UNIQUE (account_scope, remote_id),")
            old_indexes = connection.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='backtest_tasks' "
                "AND sql IS NOT NULL AND name NOT IN ('idx_backtest_request_reservation','idx_backtest_formula_reservation')").fetchall()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(old_sql.replace("backtest_tasks (", "backtest_tasks_previous (", 1))
            connection.execute("INSERT INTO backtest_tasks_previous SELECT * FROM backtest_tasks")
            connection.execute("DROP TABLE backtest_tasks")
            connection.execute("ALTER TABLE backtest_tasks_previous RENAME TO backtest_tasks")
            for (index_sql,) in old_indexes:
                connection.execute(index_sql)
        def facts(connection):
            return {name: sorted(connection.execute(f'SELECT * FROM "{name}"').fetchall(), key=repr)
                    for name in PROJECT_TABLE_NAMES}
        with sqlite3.connect(source.database_path) as connection:
            before = facts(connection)
            schema_before = connection.execute("SELECT type,name,sql FROM sqlite_master ORDER BY type,name").fetchall()
            self.assertTrue(before['platform_submitted_alphas'])
            self.assertTrue(before['backtest_yearly_stats'])
            self.assertTrue(before['automated_cycle_settlements'])
            self.assertTrue(before['backtest_mutations'])
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            with sqlite3.connect(source.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                migrate_cancelled_backtest_reservations(connection)
                raise RuntimeError("rollback")
        with sqlite3.connect(source.database_path) as connection:
            self.assertEqual(facts(connection), before)
            self.assertEqual(connection.execute("SELECT type,name,sql FROM sqlite_master ORDER BY type,name").fetchall(), schema_before)
            connection.execute("BEGIN IMMEDIATE")
            migrate_cancelled_backtest_reservations(connection)
            migrate_cancelled_backtest_reservations(connection)
            self.assertEqual(facts(connection), before)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        with open_database(source.database_path) as connection:
            initialize_database_schema(connection)

    def test_submission_check_migration_is_additive_idempotent_and_transactional(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "migration.sqlite3"
            with open_database(path) as connection:
                initialize_database_schema(connection)
                connection.execute("DROP TABLE submission_checks")
                connection.execute("INSERT INTO generation_windows (value, horizon) VALUES (22, 'month')")
            with self.assertRaisesRegex(RuntimeError, "rollback"):
                with open_database(path) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    add_submission_check_storage(connection)
                    raise RuntimeError("rollback")
            with open_database(path) as connection:
                self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name='submission_checks'").fetchone())
                connection.execute("BEGIN IMMEDIATE")
                add_submission_check_storage(connection)
                add_submission_check_storage(connection)
                self.assertEqual(tuple(connection.execute("SELECT value, horizon FROM generation_windows").fetchone()), (22, "month"))
                self.assertEqual(connection.execute("SELECT count(*) FROM submission_checks").fetchone()[0], 0)

    def test_initializes_every_current_project_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "database.sqlite3"
            with open_database(database_path) as connection:
                initialize_database_schema(connection)
                table_names = {
                    row["name"]
                    for row in connection.execute(
                        """
                        SELECT name FROM sqlite_master
                        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                        """
                    ).fetchall()
                }

        self.assertEqual(table_names, PROJECT_TABLE_NAMES)

    def test_rejects_same_named_table_with_an_old_definition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "database.sqlite3"
            with open_database(database_path) as connection:
                connection.execute(
                    "CREATE TABLE backtest_tasks (task_id TEXT PRIMARY KEY)"
                )

            with self.assertRaisesRegex(ValueError, "database_schema_incompatible"):
                with open_database(database_path) as connection:
                    initialize_database_schema(connection)

            with open_database(database_path) as connection:
                table_names = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }

        self.assertEqual(table_names, {"backtest_tasks"})

    def test_rejects_any_unexpected_table_without_modifying_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "database.sqlite3"
            with open_database(database_path) as connection:
                initialize_database_schema(connection)
                connection.execute(
                    "INSERT INTO generation_windows (value, horizon) VALUES (22, 'month')"
                )
                connection.execute(
                    "CREATE TABLE stale_candidate_cache (identity TEXT PRIMARY KEY)"
                )
                connection.execute(
                    "INSERT INTO stale_candidate_cache (identity) VALUES ('rejected')"
                )

            with open_database(database_path) as connection:
                with self.assertRaisesRegex(
                    ValueError,
                    "database_schema_incompatible:stale_candidate_cache",
                ):
                    initialize_database_schema(connection)
                retained_rows = connection.execute(
                    "SELECT identity FROM stale_candidate_cache"
                ).fetchall()

            self.assertEqual([row[0] for row in retained_rows], ["rejected"])

            with open_database(database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DROP TABLE stale_candidate_cache")
                initialize_database_schema(connection)

            with open_database(database_path) as connection:
                table_names = {
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT name FROM sqlite_master
                        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                        """
                    ).fetchall()
                }
                existing_window = connection.execute(
                    "SELECT value, horizon FROM generation_windows"
                ).fetchone()

        self.assertEqual(table_names, PROJECT_TABLE_NAMES)
        self.assertEqual(tuple(existing_window), (22, "month"))

    def test_rejects_an_unexpected_view_before_creating_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "database.sqlite3"
            with open_database(database_path) as connection:
                connection.execute("CREATE VIEW stale_view AS SELECT 1 AS value")

            with self.assertRaisesRegex(
                ValueError,
                "database_schema_incompatible:.*stale_view",
            ):
                with open_database(database_path) as connection:
                    initialize_database_schema(connection)

            with open_database(database_path) as connection:
                objects = {
                    (row["type"], row["name"])
                    for row in connection.execute(
                        """
                        SELECT type, name
                        FROM sqlite_master
                        WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
                        """
                    ).fetchall()
                }

        self.assertEqual(objects, {("view", "stale_view")})


if __name__ == "__main__":
    unittest.main()
