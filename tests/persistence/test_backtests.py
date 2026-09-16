from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persistence.backtests import (
    BacktestCheckRecord,
    BacktestMutationRecord,
    BacktestResultRecord,
    BacktestTaskRecord,
    BacktestYearlyStatRecord,
    create_backtest_mutation,
    create_backtest_task,
    get_backtest_mutation,
    get_backtest_task,
    get_backtest_yearly_stats,
    initialize_backtest_schema,
    list_active_backtest_tasks,
    list_backtest_tasks,
    list_started_backtest_task_ids,
    list_completed_backtests,
    list_terminal_backtests,
    list_backtest_formulas,
    list_backtest_mutations,
    replace_backtest_task,
    save_backtest_result,
    save_backtest_yearly_stats,
)
from persistence.database import open_database


class BacktestPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "backtests.sqlite3"
        with open_database(self.database_path) as connection:
            initialize_backtest_schema(connection)

    def test_task_owns_formula_and_exact_replay_is_stable(self) -> None:
        task = self._task("rank(close)")
        with open_database(self.database_path) as connection:
            first = create_backtest_task(connection, task)
            replayed = create_backtest_task(connection, task)

        self.assertEqual(first, replayed)
        self.assertEqual(first.formula, "rank(close)")

    def test_same_complete_identity_reuses_the_reserved_task(self) -> None:
        first = self._task("rank(close)")
        second = replace(
            first,
            task_id="another_task",
            request_fingerprint="another_request",
            created_at="2026-08-29T00:02:00+00:00",
        )
        with open_database(self.database_path) as connection:
            created = create_backtest_task(connection, first)
            replayed = create_backtest_task(connection, second)

        self.assertEqual(created, replayed)

    def test_formula_list_contains_only_prepared_backtests(self) -> None:
        with open_database(self.database_path) as connection:
            self.assertEqual(list_backtest_formulas(connection), ())
            create_backtest_task(connection, self._task("rank(close)"))
            self.assertEqual(list_backtest_formulas(connection), ("rank(close)",))

    def test_task_list_includes_all_statuses_in_stable_creation_order(self) -> None:
        active_task = replace(
            self._task("rank(high)"),
            created_at="2026-08-29T00:02:00+00:00",
        )
        completed_task = self._task("rank(close)")
        failed_task = self._task("rank(open)")
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, active_task)

            create_backtest_task(connection, failed_task)
            failed = replace(
                failed_task,
                status="failed",
                last_observed_at="2026-08-29T00:04:00+00:00",
                finished_at="2026-08-29T00:04:00+00:00",
                failure_code="platform_zero_capital",
                failure_message="没有形成有效持仓",
            )
            replace_backtest_task(
                connection,
                failed,
                expected_status="created",
            )

            create_backtest_task(connection, completed_task)
            completed_pending = self._pending(
                completed_task,
                remote_id="simulation_1",
                platform_alpha_id="alpha_1",
            )
            replace_backtest_task(
                connection,
                completed_pending,
                expected_status="created",
            )
            completed = self._complete_task(
                connection,
                completed_pending,
            )

            tasks = list_backtest_tasks(connection)
            self.assertEqual(list_backtest_tasks(connection, task_ids=frozenset()), ())
            self.assertEqual(list_backtest_tasks(connection, task_ids=frozenset({active_task.task_id})),
                             (active_task,))
            self.assertEqual(list_started_backtest_task_ids(connection), frozenset({completed.task_id}))

        expected = tuple(
            sorted(
                (active_task, completed, failed),
                key=lambda task: (task.created_at, task.task_id),
            )
        )
        self.assertEqual(tasks, expected)
        self.assertTrue(all(isinstance(task, BacktestTaskRecord) for task in tasks))
        self.assertEqual(
            {task.status for task in tasks},
            {"created", "completed", "failed"},
        )

    def test_remote_identity_is_unique_across_tasks(self) -> None:
        first = self._task("rank(close)")
        second = self._task("rank(open)")
        other_account = replace(
            self._task("rank(high)"),
            account_scope="other-account",
        )
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, first)
            create_backtest_task(connection, second)
            replace_backtest_task(
                connection,
                self._pending(first, remote_id="simulation_1"),
                expected_status="created",
            )

        with self.assertRaisesRegex(Exception, "UNIQUE constraint failed"):
            with open_database(self.database_path) as connection:
                replace_backtest_task(
                    connection,
                    self._pending(second, remote_id="simulation_1"),
                    expected_status="created",
                )

        with open_database(self.database_path) as connection:
            create_backtest_task(connection, other_account)
            pending = replace_backtest_task(
                connection,
                self._pending(other_account, remote_id="simulation_1"),
                expected_status="created",
            )

        self.assertEqual(pending.account_scope, "other-account")

    def test_result_is_one_to_one_and_conflicting_replay_fails(self) -> None:
        task = self._task("rank(close)")
        result = self._result(task.task_id)
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, task)
            pending = self._pending(task, remote_id="simulation_1")
            replace_backtest_task(connection, pending, expected_status="created")
            pending = replace(pending, platform_alpha_id="alpha_1")
            replace_backtest_task(connection, pending, expected_status="pending")
            self.assertEqual(save_backtest_result(connection, result), result)
            self.assertEqual(save_backtest_result(connection, result), result)
            self._capture_yearly_stats(connection, task.task_id)
            completed = self._finish_task(connection, pending)

        with self.assertRaisesRegex(ValueError, "backtest_result_conflict"):
            with open_database(self.database_path) as connection:
                save_backtest_result(connection, replace(result, sharpe=9.0))

    def test_check_facts_have_one_structured_source(self) -> None:
        task = self._task("rank(close)")
        result = replace(
            self._result(task.task_id),
            check_details_captured=False,
            checks=(
                BacktestCheckRecord(
                    name="LOW_SHARPE",
                    status="PASS",
                    threshold=None,
                    actual=None,
                    platform_date=None,
                ),
            ),
        )
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, task)
            pending = self._pending(task, remote_id="simulation_1")
            replace_backtest_task(connection, pending, expected_status="created")
            pending = replace(pending, platform_alpha_id="alpha_1")
            replace_backtest_task(connection, pending, expected_status="pending")
            save_backtest_result(connection, result)
            self._capture_yearly_stats(connection, task.task_id)
            completed = self._finish_task(connection, pending)
            loaded = get_backtest_task(connection, task.task_id)
            result_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(backtest_results)")
            }
            check_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_checks"
            ).fetchone()[0]

        self.assertEqual(loaded.result, result)
        self.assertNotIn("checks_json", result_columns)
        self.assertIn("check_details_captured", result_columns)
        self.assertEqual(check_count, 1)

    def test_active_query_excludes_terminal_task(self) -> None:
        created = self._task("rank(close)")
        failed = self._task("rank(open)")
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, created)
            create_backtest_task(connection, failed)
            replace_backtest_task(
                connection,
                replace(
                    failed,
                    status="failed",
                    last_observed_at="2026-08-29T00:02:00+00:00",
                    finished_at="2026-08-29T00:02:00+00:00",
                    failure_code="platform_rejected",
                    failure_message="明确拒绝",
                ),
                expected_status="created",
            )
            active = list_active_backtest_tasks(connection)

        self.assertEqual(tuple(item.task_id for item in active), (created.task_id,))

    def test_completed_query_returns_result_owned_snapshots(self) -> None:
        task = self._task("rank(close)")
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, task)
            pending = self._pending(task, remote_id="simulation_1")
            replace_backtest_task(connection, pending, expected_status="created")
            pending = replace(pending, platform_alpha_id="alpha_1")
            replace_backtest_task(connection, pending, expected_status="pending")
            completed = self._complete_task(connection, pending)
            snapshots = list_completed_backtests(connection)

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].task, completed)
        self.assertEqual(snapshots[0].result, self._result(task.task_id))

    def test_completed_batch_preserves_checks_grades_and_empty_yearly_capture(self) -> None:
        with open_database(self.database_path) as connection:
            self.assertEqual(list_completed_backtests(connection), ())
            expected = []
            for index, grade in enumerate(("GOOD", "SPECTACULAR")):
                task = self._task(f"ts_mean(close,{index + 2})")
                create_backtest_task(connection, task)
                pending = self._pending(task, remote_id=f"simulation-{index}", platform_alpha_id=f"alpha-{index}")
                replace_backtest_task(connection, pending, expected_status="created")
                result = replace(self._result(task.task_id), grade=grade, sharpe=1.5 + index,
                                 checks=(BacktestCheckRecord("LOW_SHARPE", "PASS", 1.25, 1.5 + index, None),
                                         BacktestCheckRecord("SELF_CORRELATION", "PENDING", 0.7, None, None)))
                save_backtest_result(connection, result)
                yearly = () if index else self._yearly_stats(task.task_id)
                save_backtest_yearly_stats(connection, task.task_id, yearly)
                self._finish_task(connection, pending)
                expected.append(get_backtest_task(connection, task.task_id))
            create_backtest_task(connection, self._task("rank(volume)"))
            expected.sort(key=lambda snapshot: (snapshot.task.finished_at, snapshot.task.task_id))
            self.assertEqual(list_completed_backtests(connection), tuple(expected))
            self.assertEqual(list_completed_backtests(connection, task_ids=frozenset()), ())
            for snapshot in expected:
                # Large sparse ID sets must retain the same complete facts and order.
                requested = frozenset({snapshot.task.task_id, *(f"absent-{i}" for i in range(1200))})
                self.assertEqual(list_completed_backtests(connection, task_ids=requested), (snapshot,))

    def test_completed_batch_rejects_missing_result_or_yearly_capture(self) -> None:
        with open_database(self.database_path) as connection:
            task = self._task("rank(close)")
            create_backtest_task(connection, task)
            pending = self._pending(task, remote_id="simulation-missing", platform_alpha_id="alpha-missing")
            replace_backtest_task(connection, pending, expected_status="created")
            connection.execute("SAVEPOINT missing_result")
            self._finish_task(connection, pending)
            with self.assertRaisesRegex(ValueError, "completed_backtest_result_missing"):
                list_completed_backtests(connection)
            with self.assertRaisesRegex(ValueError, "completed_backtest_result_missing"):
                list_completed_backtests(connection, task_ids=frozenset({task.task_id}))
            connection.execute("ROLLBACK TO missing_result")
            connection.execute("RELEASE missing_result")
            save_backtest_result(connection, self._result(task.task_id))
            connection.execute("SAVEPOINT missing_yearly")
            self._finish_task(connection, pending)
            with self.assertRaisesRegex(ValueError, "completed_backtest_result_missing"):
                list_completed_backtests(connection)
            connection.execute("ROLLBACK TO missing_yearly")
            connection.execute("RELEASE missing_yearly")
            save_backtest_yearly_stats(connection, task.task_id, ())
            self._finish_task(connection, pending)
            self.assertEqual(list_completed_backtests(connection)[0].yearly_stats, ())

    def test_terminal_query_includes_failed_task_without_result(self) -> None:
        completed_task = self._task("rank(close)")
        failed_task = self._task("rank(open)")
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, completed_task)
            completed_pending = self._pending(
                completed_task,
                remote_id="simulation_1",
                platform_alpha_id="alpha_1",
            )
            replace_backtest_task(
                connection,
                completed_pending,
                expected_status="created",
            )
            completed = self._complete_task(
                connection,
                completed_pending,
            )

            create_backtest_task(connection, failed_task)
            failed = replace(
                failed_task,
                status="failed",
                last_observed_at="2026-08-29T00:04:00+00:00",
                finished_at="2026-08-29T00:04:00+00:00",
                failure_code="platform_zero_capital",
                failure_message="没有形成有效持仓",
            )
            replace_backtest_task(
                connection,
                failed,
                expected_status="created",
            )
            snapshots = list_terminal_backtests(connection)

        self.assertEqual(
            tuple(item.task.status for item in snapshots),
            ("completed", "failed"),
        )
        self.assertIsNotNone(snapshots[0].result)
        self.assertEqual(
            snapshots[0].yearly_stats,
            self._yearly_stats(completed_task.task_id),
        )
        self.assertIsNone(snapshots[1].result)

    def test_yearly_stats_preserve_empty_capture_and_structured_rows(self) -> None:
        empty_task = self._task("rank(close)")
        populated_task = self._task("rank(open)")
        with open_database(self.database_path) as connection:
            for task, remote_id, alpha_id in (
                (empty_task, "simulation_1", "alpha_1"),
                (populated_task, "simulation_2", "alpha_2"),
            ):
                create_backtest_task(connection, task)
                pending = self._pending(
                    task,
                    remote_id=remote_id,
                    platform_alpha_id=alpha_id,
                )
                replace_backtest_task(connection, pending, expected_status="created")
                save_backtest_result(connection, self._result(task.task_id))

            self.assertIsNone(get_backtest_yearly_stats(connection, empty_task.task_id))
            self.assertEqual(
                save_backtest_yearly_stats(connection, empty_task.task_id, ()),
                (),
            )
            populated = self._yearly_stats(populated_task.task_id)
            self.assertEqual(
                save_backtest_yearly_stats(
                    connection,
                    populated_task.task_id,
                    populated,
                ),
                populated,
            )

        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_yearly_stats(connection, empty_task.task_id), ())
            self.assertEqual(
                get_backtest_yearly_stats(connection, populated_task.task_id),
                populated,
            )

    def test_yearly_stats_conflict_does_not_replace_captured_facts(self) -> None:
        task = self._task("rank(close)")
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, task)
            pending = self._pending(
                task,
                remote_id="simulation_1",
                platform_alpha_id="alpha_1",
            )
            replace_backtest_task(connection, pending, expected_status="created")
            captured = self._yearly_stats(task.task_id)
            save_backtest_result(connection, self._result(task.task_id))
            save_backtest_yearly_stats(connection, task.task_id, captured)

        with self.assertRaisesRegex(ValueError, "backtest_yearly_stats_conflict"):
            with open_database(self.database_path) as connection:
                save_backtest_yearly_stats(
                    connection,
                    task.task_id,
                    (replace(captured[0], sharpe=9.0),),
                )

        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_yearly_stats(connection, task.task_id), captured)

    def test_invalid_yearly_row_does_not_create_capture_marker(self) -> None:
        task = self._task("rank(close)")
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, task)
            pending = self._pending(
                task,
                remote_id="simulation_1",
                platform_alpha_id="alpha_1",
            )
            replace_backtest_task(connection, pending, expected_status="created")
            save_backtest_result(connection, self._result(task.task_id))

        valid = self._yearly_stats(task.task_id)[0]
        with self.assertRaisesRegex(ValueError, "backtest_yearly_stat_year_invalid"):
            with open_database(self.database_path) as connection:
                save_backtest_yearly_stats(
                    connection,
                    task.task_id,
                    (valid, replace(valid, year=0, stage="OS")),
                )

        with open_database(self.database_path) as connection:
            self.assertIsNone(get_backtest_yearly_stats(connection, task.task_id))

    def test_direct_mutation_lineage_is_one_to_one_and_replayable(self) -> None:
        parent = self._task("rank(close)")
        child = self._task("rank(open)")
        mutation = BacktestMutationRecord(
            child_task_id=child.task_id,
            parent_task_id=parent.task_id,
            action="field_swap",
            location="formula.arguments[0]",
            before="close",
            after="open",
        )
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, parent)
            create_backtest_task(connection, child)
            first = create_backtest_mutation(connection, mutation)
            replayed = create_backtest_mutation(connection, mutation)
            loaded = get_backtest_mutation(connection, child.task_id)
            mutations = list_backtest_mutations(connection)
            self.assertEqual(list_backtest_mutations(connection, parent_task_ids=frozenset({parent.task_id})),
                             (mutation,))
            self.assertEqual(list_backtest_mutations(connection, parent_task_ids=frozenset({child.task_id})), ())
            self.assertEqual(list_backtest_mutations(connection, parent_task_ids=frozenset()), ())

        self.assertEqual(first, mutation)
        self.assertEqual(replayed, mutation)
        self.assertEqual(loaded, mutation)
        self.assertEqual(mutations, (mutation,))

    def test_mutation_preserves_lineage_across_different_settings(self) -> None:
        parent = self._task("rank(close)")
        child = replace(
            self._task("rank(open)"),
            settings_json='{"delay":0,"region":"USA"}',
            request_fingerprint="different_settings_request",
        )
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, parent)
            create_backtest_task(connection, child)
            mutation = create_backtest_mutation(
                connection,
                BacktestMutationRecord(
                    child_task_id=child.task_id,
                    parent_task_id=parent.task_id,
                    action="field_swap",
                    location="formula.arguments[0]",
                    before="close",
                    after="open",
                ),
            )

        self.assertEqual(mutation.parent_task_id, parent.task_id)
        self.assertNotEqual(parent.settings_json, child.settings_json)

    def _task(self, formula: str) -> BacktestTaskRecord:
        formula_identity = sha256(formula.encode("utf-8")).hexdigest()
        settings_json = json.dumps(
            {"delay": 1, "region": "USA"},
            sort_keys=True,
            separators=(",", ":"),
        )
        request_identity = sha256(
            f"{formula}|{settings_json}".encode("utf-8")
        ).hexdigest()
        return BacktestTaskRecord(
            task_id=f"backtest_{formula_identity}",
            account_scope="group-account",
            formula=formula,
            formula_fingerprint=formula_identity,
            settings_json=settings_json,
            request_fingerprint=request_identity,
            status="created",
            remote_id=None,
            platform_alpha_id=None,
            created_at="2026-08-29T00:01:00+00:00",
            submission_started_at=None,
            last_observed_at=None,
            retry_not_before=None,
            finished_at=None,
            failure_code=None,
            failure_message=None,
        )

    @staticmethod
    def _pending(
        task: BacktestTaskRecord,
        *,
        remote_id: str,
        platform_alpha_id: str | None = None,
    ) -> BacktestTaskRecord:
        return replace(
            task,
            status="pending",
            remote_id=remote_id,
            platform_alpha_id=platform_alpha_id,
            submission_started_at="2026-08-29T00:02:00+00:00",
            last_observed_at="2026-08-29T00:02:00+00:00",
        )

    @staticmethod
    def _result(task_id: str) -> BacktestResultRecord:
        return BacktestResultRecord(
            task_id=task_id,
            sharpe=1.2,
            fitness=0.9,
            turnover=0.2,
            returns=0.1,
            drawdown=0.05,
            margin=0.001,
            book_size=None,
            pnl=None,
            long_count=120,
            short_count=110,
            check_details_captured=True,
            checks=(
                BacktestCheckRecord(
                    name="LOW_SHARPE",
                    status="PASS",
                    threshold=1.25,
                    actual=1.2,
                    platform_date="2021-12-30",
                ),
            ),
        )

    @classmethod
    def _capture_yearly_stats(cls, connection, task_id: str) -> None:
        save_backtest_yearly_stats(
            connection,
            task_id,
            cls._yearly_stats(task_id),
        )

    @classmethod
    def _complete_task(
        cls,
        connection,
        pending: BacktestTaskRecord,
        *,
        result: BacktestResultRecord | None = None,
    ) -> BacktestTaskRecord:
        save_backtest_result(
            connection,
            result or cls._result(pending.task_id),
        )
        cls._capture_yearly_stats(connection, pending.task_id)
        return cls._finish_task(connection, pending)

    @staticmethod
    def _finish_task(
        connection,
        pending: BacktestTaskRecord,
    ) -> BacktestTaskRecord:
        completed = replace(
            pending,
            status="completed",
            retry_not_before=None,
            last_observed_at="2026-08-29T00:03:00+00:00",
            finished_at="2026-08-29T00:03:00+00:00",
        )
        replace_backtest_task(connection, completed, expected_status="pending")
        return completed

    @staticmethod
    def _yearly_stats(task_id: str) -> tuple[BacktestYearlyStatRecord, ...]:
        return (
            BacktestYearlyStatRecord(
                task_id=task_id,
                year=2023,
                pnl=1000.0,
                book_size=20_000_000.0,
                long_count=120,
                short_count=110,
                turnover=0.2,
                sharpe=1.2,
                returns=0.1,
                drawdown=0.05,
                margin=0.001,
                fitness=0.9,
                stage="IS",
            ),
        )


if __name__ == "__main__":
    unittest.main()
