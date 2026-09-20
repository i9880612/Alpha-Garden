import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from execution.console import ConsoleReader
from execution.backtests import cancel_unsubmitted_backtest_task, fail_backtest_task, prepare_backtest_task, record_submission_accepted, record_submission_unknown
from execution.qualified_archive import synchronize_qualified_alpha_archive
from execution.seeds import load_signal_frontiers, synchronize_signal_seeds
from persistence.console import read_console_database
from persistence.database import open_database
from tests.execution.console_fixture import make_console_reader
from tests.execution.test_qualified_archive import complete_candidate, prepare_candidate, exhausted_parent, FINISHED, ARCHIVED

class ConsoleReaderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.reader = make_console_reader(self.folder.name)
        self.database = self.reader.paths.database

    def test_read_only_missing_database_never_creates_it(self):
        missing = Path(self.folder.name) / "missing.sqlite3"
        with self.assertRaises(sqlite3.OperationalError):
            with read_console_database(missing):
                pass
        self.assertFalse(missing.exists())
        with read_console_database(self.database) as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM backtest_tasks")

    def test_real_facts_account_scope_checks_and_attempt_budget(self):
        with open_database(self.database) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection))
            child = complete_candidate(connection, prepare_candidate(connection, parent=parent), sharpe=1.0)
            cancelled = prepare_candidate(connection, parent=parent, start=False)
            cancel_unsubmitted_backtest_task(connection, cancelled.task.task_id, observed_at=FINISHED)
            prepare_backtest_task(connection, account_scope="another-account", formula="rank(close)",
                                 settings={"delay": 1}, created_at=FINISHED)
        before = self.database.read_bytes()
        page = self.reader.formulas()
        self.assertEqual(page["total"], 3)
        rows = {r["task_id"]: r for r in page["items"]}
        self.assertEqual(rows[parent.task.task_id]["attempts_remaining"], 19)
        self.assertEqual(rows[child.task.task_id]["parent_alpha_id"], parent.task.platform_alpha_id)
        self.assertEqual(rows[parent.task.task_id]["full_check"], "passed")
        self.assertEqual(rows[cancelled.task.task_id]["full_check"], "pending")
        self.assertEqual(rows[cancelled.task.task_id]["source"], "mutation")
        encoded = json.dumps(page)
        self.assertNotIn(parent.task.formula, encoded)
        self.assertNotIn("synthetic-console-secret", encoded)
        self.reader.dashboard_activity()
        self.reader.dashboard_grades()
        self.reader.dashboard_research()
        self.reader.dashboard_progress()
        self.reader.dashboard_recent()
        self.reader.runs()
        self.reader.settings()
        self.reader.submissions()
        self.assertEqual(before, self.database.read_bytes())
        self.assertFalse(self.database.with_name(self.database.name + ".run.lock").exists())

    def test_optimization_reuses_frontier_and_archive_keeps_retired_candidates(self):
        with open_database(self.database) as connection:
            exhausted = exhausted_parent(connection)
            synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED)
            active = complete_candidate(connection, prepare_candidate(connection), sharpe=1.8)
            synchronize_signal_seeds(connection)
            expected = set(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids)
        self.assertIn(active.task.task_id, expected)
        self.assertEqual({r["task_id"] for r in self.reader.formulas(category="optimization")["items"]}, expected)
        archived = self.reader.formulas(category="archive")
        self.assertIn(exhausted.task.task_id, {r["task_id"] for r in archived["items"]})
        filtered = self.reader.formulas(grade="GOOD", search=exhausted.task.platform_alpha_id)
        self.assertEqual(filtered["total"], 1)
        self.assertEqual(self.reader.formulas(page=2, page_size=1)["items"].__len__(), 1)
        with self.assertRaisesRegex(ValueError, "console_run_not_found"):
            self.reader.formulas(run_id="unknown")

    def test_optimization_source_filter_combines_with_grade_search_and_pagination(self):
        with open_database(self.database) as connection:
            root = complete_candidate(connection, prepare_candidate(connection))
            synchronize_signal_seeds(connection)
            children = []
            for action in ("structural", "structural", "self_correlation_time_smoothing", "direction_reversal"):
                child = complete_candidate(connection, prepare_candidate(connection, parent=root), sharpe=1.3)
                connection.execute("UPDATE backtest_mutations SET action=? WHERE child_task_id=?", (action, child.task.task_id))
                children.append(child)
            complete_candidate(connection, prepare_candidate(connection, parent=root), qualified=False)
        before = self.database.read_bytes()
        expected = {"exploration": [root], "mutation": children[:2], "sc": children[2:3], "reversal": children[3:]}
        for source, snapshots in expected.items():
            with self.subTest(source=source):
                page = self.reader.formulas(category="optimization", source=source, page_size=1)
                self.assertEqual(page["total"], len(snapshots))
                self.assertEqual(len(page["items"]), 1)
                self.assertEqual(page["items"][0]["source"], source)
                found = self.reader.formulas(category="optimization", source=source, grade="GOOD",
                    search=snapshots[0].task.platform_alpha_id.upper())
                self.assertEqual([row["task_id"] for row in found["items"]], [snapshots[0].task.task_id])
        later = self.reader.formulas(category="optimization", source="mutation", page=2, page_size=1)
        self.assertEqual(later["total"], 2)
        self.assertEqual(len(later["items"]), 1)
        self.assertEqual(self.reader.formulas(category="optimization", source="sc", grade="AVERAGE")["total"], 0)
        with self.assertRaisesRegex(ValueError, "console_source_filter_invalid"):
            self.reader.formulas(category="optimization", source="anything")
        self.assertEqual(before, self.database.read_bytes())

    def test_unknown_grade_filter_and_empty_results_are_explicit(self):
        with open_database(self.database) as connection:
            complete_candidate(connection, prepare_candidate(connection), grade=None)
        self.assertEqual(self.reader.formulas(grade="UNKNOWN")["total"], 1)
        self.assertEqual(self.reader.formulas(search="does-not-exist")["items"], [])
        self.assertEqual(self.reader.dashboard_grades()["grades"], [])

    def test_filtered_page_keeps_parent_budget_for_children_outside_the_page(self):
        with open_database(self.database) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection))
            complete_candidate(connection, prepare_candidate(connection, parent=parent), grade="AVERAGE")
            prepare_candidate(connection, parent=parent)
            cancelled = prepare_candidate(connection, parent=parent, start=False)
            cancel_unsubmitted_backtest_task(connection, cancelled.task.task_id, observed_at=FINISHED)
            other = complete_candidate(connection, prepare_candidate(connection))
            # An unrelated historical payload must not be read to render this page.
            connection.execute("UPDATE submission_checks SET payload_json='invalid-json' WHERE task_id=?",
                               (other.task.task_id,))
        before = self.database.read_bytes()
        page = self.reader.formulas(search=parent.task.platform_alpha_id.upper(), grade="GOOD", page_size=1)
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["items"][0]["attempts_remaining"], 18)
        self.assertEqual(page["items"][0]["full_check"], "passed")
        self.assertEqual(self.reader.formulas(search=parent.task.platform_alpha_id, page=2, page_size=1)["items"], [])
        self.assertEqual(before, self.database.read_bytes())

    def test_backtest_results_inflight_and_unstarted_plans_are_separate_and_paginated(self):
        with open_database(self.database) as connection:
            completed = complete_candidate(connection, prepare_candidate(connection))
            pending = [prepare_candidate(connection) for _ in range(3)]
            unknown = prepare_candidate(connection, start=False)
            record_submission_unknown(connection, unknown.task.task_id, observed_at=FINISHED)
            planned = [prepare_candidate(connection, start=False) for _ in range(12)]
            failed = prepare_candidate(connection, start=False)
            cancel_unsubmitted_backtest_task(connection, failed.task.task_id, observed_at=FINISHED)
            prepare_backtest_task(connection, account_scope="another-account", formula="rank(close)",
                                 settings={"delay": 1}, created_at=FINISHED)
        before = self.database.read_bytes()
        records = self.reader.formulas(execution="finished", page_size=10)
        self.assertEqual(records["unstarted"], 12)
        self.assertEqual(records["inflight"], 4)
        self.assertEqual({row["task_id"] for row in records["items"]},
                         {item.task.task_id for item in (completed, failed)})
        self.assertEqual(records["total"], 2)
        inflight = self.reader.formulas(execution="inflight", page_size=2)
        later = self.reader.formulas(execution="inflight", page_size=2, page=2)
        self.assertEqual((len(inflight["items"]), len(later["items"]), inflight["total"]), (2, 2, 4))
        self.assertEqual({row["task_id"] for row in inflight["items"] + later["items"]},
                         {item.task.task_id for item in [*pending, unknown]})
        first = self.reader.formulas(execution="planned", page_size=10)
        second = self.reader.formulas(execution="planned", page_size=10, page=2)
        self.assertEqual((len(first["items"]), len(second["items"]), first["total"]), (10, 2, 12))
        self.assertEqual({row["task_id"] for row in first["items"] + second["items"]},
                         {item.task.task_id for item in planned})
        self.assertEqual(self.reader.formulas()["total"], 18)
        self.assertEqual(before, self.database.read_bytes())
        with open_database(self.database) as connection:
            complete_candidate(connection, pending[0])
        self.assertEqual(self.reader.formulas(execution="finished")["total"], 3)
        self.assertEqual(self.reader.formulas(execution="inflight")["total"], 3)
        self.assertEqual(self.reader.formulas(execution="planned")["total"], 12)

    def test_started_formulas_exclude_unsent_candidates_before_filtering_and_pagination(self):
        with open_database(self.database) as connection:
            completed = complete_candidate(connection, prepare_candidate(connection))
            pending = prepare_candidate(connection)
            unknown = prepare_candidate(connection, start=False)
            record_submission_unknown(connection, unknown.task.task_id, observed_at=FINISHED)
            failed = prepare_candidate(connection)
            fail_backtest_task(connection, failed.task.task_id, failure_code="platform_error",
                               failure_message="Synthetic platform failure", observed_at=FINISHED)
            planned = [prepare_candidate(connection, start=False) for _ in range(12)]
            cancelled = prepare_candidate(connection, start=False)
            cancel_unsubmitted_backtest_task(connection, cancelled.task.task_id, observed_at=FINISHED)
            other = prepare_backtest_task(connection, account_scope="another-account", formula="rank(close)",
                                         settings={"delay": 1}, created_at=FINISHED)
            record_submission_accepted(connection, other.task.task_id, remote_id="another-simulation", observed_at=FINISHED)
        before = self.database.read_bytes()
        first = self.reader.formulas(execution="started", page_size=2)
        second = self.reader.formulas(execution="started", page_size=2, page=2)
        self.assertEqual((first["total"], second["total"], len(first["items"]), len(second["items"])), (4, 4, 2, 2))
        self.assertEqual({row["task_id"] for row in first["items"] + second["items"]},
                         {item.task.task_id for item in (completed, pending, unknown, failed)})
        self.assertEqual(self.reader.formulas(execution="started", grade="GOOD")["total"], 1)
        self.assertEqual(self.reader.formulas(execution="started", search=pending.task.task_id)["total"], 1)
        for item in (*planned, cancelled):
            self.assertEqual(self.reader.formulas(execution="started", search=item.task.task_id)["total"], 0)
        self.assertEqual(self.reader.formulas(execution="planned")["total"], 12)
        self.assertEqual(self.reader.formulas()["total"], 17)
        self.assertEqual(before, self.database.read_bytes())
        with open_database(self.database) as connection:
            record_submission_accepted(connection, planned[0].task.task_id, remote_id="new-simulation", observed_at=FINISHED)
        self.assertEqual(self.reader.formulas(execution="started")["total"], 5)
        self.assertEqual(self.reader.formulas(execution="planned")["total"], 11)

    def test_backtest_failures_remain_visible_without_claiming_a_full_check(self):
        with open_database(self.database) as connection:
            failed, passed, pending = [complete_candidate(connection, prepare_candidate(connection)) for _ in range(3)]
            for candidate in (failed, passed, pending):
                connection.execute("DELETE FROM submission_checks WHERE task_id=?", (candidate.task.task_id,))
            for name, actual, threshold in (("LOW_SUB_UNIVERSE_SHARPE", 0.43, 0.72), ("CONCENTRATED_WEIGHT", 0.2, 0.1)):
                connection.execute("UPDATE backtest_checks SET status='FAIL', actual_value=?, threshold_value=? WHERE task_id=? AND check_name=?",
                                   (actual, threshold, failed.task.task_id, name))
            connection.execute("UPDATE backtest_checks SET status='PENDING' WHERE task_id=? AND check_name='SELF_CORRELATION'",
                               (failed.task.task_id,))
            connection.execute("UPDATE backtest_checks SET status='PENDING' WHERE task_id=?", (pending.task.task_id,))
            other = prepare_backtest_task(connection, account_scope="another-account", formula="rank(close)",
                                         settings={"delay": 1}, created_at=FINISHED)
            other = record_submission_accepted(connection, other.task.task_id, remote_id="another-simulation", observed_at=FINISHED)
            complete_candidate(connection, other, qualified=False)
        before = self.database.read_bytes()
        result = self.reader.formulas()
        rows = {row["task_id"]: row for row in result["items"]}
        self.assertEqual(set(rows), {candidate.task.task_id for candidate in (failed, passed, pending)})
        self.assertEqual(rows[failed.task.task_id]["backtest_failed_checks"], [
            {"name": "CONCENTRATED_WEIGHT", "actual": 0.2, "threshold": 0.1},
            {"name": "LOW_SUB_UNIVERSE_SHARPE", "actual": 0.43, "threshold": 0.72},
        ])
        for candidate in (failed, passed, pending):
            row = rows[candidate.task.task_id]
            self.assertEqual(row["full_check"], "pending")
            self.assertIsNone(row["check_at"])
            self.assertEqual(row["failed_checks"], [])
        self.assertEqual(rows[passed.task.task_id]["backtest_failed_checks"], [])
        self.assertEqual(rows[pending.task.task_id]["backtest_failed_checks"], [])
        page = self.reader.formulas(search=failed.task.platform_alpha_id, page_size=1)
        self.assertEqual(page["items"][0]["backtest_failed_checks"], rows[failed.task.task_id]["backtest_failed_checks"])
        self.assertEqual(self.reader.formulas(search="no-such-alpha")["items"], [])
        self.assertEqual(before, self.database.read_bytes())

    def test_activity_counts_current_account_by_local_calendar_day(self):
        now = datetime.now().astimezone().replace(hour=12, minute=0, second=0, microsecond=0)
        start = (now - timedelta(days=14)).replace(hour=0)
        week_start = (now - timedelta(days=6)).replace(hour=0)
        with open_database(self.database) as connection:
            for finished in (now, start, start - timedelta(seconds=1), week_start, week_start - timedelta(seconds=1)):
                candidate = complete_candidate(connection, prepare_candidate(connection))
                connection.execute("UPDATE backtest_tasks SET finished_at=? WHERE task_id=?",
                                   (finished.astimezone(timezone.utc).isoformat(), candidate.task.task_id))
            other = prepare_backtest_task(connection, account_scope="another-account", formula="rank(volume)",
                                         settings={"delay": 1}, created_at=FINISHED)
            record_submission_accepted(connection, other.task.task_id, remote_id="another-simulation", observed_at=FINISHED)
            complete_candidate(connection, other)
            connection.execute("UPDATE backtest_tasks SET finished_at=? WHERE task_id=?", (now.isoformat(), other.task.task_id))
        with patch("execution.console.datetime", wraps=datetime) as clock:
            clock.now.return_value = now
            activity = self.reader.dashboard_activity()
        self.assertEqual(activity["dates"], [(start.date() + timedelta(days=i)).isoformat() for i in range(15)])
        self.assertEqual(activity["trend"], [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1])
        self.assertEqual(activity["weekly_backtests"], 2)
        self.assertEqual(activity["metrics"][0]["value"], 1)
        self.assertEqual(activity["metrics"][0]["spark"], [1, 0, 0, 0, 0, 0, 1])
        self.assertTrue(all(len(metric["spark"]) == 7 for metric in activity["metrics"]))

    def test_dashboard_progress_counts_completed_batch(self):
        from tests.execution.test_submission_runner import SubmissionQueueRunnerTests
        from persistence.runs import get_automated_run_backtest_by_task

        fixture = SubmissionQueueRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        task_ids, _ = fixture._completed_run(("rank(close)", "rank(open)"))
        reader = ConsoleReader(replace(self.reader.paths, database=fixture.database_path))
        with open_database(fixture.database_path) as connection:
            link = get_automated_run_backtest_by_task(connection, task_ids[0])
        progress = reader.dashboard_progress()
        self.assertEqual(progress["run"]["run_id"], link.run_id)
        self.assertEqual(progress["cycle_number"], 1)
        self.assertEqual(progress["progress"], [
            {"source": "exploration", "completed": 2, "planned": 2},
            {"source": "sc", "completed": 0, "planned": 0},
            {"source": "mutation", "completed": 0, "planned": 0},
            {"source": "reversal", "completed": 0, "planned": 0}])

    def _progress_run(self):
        from execution.runs import AutomatedRunLimits, prepare_automated_run, start_automated_run

        run = prepare_automated_run(
            self.database, self.reader.paths.settings, account_scope="group-account",
            limits=AutomatedRunLimits(
                generation_count=10, backtest_count=10, max_cycles=3, max_backtests=30,
                max_pending_seconds=600, max_consecutive_failures=2, max_request_failures=3,
                max_in_flight_backtests=3, exploration_seed_attempt_multiplier=4,
                exploration_percent=30, self_correlation_percent=30, mutation_percent=40,
                direction_validation_percent=3, real_backtests_authorized=True,
            ), created_at="2026-09-01T00:00:00+00:00",
        )
        return start_automated_run(self.database, run.run_id, started_at="2026-09-01T00:00:01+00:00")

    def test_dashboard_progress_counts_first_batch_before_settlement(self):
        from persistence.runs import AutomatedRunBacktestRecord, attach_backtest_to_automated_run

        run = self._progress_run()
        with open_database(self.database) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection))
            mutation = prepare_candidate(connection, parent=parent, start=False)
            sc = prepare_candidate(connection, parent=parent)
            reversal = prepare_candidate(connection, parent=parent)
            connection.execute("UPDATE backtest_mutations SET action=? WHERE child_task_id=?",
                               ("self_correlation_time_smoothing", sc.task.task_id))
            connection.execute("UPDATE backtest_mutations SET action=? WHERE child_task_id=?",
                               ("direction_reversal", reversal.task.task_id))
            fail_backtest_task(connection, reversal.task.task_id, failure_code="platform_error",
                               failure_message="Test failure", observed_at=FINISHED)
            for item in (parent, mutation, sc, reversal):
                attach_backtest_to_automated_run(connection,
                    AutomatedRunBacktestRecord(run.run_id, item.task.task_id, 1))
            # Unlinked results do not belong to this batch.
            complete_candidate(connection, prepare_candidate(connection))
        before = self.database.read_bytes()
        progress = self.reader.dashboard_progress()
        self.assertEqual(progress["run"]["current_cycle"], 0)
        self.assertEqual(progress["cycle_number"], 1)
        self.assertEqual(progress["progress"], [
            {"source": "exploration", "completed": 1, "planned": 1},
            {"source": "sc", "completed": 0, "planned": 1},
            {"source": "mutation", "completed": 0, "planned": 1},
            {"source": "reversal", "completed": 1, "planned": 1}])
        self.assertEqual(before, self.database.read_bytes())

    def test_dashboard_progress_uses_latest_batch_even_with_older_unsettled_work(self):
        from persistence.runs import AutomatedRunBacktestRecord, attach_backtest_to_automated_run

        run = self._progress_run()
        with open_database(self.database) as connection:
            older = prepare_candidate(connection, start=False)
            record_submission_unknown(connection, older.task.task_id, observed_at=FINISHED)
            current = complete_candidate(connection, prepare_candidate(connection))
            queued = prepare_candidate(connection, start=False)
            for item, cycle_number in ((older, 1), (current, 2), (queued, 2)):
                attach_backtest_to_automated_run(connection,
                    AutomatedRunBacktestRecord(run.run_id, item.task.task_id, cycle_number))
        progress = self.reader.dashboard_progress()
        self.assertEqual(progress["run"]["current_cycle"], 0)
        self.assertEqual(progress["cycle_number"], 2)
        self.assertEqual(progress["progress"][0], {"source": "exploration", "completed": 1, "planned": 2})
        with open_database(self.database) as connection:
            record_submission_accepted(connection, queued.task.task_id, remote_id="next-task", observed_at=FINISHED)
            complete_candidate(connection, queued)
        self.assertEqual(self.reader.dashboard_progress()["progress"][0],
                         {"source": "exploration", "completed": 2, "planned": 2})

    def test_dashboard_progress_distinguishes_missing_run_and_unplanned_batch(self):
        empty = self.reader.dashboard_progress()
        self.assertIsNone(empty["run"])
        self.assertIsNone(empty["cycle_number"])
        run = self._progress_run()
        unplanned = self.reader.dashboard_progress()
        self.assertEqual(unplanned["run"]["run_id"], run.run_id)
        self.assertIsNone(unplanned["cycle_number"])
        self.assertTrue(all(row["planned"] == row["completed"] == 0 for row in unplanned["progress"]))
        other_env = Path(self.folder.name) / "other.env"
        other_env.write_text("WQB_ACCOUNT_SCOPE=another-account\nWQB_BASE_URL=https://example.invalid\n"
                             "WQB_SESSION_TOKEN=synthetic-console-secret\n", encoding="utf-8")
        other_reader = ConsoleReader(replace(self.reader.paths, environment=other_env))
        self.assertIsNone(other_reader.dashboard_progress()["run"])

    def test_dashboard_recent_expressions_are_limited_and_account_scoped(self):
        with open_database(self.database) as connection:
            own = [complete_candidate(connection, prepare_candidate(connection)) for _ in range(6)]
            other = prepare_backtest_task(connection, account_scope="another-account", formula="rank(volume)",
                                          settings={"delay": 1}, created_at=FINISHED)
            record_submission_accepted(connection, other.task.task_id, remote_id="another-simulation", observed_at=FINISHED)
            complete_candidate(connection, other)
        before = self.database.read_bytes()
        recent = self.reader.dashboard_recent()["items"]
        self.assertEqual(len(recent), 4)
        expressions = {item.task.task_id: item.task.formula for item in own}
        for item in recent:
            self.assertEqual(item["formula"], expressions[item["task_id"]])
        self.assertNotIn(other.task.formula, json.dumps(recent))
        self.assertEqual(before, self.database.read_bytes())

    def test_dashboard_research_reuses_optimization_and_exhausted_archive_candidates(self):
        from tests.execution.test_submission_runner import SubmissionQueueRunnerTests

        fixture = SubmissionQueueRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture._qualified_batch()
        reader = ConsoleReader(replace(self.reader.paths, database=fixture.database_path))
        with open_database(fixture.database_path) as connection:
            complete_candidate(connection, prepare_candidate(connection), grade="EXCELLENT")
            complete_candidate(connection, prepare_candidate(connection), grade=None)
            synchronize_signal_seeds(connection)
            active_ids = set(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids)
        research = reader.dashboard_research()
        self.assertEqual(reader.dashboard_grades()["grades"], [])
        self.assertEqual(research["manual_candidates"], 2)
        self.assertEqual(research["optimization_parents"], len(active_ids))
        self.assertEqual(research["optimization_attempts"], 20 * len(active_ids))

    def test_research_manual_candidates_require_exhausted_budget(self):
        from tests.execution.test_submission_runner import SubmissionQueueRunnerTests

        fixture = SubmissionQueueRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture._qualified_batch(attempts=19)
        reader = ConsoleReader(replace(self.reader.paths, database=fixture.database_path))
        self.assertEqual(reader.dashboard_grades()["grades"], [])
        self.assertEqual(reader.dashboard_research()["manual_candidates"], 0)

    def test_dashboard_grades_use_submitted_platform_records_and_preserve_unknown(self):
        from persistence.submissions import PlatformSubmittedAlphaRecord, record_platform_submitted_alphas

        with open_database(self.database) as connection:
            local = complete_candidate(connection, prepare_candidate(connection), grade="EXCELLENT")
            complete_candidate(connection, prepare_candidate(connection), grade="INFERIOR")
            records = []
            for account, alpha_id, formula, grade in [
                ("group-account", local.task.platform_alpha_id, local.task.formula, "GOOD"),
                ("group-account", "submitted-spectacular", "rank(volume)", "SPECTACULAR"),
                ("group-account", "submitted-good", "rank(open)", "GOOD"),
                ("group-account", "submitted-unknown", "rank(close)", None),
                ("another-account", "other-excellent", "rank(vwap)", "EXCELLENT"),
            ]:
                payload = {"id": alpha_id, "status": "ACTIVE", "dateSubmitted": FINISHED,
                           "hidden": False, "regular": {"code": formula}, "grade": grade}
                records.append(PlatformSubmittedAlphaRecord(account, alpha_id, formula, "ACTIVE", FINISHED,
                                                            False, payload, FINISHED))
            record_platform_submitted_alphas(connection, tuple(records))
        before = self.database.read_bytes()
        grades = self.reader.dashboard_grades()["grades"]
        self.assertEqual(grades, [
            {"label": "SPECTACULAR", "value": 1}, {"label": "GOOD", "value": 2}, {"label": "UNKNOWN", "value": 1}])
        self.assertEqual(sum(row["value"] for row in grades), self.reader.dashboard_activity()["metrics"][3]["value"])
        self.assertEqual(self.reader.formulas(category="submitted", grade="GOOD")["total"], 2)
        self.assertEqual(before, self.database.read_bytes())


class ResearchConsoleTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.reader = make_console_reader(folder.name)
        self.database = self.reader.paths.database

    def test_formula_detail_contains_local_evidence(self):
        from persistence.pnl import PnlSeriesRecord, save_pnl_series
        with open_database(self.database) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection))
            child = complete_candidate(connection, prepare_candidate(connection, parent=parent))
            synchronize_signal_seeds(connection)
            other = prepare_backtest_task(connection, account_scope="other-account", formula="rank(volume)", settings={"delay": 1}, created_at=FINISHED)
            connection.execute("DELETE FROM platform_pnl_series WHERE platform_alpha_id=?", (parent.task.platform_alpha_id,))
            connection.execute("DELETE FROM platform_pnl_series WHERE platform_alpha_id=?", (child.task.platform_alpha_id,))
            save_pnl_series(connection, PnlSeriesRecord("group-account", parent.task.platform_alpha_id, FINISHED,
                (("2026-01-01", -10.0), ("2026-01-02", 25.0))))
        before = self.database.read_bytes()
        with patch("worldquant.client.WorldQuantClient.authenticate", side_effect=AssertionError("No platform access")):
            detail = self.reader.formula_detail(parent.task.task_id)
            self.assertEqual(detail["expression"], parent.task.formula)
            self.assertEqual(detail["references"], {"names": ["close"], "operators": ["ts_mean"]})
            self.assertEqual(detail["pnl"]["points"], [["2026-01-01", -10.0], ["2026-01-02", 25.0]])
            self.assertNotIn("children", detail)
            self.assertIsNotNone(detail["seed_promoted_at"])
            self.assertEqual(detail["yearly"][0]["year"], 2025)
            child_detail = self.reader.formula_detail(child.task.task_id)
            self.assertEqual(child_detail["summary"]["parent_task_id"], parent.task.task_id)
            self.assertEqual(child_detail["mutation"]["before"], parent.task.formula)
            self.assertIsNone(child_detail["pnl"])
            with self.assertRaisesRegex(ValueError, "console_formula_not_found"):
                self.reader.formula_detail(other.task.task_id)
        self.assertEqual(before, self.database.read_bytes())

    def test_formula_detail_displays_full_check_values_and_preserves_backtest_snapshot(self):
        with open_database(self.database) as connection:
            candidate = complete_candidate(connection, prepare_candidate(connection))
            connection.execute("UPDATE backtest_checks SET status='PENDING' WHERE task_id=? AND check_name='SELF_CORRELATION'", (candidate.task.task_id,))
        for status, value, expected in [("PASS", 0.42, "passed"), ("FAIL", 0.81, "failed"), ("PENDING", None, "pending"), ("ERROR", None, "pending")]:
            with self.subTest(status=status):
                with open_database(self.database) as connection:
                    payload = json.loads(connection.execute("SELECT payload_json FROM submission_checks WHERE task_id=?", (candidate.task.task_id,)).fetchone()[0])
                    for check in payload["is"]["checks"]:
                        if check["name"] == "SELF_CORRELATION":
                            check.update(result=status, value=value, limit=0.7)
                    payload["private_field"] = "must-not-leak"
                    connection.execute("UPDATE submission_checks SET payload_json=?,observed_at=? WHERE task_id=?", (json.dumps(payload), ARCHIVED, candidate.task.task_id))
                before = self.database.read_bytes()
                with patch("worldquant.client.WorldQuantClient.authenticate", side_effect=AssertionError("No platform access")):
                    detail = self.reader.formula_detail(candidate.task.task_id)
                self.assertEqual(detail["checks"]["source"], "full")
                self.assertEqual(detail["checks"]["observed_at"], ARCHIVED)
                self.assertEqual(detail["summary"]["full_check"], expected)
                sc = next(check for check in detail["checks"]["items"] if check["name"] == "SELF_CORRELATION")
                self.assertEqual(sc, {"name": "SELF_CORRELATION", "status": status, "actual": value, "threshold": 0.7})
                self.assertEqual(next(check for check in detail["result"]["checks"] if check["name"] == "SELF_CORRELATION")["status"], "PENDING")
                self.assertNotIn("must-not-leak", json.dumps(detail))
                self.assertEqual(before, self.database.read_bytes())

    def test_formula_detail_without_full_check_keeps_pending_and_errors_do_not_reuse_old_passes(self):
        with open_database(self.database) as connection:
            candidate = complete_candidate(connection, prepare_candidate(connection))
            connection.execute("UPDATE backtest_checks SET status='PENDING' WHERE task_id=? AND check_name='SELF_CORRELATION'", (candidate.task.task_id,))
            connection.execute("DELETE FROM submission_checks WHERE task_id=?", (candidate.task.task_id,))
        detail = self.reader.formula_detail(candidate.task.task_id)
        self.assertEqual(detail["checks"]["source"], "backtest")
        self.assertEqual(next(check for check in detail["checks"]["items"] if check["name"] == "SELF_CORRELATION")["status"], "PENDING")
        from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
        with open_database(self.database) as connection:
            save_submission_check(connection, SubmissionCheckRecord(candidate.task.task_id, ARCHIVED, None, "request_failed"))
        detail = self.reader.formula_detail(candidate.task.task_id)
        self.assertEqual(detail["checks"], {"source": "full", "observed_at": ARCHIVED, "error": "request_failed", "items": []})
        self.assertEqual(detail["summary"]["full_check"], "pending")

    def test_formula_detail_uses_newest_check_across_research_and_submission(self):
        from tests.execution.test_submission_runner import SubmissionQueueRunnerTests
        from persistence.runs import get_automated_run_backtest_by_task
        from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
        fixture = SubmissionQueueRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        task_ids, _ = fixture._completed_run(("rank(close)",))
        task_id = task_ids[0]
        reader = ConsoleReader(replace(self.reader.paths, database=fixture.database_path))
        payload = {"is": {"checks": [{"name": "SELF_CORRELATION", "result": "FAIL", "value": 0.88, "limit": 0.7}]}}
        with open_database(fixture.database_path) as connection:
            link = get_automated_run_backtest_by_task(connection, task_id)
            connection.execute("""INSERT INTO formal_submission_attempts
                (task_id,run_id,cycle_number,family_root_task_id,submission_mode,status,check_attempt_count,
                 check_payload_json,check_observed_at,created_at,updated_at)
                VALUES (?,?,?,?, 'manual','ineligible',1,?,?,?,?)""",
                (task_id, link.run_id, link.cycle_number, task_id, json.dumps(payload), "2026-10-01T00:00:00+00:00", ARCHIVED, "2026-10-01T00:00:00+00:00"))
        detail = reader.formula_detail(task_id)
        self.assertEqual(detail["checks"]["items"][0]["actual"], 0.88)
        self.assertEqual(detail["summary"]["check_at"], detail["checks"]["observed_at"])
        with open_database(fixture.database_path) as connection:
            connection.execute("DELETE FROM submission_checks WHERE task_id=?", (task_id,))
            save_submission_check(connection, SubmissionCheckRecord(task_id, "2026-10-02T00:00:00+00:00", None, "request_failed"))
        detail = reader.formula_detail(task_id)
        self.assertEqual(detail["checks"]["items"], [])
        self.assertEqual(detail["checks"]["error"], "request_failed")
        self.assertEqual(detail["summary"]["full_check"], "pending")

    def test_formula_children_paginate_independently_and_remain_account_scoped(self):
        with open_database(self.database) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection))
            children = [prepare_candidate(connection, parent=parent, start=index == 0) for index in range(16)]
            complete_candidate(connection, children[0])
            other = prepare_backtest_task(connection, account_scope="other-account", formula="rank(volume)",
                                          settings={"delay": 1}, created_at=FINISHED)
            from persistence.backtests import BacktestMutationRecord, create_backtest_mutation
            create_backtest_mutation(connection, BacktestMutationRecord(
                other.task.task_id, parent.task.task_id, "structural", "formula", parent.task.formula, other.task.formula))
        before = self.database.read_bytes()
        with patch("worldquant.client.WorldQuantClient.authenticate", side_effect=AssertionError("No platform access")):
            first = self.reader.formula_children(parent.task.task_id)
            second = self.reader.formula_children(parent.task.task_id, page=2)
            self.assertEqual((first["total"], second["total"]), (16, 16))
            self.assertEqual((len(first["items"]), len(second["items"])), (10, 6))
            self.assertEqual((first["page"], second["page"], first["page_size"]), (1, 2, 10))
            ids = [row["task_id"] for row in first["items"] + second["items"]]
            self.assertEqual(len(set(ids)), 16)
            self.assertEqual(set(ids), {child.task.task_id for child in children})
            self.assertEqual(first["items"][0]["status"], "completed")
            self.assertEqual(self.reader.formula_children(parent.task.task_id, page=3)["items"], [])
            self.assertEqual(self.reader.formula_children(children[0].task.task_id)["total"], 0)
            for task_id in (other.task.task_id, "missing"):
                with self.assertRaisesRegex(ValueError, "console_formula_not_found"):
                    self.reader.formula_children(task_id)
        self.assertEqual(before, self.database.read_bytes())

    def test_seed_views_use_existing_membership_and_engine_frontiers_without_promoting(self):
        with open_database(self.database) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection))
            synchronize_signal_seeds(connection)
            unadmitted = complete_candidate(connection, prepare_candidate(connection))
            expected = set(load_signal_frontiers(connection).active_branch_task_ids)
        before = self.database.read_bytes()
        roots = self.reader.seeds()
        self.assertEqual(roots["root_count"], 1)
        self.assertEqual(roots["items"][0]["task_id"], parent.task.task_id)
        self.assertNotIn(unadmitted.task.task_id, {row["task_id"] for row in roots["items"]})
        active = self.reader.seeds(mode="normal")
        self.assertEqual({row["task_id"] for row in active["items"]}, expected)
        self.assertEqual(self.reader.seeds(search="missing")["items"], [])
        self.assertEqual(before, self.database.read_bytes())
