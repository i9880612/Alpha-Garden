from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

from execution.submission_checks import (check_completed_backtest, advance_deferred_submission_check,
                                         remaining_submission_check_seconds)
from persistence.backtests import get_backtest_task
from execution.run_recovery import advance_stopped_run_check, settle_stopped_run_results
from execution.runs import fail_automated_run
from execution.submission_queue import synchronize_submission_queue
from persistence.database import open_database
from persistence.submission_checks import get_submission_check
from persistence.submission_queue import list_formal_submission_queue
from persistence.submissions import list_formal_submission_attempts
from tests.execution import test_submission_runner as fixture
from worldquant.backtests import STANDARD_REGULAR_CHECK_NAMES
from worldquant.client import WorldQuantRequestError
from worldquant.submissions import FormalCheckObservation


class SubmissionCheckTests(unittest.TestCase):
    def test_late_sc_pass_archives_exhausted_parent_after_original_cycle_finished(self):
        from tests.execution import test_qualified_archive as archive_fixture
        from persistence.qualified_archive import list_qualified_alpha_archive

        with open_database(self.database_path) as connection:
            parent = archive_fixture.exhausted_parent(connection)
            connection.execute("DELETE FROM submission_checks WHERE task_id=?", (parent.task.task_id,))
        client = SimpleNamespace(fetch_formal_submission_check=lambda **_: FormalCheckObservation(
            {"is": {"checks": [{"name": name, "result": "PASS"}
                                for name in STANDARD_REGULAR_CHECK_NAMES]}}, None,
        ))
        self.assertTrue(check_completed_backtest(self.database_path, client, parent.task.task_id,
            observed_at=archive_fixture.ARCHIVED, recovered=True, include_deferred=True))
        with open_database(self.database_path) as connection:
            self.assertEqual(tuple(row.task_id for row in list_qualified_alpha_archive(connection)), (parent.task.task_id,))
            self.assertEqual(list_formal_submission_queue(connection), ())

    def setUp(self):
        self.fixture = fixture.SubmissionQueueRunnerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.database_path = self.fixture.database_path

    def test_only_spectacular_enters_queue_without_changing_other_results(self):
        grades = ("SPECTACULAR", "EXCELLENT", "GOOD", "AVERAGE", "INFERIOR", None, "UNKNOWN")
        formulas = tuple(f"ts_mean(close,{i})" for i in range(2, 9))
        tasks, _ = self.fixture._completed_run(formulas, grades=grades)
        with open_database(self.database_path) as connection:
            self.assertEqual([item.task_id for item in list_formal_submission_queue(connection)], [tasks[0]])
            for task, grade in zip(tasks, grades):
                snapshot = get_backtest_task(connection, task)
                self.assertEqual(snapshot.task.status, "completed")
                self.assertEqual(snapshot.result.grade, grade)
                self.assertEqual(snapshot.result.fitness, 1.1)
                self.assertTrue(snapshot.yearly_stats)
            connection.execute("DELETE FROM submission_checks")
        for task in tasks[1:]:
            # SC observations still serve research, independently of queue admission.
            client = SimpleNamespace(fetch_formal_submission_check=lambda **_: FormalCheckObservation(
                {"is": {"checks": [{"name": name, "result": "PASS"}
                                    for name in STANDARD_REGULAR_CHECK_NAMES]}}, None,
            ))
            with self.assertLogs("execution.progress", level="INFO") as captured:
                self.assertTrue(check_completed_backtest(
                    self.database_path, client, task, observed_at="2026-09-04T00:08:00+00:00",
                ))
            self.assertIn("不入队", " ".join(captured.output))
        with open_database(self.database_path) as connection:
            self.assertEqual([item.task_id for item in list_formal_submission_queue(connection)], [tasks[0]])
            for task in tasks[1:]:
                self.assertIn('"PASS"', get_submission_check(connection, task).payload_json)

    def test_only_official_pass_enters_queue_and_replay_never_rechecks(self):
        tasks, _ = self.fixture._completed_run(("rank(close)", "rank(open)", "rank(high)", "rank(low)"))
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM submission_queue")
            connection.execute("DELETE FROM submission_checks")
            synchronize_submission_queue(connection, candidate_task_ids=tasks, enqueued_at="2026-09-04T00:07:00+00:00")
            self.assertEqual(list_formal_submission_queue(connection), ())
        observed = []
        for task_id, state in zip(tasks, ("PASS", "FAIL", "PENDING", "ERROR")):
            def check(*, platform_alpha_id, state=state):
                observed.append(platform_alpha_id)
                if state == "ERROR":
                    raise WorldQuantRequestError("network_error", retryable=True, outcome_unknown=False)
                return FormalCheckObservation({"is": {"checks": [
                    {"name": name, "result": state if name == "SELF_CORRELATION" else "PASS"}
                    for name in STANDARD_REGULAR_CHECK_NAMES
                ]}}, None)
            client = SimpleNamespace(fetch_formal_submission_check=check)
            self.assertTrue(check_completed_backtest(
                self.database_path, client, task_id, observed_at="2026-09-04T00:08:00+00:00",
            ))
            self.assertFalse(check_completed_backtest(
                self.database_path, client, task_id, observed_at="2026-09-04T00:08:01+00:00",
            ))
        with open_database(self.database_path) as connection:
            synchronize_submission_queue(connection, candidate_task_ids=tasks, enqueued_at="2026-09-04T00:10:00+00:00")
            self.assertEqual([row.task_id for row in list_formal_submission_queue(connection)], [tasks[0]])
            self.assertEqual(list_formal_submission_attempts(connection), ())
            error = get_submission_check(connection, tasks[3])
            self.assertEqual(error.error_code, "network_error")
            self.assertIsNone(error.payload_json)
            pending = json.loads(get_submission_check(connection, tasks[2]).payload_json)
            self.assertIn("PENDING", str(pending))
        self.assertEqual(len(observed), 4)

    def test_incomplete_backtest_does_not_call_check(self):
        tasks, _ = self.fixture._completed_run(("rank(close)",), include_yearly_stats=False)
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM submission_checks")
        self.assertFalse(check_completed_backtest(
            self.database_path, SimpleNamespace(), tasks[0], observed_at="2026-09-04T00:08:00+00:00",
        ))

    def test_transient_reads_respect_retry_after_and_survive_restart(self):
        tasks, _ = self.fixture._completed_run(("rank(close)",))
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM submission_queue")
            passed_payload = json.loads(get_submission_check(connection, tasks[0]).payload_json)
            connection.execute("DELETE FROM submission_checks")
        replies = iter([
            WorldQuantRequestError("rate_limited", retryable=True, outcome_unknown=False, retry_after_seconds=120),
            FormalCheckObservation({"is": {"checks": []}}, 90),
            FormalCheckObservation(passed_payload, None),
        ])
        calls = []

        def check(**kwargs):
            calls.append(kwargs)
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        base = datetime.fromisoformat("2026-09-04T00:08:00+00:00")
        for seconds, expected in [(0, True), (119, False), (120, True), (209, False), (210, True), (400, False)]:
            # A fresh caller each time relies only on persisted retry state.
            self.assertEqual(check_completed_backtest(self.database_path,
                SimpleNamespace(fetch_formal_submission_check=check), tasks[0],
                observed_at=(base + timedelta(seconds=seconds)).isoformat()), expected)
        self.assertEqual(len(calls), 3)
        with open_database(self.database_path) as connection:
            saved = get_submission_check(connection, tasks[0])
            self.assertEqual(saved.attempt_count, 3)
            self.assertIsNone(saved.retry_not_before)
            self.assertEqual([row.task_id for row in list_formal_submission_queue(connection)], list(tasks))

    def test_exhausted_pending_is_unknown_not_queued_or_a_failed_backtest(self):
        tasks, _ = self.fixture._completed_run(("rank(close)",))
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM submission_queue")
            connection.execute("DELETE FROM submission_checks")
        client = SimpleNamespace(fetch_formal_submission_check=lambda **_: FormalCheckObservation({}, None))
        for minute in (8, 9, 10):
            self.assertTrue(check_completed_backtest(self.database_path, client, tasks[0],
                observed_at=f"2026-09-04T00:{minute:02}:00+00:00"))
        self.assertFalse(check_completed_backtest(self.database_path, client, tasks[0],
            observed_at="2026-09-04T01:00:00+00:00"))
        with open_database(self.database_path) as connection:
            self.assertEqual(list_formal_submission_queue(connection), ())
            self.assertEqual(get_submission_check(connection, tasks[0]).payload_json, "{}")
            self.assertEqual(connection.execute("SELECT status FROM backtest_tasks WHERE task_id=?", tasks).fetchone()[0], "completed")
            saved = get_submission_check(connection, tasks[0])
            self.assertEqual(saved.retry_not_before, "2026-09-04T00:20:00+00:00")
            self.assertIsNone(remaining_submission_check_seconds(
                connection, get_backtest_task(connection, tasks[0]), "2026-09-04T00:20:00+00:00"))

    def test_deferred_check_survives_settlement_and_enters_queue_on_late_pass(self):
        tasks, _ = self.fixture._completed_run(("rank(close)",))
        with open_database(self.database_path) as connection:
            passed = json.loads(get_submission_check(connection, tasks[0]).payload_json)
            connection.execute("DELETE FROM submission_queue")
            connection.execute("DELETE FROM submission_checks")
        pending = SimpleNamespace(fetch_formal_submission_check=lambda **_: FormalCheckObservation(None, 1200))
        for minute in (8, 28, 48):
            check_completed_backtest(self.database_path, pending, tasks[0],
                                     observed_at=f"2026-09-04T00:{minute:02}:00+00:00")
        client = SimpleNamespace(fetch_formal_submission_check=lambda **_: FormalCheckObservation(passed, None))
        self.assertFalse(advance_deferred_submission_check(self.database_path, client,
            account_scope="another-account", observed_at="2026-09-04T01:08:00+00:00"))
        self.assertFalse(advance_deferred_submission_check(self.database_path, client,
            account_scope="group-account", observed_at="2026-09-04T01:07:59+00:00"))
        self.assertTrue(advance_deferred_submission_check(self.database_path, client,
            account_scope="group-account", observed_at="2026-09-04T01:08:00+00:00"))
        self.assertFalse(advance_deferred_submission_check(self.database_path, SimpleNamespace(),
            account_scope="group-account", observed_at="2026-09-04T02:08:00+00:00"))
        with open_database(self.database_path) as connection:
            self.assertEqual([r.task_id for r in list_formal_submission_queue(connection)], list(tasks))
            self.assertEqual(get_submission_check(connection, tasks[0]).attempt_count, 4)
            self.assertEqual(get_backtest_task(connection, tasks[0]).task.status, "completed")

    def test_legacy_exhausted_empty_check_is_revisited_one_at_a_time(self):
        tasks, _ = self.fixture._completed_run(("rank(close)", "rank(open)"))
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM submission_queue")
            connection.execute("UPDATE submission_checks SET payload_json='null', attempt_count=3, "
                               "retry_not_before=NULL, observed_at='2026-09-04T00:10:00+00:00'")
        calls = []
        def check(**kwargs):
            calls.append(kwargs)
            return FormalCheckObservation(None, None)
        self.assertFalse(advance_deferred_submission_check(self.database_path, SimpleNamespace(),
            account_scope="group-account", observed_at="2026-09-04T00:19:59+00:00"))
        self.assertTrue(advance_deferred_submission_check(self.database_path,
            SimpleNamespace(fetch_formal_submission_check=check),
            account_scope="group-account", observed_at="2026-09-04T00:20:00+00:00"))
        self.assertEqual(len(calls), 1)
        with open_database(self.database_path) as connection:
            self.assertEqual(sorted(get_submission_check(connection, t).attempt_count for t in tasks), [3, 4])
            self.assertEqual(list_formal_submission_queue(connection), ())

    def test_stopped_completed_result_gets_its_missing_check_without_resimulation(self):
        tasks, _ = self.fixture._completed_run(("rank(close)",))
        with open_database(self.database_path) as connection:
            passed_payload = json.loads(get_submission_check(connection, tasks[0]).payload_json)
            run_id = connection.execute("SELECT run_id FROM automated_run_backtests WHERE task_id=?", tasks).fetchone()[0]
            connection.execute("DELETE FROM submission_queue")
            connection.execute("DELETE FROM submission_checks")
            # Reproduce interruption after results but before run settlement.
            connection.execute("DELETE FROM automated_cycle_settlements WHERE run_id=?", (run_id,))
            connection.execute("UPDATE automated_runs SET status='running', current_cycle=0, finished_at=NULL, stop_reason=NULL WHERE run_id=?", (run_id,))
        fail_automated_run(self.database_path, run_id, failed_at="2026-09-04T00:07:00+00:00", reason="replaced_by_new_run")
        client = SimpleNamespace(fetch_formal_submission_check=lambda **_: FormalCheckObservation(passed_payload, None))
        settle_stopped_run_results(self.database_path, account_scope="group-account",
                                   observed_at="2026-09-04T00:07:00+00:00")
        advanced = advance_stopped_run_check(self.database_path, client,
            account_scope="group-account", observed_at="2026-09-04T00:08:00+00:00")
        self.assertEqual(advanced.action, "submission_check_observed")
        with open_database(self.database_path) as connection:
            self.assertEqual([row.task_id for row in list_formal_submission_queue(connection)], list(tasks))

    def test_settled_history_is_not_reopened_even_with_missing_or_pending_check(self):
        tasks, _ = self.fixture._completed_run(("rank(close)", "rank(open)"))
        with open_database(self.database_path) as connection:
            run_id = connection.execute("SELECT run_id FROM automated_run_backtests WHERE task_id=?", (tasks[0],)).fetchone()[0]
            connection.execute("DELETE FROM submission_checks")
            connection.execute("UPDATE automated_runs SET status='failed', stop_reason='replaced_by_new_run' WHERE run_id=?", (run_id,))
        pending = SimpleNamespace(fetch_formal_submission_check=lambda **_: FormalCheckObservation({}, 30))
        check_completed_backtest(self.database_path, pending, tasks[0], observed_at="2026-09-04T00:08:00+00:00")
        self.assertIsNone(advance_stopped_run_check(self.database_path, SimpleNamespace(),
            account_scope="group-account", observed_at="2026-09-04T01:00:00+00:00"))
        with open_database(self.database_path) as connection:
            self.assertEqual(get_submission_check(connection, tasks[0]).attempt_count, 1)
            self.assertIsNone(get_submission_check(connection, tasks[1]))

    def test_formal_checks_and_submitted_formulas_are_not_prechecked_again(self):
        tasks, formulas = self.fixture._completed_run(("rank(close)", "rank(open)"))
        client = fixture.SubmissionClient(formulas, failed_check_ids=frozenset({"alpha-1"}))
        self.fixture._submit(client, fixture.FakeTime("2026-09-04T01:00:00+00:00"))
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM submission_checks")
            self.assertEqual({a.status for a in list_formal_submission_attempts(connection)}, {"submitted", "ineligible"})
        for task_id in tasks:
            self.assertFalse(check_completed_backtest(self.database_path, SimpleNamespace(), task_id,
                observed_at="2026-09-04T02:00:00+00:00"))

    def test_deferred_authentication_failure_is_not_saved_as_formula_failure(self):
        tasks, _ = self.fixture._completed_run(("rank(close)",))
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM submission_queue")
            connection.execute("UPDATE submission_checks SET payload_json='null',attempt_count=3,retry_not_before=NULL")
            before = get_submission_check(connection, tasks[0])
        def check(**kwargs):
            raise WorldQuantRequestError("authentication_failed", retryable=False, outcome_unknown=False, status_code=401)
        with self.assertRaises(WorldQuantRequestError):
            advance_deferred_submission_check(self.database_path, SimpleNamespace(fetch_formal_submission_check=check),
                account_scope="group-account", observed_at="2026-09-04T01:00:00+00:00")
        with open_database(self.database_path) as connection:
            self.assertEqual(get_submission_check(connection, tasks[0]), before)
            self.assertEqual(list_formal_submission_queue(connection), ())

    def test_imported_submitted_formula_is_skipped_without_a_local_attempt(self):
        tasks, _ = self.fixture._completed_run(("rank(close)",))
        self.fixture._record_submitted_alpha(platform_alpha_id="imported-alpha", formula="rank(close)")
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM submission_checks")
        self.assertFalse(check_completed_backtest(self.database_path, SimpleNamespace(), tasks[0],
            observed_at="2026-09-04T02:00:00+00:00"))
