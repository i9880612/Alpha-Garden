import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from execution.backtest_batches import AutomatedCandidateBacktest, prepare_automated_candidate_backtest_batch
from execution.backtests import record_submission_accepted, record_submission_unknown
from execution.launch import launch_automated_run
from execution.real_backtests import advance_real_backtest, account_in_flight_backtest_count
from execution.run_recovery import advance_stopped_run_backtest
from execution.runs import prepare_automated_run, start_automated_run, fail_automated_run
from persistence.backtests import get_backtest_task
from persistence.database import open_database
from persistence.runs import get_automated_run, list_automated_run_backtests
from persistence.seeds import list_signal_seeds
from worldquant.backtests import BacktestPollObservation, BacktestSubmissionObservation
from worldquant.client import WorldQuantRequestError
from worldquant.backtests import BacktestYearlyStatsObservation, STANDARD_REGULAR_CHECK_NAMES
from worldquant.submissions import FormalCheckObservation
from tests.execution import test_launch as fixture
from tests.execution import test_submission_runner as submission_fixture


class RecoveryClient(fixture.LaunchClient):
    def __init__(self, old_state="completed"):
        super().__init__()
        self.old_state = old_state
        self.sent_formulas = []
        self.polled_remotes = []

    def submit_backtest(self, *, formula, settings):
        self.calls.append("submit")
        self.sent_formulas.append(formula)
        return BacktestSubmissionObservation("accepted",
            "https://api.worldquantbrain.com/simulations/new", None)

    def poll_backtest(self, remote_id):
        self.calls.append("poll")
        self.polled_remotes.append(remote_id)
        suffix = remote_id.rsplit("/", 1)[-1]
        if suffix == "old":
            if self.old_state == "error":
                raise WorldQuantRequestError("old_poll_error", status_code=404,
                                            retryable=False, outcome_unknown=False)
            if self.old_state == "pending":
                return BacktestPollObservation("pending", None, "PENDING", None)
        return BacktestPollObservation("completed", "alpha-"+suffix, "COMPLETE", None)


class RunRecoveryTests(unittest.TestCase):
    def test_stopped_result_recovery_also_archives_exhausted_qualified_parents(self):
        from tests.execution import test_qualified_archive as archive_fixture
        from execution.run_recovery import settle_stopped_run_results
        from persistence.qualified_archive import list_qualified_alpha_archive

        self._old(failed=True)
        with open_database(self.db) as connection:
            parent = archive_fixture.exhausted_parent(connection)
        settle_stopped_run_results(self.db, account_scope="group-account",
                                   observed_at=archive_fixture.ARCHIVED)
        with open_database(self.db) as connection:
            self.assertEqual(tuple(row.task_id for row in list_qualified_alpha_archive(connection)),
                             (parent.task.task_id,))

    def setUp(self):
        self.fixture = fixture.AutomatedRunLaunchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.db = self.fixture.database_path
        self.time = fixture.FakeTime("2026-08-30T00:00:00+08:00")

    def _old(self, *, mixed=False, failed=False, elapsed_seconds=120):
        limits = replace(self.fixture._limits(), generation_count=5,
                         backtest_count=4 if mixed else 1, max_cycles=-1, max_backtests=0)
        run = prepare_automated_run(self.db, self.fixture.settings_path,
            account_scope="group-account", limits=limits, created_at=self.time.now().isoformat())
        start_automated_run(self.db, run.run_id, started_at=self.time.now().isoformat())
        formulas = ("rank(close)", "rank(open)", "rank(volume)", "ts_rank(close,22)") if mixed else ("rank(close)",)
        tasks = prepare_automated_candidate_backtest_batch(self.db, run_id=run.run_id,
            candidates=tuple(AutomatedCandidateBacktest(self.fixture._candidate(f),
                self.fixture.settings) for f in formulas), created_at=self.time.now().isoformat())
        with open_database(self.db) as c:
            record_submission_accepted(c, tasks[0].task.task_id,
                remote_id="https://api.worldquantbrain.com/simulations/old",
                observed_at=self.time.now().isoformat())
            if mixed:
                record_submission_unknown(c, tasks[1].task.task_id, observed_at=self.time.now().isoformat())
                record_submission_accepted(c, tasks[3].task.task_id,
                    remote_id="https://api.worldquantbrain.com/simulations/finished",
                    observed_at=self.time.now().isoformat())
        if mixed:
            client = RecoveryClient()
            client.authenticated = True
            for _ in range(3):
                advance_real_backtest(self.db, client, tasks[3].task.task_id,
                    observed_at=self.time.now().isoformat(), allow_submission=False)
        if failed:
            fail_automated_run(self.db, run.run_id, failed_at=self.time.now().isoformat(),
                reason="platform_request_not_retryable:test", request_failure_code="test",
                request_status_code=404)
        self.time.wait(elapsed_seconds)
        return run, tasks

    def _launch(self, client):
        return launch_automated_run(self.db, self.fixture.settings_path,
            self.fixture.environment_path, limits=self.fixture._limits(),
            clock=self.time.now, waiter=self.time.wait, client_factory=lambda _: client)

    def test_new_limits_and_old_mixed_facts_remain_separate(self):
        old, tasks = self._old(mixed=True)
        with open_database(self.db) as c:
            unknown = get_backtest_task(c, tasks[1].task.task_id)
            completed = get_backtest_task(c, tasks[3].task.task_id)
        client = RecoveryClient()
        result = self._launch(client)
        self.assertEqual(result.run.status, "completed")
        self.assertEqual(result.run.max_cycles, 1)
        self.assertEqual(result.run.max_backtests, 1)
        self.assertEqual(len(client.sent_formulas), 1)
        self.assertNotIn(client.sent_formulas[0], [t.task.formula for t in tasks])
        with open_database(self.db) as c:
            previous = get_automated_run(c, old.run_id)
            self.assertEqual(previous.status, "failed")
            self.assertEqual(previous.max_cycles, -1)
            self.assertEqual(previous.stop_reason, "replaced_by_new_run")
            self.assertEqual(get_backtest_task(c, tasks[0].task.task_id).task.status, "completed")
            expired = get_backtest_task(c, tasks[1].task.task_id)
            self.assertEqual(expired.task.status, "failed")
            self.assertEqual(expired.task.failure_code, "submission_outcome_timeout")
            self.assertEqual(expired.task.request_fingerprint, unknown.task.request_fingerprint)
            self.assertIsNone(expired.result)
            cancelled = get_backtest_task(c, tasks[2].task.task_id)
            self.assertEqual(cancelled.task.status, "failed")
            self.assertIsNone(cancelled.task.submission_started_at)
            self.assertEqual(get_backtest_task(c, tasks[3].task.task_id), completed)
            self.assertEqual(len(list_automated_run_backtests(c, result.run.run_id)), 1)

    def test_hard_failed_owner_is_not_revived_or_its_error_cleared(self):
        old, tasks = self._old(failed=True)
        client = RecoveryClient()
        result = self._launch(client)
        self.assertEqual(result.run.status, "completed")
        with open_database(self.db) as c:
            previous = get_automated_run(c, old.run_id)
            self.assertEqual(previous.status, "failed")
            self.assertEqual(previous.stop_reason, "platform_request_not_retryable:test")
            self.assertEqual(previous.last_request_failure_code, "test")
            self.assertEqual(previous.last_request_status_code, 404)
            self.assertEqual(previous.current_cycle, 1)
            self.assertEqual(get_backtest_task(c, tasks[0].task.task_id).task.status, "completed")

    def test_expired_old_task_releases_slot_and_preserves_request(self):
        old, tasks = self._old()
        client = RecoveryClient("pending")
        result = self._launch(client)
        self.assertEqual(result.run.status, "completed")
        self.assertEqual(len(client.sent_formulas), 1)
        with open_database(self.db) as c:
            task = get_backtest_task(c, tasks[0].task.task_id).task
            self.assertEqual(task.status, "failed")
            self.assertEqual(task.failure_code, "platform_pending_timeout")
            self.assertEqual(task.remote_id, "https://api.worldquantbrain.com/simulations/old")
            self.assertEqual(account_in_flight_backtest_count(c, "group-account"), 0)
            self.assertEqual(get_automated_run(c, old.run_id).status, "failed")

    def test_old_fetch_error_does_not_fail_new_plan_or_fake_a_terminal_result(self):
        _, tasks = self._old(failed=True, elapsed_seconds=10)
        client = RecoveryClient("error")
        with self.assertLogs("execution.run_recovery", level="WARNING"):
            result = self._launch(client)
        self.assertEqual(result.run.status, "completed")
        with open_database(self.db) as c:
            old = get_backtest_task(c, tasks[0].task.task_id)
            self.assertEqual(old.task.status, "pending")
            self.assertIsNone(old.task.finished_at)
            self.assertIsNone(old.result)

    def test_recovery_does_not_touch_a_different_account(self):
        _, tasks = self._old(failed=True)
        with open_database(self.db) as c:
            before = get_backtest_task(c, tasks[0].task.task_id)
        client = RecoveryClient()
        client.authenticated = True
        self.assertIsNone(advance_stopped_run_backtest(self.db, client,
            account_scope="different", observed_at=self.time.now().isoformat()))
        self.assertEqual(client.calls, [])
        with open_database(self.db) as c:
            self.assertEqual(get_backtest_task(c, tasks[0].task.task_id), before)

    def test_three_expired_old_requests_release_capacity_for_new_run(self):
        old, tasks = self._old(mixed=True)
        with open_database(self.db) as c:
            for index in (1, 2):
                record_submission_accepted(
                    c, tasks[index].task.task_id,
                    remote_id=f"https://api.worldquantbrain.com/simulations/old-{index}",
                    observed_at="2026-08-30T00:00:00+08:00",
                )
            self.assertEqual(account_in_flight_backtest_count(c, "group-account"), 3)
        class AllOldPending(RecoveryClient):
            def poll_backtest(self, remote_id):
                if remote_id.rsplit("/", 1)[-1].startswith("old"):
                    self.polled_remotes.append(remote_id)
                    return BacktestPollObservation("pending", None, "RUNNING", 0.1)
                return super().poll_backtest(remote_id)
        client = AllOldPending()
        start = self.time.now()
        original_wait = self.time.wait
        def bounded_wait(seconds):
            original_wait(seconds)
            self.assertLess((self.time.now() - start).total_seconds(), 180)
        self.time.wait = bounded_wait
        result = self._launch(client)
        self.assertEqual(result.run.status, "completed")
        self.assertEqual(len(client.sent_formulas), 1)
        self.assertNotIn(client.sent_formulas[0], [item.task.formula for item in tasks])
        with open_database(self.db) as c:
            for item in tasks[:3]:
                expired = get_backtest_task(c, item.task.task_id)
                self.assertEqual(expired.task.failure_code, "platform_pending_timeout")
                self.assertIsNotNone(expired.task.remote_id)
                self.assertIsNone(expired.result)
            self.assertEqual(account_in_flight_backtest_count(c, "group-account"), 0)

    def test_expired_old_read_error_releases_local_slot_without_fake_result(self):
        _, tasks = self._old(failed=True)
        client = RecoveryClient("error")
        client.authenticated = True
        result = advance_stopped_run_backtest(
            self.db, client, account_scope="group-account",
            observed_at=self.time.now().isoformat(),
        )
        self.assertEqual(result.action, "pending_timeout")
        self.assertEqual(result.snapshot.task.failure_code, "platform_pending_timeout")
        self.assertIsNone(result.snapshot.result)
        self.assertEqual(client.sent_formulas, [])

    def test_full_old_capacity_is_drained_before_new_post(self):
        _, tasks = self._old(mixed=True)
        with open_database(self.db) as c:
            record_submission_accepted(c, tasks[2].task.task_id,
                remote_id="https://api.worldquantbrain.com/simulations/old-second",
                observed_at=self.time.now().isoformat())
            self.assertEqual(account_in_flight_backtest_count(c, "group-account"), 3)
        client = RecoveryClient()
        original_submit = client.submit_backtest
        observed_counts = []

        def submit(**arguments):
            with open_database(self.db) as c:
                observed_counts.append(account_in_flight_backtest_count(c, "group-account"))
            return original_submit(**arguments)

        client.submit_backtest = submit
        result = self._launch(client)
        self.assertEqual(result.run.status, "completed")
        self.assertEqual(len(observed_counts), 1)
        self.assertLessEqual(observed_counts[0], 3)
        with open_database(self.db) as c:
            self.assertEqual(get_backtest_task(c, tasks[1].task.task_id).task.failure_code,
                             "submission_outcome_timeout")

    def test_completed_old_result_is_seeded_and_settled_without_any_pending(self):
        old, tasks = self._old()
        client = RecoveryClient()
        client.authenticated = True
        for _ in range(3):
            advance_real_backtest(self.db, client, tasks[0].task.task_id,
                observed_at=self.time.now().isoformat(), allow_submission=False)
        with open_database(self.db) as c:
            self.assertEqual(get_automated_run(c, old.run_id).current_cycle, 0)
            self.assertNotIn(tasks[0].task.task_id, {s.root_task_id for s in list_signal_seeds(c)})
            original = get_backtest_task(c, tasks[0].task.task_id)
        result = self._launch(RecoveryClient())
        self.assertEqual(result.run.status, "completed")
        with open_database(self.db) as c:
            self.assertEqual(get_automated_run(c, old.run_id).current_cycle, 1)
            self.assertIn(tasks[0].task.task_id, {s.root_task_id for s in list_signal_seeds(c)})
            self.assertEqual(get_backtest_task(c, tasks[0].task.task_id), original)

    def test_full_capacity_slow_first_task_does_not_starve_completable_later_task(self):
        _, tasks = self._old(mixed=True)
        with open_database(self.db) as c:
            record_submission_accepted(c, tasks[2].task.task_id,
                remote_id="https://api.worldquantbrain.com/simulations/old-second",
                observed_at=self.time.now().isoformat())
        client = RecoveryClient("pending")
        start = self.time.now()
        original_wait = self.time.wait

        def bounded_wait(seconds):
            original_wait(seconds)
            self.assertLess((self.time.now()-start).total_seconds(), 600,
                            "a slow old task starved independent completed work")

        self.time.wait = bounded_wait
        result = self._launch(client)
        self.assertEqual(result.run.status, "completed")
        self.assertIn("https://api.worldquantbrain.com/simulations/old-second", client.polled_remotes)
        self.assertEqual(len(client.sent_formulas), 1)
        with open_database(self.db) as c:
            self.assertEqual(get_backtest_task(c, tasks[0].task.task_id).task.failure_code,
                             "platform_pending_timeout")
            self.assertEqual(get_backtest_task(c, tasks[2].task.task_id).task.status, "completed")
            self.assertIn(tasks[3].task.task_id, {s.root_task_id for s in list_signal_seeds(c)})

    def test_old_checks_finish_before_new_batch_and_new_checks_wait_for_all_results(self):
        old, tasks = self._old()
        test = self

        class PhasedClient(RecoveryClient):
            old_checks = 0
            checked = []

            def fetch_backtest_detail(self, **kwargs):
                return replace(super().fetch_backtest_detail(**kwargs), grade="SPECTACULAR")

            def fetch_backtest_yearly_stats(self, *, platform_alpha_id):
                self.calls.append("yearly_stats")
                return BacktestYearlyStatsObservation("ready", (submission_fixture.SubmissionQueueRunnerTests._yearly_stat(1.3),), None)

            def fetch_formal_submission_check(self, *, platform_alpha_id):
                self.checked.append(platform_alpha_id)
                if platform_alpha_id == "alpha-old":
                    self.old_checks += 1
                    test.assertEqual(self.sent_formulas, [])
                    if self.old_checks == 1:
                        return FormalCheckObservation({}, 40)
                else:
                    with open_database(test.db) as c:
                        states = c.execute("SELECT t.status FROM backtest_tasks t JOIN automated_run_backtests l USING(task_id) WHERE l.run_id<>?", (old.run_id,)).fetchall()
                    test.assertEqual(len(states), 2)
                    test.assertTrue(all(row[0] == "completed" for row in states))
                return FormalCheckObservation({"is": {"checks": [
                    {"name": name, "result": "PASS"} for name in STANDARD_REGULAR_CHECK_NAMES
                ]}}, None)

            def submit_backtest(self, *, formula, settings):
                test.assertEqual(self.old_checks, 2)
                with open_database(test.db) as c:
                    test.assertIsNotNone(c.execute("SELECT task_id FROM submission_queue WHERE task_id=?", (tasks[0].task.task_id,)).fetchone())
                self.sent_formulas.append(formula)
                return BacktestSubmissionObservation("accepted",
                    f"https://api.worldquantbrain.com/simulations/new-{len(self.sent_formulas)}", None)

        client = PhasedClient()
        client.authenticated = True
        for _ in range(3):
            advance_real_backtest(self.db, client, tasks[0].task.task_id,
                observed_at=self.time.now().isoformat(), allow_submission=False)
        result = launch_automated_run(self.db, self.fixture.settings_path, self.fixture.environment_path,
            limits=replace(self.fixture._limits(), generation_count=5, backtest_count=2, max_backtests=2),
            clock=self.time.now, waiter=self.time.wait, client_factory=lambda _: client)
        self.assertEqual(result.run.status, "completed")
        self.assertEqual(client.checked[:2], ["alpha-old", "alpha-old"])
        self.assertEqual(set(client.checked[2:]), {"alpha-new-1", "alpha-new-2"})
        self.assertEqual(len(client.checked), 4)
        self.assertTrue(any(delay >= 30 for delay in self.time.waits))
