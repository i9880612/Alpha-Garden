from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.execution.catalog_fixture import initialize_test_generation_catalog
from execution.backtests import record_submission_accepted, record_submission_unknown
from execution.backtest_batches import (
    AutomatedCandidateBacktest,
    prepare_automated_candidate_backtest_batch,
)
from execution.runner import run_automated_run
from execution.runs import (
    AutomatedRunLimits,
    fail_automated_run,
    prepare_automated_run,
    start_automated_run,
)
from generation.candidate import FormulaCandidate, exploration_candidate
from generation.parser import parse_formula
from persistence.backtests import get_backtest_task, initialize_backtest_schema
from persistence.database import open_database
from persistence.runs import (
    get_automated_run,
    get_automated_cycle_settlement,
    initialize_run_schema,
    list_automated_run_backtests,
)
from persistence.submission_queue import initialize_submission_queue_schema
from persistence.submissions import initialize_submission_schema
from worldquant.backtests import (
    BacktestCheck,
    BacktestDetail,
    BacktestPollObservation,
    BacktestSettings,
    BacktestSubmissionObservation,
    BacktestYearlyStat,
    BacktestYearlyStatsObservation,
    STANDARD_REGULAR_CHECK_NAMES,
)
from worldquant.alphas import UserAlphaPage, UserAlphaRecord
from worldquant.client import WorldQuantRequestError
from worldquant.submissions import (
    AlphaDetailObservation,
    FormalCheckObservation,
    FormalSubmissionObservation,
)


class FakeTime:
    def __init__(self, value: str) -> None:
        self.value = datetime.fromisoformat(value)
        self.waits: list[float] = []

    def now(self) -> datetime:
        return self.value

    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.value += timedelta(seconds=seconds)

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class RunnerClient:
    def __init__(
        self,
        *submissions,
        poll_pending: bool = False,
        polls=(),
        poll_observer=None,
        authentications=(),
        authentication_observer=None,
        alpha_records=(),
    ) -> None:
        self.authenticated = False
        self.submissions = list(submissions)
        self.poll_pending = poll_pending
        self.polls = list(polls)
        self.poll_observer = poll_observer
        self.authentications = list(authentications)
        self.authentication_observer = authentication_observer
        self.alpha_records = tuple(alpha_records)
        self.calls: list[str] = []
        self.formulas: dict[str, str] = {}
        self.formal_submission_ids: list[str] = []

    def authenticate(self):
        self.calls.append("authenticate")
        if self.authentications:
            result = self.authentications.pop(0)
            if isinstance(result, Exception):
                raise result
        if self.authentication_observer is not None:
            self.authentication_observer()
        self.authenticated = True

    def submit_backtest(self, *, formula, settings):
        self.calls.append("submit")
        result = self.submissions.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def poll_backtest(self, remote_id):
        self.calls.append("poll")
        if self.poll_observer is not None:
            self.poll_observer()
        if self.polls:
            result = self.polls.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        if self.poll_pending:
            return BacktestPollObservation("pending", None, "RUNNING", 0.5)
        return BacktestPollObservation("completed", "alpha-1", "COMPLETE", None)

    def fetch_backtest_detail(
        self,
        *,
        platform_alpha_id,
        expected_formula,
        expected_settings,
    ):
        self.calls.append("detail")
        self.formulas[platform_alpha_id] = expected_formula
        return BacktestDetail(
            platform_alpha_id=platform_alpha_id,
            sharpe=1.3,
            fitness=1.1,
            turnover=0.12,
            returns=0.08,
            drawdown=0.04,
            margin=0.001,
            book_size=20_000_000,
            pnl=100_000,
            grade="SPECTACULAR",
            checks=tuple(
                BacktestCheck(
                    name=name,
                    status="PASS",
                    threshold=None,
                    actual=None,
                    platform_date=None,
                )
                for name in sorted(STANDARD_REGULAR_CHECK_NAMES)
            ),
        )

    def fetch_backtest_yearly_stats(self, *, platform_alpha_id):
        self.calls.append("yearly_stats")
        return BacktestYearlyStatsObservation(
            state="ready",
            stats=(
                BacktestYearlyStat(
                    year=2025,
                    pnl=100_000,
                    book_size=20_000_000,
                    turnover=0.12,
                    sharpe=1.3,
                    returns=0.08,
                    drawdown=0.04,
                    margin=0.001,
                    fitness=1.1,
                    long_count=100,
                    short_count=100,
                    stage="IS",
                ),
            ),
            retry_after_seconds=None,
        )

    def fetch_formal_submission_check(self, *, platform_alpha_id):
        self.calls.append("formal_check")
        return FormalCheckObservation(
            payload={
                "is": {
                    "checks": [
                        {"name": name, "result": "PASS"}
                        for name in sorted(STANDARD_REGULAR_CHECK_NAMES)
                    ]
                }
            },
            retry_after_seconds=None,
        )

    def submit_formal_alpha(self, *, platform_alpha_id):
        self.calls.append("formal_submit")
        self.formal_submission_ids.append(platform_alpha_id)
        return FormalSubmissionObservation(status_code=201, payload=None)

    def fetch_alpha_detail(self, *, platform_alpha_id):
        self.calls.append("formal_confirmation")
        status = (
            "ACTIVE"
            if platform_alpha_id in self.formal_submission_ids
            else "UNSUBMITTED"
        )
        payload = {
            "id": platform_alpha_id,
            "status": status,
            "grade": "SPECTACULAR",
            "regular": {"code": self.formulas[platform_alpha_id]},
        }
        if status == "ACTIVE":
            payload.update(
                {
                    "dateSubmitted": "2026-08-29T16:10:00+00:00",
                    "hidden": False,
                }
            )
        return AlphaDetailObservation(payload=payload)

    def fetch_user_alpha_page(self, *, limit, offset, hidden, status=None):
        self.calls.append("alpha_list")
        records = self.alpha_records if status is None and hidden is False else ()
        return UserAlphaPage(
            total_count=len(records),
            records=records[offset : offset + limit],
            has_next=offset + limit < len(records),
        )


class AutomatedRunnerTests(unittest.TestCase):
    def test_deferred_checks_do_not_stop_new_batches_and_late_pass_is_queued(self):
        old_run = self._prepare_running_task(max_pending_seconds=60)
        clock = FakeTime("2026-08-30T00:01:31+08:00")
        run_automated_run(self.database_path, RunnerClient(self._accepted()), old_run,
                          clock=clock.now, waiter=clock.wait)
        with open_database(self.database_path) as connection:
            old_id = list_automated_run_backtests(connection, old_run)[0].task_id
            old_snapshot = get_backtest_task(connection, old_id)
            connection.execute("DELETE FROM submission_queue WHERE task_id=?", (old_id,))
            connection.execute("UPDATE submission_checks SET payload_json='null',attempt_count=3,retry_not_before=NULL, "
                               "observed_at='2026-08-30T00:02:00+08:00' WHERE task_id=?", (old_id,))
        class Client(RunnerClient):
            pending = True
            def fetch_formal_submission_check(self, *, platform_alpha_id):
                if platform_alpha_id == "alpha-1" and self.pending:
                    self.calls.append("deferred_check")
                    return FormalCheckObservation(None, None)
                return super().fetch_formal_submission_check(platform_alpha_id=platform_alpha_id)
        for number, formula in ((2, "rank(open)"), (3, "rank(volume)")):
            run_id = self._prepare_running_task(max_pending_seconds=59+number, formulas=(formula,))
            client = Client(self._accepted(f"simulation-{number}"), polls=(
                BacktestPollObservation("completed", f"alpha-{number}", "COMPLETE", None),))
            client.pending = number == 2
            clock = FakeTime(f"2026-08-30T0{number}:00:00+08:00")
            result = run_automated_run(self.database_path, client, run_id, clock=clock.now, waiter=clock.wait)
            self.assertEqual(result.run.status, "completed")
            self.assertEqual(client.calls.count("submit"), 1)
            self.assertNotIn("formal_submit", client.calls)
            with open_database(self.database_path) as connection:
                self.assertEqual(get_backtest_task(connection, old_id), old_snapshot)
                queued = connection.execute("SELECT 1 FROM submission_queue WHERE task_id=?", (old_id,)).fetchone()
                self.assertEqual(queued is not None, number == 3)
                self.assertEqual(connection.execute("SELECT attempt_count FROM submission_checks WHERE task_id=?", (old_id,)).fetchone()[0], number+2)

    def test_expired_unknown_does_not_resume_its_failed_owner(self):
        run_id = self._prepare_running_task(max_pending_seconds=600,
            formulas=("rank(close)", "rank(open)"))
        with open_database(self.database_path) as connection:
            unknown_id, created_id = [link.task_id for link in
                list_automated_run_backtests(connection, run_id)]
            record_submission_unknown(connection, unknown_id,
                observed_at="2026-08-30T00:02:00+08:00")
        failed = fail_automated_run(self.database_path, run_id,
            failed_at="2026-08-30T00:03:00+08:00", reason="submission_reconciliation_required")
        client = RunnerClient()
        clock = FakeTime("2026-08-30T00:12:00+08:00")
        completion = run_automated_run(self.database_path, client, run_id,
            clock=clock.now, waiter=clock.wait)
        self.assertEqual(completion.run, failed)
        self.assertFalse(completion.authentication_performed)
        self.assertEqual(client.calls, [])
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, unknown_id).task.failure_code,
                "submission_outcome_timeout")
            self.assertIsNone(get_backtest_task(connection, created_id).task.submission_started_at)

    def test_submission_claim_expires_old_slots_in_the_same_transaction(self):
        from execution.real_backtests import advance_real_backtest, account_in_flight_backtest_count
        old_run = self._prepare_running_task(max_pending_seconds=600,
            formulas=("rank(close)", "rank(open)", "rank(high)"))
        with open_database(self.database_path) as connection:
            old_ids = [link.task_id for link in list_automated_run_backtests(connection, old_run)]
            for task_id in old_ids:
                record_submission_unknown(connection, task_id, observed_at="2026-08-30T00:02:00+08:00")
        fail_automated_run(self.database_path, old_run,
            failed_at="2026-08-30T00:03:00+08:00", reason="submission_reconciliation_required")
        run_id = self._prepare_running_task(max_pending_seconds=600, formulas=("rank(volume)",))
        with open_database(self.database_path) as connection:
            task_id = list_automated_run_backtests(connection, run_id)[0].task_id
            self.assertEqual(account_in_flight_backtest_count(connection, "group-account"), 3)
        client = RunnerClient(self._accepted())
        client.authenticated = True
        advanced = advance_real_backtest(self.database_path, client, task_id,
            observed_at="2026-08-30T00:12:00+08:00", allow_submission=True)
        self.assertEqual(advanced.action, "submitted")
        self.assertEqual(client.calls, ["submit"])
        with open_database(self.database_path) as connection:
            self.assertEqual(account_in_flight_backtest_count(connection, "group-account"), 1)
            self.assertTrue(all(get_backtest_task(connection, item).task.failure_code == "submission_outcome_timeout" for item in old_ids))

    def test_old_unknown_times_out_and_new_run_fills_three_slots(self):
        old_run = self._prepare_running_task(max_pending_seconds=600)
        with open_database(self.database_path) as connection:
            old_id = list_automated_run_backtests(connection, old_run)[0].task_id
            original = record_submission_unknown(connection, old_id, observed_at="2026-08-30T00:02:00+08:00")
        fail_automated_run(self.database_path, old_run,
            failed_at="2026-08-30T00:03:00+08:00", reason="submission_reconciliation_required")
        run_id = self._prepare_running_task(max_pending_seconds=600,
            formulas=("rank(open)", "rank(high)", "rank(volume)"))
        client = RunnerClient(*(self._accepted(f"fresh-{i}") for i in range(3)))
        clock = FakeTime("2026-08-30T00:12:00+08:00")
        completion = run_automated_run(self.database_path, client, run_id,
            clock=clock.now, waiter=clock.wait)
        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(client.calls[:4], ["authenticate", "submit", "submit", "submit"])
        self.assertEqual(client.calls.count("submit"), 3)
        self.assertNotIn("alpha_list", client.calls)
        with open_database(self.database_path) as connection:
            expired = get_backtest_task(connection, old_id)
            self.assertEqual(expired.task.failure_code, "submission_outcome_timeout")
            self.assertEqual(expired.task.request_fingerprint, original.task.request_fingerprint)
            self.assertIsNone(expired.result)
            self.assertEqual(get_automated_run(connection, old_run).status, "failed")

    def test_mixed_unknown_capacity_reconciles_current_run_and_sends_remaining_task(self):
        old_run = self._prepare_running_task(max_pending_seconds=600)
        with open_database(self.database_path) as connection:
            old_id = list_automated_run_backtests(connection, old_run)[0].task_id
            record_submission_unknown(connection, old_id, observed_at="2026-08-30T00:02:00+08:00")
            old_snapshot = get_backtest_task(connection, old_id)
        fail_automated_run(self.database_path, old_run,
            failed_at="2026-08-30T00:03:00+08:00", reason="submission_reconciliation_required")
        run_id = self._prepare_running_task(max_pending_seconds=600,
            formulas=("rank(open)", "rank(high)", "rank(volume)"))
        with open_database(self.database_path) as connection:
            links = list_automated_run_backtests(connection, run_id)
            unknown_ids = [link.task_id for link in links[:2]]
            records = []
            for index, task_id in enumerate(unknown_ids):
                record_submission_unknown(connection, task_id, observed_at="2026-08-30T00:04:00+08:00")
                task = get_backtest_task(connection, task_id).task
                records.append(UserAlphaRecord(
                    platform_alpha_id=f"reconciled-{index}", alpha_type="REGULAR",
                    status="UNSUBMITTED", formula=task.formula,
                    settings=self.settings.as_platform_dict(), hidden=False,
                    created_at=datetime.fromisoformat("2026-08-30T00:04:01+08:00")))
        client = RunnerClient(self._accepted(), alpha_records=records)
        clock = FakeTime("2026-08-30T00:04:20+08:00")

        def wait(seconds):
            if not clock.waits:
                self.assertEqual(seconds, 60)
                self.assertNotIn("submit", client.calls)
                with open_database(self.database_path) as connection:
                    self.assertEqual(get_automated_run(connection, run_id).status, "running")
                    self.assertTrue(all(get_backtest_task(connection, task_id).task.status ==
                        "submission_unknown" for task_id in unknown_ids))
            clock.wait(seconds)

        completion = run_automated_run(self.database_path, client, run_id,
            clock=clock.now, waiter=wait)
        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(client.calls.count("submit"), 1)
        self.assertGreaterEqual(clock.waits.count(60), 1)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, old_id), old_snapshot)
            self.assertEqual(get_automated_run(connection, old_run).status, "failed")
            for task_id in unknown_ids:
                snapshot = get_backtest_task(connection, task_id)
                self.assertEqual(snapshot.task.failure_code, "remote_accepted_without_simulation_id")
                self.assertIsNone(snapshot.result)
            self.assertEqual(get_backtest_task(connection, links[2].task_id).task.status, "completed")

    def test_historical_capacity_exhaustion_waits_and_reconciles_without_stopping(self):
        old_run = self._prepare_running_task(max_pending_seconds=600,
            formulas=("rank(close)", "rank(open)"), max_in_flight_backtests=2)
        with open_database(self.database_path) as connection:
            ids = [link.task_id for link in list_automated_run_backtests(connection, old_run)]
            for task_id in ids:
                record_submission_unknown(connection, task_id, observed_at="2026-08-30T00:02:00+08:00")
            before = [get_backtest_task(connection, task_id) for task_id in ids]
        fail_automated_run(self.database_path, old_run, failed_at="2026-08-30T00:03:00+08:00", reason="submission_reconciliation_required")
        run_id = self._prepare_running_task(max_pending_seconds=600,
            formulas=("rank(high)",), max_in_flight_backtests=2)
        client = RunnerClient()
        client.alpha_records = ()
        clock = FakeTime("2026-08-30T00:10:00+08:00")
        waits = []

        def wait(seconds):
            waits.append(seconds)
            clock.wait(seconds)
            if len(waits) == 2:
                raise InterruptedError("test_stop_waiting")

        with self.assertRaisesRegex(InterruptedError, "test_stop_waiting"):
            run_automated_run(self.database_path, client, run_id, clock=clock.now, waiter=wait)
        self.assertEqual(waits, [60, 60])
        self.assertEqual(client.calls.count("alpha_list"), 8)
        self.assertNotIn("submit", client.calls)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_automated_run(connection, run_id).status, "running")
            self.assertEqual(get_automated_run(connection, old_run).status, "failed")
            self.assertEqual([get_backtest_task(connection, task_id) for task_id in ids], before)

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.database_path = root / "runner.sqlite3"
        self.settings_path = root / "backtest.json"
        self.settings = BacktestSettings(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
            decay=4,
            neutralization="SECTOR",
            truncation=0.08,
            pasteurization="ON",
            unit_handling="VERIFY",
            nan_handling="OFF",
            language="FASTEXPR",
            visualization=False,
            max_trade="OFF",
            max_position="OFF",
        )
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
        with open_database(self.database_path) as connection:
            initialize_backtest_schema(connection)
            initialize_run_schema(connection)
            initialize_submission_schema(connection)
            initialize_submission_queue_schema(connection)
            initialize_test_generation_catalog(connection)

    def test_unlimited_run_expires_unavailable_task_and_plans_next_cycle(
        self,
    ) -> None:
        run_id = self._prepare_running_task(
            max_pending_seconds=60,
            unlimited=True,
            max_in_flight_backtests=1,
        )
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")
        failure = WorldQuantRequestError(
            "worldquant_poll_request_failed",
            retryable=True,
            outcome_unknown=False,
            transport_error_type="TimeoutError",
        )
        observed_counts = []

        def observe_poll():
            with open_database(self.database_path) as connection:
                run = get_automated_run(connection, run_id)
                observed_counts.append(run.request_failure_count)
                self.assertEqual(run.status, "running")
            fake_time.advance(30)

        client = RunnerClient(
            self._accepted(),
            polls=[failure] * 8,
            poll_observer=observe_poll,
        )
        # Interrupt only when the real driver starts planning the NEXT cycle.
        with patch(
            "execution.driver.plan_automated_cycle", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                run_automated_run(
                    self.database_path,
                    client,
                    run_id,
                    clock=fake_time.now,
                    waiter=fake_time.wait,
                )

        self.assertEqual(client.calls.count("submit"), 1)
        self.assertLess(client.calls.count("poll"), 8)
        self.assertLess(max(observed_counts), 8)
        self.assertLess((fake_time.now() - datetime.fromisoformat(
            "2026-08-30T00:01:31+08:00")).total_seconds(), 120)
        with open_database(self.database_path) as connection:
            run = get_automated_run(connection, run_id)
            task = get_backtest_task(
                connection,
                list_automated_run_backtests(connection, run_id)[0].task_id,
            )
        self.assertEqual(run.status, "running")
        self.assertEqual(run.current_cycle, 1)
        self.assertEqual(run.request_failure_count, 0)
        self.assertIsNone(run.last_request_failure_code)
        self.assertEqual(task.task.status, "failed")
        self.assertEqual(task.task.failure_code, "platform_pending_timeout")
        self.assertIsNone(task.result)
        self.assertEqual(task.task.remote_id, self._accepted().remote_id)

    def test_unlimited_retry_wait_survives_interruption_and_reentry(self) -> None:
        run_id = self._prepare_running_task(
            max_pending_seconds=60,
            unlimited=True,
            max_in_flight_backtests=1,
        )
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")
        failure = WorldQuantRequestError(
            "worldquant_poll_http_error",
            status_code=429,
            retryable=True,
            outcome_unknown=False,
            retry_after_seconds=180,
        )
        client = RunnerClient(self._accepted(), polls=[failure])

        def interrupt_on_cooldown(seconds):
            if seconds > 1:
                raise KeyboardInterrupt
            fake_time.wait(seconds)

        with self.assertRaises(KeyboardInterrupt):
            run_automated_run(
                self.database_path,
                client,
                run_id,
                clock=fake_time.now,
                waiter=interrupt_on_cooldown,
            )
        with open_database(self.database_path) as connection:
            paused = get_automated_run(connection, run_id)
            pending = get_backtest_task(connection, list_automated_run_backtests(connection, run_id)[0].task_id)
        self.assertEqual(paused.status, "running")
        self.assertEqual(paused.request_failure_count, 0)
        self.assertEqual(pending.task.status, "pending")
        self.assertGreaterEqual(datetime.fromisoformat(pending.task.retry_not_before),
                                fake_time.now() + timedelta(seconds=180))

        with patch(
            "execution.driver.plan_automated_cycle", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                run_automated_run(
                    self.database_path,
                    client,
                    run_id,
                    clock=fake_time.now,
                    waiter=fake_time.wait,
                )
        self.assertLessEqual(sum(fake_time.waits), 60.0)
        self.assertEqual(client.calls.count("submit"), 1)
        self.assertEqual(client.calls.count("poll"), 1)
        with open_database(self.database_path) as connection:
            recovered = get_automated_run(connection, run_id)
            expired = get_backtest_task(connection, pending.task.task_id)
        self.assertEqual(expired.task.failure_code, "platform_pending_timeout")
        self.assertEqual(recovered.current_cycle, 1)
        self.assertEqual(recovered.request_failure_count, 0)

    def test_authenticates_once_and_runs_until_terminal(self) -> None:
        run_id = self._prepare_running_task(max_pending_seconds=60)
        client = RunnerClient(self._accepted())
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(completion.run.current_cycle, 1)
        self.assertEqual(completion.step_count, 6)
        self.assertEqual(completion.platform_request_count, 5)
        self.assertTrue(completion.authentication_performed)
        self.assertEqual(
            client.calls,
            ["authenticate", "submit", "poll", "detail", "yearly_stats", "formal_check"],
        )
        self.assertEqual(fake_time.waits, [1.0, 1.0, 1.0, 1.0, 1.0])

        repeated_client = RunnerClient()
        repeated = run_automated_run(
            self.database_path,
            repeated_client,
            run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )
        self.assertEqual(repeated.run.status, "completed")
        self.assertFalse(repeated.authentication_performed)
        self.assertEqual(repeated_client.calls, [])

    def test_unresolved_response_does_not_block_independent_recovery(self):
        run_id = self._prepare_running_task(
            max_pending_seconds=600,
            formulas=("rank(close)", "rank(open)", "rank(volume)"),
        )
        with open_database(self.database_path) as connection:
            links = list_automated_run_backtests(connection, run_id)
            unknown_id = links[0].task_id
            before = record_submission_unknown(
                connection, unknown_id, observed_at="2026-08-30T00:01:31+08:00"
            )
        fail_automated_run(
            self.database_path, run_id, failed_at="2026-08-30T00:02:00+08:00",
            reason="submission_reconciliation_required",
        )
        client = RunnerClient(self._accepted("simulation-2"), self._accepted("simulation-3"))
        fake_time = FakeTime("2026-08-30T00:04:01+08:00")
        completion = run_automated_run(
            self.database_path, client, run_id, clock=fake_time.now, waiter=fake_time.wait
        )
        self.assertEqual(completion.run.stop_reason, "submission_reconciliation_required")
        self.assertEqual(client.calls.count("submit"), 2)
        self.assertEqual(client.calls.count("alpha_list"), 4)
        self.assertLess(client.calls.index("submit"), client.calls.index("alpha_list"))
        with open_database(self.database_path) as connection:
            after = get_backtest_task(connection, unknown_id)
            completed = connection.execute(
                "SELECT COUNT(*) FROM backtest_tasks WHERE status='completed'"
            ).fetchone()[0]
        self.assertEqual(after, before)
        self.assertEqual(completed, 2)
        repeated = RunnerClient()
        run_automated_run(
            self.database_path, repeated, run_id, clock=fake_time.now, waiter=fake_time.wait
        )
        self.assertEqual(repeated.calls.count("submit"), 0)

    def test_unknown_does_not_hide_non_retryable_failure(self):
        run_id = self._prepare_running_task(
            max_pending_seconds=600,
            formulas=("rank(close)", "rank(open)", "rank(volume)"),
        )
        client = RunnerClient(
            BacktestSubmissionObservation("unknown", None, "missing_location"),
            WorldQuantRequestError("unauthorized", retryable=False, outcome_unknown=False),
        )
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")
        completion = run_automated_run(
            self.database_path, client, run_id, clock=fake_time.now, waiter=fake_time.wait
        )
        self.assertEqual(completion.run.stop_reason, "platform_request_not_retryable:unauthorized")
        self.assertEqual(client.calls.count("submit"), 2)
        self.assertEqual(client.calls.count("alpha_list"), 0)
        with open_database(self.database_path) as connection:
            statuses = [row[0] for row in connection.execute("SELECT status FROM backtest_tasks")]
        self.assertEqual(statuses.count("submission_unknown"), 1)
        self.assertEqual(statuses.count("created"), 0)
        with open_database(self.database_path) as connection:
            unknown = connection.execute(
                "SELECT formula FROM backtest_tasks WHERE status='submission_unknown'"
            ).fetchone()[0]
        reconciliation_client = RunnerClient(alpha_records=(UserAlphaRecord(
            platform_alpha_id="alpha-reconciled", alpha_type="REGULAR",
            status="UNSUBMITTED", formula=unknown,
            settings=self.settings.as_platform_dict(),
            created_at=datetime.fromisoformat("2026-08-30T00:01:31.500000+08:00"),
            hidden=False,
        ),))
        fake_time.advance(90)
        reconciled = run_automated_run(
            self.database_path, reconciliation_client, run_id,
            clock=fake_time.now, waiter=fake_time.wait,
        )
        self.assertEqual(reconciled.run.stop_reason, completion.run.stop_reason)
        self.assertEqual(reconciled.run.status, "failed")
        self.assertEqual(reconciliation_client.calls.count("submit"), 0)
        self.assertEqual(reconciliation_client.calls.count("alpha_list"), 4)
        with open_database(self.database_path) as connection:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM backtest_tasks WHERE status='submission_unknown'"
            ).fetchone()[0], 0)

    def test_all_unknown_slots_stop_without_resending_or_overcommitting(self):
        run_id = self._prepare_running_task(
            max_pending_seconds=600,
            formulas=("rank(close)", "rank(open)", "rank(volume)", "ts_rank(close,5)"),
        )
        client = RunnerClient(*(
            BacktestSubmissionObservation("unknown", None, "missing_location")
            for _ in range(3)
        ))
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")
        completion = run_automated_run(
            self.database_path, client, run_id, clock=fake_time.now, waiter=fake_time.wait
        )
        self.assertEqual(completion.run.stop_reason, "submission_reconciliation_required")
        self.assertEqual(client.calls.count("submit"), 3)
        with open_database(self.database_path) as connection:
            statuses = [row[0] for row in connection.execute("SELECT status FROM backtest_tasks")]
        self.assertEqual(statuses.count("submission_unknown"), 3)
        self.assertEqual(statuses.count("created"), 1)

    def test_confirming_unknown_cannot_resume_a_masked_historical_hard_stop(self):
        run_id = self._prepare_running_task(
            max_pending_seconds=600,
            formulas=("rank(close)", "rank(open)", "rank(volume)"),
        )
        with open_database(self.database_path) as connection:
            first = list_automated_run_backtests(connection, run_id)[0]
            unknown = record_submission_unknown(
                connection, first.task_id, observed_at="2026-08-30T00:01:31+08:00"
            )
        # Persist the old version's observed facts: hard request failure masked by unknown.
        fail_automated_run(
            self.database_path, run_id, failed_at="2026-08-30T00:01:32+08:00",
            reason="submission_reconciliation_required", request_failure_code="unauthorized",
        )
        client = RunnerClient(alpha_records=(UserAlphaRecord(
            platform_alpha_id="alpha-reconciled", alpha_type="REGULAR",
            status="UNSUBMITTED", formula=unknown.task.formula,
            settings=self.settings.as_platform_dict(),
            created_at=datetime.fromisoformat("2026-08-30T00:01:31.500000+08:00"),
            hidden=False,
        ),))
        fake_time = FakeTime("2026-08-30T00:03:01+08:00")
        completion = run_automated_run(
            self.database_path, client, run_id, clock=fake_time.now, waiter=fake_time.wait
        )
        self.assertEqual(completion.run.status, "failed")
        self.assertEqual(client.calls.count("alpha_list"), 4)
        self.assertEqual(client.calls.count("submit"), 0)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, first.task_id).task.status, "failed")
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM backtest_tasks WHERE status='created'"
            ).fetchone()[0], 2)
        from execution.runs import resume_automated_run_after_submission_reconciliation
        with self.assertRaisesRegex(ValueError, "automated_run_reconciliation_resume_blocked"):
            resume_automated_run_after_submission_reconciliation(self.database_path, run_id)

    def test_unknown_does_not_hide_request_failure_limit(self):
        run_id = self._prepare_running_task(
            max_pending_seconds=600,
            formulas=("rank(close)", "rank(open)", "rank(volume)"),
        )
        client = RunnerClient(
            BacktestSubmissionObservation("unknown", None, "missing_location"),
            *(WorldQuantRequestError("rate_limit", retryable=True, outcome_unknown=False)
              for _ in range(3)),
        )
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")
        completion = run_automated_run(
            self.database_path, client, run_id, clock=fake_time.now, waiter=fake_time.wait
        )
        self.assertEqual(completion.run.stop_reason, "request_failure_limit_reached")
        self.assertEqual(client.calls.count("submit"), 4)
        self.assertEqual(client.calls.count("alpha_list"), 0)

    def test_positive_reconciliation_resumes_without_reposting_unknown_task(
        self,
    ) -> None:
        run_id = self._prepare_running_task(
            max_pending_seconds=600,
            formulas=("rank(close)", "rank(open)"),
        )
        with open_database(self.database_path) as connection:
            links = list_automated_run_backtests(connection, run_id)
            unknown_id = next(
                link.task_id
                for link in links
                if (
                    (snapshot := get_backtest_task(connection, link.task_id))
                    is not None
                    and snapshot.task.formula == "rank(close)"
                )
            )
            record_submission_unknown(
                connection,
                unknown_id,
                observed_at="2026-08-30T00:01:31+08:00",
            )
        fail_automated_run(
            self.database_path,
            run_id,
            failed_at="2026-08-30T00:02:00+08:00",
            reason="submission_reconciliation_required",
        )
        client = RunnerClient(
            self._accepted("simulation-2"),
            alpha_records=(
                UserAlphaRecord(
                    platform_alpha_id="alpha-reconciled",
                    alpha_type="REGULAR",
                    status="UNSUBMITTED",
                    formula="rank(close)",
                    settings={
                        **self.settings.as_platform_dict(),
                        "startDate": "2020-01-01",
                        "endDate": "2025-01-01",
                    },
                    created_at=datetime.fromisoformat("2026-08-30T00:01:45+08:00"),
                    hidden=False,
                ),
            ),
        )
        fake_time = FakeTime("2026-08-30T00:04:01+08:00")

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(client.calls.count("alpha_list"), 4)
        self.assertEqual(client.calls.count("submit"), 1)
        with open_database(self.database_path) as connection:
            unknown = get_backtest_task(connection, unknown_id)
        assert unknown is not None
        self.assertEqual(unknown.task.status, "failed")
        self.assertEqual(unknown.task.platform_alpha_id, "alpha-reconciled")
        self.assertEqual(
            unknown.task.failure_code,
            "remote_accepted_without_simulation_id",
        )

    def test_unknown_request_error_reconciles_same_timestamp_without_repost(
        self,
    ) -> None:
        run_id = self._prepare_running_task(
            max_pending_seconds=600,
            formulas=("rank(close)", "rank(open)"),
            max_in_flight_backtests=1,
        )
        first_client = RunnerClient(
            WorldQuantRequestError(
                "lost_backtest_response",
                retryable=True,
                outcome_unknown=True,
            )
        )
        first_time = FakeTime("2026-08-30T00:01:31+08:00")

        first_completion = run_automated_run(
            self.database_path,
            first_client,
            run_id,
            clock=first_time.now,
            waiter=first_time.wait,
        )

        self.assertEqual(first_completion.run.status, "failed")
        self.assertEqual(
            first_completion.run.stop_reason,
            "submission_reconciliation_required",
        )
        self.assertEqual(first_client.calls.count("submit"), 1)
        self.assertEqual(first_client.calls.count("alpha_list"), 0)
        with open_database(self.database_path) as connection:
            links = list_automated_run_backtests(connection, run_id)
            snapshots = tuple(
                get_backtest_task(connection, link.task_id) for link in links
            )
        unknown = next(
            snapshot
            for snapshot in snapshots
            if snapshot is not None and snapshot.task.status == "submission_unknown"
        )
        self.assertEqual(
            unknown.task.submission_started_at,
            first_completion.run.finished_at,
        )

        second_client = RunnerClient(
            self._accepted("simulation-2"),
            alpha_records=(
                UserAlphaRecord(
                    platform_alpha_id="alpha-reconciled",
                    alpha_type="REGULAR",
                    status="ACTIVE",
                    formula=unknown.task.formula,
                    settings={
                        **self.settings.as_platform_dict(),
                        "startDate": "2020-01-01",
                        "endDate": "2025-01-01",
                    },
                    created_at=datetime.fromisoformat("2026-08-30T00:01:45+08:00"),
                    hidden=False,
                ),
            ),
        )
        second_time = FakeTime("2026-08-30T00:04:01+08:00")

        completion = run_automated_run(
            self.database_path,
            second_client,
            run_id,
            clock=second_time.now,
            waiter=second_time.wait,
        )

        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(second_client.calls.count("alpha_list"), 4)
        self.assertEqual(second_client.calls.count("submit"), 1)
        with open_database(self.database_path) as connection:
            reconciled = get_backtest_task(connection, unknown.task.task_id)
        assert reconciled is not None
        self.assertEqual(reconciled.task.status, "failed")
        self.assertEqual(
            reconciled.task.platform_alpha_id,
            "alpha-reconciled",
        )

    def test_enabled_automatic_submission_runs_to_confirmed_completion(self) -> None:
        run_id = self._prepare_running_task(
            max_pending_seconds=60,
            automatic_submissions_enabled=True,
        )
        client = RunnerClient(self._accepted())
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(completion.platform_request_count, 9)
        self.assertEqual(completion.formal_submission_claimed_count, 1)
        self.assertEqual(completion.formal_submission_confirmed_count, 1)
        self.assertEqual(completion.formal_submission_unresolved_count, 0)
        self.assertEqual(client.calls.count("formal_submit"), 1)
        self.assertEqual(client.calls[-1], "formal_confirmation")

    def test_expired_submission_session_reauthenticates_without_resetting_rate_limit_count(self):
        run_id = self._prepare_running_task(max_pending_seconds=60)
        class ExpiringClient(RunnerClient):
            def submit_backtest(self, **kwargs):
                try:
                    return super().submit_backtest(**kwargs)
                except WorldQuantRequestError as exc:
                    if exc.code == "worldquant_authentication_expired":
                        self.authenticated = False
                    raise
        rate_limit = WorldQuantRequestError(
            "worldquant_submission_http_error", status_code=429, retryable=True,
            outcome_unknown=False, retry_after_seconds=1,
        )
        counts_at_login = []
        def observe_login():
            with open_database(self.database_path) as connection:
                counts_at_login.append(get_automated_run(connection, run_id).request_failure_count)
        client = ExpiringClient(
            rate_limit, WorldQuantRequestError("worldquant_authentication_expired",
                status_code=401, retryable=True, outcome_unknown=False),
            rate_limit, self._accepted(), authentication_observer=observe_login,
        )
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")
        completion = run_automated_run(self.database_path, client, run_id,
            clock=fake_time.now, waiter=fake_time.wait)
        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(counts_at_login, [0, 1])
        self.assertEqual(client.calls.count("submit"), 4)
        with open_database(self.database_path) as connection:
            tasks = list_automated_run_backtests(connection, run_id)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(get_backtest_task(connection, tasks[0].task_id).task.status, "completed")

    def test_retry_wait_is_executed_without_losing_the_run(self) -> None:
        run_id = self._prepare_running_task(max_pending_seconds=60)
        client = RunnerClient(
            WorldQuantRequestError(
                "temporary",
                retryable=True,
                outcome_unknown=False,
            ),
            self._accepted(),
        )
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(completion.platform_request_count, 6)
        self.assertEqual(completion.run.request_failure_count, 0)
        self.assertEqual(fake_time.waits[0], 1.0)
        self.assertEqual(client.calls.count("authenticate"), 1)
        self.assertEqual(client.calls.count("submit"), 2)

    def test_pending_deadline_caps_poll_wait_and_fails_the_task(self) -> None:
        run_id = self._prepare_running_task(max_pending_seconds=2)
        client = RunnerClient(self._accepted(), poll_pending=True)
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            poll_interval_seconds=10,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertEqual(completion.run.status, "failed")
        self.assertEqual(completion.run.stop_reason, "consecutive_failures_reached")
        self.assertEqual(completion.platform_request_count, 2)
        self.assertEqual(client.calls, ["authenticate", "submit", "poll"])
        self.assertEqual(fake_time.waits[0], 2.0)
        with open_database(self.database_path) as connection:
            link = list_automated_run_backtests(connection, run_id)[0]
            snapshot = get_backtest_task(connection, link.task_id)
            settlement = get_automated_cycle_settlement(connection, run_id, 1)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.task.status, "failed")
        self.assertEqual(snapshot.task.failure_code, "platform_pending_timeout")
        self.assertIsNotNone(settlement)
        self.assertEqual(completion.run.current_cycle, 1)

    def test_slow_poll_gets_a_final_read_with_positive_poll_wait(self) -> None:
        run_id = self._prepare_running_task(max_pending_seconds=5)
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")
        client = RunnerClient(
            self._accepted(),
            poll_pending=True,
            poll_observer=lambda: fake_time.advance(4),
        )

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertEqual(completion.run.status, "failed")
        self.assertEqual(client.calls, ["authenticate", "submit", "poll", "poll"])
        self.assertEqual(fake_time.waits, [1.0, 1.0, 1.0])
        with open_database(self.database_path) as connection:
            link = list_automated_run_backtests(connection, run_id)[0]
            snapshot = get_backtest_task(connection, link.task_id)
        assert snapshot is not None
        self.assertIsNotNone(snapshot.task.finished_at)

    def test_pending_timeout_continues_unsent_tasks(
        self,
    ) -> None:
        run_id = self._prepare_running_task(
            max_pending_seconds=5,
            formulas=("rank(close)", "rank(open)", "rank(volume)"),
            max_in_flight_backtests=2,
        )
        client = RunnerClient(
            self._accepted("simulation-1"),
            self._accepted("simulation-2"),
            self._accepted("simulation-3"),
            poll_pending=True,
        )
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertEqual(completion.run.status, "failed")
        self.assertEqual(completion.run.stop_reason, "consecutive_failures_reached")
        self.assertEqual(client.calls.count("submit"), 3)
        with open_database(self.database_path) as connection:
            links = list_automated_run_backtests(connection, run_id)
            snapshots = tuple(
                get_backtest_task(connection, link.task_id) for link in links
            )
        self.assertTrue(all(snapshot is not None for snapshot in snapshots))
        self.assertEqual(
            sorted(snapshot.task.failure_code for snapshot in snapshots if snapshot),
            [
                "platform_pending_timeout",
                "platform_pending_timeout",
                "platform_pending_timeout",
            ],
        )

    def test_unlimited_unknown_requests_reconcile_or_expire_without_manual_resume(self):
        run_id = self._prepare_running_task(
            max_pending_seconds=60, unlimited=True,
            formulas=("rank(close)", "rank(open)", "rank(volume)", "rank(high)"),
        )
        client = RunnerClient(
            *(BacktestSubmissionObservation("unknown", None, None) for _ in range(3)),
            self._accepted("fourth"),
        )
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")
        original_wait = fake_time.wait
        def bounded_wait(seconds):
            original_wait(seconds)
            self.assertLess((fake_time.now() - datetime.fromisoformat(
                "2026-08-30T00:01:31+08:00")).total_seconds(), 240)
        with patch("execution.driver.plan_automated_cycle", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_automated_run(self.database_path, client, run_id,
                    clock=fake_time.now, waiter=bounded_wait)
        self.assertEqual(client.calls.count("submit"), 4)
        with open_database(self.database_path) as connection:
            run = get_automated_run(connection, run_id)
            tasks = [get_backtest_task(connection, link.task_id)
                     for link in list_automated_run_backtests(connection, run_id)]
        self.assertEqual(run.status, "running")
        self.assertEqual(run.current_cycle, 1)
        self.assertEqual(sum(item.task.status == "completed" for item in tasks), 1)
        self.assertEqual(sum(item.task.failure_code == "submission_outcome_timeout"
                             for item in tasks), 3)


    def test_expired_pending_authenticates_and_recovers_remote_result(self) -> None:
        run_id = self._prepare_running_task(max_pending_seconds=30)
        with open_database(self.database_path) as connection:
            link = list_automated_run_backtests(connection, run_id)[0]
            record_submission_accepted(
                connection,
                link.task_id,
                remote_id=("https://api.worldquantbrain.com/simulations/simulation-1"),
                observed_at="2026-08-30T00:01:31+08:00",
            )
        states_at_authentication: list[str] = []

        def observe_authentication() -> None:
            with open_database(self.database_path) as connection:
                snapshot = get_backtest_task(connection, link.task_id)
            assert snapshot is not None
            states_at_authentication.append(snapshot.task.status)

        client = RunnerClient(authentication_observer=observe_authentication)
        fake_time = FakeTime("2026-08-30T00:02:01+08:00")

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(completion.run.stop_reason, "max_cycles_reached")
        self.assertTrue(completion.authentication_performed)
        self.assertGreaterEqual(completion.platform_request_count, 3)
        self.assertNotIn("submit", client.calls)
        self.assertEqual(states_at_authentication, ["pending"])

    def test_authentication_retry_uses_the_persisted_failure_boundary(self) -> None:
        run_id = self._prepare_running_task(max_pending_seconds=60)
        client = RunnerClient(
            self._accepted(),
            authentications=(
                WorldQuantRequestError(
                    "temporary_authentication_failure",
                    retryable=True,
                    outcome_unknown=False,
                    retry_after_seconds=2,
                ),
            ),
        )
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(completion.run.request_failure_count, 0)
        self.assertTrue(completion.authentication_performed)
        self.assertEqual(client.calls.count("authenticate"), 2)
        self.assertEqual(client.calls.count("submit"), 1)
        self.assertEqual(fake_time.waits, [2, 1.0, 1.0, 1.0, 1.0, 1.0])

    def test_authentication_failure_limit_stops_without_submitting(self) -> None:
        run_id = self._prepare_running_task(max_pending_seconds=60)
        failure = WorldQuantRequestError(
            "temporary_authentication_failure",
            retryable=True,
            outcome_unknown=False,
        )
        client = RunnerClient(
            self._accepted(),
            authentications=(failure, failure, failure),
        )
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertEqual(completion.run.status, "failed")
        self.assertEqual(
            completion.run.stop_reason,
            "request_failure_limit_reached",
        )
        self.assertFalse(completion.authentication_performed)
        self.assertEqual(
            client.calls,
            ["authenticate", "authenticate", "authenticate"],
        )
        self.assertEqual(completion.platform_request_count, 0)

    def test_authentication_failure_preserves_pending_until_explicit_resume(
        self,
    ) -> None:
        run_id = self._prepare_running_task(max_pending_seconds=5)
        with open_database(self.database_path) as connection:
            link = list_automated_run_backtests(connection, run_id)[0]
            record_submission_accepted(
                connection,
                link.task_id,
                remote_id=("https://api.worldquantbrain.com/simulations/simulation-1"),
                observed_at="2026-08-30T00:01:31+08:00",
            )
        failure = WorldQuantRequestError(
            "temporary_authentication_failure",
            retryable=True,
            outcome_unknown=False,
        )
        client = RunnerClient(authentications=(failure, failure, failure))
        fake_time = FakeTime("2026-08-30T00:01:32+08:00")

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertEqual(completion.run.status, "failed")
        self.assertEqual(
            completion.run.stop_reason,
            "request_failure_limit_reached",
        )
        self.assertEqual(client.calls, ["authenticate", "authenticate", "authenticate"])
        self.assertEqual(fake_time.waits, [1.0, 2.0])
        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, link.task_id)
        assert snapshot is not None
        self.assertEqual(snapshot.task.status, "pending")
        self.assertIsNone(snapshot.task.failure_code)

    def test_request_failure_stop_preserves_unknown_remote_result(
        self,
    ) -> None:
        run_id = self._prepare_running_task(
            max_pending_seconds=5,
            formulas=("rank(close)", "rank(open)"),
            max_in_flight_backtests=1,
        )
        failure = WorldQuantRequestError(
            "temporary_poll_failure",
            retryable=True,
            outcome_unknown=False,
        )
        client = RunnerClient(
            self._accepted(),
            polls=(failure, failure, failure),
        )
        fake_time = FakeTime("2026-08-30T00:01:31+08:00")

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertEqual(completion.run.status, "failed")
        self.assertEqual(
            completion.run.stop_reason,
            "request_failure_limit_reached",
        )
        self.assertEqual(completion.platform_request_count, 4)
        self.assertEqual(fake_time.waits, [1.0, 1.0, 2.0])
        with open_database(self.database_path) as connection:
            links = list_automated_run_backtests(connection, run_id)
            snapshots = tuple(
                get_backtest_task(connection, link.task_id) for link in links
            )
            settlement = get_automated_cycle_settlement(connection, run_id, 1)
        self.assertTrue(all(snapshot is not None for snapshot in snapshots))
        self.assertEqual(
            {snapshot.task.failure_code for snapshot in snapshots if snapshot},
            {
                "automated_run_stopped_before_submission",
                None,
            },
        )
        self.assertIsNone(settlement)
        self.assertEqual(completion.run.current_cycle, 0)

    def _prepare_running_task(
        self,
        *,
        max_pending_seconds: int,
        formulas: tuple[str, ...] = ("rank(close)",),
        max_in_flight_backtests: int = 3,
        automatic_submissions_enabled: bool = False,
        unlimited: bool = False,
    ) -> str:
        run = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=5,
                backtest_count=len(formulas),
                max_cycles=-1 if unlimited else 1,
                max_backtests=0 if unlimited else len(formulas),
                max_pending_seconds=max_pending_seconds,
                max_consecutive_failures=1,
                max_request_failures=3,
                max_in_flight_backtests=max_in_flight_backtests,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
                automatic_submissions_enabled=automatic_submissions_enabled,
            ),
            created_at="2026-08-30T00:00:00+08:00",
        )
        start_automated_run(
            self.database_path,
            run.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        prepare_automated_candidate_backtest_batch(
            self.database_path,
            run_id=run.run_id,
            candidates=tuple(
                AutomatedCandidateBacktest(self._candidate(formula), self.settings)
                for formula in formulas
            ),
            created_at="2026-08-30T00:01:30+08:00",
        )
        return run.run_id

    @staticmethod
    def _candidate(formula: str) -> FormulaCandidate:
        return exploration_candidate(parse_formula(formula).expression)

    @staticmethod
    def _accepted(
        remote_name: str = "simulation-1",
    ) -> BacktestSubmissionObservation:
        return BacktestSubmissionObservation(
            "accepted",
            f"https://api.worldquantbrain.com/simulations/{remote_name}",
            None,
        )


if __name__ == "__main__":
    unittest.main()
