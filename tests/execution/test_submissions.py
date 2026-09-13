import unittest
from datetime import datetime, timedelta

from execution.driver import advance_automated_run
from execution.runner import run_automated_run
from persistence.database import open_database
from persistence.submissions import list_formal_submission_attempts
from tests.execution import test_driver as fixture
from tests.execution.test_runner import FakeTime
from worldquant.client import WorldQuantRequestError
from worldquant.submissions import AlphaDetailObservation, FormalCheckObservation


class AutomatedSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.AutomatedRunDriverTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.db = self.fixture.database_path

    def test_unavailable_pre_submission_detail_does_not_hold_the_queue(self):
        self._assert_pre_submission_wait_is_bounded("detail")

    def test_pending_check_releases_queue_without_breaking_retry_after(self):
        self._assert_pre_submission_wait_is_bounded("check")

    def test_missing_alpha_does_not_block_independent_automatic_submission(self):
        self._assert_pre_submission_wait_is_bounded("missing_detail")

    def test_lower_grade_does_not_block_independent_automatic_submission(self):
        self._assert_pre_submission_wait_is_bounded("grade")

    def test_rejected_login_releases_post_claim_then_renews_before_sending(self):
        run_id = self.fixture._prepare_run(max_pending_seconds=60,
                                          automatic_submissions_enabled=True)
        class ExpiringClient(fixture.DriverClient):
            authenticated = False
            rejected_once = False

            def authenticate(self):
                self.calls.append("authenticate")
                self.authenticated = True

            def submit_formal_alpha(self, *, platform_alpha_id):
                if not self.rejected_once:
                    self.rejected_once = True
                    self.authenticated = False
                    self.calls.append("rejected_login")
                    raise WorldQuantRequestError("worldquant_authentication_expired",
                        status_code=401, retryable=True, outcome_unknown=False)
                return super().submit_formal_alpha(platform_alpha_id=platform_alpha_id)
        client = ExpiringClient(self.fixture._accepted())
        self.fixture._advance(client, minute=1)
        self.fixture._prepare_backtests(run_id, ("rank(close)",),
                                       created_at="2026-08-30T00:01:30+08:00")
        time = FakeTime("2026-08-30T00:01:31+08:00")
        result = run_automated_run(self.db, client, run_id, clock=time.now, waiter=time.wait)
        self.assertEqual(result.run.status, "completed")
        self.assertEqual(client.calls.count("authenticate"), 2)
        self.assertEqual(client.calls.count("rejected_login"), 1)
        self.assertEqual(len(client.formal_submission_ids), 1)
        with open_database(self.db) as connection:
            attempts = list_formal_submission_attempts(connection, run_id=run_id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].status, "submitted")
        report = self.db.with_name("submitted-formulas.md").read_text(encoding="utf-8")
        self.assertIn("共 **1** 条", report)
        self.assertIn(f"| {client.formal_submission_ids[0]} |", report)

    def _assert_pre_submission_wait_is_bounded(self, blocked_phase):
        run_id = self.fixture._prepare_run(
            max_cycles=-1, max_backtests=0, backtest_count=2,
            max_pending_seconds=20, automatic_submissions_enabled=True,
        )
        class BlockingClient(fixture.DriverClient):
            blocked_alpha = None
            blocked_check_reads = 0

            def fetch_alpha_detail(self, *, platform_alpha_id):
                if platform_alpha_id == self.blocked_alpha and blocked_phase == "missing_detail":
                    raise WorldQuantRequestError(
                        "worldquant_alpha_detail_http_error", status_code=404,
                        retryable=False, outcome_unknown=False,
                    )
                if platform_alpha_id == self.blocked_alpha and blocked_phase == "detail":
                    raise WorldQuantRequestError(
                        "worldquant_alpha_detail_http_error", status_code=503,
                        retryable=True, outcome_unknown=False,
                    )
                if platform_alpha_id == self.blocked_alpha and blocked_phase == "grade":
                    observation = super().fetch_alpha_detail(platform_alpha_id=platform_alpha_id)
                    return AlphaDetailObservation(dict(observation.payload, grade="EXCELLENT"))
                return super().fetch_alpha_detail(platform_alpha_id=platform_alpha_id)

            def fetch_formal_submission_check(self, *, platform_alpha_id):
                if platform_alpha_id == self.blocked_alpha and blocked_phase == "check":
                    self.blocked_check_reads += 1
                    return FormalCheckObservation({}, 600)
                return super().fetch_formal_submission_check(platform_alpha_id=platform_alpha_id)

        client = BlockingClient(self.fixture._accepted("one"), self.fixture._accepted("two"))
        self.fixture._advance(client, minute=1)
        self.fixture._prepare_backtests(
            run_id, ("rank(close)", "rank(open)"), created_at="2026-08-30T00:01:30+08:00",
        )
        observed = datetime.fromisoformat("2026-08-30T00:01:31+08:00")
        blocked_at = None
        for _ in range(120):
            advanced = advance_automated_run(self.db, client, run_id, observed_at=observed.isoformat())
            self.assertEqual(advanced.run.status, "running")
            with open_database(self.db) as connection:
                attempts = list_formal_submission_attempts(connection, run_id=run_id)
                if attempts and client.blocked_alpha is None:
                    first = attempts[0]
                    client.blocked_alpha = connection.execute(
                        "SELECT platform_alpha_id FROM backtest_tasks WHERE task_id=?", (first.task_id,),
                    ).fetchone()[0]
                    blocked_at = observed
            if len(attempts) == 2 and any(item.status == "submitted" for item in attempts):
                break
            if blocked_at is not None:
                self.assertLess((observed - blocked_at).total_seconds(), 50,
                                "one candidate blocked the independent submission queue")
            observed += timedelta(seconds=max(1, advanced.retry_after_seconds or 0))
        self.assertEqual(len(attempts), 2)
        expired = next(item for item in attempts if item.task_id == first.task_id)
        self.assertEqual(expired.status, "ineligible" if blocked_phase == "grade" else "failed")
        self.assertEqual(expired.failure_code,
                         "formal_submission_grade_below_target:EXCELLENT" if blocked_phase == "grade" else
                         "formal_submission_read_failed:worldquant_alpha_detail_http_error"
                         if blocked_phase == "missing_detail" else "formal_submission_check_timeout")
        self.assertIsNone(expired.submission_claimed_at)
        self.assertEqual(len(client.formal_submission_ids), 1)
        self.assertNotIn(client.blocked_alpha, client.formal_submission_ids)
        if blocked_phase == "check":
            self.assertEqual(client.blocked_check_reads, 1)
