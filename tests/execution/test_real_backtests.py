from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from unittest.mock import patch
from urllib.request import Request

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.real_backtests import advance_real_backtest, prepare_real_backtest, expire_real_backtest_if_pending_timeout
from execution.backtests import (
    record_pending_observation,
    record_submission_accepted,
    record_submission_unknown,
)
from persistence.backtests import (
    get_backtest_task,
    initialize_backtest_schema,
    list_completed_backtests,
    list_terminal_backtests,
)
from persistence.database import open_database
from persistence.runs import initialize_run_schema
from persistence.submissions import initialize_submission_schema
from worldquant.backtests import BacktestSettings, WorldQuantProtocolError
from worldquant.client import (
    WorldQuantClient,
    WorldQuantCredentials,
    WorldQuantRequestError,
    WorldQuantResponse,
)


class FakeExecutor:
    def __init__(
        self,
        *responses: WorldQuantResponse | Exception,
        observer=None,
    ) -> None:
        self.responses = list(responses)
        self.observer = observer
        self.requests: list[Request] = []

    def __call__(self, request: Request, timeout: float) -> WorldQuantResponse:
        self.requests.append(request)
        if self.observer is not None:
            self.observer(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RealBacktestExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "real.sqlite3"
        self.formula = "rank(close)"
        with open_database(self.database_path) as connection:
            initialize_backtest_schema(connection)
            initialize_run_schema(connection)
            initialize_submission_schema(connection)
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

    def test_prepare_commits_intent_without_platform_request(self) -> None:
        prepared = self._prepare()

        with open_database(self.database_path) as connection:
            persisted = get_backtest_task(connection, prepared.task.task_id)

        self.assertEqual(prepared.task.status, "created")
        self.assertEqual(persisted, prepared)

    def test_created_task_requires_explicit_submission_confirmation(self) -> None:
        prepared = self._prepare()
        executor = FakeExecutor()
        client = self._client(executor)

        blocked = advance_real_backtest(
            self.database_path,
            client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:02:00+00:00",
        )

        self.assertEqual(blocked.action, "submission_confirmation_required")
        self.assertFalse(blocked.platform_request_performed)
        self.assertEqual(blocked.snapshot.task.status, "created")
        self.assertEqual(executor.requests, [])

    def test_explicit_submission_requires_authentication_before_guarding_task(
        self,
    ) -> None:
        prepared = self._prepare()
        executor = FakeExecutor()
        client = self._client(executor)

        with self.assertRaisesRegex(
            RuntimeError,
            "worldquant_authentication_required",
        ):
            advance_real_backtest(
                self.database_path,
                client,
                prepared.task.task_id,
                observed_at="2026-08-29T00:02:00+00:00",
                allow_submission=True,
            )

        with open_database(self.database_path) as connection:
            persisted = get_backtest_task(connection, prepared.task.task_id)
        self.assertEqual(persisted.task.status, "created")
        self.assertEqual(executor.requests, [])

    def test_expired_login_during_submission_preserves_the_guarded_task(self):
        prepared = self._prepare()
        remote = "https://api.worldquantbrain.com/simulations/refreshed"
        executor = FakeExecutor(
            self._response(201, {}), self._response(401, {}),
            self._response(201, {}), self._response(201, {}, headers={"Location": remote}),
        )
        client = self._client(executor)
        client.authenticate()
        with self.assertRaises(WorldQuantRequestError) as expired:
            advance_real_backtest(
                self.database_path, client, prepared.task.task_id,
                observed_at="2026-08-29T00:02:00+00:00", allow_submission=True,
            )
        self.assertEqual(expired.exception.code, "worldquant_authentication_expired")
        with open_database(self.database_path) as connection:
            unclaimed = get_backtest_task(connection, prepared.task.task_id)
        self.assertEqual(unclaimed.task.status, "created")
        self.assertIsNone(unclaimed.task.submission_started_at)
        client.authenticate()
        advanced = advance_real_backtest(
            self.database_path, client, prepared.task.task_id,
            observed_at="2026-08-29T00:02:01+00:00", allow_submission=True,
        )
        self.assertEqual(advanced.snapshot.task.status, "pending")
        self.assertEqual(advanced.snapshot.task.task_id, prepared.task.task_id)
        self.assertEqual(advanced.snapshot.task.remote_id, remote)
        self.assertIsNotNone(advanced.snapshot.task.submission_started_at)
        self.assertIsNone(advanced.snapshot.result)
        self.assertEqual(len(executor.requests), 4)
        self.assertEqual([r.full_url for r in executor.requests].count(
            "https://api.worldquantbrain.com/simulations"), 2)

    def test_submission_rejects_timeout_longer_than_claim_window(self) -> None:
        prepared = self._prepare()
        executor = FakeExecutor(self._response(201, {}))
        client = self._client(executor, timeout_seconds=31.0)
        client.authenticate()

        with self.assertRaisesRegex(
            ValueError,
            "real_backtest_submission_timeout_unsupported",
        ):
            advance_real_backtest(
                self.database_path,
                client,
                prepared.task.task_id,
                observed_at="2026-08-29T00:02:00+00:00",
                allow_submission=True,
            )

        with open_database(self.database_path) as connection:
            persisted = get_backtest_task(connection, prepared.task.task_id)
        self.assertEqual(persisted.task.status, "created")
        self.assertEqual(len(executor.requests), 1)

    def test_explicit_advances_survive_restart_and_store_one_real_result(self) -> None:
        prepared = self._prepare()

        def assert_guard_committed(request: Request) -> None:
            if not request.full_url.endswith("/simulations"):
                return
            with open_database(self.database_path) as connection:
                snapshot = get_backtest_task(connection, prepared.task.task_id)
            self.assertEqual(snapshot.task.status, "submission_unknown")

        submit_client, submit_executor = self._authenticated_client(
            self._response(
                201,
                {},
                headers={"Location": "/simulations/simulation-1"},
            ),
            observer=assert_guard_committed,
        )
        submitted = advance_real_backtest(
            self.database_path,
            submit_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:02:00+00:00",
            allow_submission=True,
        )

        poll_client, _ = self._authenticated_client(
            self._response(200, {"progress": 0.35})
        )
        pending = advance_real_backtest(
            self.database_path,
            poll_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:03:00+00:00",
        )

        completed_poll_client, _ = self._authenticated_client(
            self._response(200, {"status": "COMPLETE", "alpha": "alpha-1"})
        )
        detail_ready = advance_real_backtest(
            self.database_path,
            completed_poll_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:04:00+00:00",
        )

        detail_client, _ = self._authenticated_client(
            self._response(200, self._detail_payload())
        )
        detail_captured = advance_real_backtest(
            self.database_path,
            detail_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:04:30+00:00",
        )

        yearly_client, _ = self._authenticated_client(
            self._response(200, self._yearly_payload())
        )
        completed = advance_real_backtest(
            self.database_path,
            yearly_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:05:00+00:00",
        )

        with open_database(self.database_path) as connection:
            task_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_tasks"
            ).fetchone()[0]
            result_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_results"
            ).fetchone()[0]
        self.assertEqual(submitted.action, "submitted")
        self.assertEqual(pending.action, "pending")
        self.assertEqual(detail_ready.action, "detail_ready")
        self.assertEqual(detail_captured.action, "detail_captured")
        self.assertEqual(completed.action, "completed")
        self.assertEqual(completed.snapshot.result.sharpe, 1.3)
        self.assertEqual(completed.snapshot.result.long_count, 740)
        self.assertEqual(completed.snapshot.result.short_count, 700)
        self.assertEqual((task_count, result_count), (1, 1))
        self.assertEqual(len(completed.snapshot.yearly_stats), 1)
        self.assertEqual(completed.snapshot.yearly_stats[0].year, 2024)
        self.assertEqual(completed.snapshot.yearly_stats[0].long_count, 731)
        self.assertEqual(completed.snapshot.yearly_stats[0].short_count, 694)
        self.assertEqual(completed.snapshot.yearly_stats[0].stage, "IS")
        self.assertEqual(len(submit_executor.requests), 2)

    def test_unknown_submit_outcome_blocks_second_submission(self) -> None:
        prepared = self._prepare()
        client, _ = self._authenticated_client(OSError("connection lost"))

        with self.assertRaises(WorldQuantRequestError) as raised:
            advance_real_backtest(
                self.database_path,
                client,
                prepared.task.task_id,
                observed_at="2026-08-29T00:02:00+00:00",
                allow_submission=True,
            )

        blocked_executor = FakeExecutor()
        blocked_client = self._client(blocked_executor)
        blocked = advance_real_backtest(
            self.database_path,
            blocked_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:04:01+00:00",
        )
        self.assertTrue(raised.exception.outcome_unknown)
        self.assertEqual(blocked.action, "reconciliation_required")
        self.assertEqual(blocked.snapshot.task.status, "submission_unknown")
        self.assertEqual(blocked_executor.requests, [])

    def test_brief_lock_before_or_after_post_waits_without_resending(self) -> None:
        for lock_after_post in (False, True):
            with self.subTest(lock_after_post=lock_after_post):
                self.formula = "rank(open)" if lock_after_post else "rank(close)"
                prepared = self._prepare()
                reader = sqlite3.connect(self.database_path.as_uri() + "?mode=ro",
                                         uri=True, check_same_thread=False)
                locked = Event()

                def hold_reader():
                    reader.execute("BEGIN")
                    reader.execute("SELECT * FROM backtest_tasks").fetchall()
                    locked.set()

                def observe(request):
                    if lock_after_post and request.full_url.endswith("/simulations"):
                        hold_reader()

                client, executor = self._authenticated_client(
                    self._response(201, {}, headers={"Location": f"/simulations/lock-test-{lock_after_post}"}),
                    observer=observe,
                )
                if not lock_after_post:
                    hold_reader()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        advance_real_backtest, self.database_path, client,
                        prepared.task.task_id, observed_at="2026-08-29T00:02:00+00:00",
                        allow_submission=True,
                    )
                    try:
                        self.assertTrue(locked.wait(timeout=5))
                        with self.assertRaises(TimeoutError):
                            future.result(timeout=6)
                        posts = [r for r in executor.requests if r.full_url.endswith("/simulations")]
                        self.assertEqual(len(posts), int(lock_after_post))
                    finally:
                        reader.close()
                    advanced = future.result(timeout=5)
                self.assertEqual(advanced.action, "submitted")
                with open_database(self.database_path) as connection:
                    persisted = get_backtest_task(connection, prepared.task.task_id)
                self.assertEqual(persisted.task.status, "pending")
                self.assertTrue(persisted.task.remote_id.endswith(f"/simulations/lock-test-{lock_after_post}"))
                self.assertEqual(len([r for r in executor.requests if r.full_url.endswith("/simulations")]), 1)

    def test_lock_wait_exhaustion_before_or_after_post_preserves_send_guard(self) -> None:
        @contextmanager
        def short_wait_database(path):
            with open_database(path) as connection:
                connection.execute("PRAGMA busy_timeout=50")
                yield connection

        for lock_after_post in (False, True):
            with self.subTest(lock_after_post=lock_after_post):
                self.formula = "rank(open)" if lock_after_post else "rank(close)"
                prepared = self._prepare()
                reader = sqlite3.connect(self.database_path.as_uri() + "?mode=ro", uri=True)

                def hold_reader():
                    reader.execute("BEGIN")
                    reader.execute("SELECT * FROM backtest_tasks").fetchall()

                def observe(request):
                    if lock_after_post and request.full_url.endswith("/simulations"):
                        hold_reader()

                client, executor = self._authenticated_client(
                    self._response(201, {}, headers={"Location": "/simulations/lock-test"}),
                    observer=observe,
                )
                if not lock_after_post:
                    hold_reader()
                try:
                    with patch("execution.real_backtests.open_database", short_wait_database):
                        with self.assertRaises(sqlite3.OperationalError) as raised:
                            advance_real_backtest(
                                self.database_path, client, prepared.task.task_id,
                                observed_at="2026-08-29T00:02:00+00:00", allow_submission=True,
                            )
                    self.assertEqual(raised.exception.sqlite_errorcode, sqlite3.SQLITE_BUSY)
                finally:
                    reader.close()
                with open_database(self.database_path) as connection:
                    persisted = get_backtest_task(connection, prepared.task.task_id)
                self.assertEqual(persisted.task.status, "submission_unknown" if lock_after_post else "created")
                self.assertEqual(len([r for r in executor.requests if r.full_url.endswith("/simulations")]), int(lock_after_post))
                if lock_after_post:
                    blocked_executor = FakeExecutor()
                    blocked = advance_real_backtest(
                        self.database_path, self._client(blocked_executor), prepared.task.task_id,
                        observed_at="2026-08-29T00:04:01+00:00", allow_submission=True,
                    )
                    self.assertEqual(blocked.action, "reconciliation_required")
                    self.assertEqual(blocked_executor.requests, [])

    def test_detail_identity_mismatch_does_not_capture_yearly_stats(self) -> None:
        prepared = self._prepare()
        submit_client, _ = self._authenticated_client(
            self._response(
                201,
                {},
                headers={"Location": "/simulations/simulation-1"},
            )
        )
        advance_real_backtest(
            self.database_path,
            submit_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:02:00+00:00",
            allow_submission=True,
        )
        poll_client, _ = self._authenticated_client(
            self._response(200, {"status": "COMPLETE", "alpha": "alpha-1"})
        )
        advance_real_backtest(
            self.database_path,
            poll_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:03:00+00:00",
        )
        payload = self._detail_payload()
        payload["settings"]["maxPosition"] = "ON"
        settings_client, _ = self._authenticated_client(
            self._response(200, payload)
        )
        with self.assertRaises(WorldQuantProtocolError):
            advance_real_backtest(
                self.database_path,
                settings_client,
                prepared.task.task_id,
                observed_at="2026-08-29T00:04:30+00:00",
            )

        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, prepared.task.task_id)
        self.assertEqual(snapshot.task.status, "pending")
        self.assertIsNone(snapshot.result)
        self.assertIsNone(snapshot.yearly_stats)

    def test_invalid_task_response_exhaustion_fails_locally_without_reposting(self):
        for stage in ("poll", "detail"):
            with self.subTest(stage=stage):
                self.formula = "rank(close)" if stage == "detail" else "rank(open)"
                prepared = self._prepare()
                with open_database(self.database_path) as connection:
                    record_submission_accepted(
                        connection, prepared.task.task_id,
                        remote_id=f"https://api.worldquantbrain.com/simulations/{stage}",
                        platform_alpha_id="alpha-1" if stage == "detail" else None,
                        observed_at="2026-08-29T00:02:00+00:00",
                    )
                payload = self._detail_payload()
                payload["regular"]["code"] = "rank(high)"
                bad = self._response(200, payload if stage == "detail" else {"status": "UNEXPECTED"})
                executor = FakeExecutor(self._response(201, {}), bad, bad, bad)
                client = self._client(executor)
                client.authenticate()
                failed = advance_real_backtest(
                    self.database_path, client, prepared.task.task_id,
                    observed_at="2026-08-29T00:03:00+00:00",
                )
                self.assertEqual(failed.action, "failed")
                self.assertEqual(failed.snapshot.task.failure_code, "platform_response_retry_exhausted")
                self.assertIn("3", failed.snapshot.task.failure_message)
                self.assertIsNone(failed.snapshot.result)
                self.assertIsNone(failed.snapshot.yearly_stats)
                self.assertEqual([r.get_method() for r in executor.requests[1:]], ["GET"] * 3)
                # A fresh client cannot restart or resubmit this terminal task.
                again = advance_real_backtest(
                    self.database_path, self._client(FakeExecutor()), prepared.task.task_id,
                    observed_at="2026-08-29T00:04:00+00:00", allow_submission=True,
                )
                self.assertEqual(again.action, "already_failed")

    def test_invalid_poll_can_recover_before_attempt_limit(self):
        prepared = self._prepare()
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection, prepared.task.task_id,
                remote_id="https://api.worldquantbrain.com/simulations/one",
                observed_at="2026-08-29T00:02:00+00:00",
            )
        executor = FakeExecutor(
            self._response(201, {}),
            self._response(200, {"status": "UNEXPECTED"}),
            self._response(200, {"status": "COMPLETE", "alpha": "alpha-1"}),
        )
        client = self._client(executor)
        client.authenticate()
        recovered = advance_real_backtest(
            self.database_path, client, prepared.task.task_id,
            observed_at="2026-08-29T00:03:00+00:00",
        )
        self.assertEqual(recovered.action, "detail_ready")
        self.assertIsNone(recovered.snapshot.task.failure_code)
        self.assertEqual(len(executor.requests), 3)

    def test_rate_limit_after_invalid_response_keeps_existing_retry_semantics(self):
        prepared = self._prepare()
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection, prepared.task.task_id,
                remote_id="https://api.worldquantbrain.com/simulations/one",
                observed_at="2026-08-29T00:02:00+00:00",
            )
        executor = FakeExecutor(
            self._response(201, {}),
            self._response(200, {"status": "UNEXPECTED"}),
            self._response(429, {}, headers={"Retry-After": "15"}),
        )
        client = self._client(executor)
        client.authenticate()
        with self.assertRaises(WorldQuantRequestError) as caught:
            advance_real_backtest(
                self.database_path, client, prepared.task.task_id,
                observed_at="2026-08-29T00:03:00+00:00",
            )
        self.assertEqual(caught.exception.retry_after_seconds, 15)
        with open_database(self.database_path) as connection:
            saved = get_backtest_task(connection, prepared.task.task_id)
        self.assertEqual(saved.task.status, "pending")
        self.assertIsNone(saved.task.failure_code)
        self.assertEqual(len(executor.requests), 3)

    def test_unavailable_required_detail_metric_fails_only_that_task(self) -> None:
        prepared = self._prepare()
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection,
                prepared.task.task_id,
                remote_id=("https://api.worldquantbrain.com/simulations/simulation-1"),
                observed_at="2026-08-29T00:02:00+00:00",
            )
            record_pending_observation(
                connection,
                prepared.task.task_id,
                observed_at="2026-08-29T00:03:00+00:00",
                platform_alpha_id="alpha-1",
            )
        payload = self._detail_payload()
        payload["is"]["fitness"] = None
        detail_client, _ = self._authenticated_client(
            self._response(200, payload)
        )
        failed = advance_real_backtest(
            self.database_path,
            detail_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:03:30+00:00",
        )

        self.assertEqual(failed.action, "failed")
        self.assertEqual(failed.snapshot.task.status, "failed")
        self.assertEqual(
            failed.snapshot.task.failure_code,
            "platform_detail_metric_unavailable",
        )
        self.assertEqual(failed.snapshot.task.platform_alpha_id, "alpha-1")
        self.assertIsNone(failed.snapshot.result)

    def test_partial_passed_check_set_remains_pending_until_complete(self) -> None:
        prepared = self._prepare()
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection,
                prepared.task.task_id,
                remote_id=("https://api.worldquantbrain.com/simulations/simulation-1"),
                observed_at="2026-08-29T00:02:00+00:00",
            )
            record_pending_observation(
                connection,
                prepared.task.task_id,
                observed_at="2026-08-29T00:03:00+00:00",
                platform_alpha_id="alpha-1",
            )
        partial_payload = self._detail_payload()
        partial_payload["is"]["checks"] = [
            {"name": "LOW_SHARPE", "result": "PASS"}
        ]
        partial_client, _ = self._authenticated_client(
            self._response(200, partial_payload)
        )

        pending = advance_real_backtest(
            self.database_path,
            partial_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:04:00+00:00",
        )

        self.assertEqual(pending.action, "checks_pending")
        self.assertEqual(pending.snapshot.task.status, "pending")
        self.assertEqual(
            pending.snapshot.task.last_observed_at,
            "2026-08-29T00:04:00+00:00",
        )
        self.assertIsNone(pending.snapshot.result)

        complete_client, _ = self._authenticated_client(
            self._response(200, self._detail_payload())
        )
        completed = advance_real_backtest(
            self.database_path,
            complete_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:05:00+00:00",
        )

        yearly_client, _ = self._authenticated_client(
            self._response(200, self._yearly_payload())
        )
        completed = advance_real_backtest(
            self.database_path,
            yearly_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:05:30+00:00",
        )

        self.assertEqual(completed.action, "completed")
        self.assertEqual(completed.snapshot.task.status, "completed")

    def test_partial_failed_check_set_is_saved_as_failure_evidence(self) -> None:
        prepared = self._prepare()
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection,
                prepared.task.task_id,
                remote_id=("https://api.worldquantbrain.com/simulations/simulation-1"),
                observed_at="2026-08-29T00:02:00+00:00",
            )
            record_pending_observation(
                connection,
                prepared.task.task_id,
                observed_at="2026-08-29T00:03:00+00:00",
                platform_alpha_id="alpha-1",
            )
        payload = self._detail_payload()
        payload["is"]["checks"] = [
            {"name": "LOW_SHARPE", "result": "FAIL"}
        ]
        yearly_client, _ = self._authenticated_client(
            self._response(200, payload)
        )
        self.assertEqual(
            advance_real_backtest(
                self.database_path,
                yearly_client,
                prepared.task.task_id,
                observed_at="2026-08-29T00:03:30+00:00",
            ).action,
            "detail_captured",
        )
        client, _ = self._authenticated_client(
            self._response(200, self._yearly_payload())
        )

        completed = advance_real_backtest(
            self.database_path,
            client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:04:00+00:00",
        )

        self.assertEqual(completed.action, "completed")
        self.assertEqual(completed.snapshot.task.status, "completed")
        self.assertIsNotNone(completed.snapshot.result)
        assert completed.snapshot.result is not None
        self.assertEqual(
            tuple(
                (check.name, check.status)
                for check in completed.snapshot.result.checks
            ),
            (("LOW_SHARPE", "FAIL"),),
        )

    def test_zero_capital_detail_uses_specific_failure_code(self) -> None:
        prepared = self._prepare()
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection,
                prepared.task.task_id,
                remote_id=("https://api.worldquantbrain.com/simulations/simulation-1"),
                observed_at="2026-08-29T00:02:00+00:00",
            )
            record_pending_observation(
                connection,
                prepared.task.task_id,
                observed_at="2026-08-29T00:03:00+00:00",
                platform_alpha_id="alpha-1",
            )
        payload = self._detail_payload()
        payload["is"].update(
            {
                "pnl": 0,
                "bookSize": 0,
                "turnover": 0.0,
                "returns": 0.0,
                "drawdown": 0.0,
                "margin": 0.0,
                "sharpe": 0.0,
                "fitness": None,
            }
        )
        detail_client, _ = self._authenticated_client(self._response(200, payload))

        failed = advance_real_backtest(
            self.database_path,
            detail_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:04:00+00:00",
        )

        self.assertEqual(failed.action, "failed")
        self.assertEqual(failed.snapshot.task.status, "failed")
        self.assertEqual(
            failed.snapshot.task.failure_code,
            "platform_zero_capital",
        )
        self.assertEqual(failed.snapshot.task.platform_alpha_id, "alpha-1")
        self.assertIsNone(failed.snapshot.result)

    def test_explicit_rejection_returns_task_to_created(self) -> None:
        prepared = self._prepare()
        client, _ = self._authenticated_client(self._response(429, {}))

        with self.assertRaises(WorldQuantRequestError) as raised:
            advance_real_backtest(
                self.database_path,
                client,
                prepared.task.task_id,
                observed_at="2026-08-29T00:02:00+00:00",
                allow_submission=True,
            )

        with open_database(self.database_path) as connection:
            persisted = get_backtest_task(connection, prepared.task.task_id)
        self.assertFalse(raised.exception.outcome_unknown)
        self.assertEqual(persisted.task.status, "created")

    def test_poll_failure_keeps_remote_task_pending(self) -> None:
        prepared = self._prepare()
        submit_client, _ = self._authenticated_client(
            self._response(
                201,
                {},
                headers={"Location": "/simulations/simulation-1"},
            )
        )
        advance_real_backtest(
            self.database_path,
            submit_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:02:00+00:00",
            allow_submission=True,
        )
        poll_client, _ = self._authenticated_client(OSError("temporary"))

        with self.assertRaises(WorldQuantRequestError):
            advance_real_backtest(
                self.database_path,
                poll_client,
                prepared.task.task_id,
                observed_at="2026-08-29T00:03:00+00:00",
            )

        with open_database(self.database_path) as connection:
            persisted = get_backtest_task(connection, prepared.task.task_id)
        self.assertEqual(persisted.task.status, "pending")
        self.assertEqual(
            persisted.task.remote_id,
            "https://api.worldquantbrain.com/simulations/simulation-1",
        )

    def test_pending_timeout_uses_submission_start_and_poll_does_not_reset_it(
        self,
    ) -> None:
        prepared = self._prepare()
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection,
                prepared.task.task_id,
                remote_id=("https://api.worldquantbrain.com/simulations/simulation-1"),
                observed_at="2026-08-29T00:02:00+00:00",
            )
        poll_client, poll_executor = self._authenticated_client(
            self._response(200, {"progress": 0.35})
        )

        pending = advance_real_backtest(
            self.database_path,
            poll_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:11:30+00:00",
            max_pending_seconds=600,
        )
        timed_out = advance_real_backtest(
            self.database_path,
            self._authenticated_client(self._response(200, {"progress": 0.5}))[0],
            prepared.task.task_id,
            observed_at="2026-08-29T00:12:00+00:00",
            max_pending_seconds=600,
        )

        self.assertEqual(pending.action, "pending")
        self.assertEqual(len(poll_executor.requests), 2)
        self.assertEqual(timed_out.action, "pending_timeout")
        self.assertTrue(timed_out.platform_request_performed)
        self.assertEqual(timed_out.snapshot.task.status, "failed")
        self.assertEqual(
            timed_out.snapshot.task.failure_code,
            "platform_pending_timeout",
        )
        self.assertEqual(
            timed_out.snapshot.task.submission_started_at,
            "2026-08-29T00:02:00+00:00",
        )
        self.assertEqual(
            timed_out.snapshot.task.remote_id,
            "https://api.worldquantbrain.com/simulations/simulation-1",
        )

    def test_submission_unknown_is_not_a_pending_timeout(self) -> None:
        prepared = self._prepare()
        with open_database(self.database_path) as connection:
            record_submission_unknown(
                connection,
                prepared.task.task_id,
                observed_at="2026-08-29T00:02:00+00:00",
            )

        advanced = advance_real_backtest(
            self.database_path,
            self._client(FakeExecutor()),
            prepared.task.task_id,
            observed_at="2026-08-29T00:12:00+00:00",
            max_pending_seconds=1,
        )

        self.assertEqual(advanced.action, "reconciliation_required")
        self.assertEqual(advanced.snapshot.task.status, "submission_unknown")
        self.assertFalse(advanced.platform_request_performed)

    def test_concurrent_timeout_advancers_commit_one_terminal_transition(
        self,
    ) -> None:
        prepared = self._prepare()
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection,
                prepared.task.task_id,
                remote_id=("https://api.worldquantbrain.com/simulations/simulation-1"),
                observed_at="2026-08-29T00:02:00+00:00",
            )

        def advance():
            return expire_real_backtest_if_pending_timeout(
                self.database_path,
                prepared.task.task_id,
                observed_at="2026-08-29T00:12:00+00:00",
                max_pending_seconds=600,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(lambda _: advance(), range(2)))

        self.assertEqual(sum(result is not None for result in results), 1)

    def test_late_poll_response_cannot_overwrite_a_concurrent_timeout(self) -> None:
        prepared = self._prepare()
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection,
                prepared.task.task_id,
                remote_id=("https://api.worldquantbrain.com/simulations/simulation-1"),
                observed_at="2026-08-29T00:02:00+00:00",
            )
        timeout_actions: list[str] = []

        def expire_while_polling(request: Request) -> None:
            if not request.full_url.endswith("/simulations/simulation-1"):
                return
            timeout = expire_real_backtest_if_pending_timeout(
                self.database_path,
                prepared.task.task_id,
                observed_at="2026-08-29T00:12:00+00:00",
                max_pending_seconds=600,
            )
            timeout_actions.append("pending_timeout" if timeout is not None else "no_change")

        client, _ = self._authenticated_client(
            self._response(200, {"progress": 0.8}),
            observer=expire_while_polling,
        )

        advanced = advance_real_backtest(
            self.database_path,
            client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:11:59+00:00",
            max_pending_seconds=600,
        )

        self.assertEqual(timeout_actions, ["pending_timeout"])
        self.assertEqual(advanced.action, "pending_timeout")
        self.assertTrue(advanced.platform_request_performed)
        self.assertEqual(advanced.snapshot.task.status, "failed")
        self.assertEqual(
            advanced.snapshot.task.last_observed_at,
            "2026-08-29T00:12:00+00:00",
        )

    def test_empty_yearly_stats_body_is_pending_and_honors_retry_after(self) -> None:
        prepared = self._prepare()
        submit_client, _ = self._authenticated_client(
            self._response(
                201,
                {},
                headers={"Location": "/simulations/simulation-1"},
            )
        )
        advance_real_backtest(
            self.database_path,
            submit_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:02:00+00:00",
            allow_submission=True,
        )
        poll_client, _ = self._authenticated_client(
            self._response(200, {"status": "COMPLETE", "alpha": "alpha-1"})
        )
        advance_real_backtest(
            self.database_path,
            poll_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:03:00+00:00",
        )
        detail_client, _ = self._authenticated_client(
            self._response(200, self._detail_payload())
        )
        detail_captured = advance_real_backtest(
            self.database_path,
            detail_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:03:30+00:00",
        )
        self.assertEqual(detail_captured.action, "detail_captured")
        empty_client, empty_executor = self._authenticated_client(
            self._empty_response(200, headers={"Retry-After": "7"})
        )
        pending = advance_real_backtest(
            self.database_path,
            empty_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:04:00+00:00",
        )

        self.assertEqual(pending.action, "yearly_stats_pending")
        self.assertEqual(pending.retry_after_seconds, 7.0)
        self.assertIsNone(pending.snapshot.yearly_stats)
        self.assertEqual(len(empty_executor.requests), 2)

        before_deadline_executor = FakeExecutor()
        before_deadline = advance_real_backtest(
            self.database_path,
            self._client(before_deadline_executor),
            prepared.task.task_id,
            observed_at="2026-08-29T00:04:06+00:00",
        )
        self.assertEqual(before_deadline.action, "retry_wait")
        self.assertEqual(before_deadline.retry_after_seconds, 1.0)
        self.assertEqual(before_deadline_executor.requests, [])

        after_deadline_client, _ = self._authenticated_client(
            self._response(200, self._yearly_payload())
        )
        completed = advance_real_backtest(
            self.database_path,
            after_deadline_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:04:07+00:00",
        )
        self.assertEqual(completed.action, "completed")

    def test_yearly_retry_is_capped_and_timeout_preserves_partial_result(
        self,
    ) -> None:
        prepared = self._prepare()
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection,
                prepared.task.task_id,
                remote_id=("https://api.worldquantbrain.com/simulations/simulation-1"),
                observed_at="2026-08-29T00:02:00+00:00",
                platform_alpha_id="alpha-1",
            )
        detail_client, _ = self._authenticated_client(
            self._response(200, self._detail_payload())
        )
        detail = advance_real_backtest(
            self.database_path,
            detail_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:03:30+00:00",
            max_pending_seconds=125,
        )
        self.assertEqual(detail.action, "detail_captured")
        yearly_client, _ = self._authenticated_client(
            self._empty_response(200, headers={"Retry-After": "7"})
        )

        pending = advance_real_backtest(
            self.database_path,
            yearly_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:04:00+00:00",
            max_pending_seconds=125,
        )
        retry_client, no_request_executor = self._authenticated_client(self._empty_response(200))
        timed_out = advance_real_backtest(
            self.database_path,
            retry_client,
            prepared.task.task_id,
            observed_at="2026-08-29T00:04:07+00:00",
            max_pending_seconds=125,
        )

        self.assertEqual(pending.action, "yearly_stats_pending")
        self.assertEqual(pending.retry_after_seconds, 5.0)
        self.assertEqual(timed_out.action, "pending_timeout")
        self.assertEqual(timed_out.snapshot.task.status, "failed")
        self.assertIsNotNone(timed_out.snapshot.result)
        self.assertIsNone(timed_out.snapshot.yearly_stats)
        self.assertIsNone(timed_out.snapshot.task.retry_not_before)
        self.assertEqual(len(no_request_executor.requests), 2)
        with open_database(self.database_path) as connection:
            self.assertEqual(list_completed_backtests(connection), ())
            terminal = list_terminal_backtests(connection)
        self.assertEqual(terminal, (timed_out.snapshot,))

    def _prepare(self):
        return prepare_real_backtest(
            self.database_path,
            account_scope="group-account",
            formula=self.formula,
            settings=self.settings,
            created_at="2026-08-29T00:01:00+00:00",
        )

    def _authenticated_client(self, response, *, observer=None):
        executor = FakeExecutor(self._response(201, {}), response, observer=observer)
        client = self._client(executor)
        client.authenticate()
        return client, executor

    @staticmethod
    def _client(
        executor: FakeExecutor,
        *,
        timeout_seconds: float = 30.0,
    ) -> WorldQuantClient:
        return WorldQuantClient(
            base_url="https://api.worldquantbrain.com",
            credentials=WorldQuantCredentials(
                username="user@example.com",
                password="secret",
            ),
            timeout_seconds=timeout_seconds,
            request_executor=executor,
        )

    @staticmethod
    def _response(
        status_code: int,
        payload: object,
        *,
        headers: dict[str, str] | None = None,
    ) -> WorldQuantResponse:
        return WorldQuantResponse(
            status_code=status_code,
            headers=headers or {},
            body=json.dumps(payload).encode("utf-8"),
        )

    def _detail_payload(self) -> dict[str, object]:
        return {
            "id": "alpha-1",
            "type": "REGULAR",
            "settings": self.settings.as_platform_dict(),
            "regular": {"code": "rank(close)"},
            "status": "UNSUBMITTED",
            "is": {
                "pnl": 100_000,
                "bookSize": 20_000_000,
                "turnover": 0.12,
                "returns": 0.08,
                "drawdown": 0.04,
                "margin": 0.001,
                "sharpe": 1.3,
                "fitness": 1.1,
                "longCount": 740,
                "shortCount": 700,
                "checks": [
                    {"name": "LOW_SHARPE", "result": "PASS"},
                    {"name": "LOW_FITNESS", "result": "PASS"},
                    {"name": "LOW_TURNOVER", "result": "PASS"},
                    {"name": "HIGH_TURNOVER", "result": "PASS"},
                    {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
                    {
                        "name": "LOW_SUB_UNIVERSE_SHARPE",
                        "result": "PASS",
                    },
                    {"name": "SELF_CORRELATION", "result": "PASS"},
                    {"name": "MATCHES_COMPETITION", "result": "PASS"},
                ],
            },
        }

    @staticmethod
    def _yearly_payload() -> dict[str, object]:
        properties = (
            ("year", "year"),
            ("pnl", "amount"),
            ("bookSize", "amount"),
            ("longCount", "integer"),
            ("shortCount", "integer"),
            ("turnover", "percent"),
            ("sharpe", "decimal"),
            ("returns", "percent"),
            ("drawdown", "percent"),
            ("margin", "permyriad"),
            ("fitness", "decimal"),
            ("stage", "string"),
        )
        return {
            "schema": {
                "name": "yearly-stats",
                "title": "Yearly stats",
                "properties": [
                    {"name": name, "title": name, "type": kind}
                    for name, kind in properties
                ],
            },
            "records": [
                [
                    "2024",
                    25_000,
                    20_000_000,
                    731,
                    694,
                    0.12,
                    1.1,
                    0.04,
                    0.06,
                    0.0009,
                    0.7,
                    "IS",
                ]
            ],
        }

    @staticmethod
    def _empty_response(
        status_code: int,
        *,
        headers: dict[str, str] | None = None,
    ) -> WorldQuantResponse:
        return WorldQuantResponse(
            status_code=status_code,
            headers=headers or {},
            body=b"",
        )


if __name__ == "__main__":
    unittest.main()
