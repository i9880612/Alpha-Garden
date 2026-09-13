from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from dataclasses import replace
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.backtest_batches import (
    AutomatedCandidateBacktest,
    prepare_automated_candidate_backtest_batch,
)
from alpha_garden.console import run_progress_console
from execution.backtests import record_submission_accepted, record_submission_unknown
from execution.cycle_backtests import (
    advance_automated_cycle_backtests,
    settle_failed_automated_cycle_if_terminal,
)
from execution.launch import launch_automated_run, resume_automated_run
from execution.real_backtests import expire_real_backtest_if_pending_timeout
from execution.process_lock import exclusive_run_process
from execution.runs import (
    AutomatedRunLimits,
    complete_automated_run_after_candidate_planning_stop,
    fail_automated_run,
    prepare_automated_run,
    start_automated_run,
)
from generation.candidate import FormulaCandidate, exploration_candidate
from generation.parser import parse_formula
from persistence.catalog import (
    FieldCatalogContext,
    FieldCatalogRecord,
    OperatorCatalogRecord,
    OperatorOutputRecord,
    OperatorRoleRecord,
    PlatformCatalogSyncRecord,
    WindowCatalogRecord,
    replace_operator_outputs,
    replace_operator_roles,
    replace_platform_catalog,
    replace_window_catalog,
)
from persistence.backtests import get_backtest_task
from persistence.database import open_database
from persistence.run_diagnostics import (
    CandidatePlanningDiagnostic,
    CandidatePlanningExclusionCount,
)
from persistence.runs import list_active_automated_runs, get_automated_run
from persistence.schema import initialize_database_schema
from worldquant.backtests import (
    BacktestCheck,
    BacktestDetail,
    BacktestPollObservation,
    BacktestSettings,
    BacktestSubmissionObservation,
    BacktestYearlyStatsObservation,
    STANDARD_REGULAR_CHECK_NAMES,
)
from worldquant.alphas import UserAlphaPage
from worldquant.client import WorldQuantRequestError


class FakeTime:
    def __init__(self, value: str) -> None:
        self.value = datetime.fromisoformat(value)
        self.waits: list[float] = []

    def now(self) -> datetime:
        return self.value

    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.value += timedelta(seconds=seconds)


class LaunchClient:
    def __init__(self) -> None:
        self.authenticated = False
        self.calls: list[str] = []

    def authenticate(self):
        self.calls.append("authenticate")
        self.authenticated = True

    def submit_backtest(self, *, formula, settings):
        self.calls.append("submit")
        return BacktestSubmissionObservation(
            "accepted",
            "https://api.worldquantbrain.com/simulations/simulation-1",
            None,
        )

    def poll_backtest(self, remote_id):
        self.calls.append("poll")
        return BacktestPollObservation("completed", "alpha-1", "COMPLETE", None)

    def fetch_backtest_yearly_stats(self, *, platform_alpha_id):
        self.calls.append("yearly_stats")
        return BacktestYearlyStatsObservation(
            state="ready",
            stats=(),
            retry_after_seconds=None,
        )

    def fetch_backtest_detail(
        self,
        *,
        platform_alpha_id,
        expected_formula,
        expected_settings,
    ):
        self.calls.append("detail")
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

    def fetch_user_alpha_page(self, *, limit, offset, hidden, status=None):
        self.calls.append("alpha_list")
        return UserAlphaPage(total_count=0, records=(), has_next=False)


class AutomatedRunLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.database_path = root / "launch.sqlite3"
        self.settings_path = root / "backtest.json"
        self.environment_path = root / ".env"
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
            initialize_database_schema(connection)
        self._initialize_catalog()

    def test_explicit_parameters_create_and_complete_one_run(self) -> None:
        fake_time = FakeTime("2026-08-30T00:00:00+08:00")
        client = LaunchClient()
        received_connections = []

        def make_client(settings):
            received_connections.append(settings)
            return client

        output = io.StringIO()
        with redirect_stdout(output), run_progress_console():
            completion = launch_automated_run(
                self.database_path,
                self.settings_path,
                self.environment_path,
                limits=self._limits(),
                clock=fake_time.now,
                waiter=fake_time.wait,
                client_factory=make_client,
            )
        rendered = output.getvalue()
        for expected in ("阶段[1] 公式生成中", "阶段[1] 公式生成完成", "阶段[2] 筛选完成",
                         "[通过]   第1轮 01/1 [探索]   alpha-1    S  1.30 F  1.10 T 12.0%",
                         "阶段[5] 结算与学习反馈完成"):
            self.assertIn(expected, rendered)
        self.assertEqual(rendered.count("alpha-1    S  1.30 F  1.10 T 12.0%"), 1)

        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(completion.run.stop_reason, "max_cycles_reached")
        self.assertEqual(completion.run.generation_count, 3)
        self.assertEqual(completion.run.max_backtests, 1)
        self.assertTrue(completion.run.real_backtests_authorized)
        self.assertFalse(completion.run.automatic_submissions_enabled)
        self.assertEqual(completion.platform_request_count, 4)
        self.assertEqual(
            client.calls,
            ["authenticate", "submit", "poll", "detail", "yearly_stats"],
        )
        self.assertEqual(fake_time.waits, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(
            received_connections[0].base_url,
            "https://api.worldquantbrain.com",
        )
        with open_database(self.database_path) as connection:
            task_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_tasks"
            ).fetchone()[0]
            active_runs = list_active_automated_runs(connection)
        self.assertEqual(task_count, 1)
        self.assertEqual(active_runs, ())

    def test_missing_backtest_authorization_stops_before_reading_credentials(self) -> None:
        limits = AutomatedRunLimits(
            exploration_percent=30,
            self_correlation_percent=30,
            mutation_percent=40,
            direction_validation_percent=3,
            generation_count=3,
            backtest_count=1,
            max_cycles=1,
            max_backtests=0,
            max_pending_seconds=60,
            max_consecutive_failures=1,
            max_request_failures=3,
            max_in_flight_backtests=3,
            exploration_seed_attempt_multiplier=4,
            real_backtests_authorized=False,
        )

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_backtests_not_authorized",
        ):
            launch_automated_run(
                self.database_path,
                self.settings_path,
                self.environment_path.with_name("missing.env"),
                limits=limits,
            )

    def test_invalid_catalog_stops_before_platform_client_creation(self) -> None:
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM platform_catalog_syncs")

        with self.assertRaisesRegex(ValueError, "generation_catalog_sync_missing"):
            launch_automated_run(
                self.database_path,
                self.settings_path,
                self.environment_path,
                limits=self._limits(),
                client_factory=lambda settings: self.fail(
                    "目录身份缺失时不应创建平台客户端"
                ),
            )

        with open_database(self.database_path) as connection:
            run_count = connection.execute(
                "SELECT COUNT(*) FROM automated_runs"
            ).fetchone()[0]
        self.assertEqual(run_count, 0)

    def test_new_command_retires_unstarted_plan_without_inventing_a_start(self) -> None:
        old = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=self._limits(),
            created_at="2026-08-30T00:00:00+08:00",
        )
        fake_time = FakeTime("2026-08-30T00:01:00+08:00")
        client = LaunchClient()

        completion = launch_automated_run(
            self.database_path,
            self.settings_path,
            self.environment_path,
            limits=self._limits(),
            clock=fake_time.now,
            waiter=fake_time.wait,
            client_factory=lambda settings: client,
        )
        self.assertEqual(completion.run.status, "completed")
        self.assertNotEqual(completion.run.run_id, old.run_id)
        with open_database(self.database_path) as connection:
            previous = get_automated_run(connection, old.run_id)
            self.assertEqual(previous.status, "failed")
            self.assertEqual(previous.stop_reason, "replaced_by_new_run")
            self.assertIsNone(previous.started_at)
            self.assertEqual(list_active_automated_runs(connection), ())

    def test_live_driver_blocks_run_resume_and_submit_without_authentication(self):
        from execution.submission_runner import submit_queued_alphas

        old = prepare_automated_run(self.database_path, self.settings_path,
            account_scope="group-account", limits=self._limits(),
            created_at="2026-08-30T00:00:00+08:00")
        client = LaunchClient()
        with exclusive_run_process(self.database_path):
            for operation in (
                lambda: launch_automated_run(self.database_path, self.settings_path,
                    self.environment_path, limits=self._limits(),
                    client_factory=lambda _: client),
                lambda: resume_automated_run(self.database_path, self.environment_path,
                    old.run_id, client_factory=lambda _: client),
                lambda: submit_queued_alphas(self.database_path, self.environment_path,
                    client_factory=lambda _: client),
            ):
                with self.assertRaisesRegex(ValueError, "已有命令"):
                    operation()
        self.assertEqual(client.calls, [])
        with open_database(self.database_path) as connection:
            self.assertEqual(get_automated_run(connection, old.run_id), old)

    def test_keyboard_interrupt_releases_lock_and_next_run_uses_new_limits(self):
        fake_time = FakeTime("2026-08-30T00:00:00+08:00")
        old_limits = replace(self._limits(), max_cycles=-1, max_backtests=0,
                             automatic_submissions_enabled=True)
        observed = []

        def interrupted(_):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            launch_automated_run(self.database_path, self.settings_path,
                self.environment_path, limits=old_limits,
                clock=fake_time.now, client_factory=interrupted,
                run_created=observed.append)
        fake_time.wait(1)
        client = LaunchClient()
        result = launch_automated_run(self.database_path, self.settings_path,
            self.environment_path, limits=self._limits(), clock=fake_time.now,
            waiter=fake_time.wait, client_factory=lambda _: client)
        self.assertEqual(result.run.status, "completed")
        self.assertEqual(result.run.max_cycles, 1)
        self.assertEqual(result.run.max_backtests, 1)
        self.assertFalse(result.run.automatic_submissions_enabled)
        self.assertNotEqual(result.run.run_id, observed[0].run_id)
        with open_database(self.database_path) as connection:
            previous = get_automated_run(connection, observed[0].run_id)
            self.assertEqual(previous.max_cycles, -1)
            self.assertTrue(previous.automatic_submissions_enabled)

    def test_invalid_new_policy_does_not_retire_old_plan(self):
        old = prepare_automated_run(self.database_path, self.settings_path,
            account_scope="group-account", limits=self._limits(),
            created_at="2026-08-30T00:00:00+08:00")
        with self.assertRaises(ValueError):
            launch_automated_run(self.database_path, self.settings_path.with_name('missing'),
                self.environment_path, limits=self._limits())
        with open_database(self.database_path) as connection:
            self.assertEqual(get_automated_run(connection, old.run_id), old)
        with self.assertRaises(ValueError):
            launch_automated_run(self.database_path, self.settings_path,
                self.environment_path, limits=self._limits(), poll_interval_seconds=0)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_automated_run(connection, old.run_id), old)

    def test_created_run_is_observable_before_client_creation(self) -> None:
        observed_runs = []

        def stop_after_observation(settings):
            self.assertEqual(len(observed_runs), 1)
            raise RuntimeError("stop_after_run_created")

        with self.assertRaisesRegex(RuntimeError, "stop_after_run_created"):
            launch_automated_run(
                self.database_path,
                self.settings_path,
                self.environment_path,
                limits=self._limits(),
                client_factory=stop_after_observation,
                run_created=observed_runs.append,
            )

        with open_database(self.database_path) as connection:
            active_runs = list_active_automated_runs(connection)
        self.assertEqual(len(observed_runs), 1)
        self.assertEqual(observed_runs[0], active_runs[0])

    def test_automatic_submission_switch_is_accepted_and_frozen(self) -> None:
        limits = AutomatedRunLimits(
            exploration_percent=30,
            self_correlation_percent=30,
            mutation_percent=40,
            direction_validation_percent=3,
            generation_count=3,
            backtest_count=1,
            max_cycles=1,
            max_backtests=1,
            max_pending_seconds=60,
            max_consecutive_failures=1,
            max_request_failures=3,
            max_in_flight_backtests=3,
            exploration_seed_attempt_multiplier=4,
            real_backtests_authorized=True,
            automatic_submissions_enabled=True,
        )

        observed_runs = []

        def stop_after_creation(settings):
            raise RuntimeError("stop_after_formal_authority_was_frozen")

        with self.assertRaisesRegex(
            RuntimeError,
            "stop_after_formal_authority_was_frozen",
        ):
            launch_automated_run(
                self.database_path,
                self.settings_path,
                self.environment_path,
                limits=limits,
                client_factory=stop_after_creation,
                run_created=observed_runs.append,
            )

        self.assertEqual(len(observed_runs), 1)
        self.assertTrue(observed_runs[0].automatic_submissions_enabled)

    def test_resumes_an_existing_run_without_creating_another_plan(self) -> None:
        run = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=self._limits(),
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
            candidates=(
                AutomatedCandidateBacktest(
                    self._candidate("rank(close)"),
                    self.settings,
                ),
            ),
            created_at="2026-08-30T00:01:30+08:00",
        )
        initial_client = LaunchClient()
        initial_client.authenticated = True
        advance_automated_cycle_backtests(
            self.database_path,
            initial_client,
            run.run_id,
            observed_at="2026-08-30T00:01:31+08:00",
        )
        client = LaunchClient()
        fake_time = FakeTime("2026-08-30T00:01:32+08:00")

        completion = resume_automated_run(
            self.database_path,
            self.environment_path,
            run.run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
            client_factory=lambda settings: client,
        )

        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(initial_client.calls, ["submit"])
        self.assertEqual(
            client.calls,
            ["authenticate", "poll", "detail", "yearly_stats"],
        )
        with open_database(self.database_path) as connection:
            run_count = connection.execute(
                "SELECT COUNT(*) FROM automated_runs"
            ).fetchone()[0]
        self.assertEqual(run_count, 1)

    def test_resume_rejects_a_different_account_before_client_creation(self) -> None:
        run = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=self._limits(),
            created_at="2026-08-30T00:00:00+08:00",
        )
        other_environment = self.environment_path.with_name("other.env")
        other_environment.write_text(
            self.environment_path.read_text(encoding="utf-8").replace(
                "WQB_ACCOUNT_SCOPE=group-account",
                "WQB_ACCOUNT_SCOPE=other-account",
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_account_scope_mismatch",
        ):
            resume_automated_run(
                self.database_path,
                other_environment,
                run.run_id,
                client_factory=lambda settings: self.fail(
                    "账号不匹配时不应创建平台客户端"
                ),
            )

    def test_resume_rejects_terminal_runs_before_client_creation(self) -> None:
        completed = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=self._limits(),
            created_at="2026-08-30T00:00:00+08:00",
        )
        start_automated_run(
            self.database_path,
            completed.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        complete_automated_run_after_candidate_planning_stop(
            self.database_path,
            completed.run_id,
            completed_at="2026-08-30T00:02:00+08:00",
            reason="generation_attempt_budget_exhausted",
            diagnostic=CandidatePlanningDiagnostic(
                cycle_number=1,
                exploration_generation_target_count=3,
                exploration_backtest_target_count=1,
                seed_attempt_limit=12,
                attempted_seed_count=12,
                generated_candidate_count=0,
                selected_candidate_count=None,
                generation_exclusions=(
                    CandidatePlanningExclusionCount("test_rejection", 12),
                ),
                selection_rejections=(),
            ),
        )
        with self.assertRaisesRegex(ValueError, "automated_run_already_completed"):
            resume_automated_run(
                self.database_path,
                self.environment_path,
                completed.run_id,
                client_factory=lambda settings: self.fail(
                    "终态运行不应创建客户端"
                ),
            )

        failed = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=self._limits(),
            created_at="2026-08-30T00:03:00+08:00",
        )
        start_automated_run(
            self.database_path,
            failed.run_id,
            started_at="2026-08-30T00:04:00+08:00",
        )
        fail_automated_run(
            self.database_path,
            failed.run_id,
            failed_at="2026-08-30T00:05:00+08:00",
            reason="test_failed",
        )
        with self.assertRaisesRegex(ValueError, "automated_run_already_failed"):
            resume_automated_run(
                self.database_path,
                self.environment_path,
                failed.run_id,
                client_factory=lambda settings: self.fail(
                    "失败运行不应创建客户端"
                ),
            )

    def test_resume_expires_unknown_without_platform_requests(self) -> None:
        run = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=self._limits(),
            created_at="2026-08-30T00:00:00+08:00",
        )
        start_automated_run(
            self.database_path,
            run.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        prepared = prepare_automated_candidate_backtest_batch(
            self.database_path,
            run_id=run.run_id,
            candidates=(
                AutomatedCandidateBacktest(
                    self._candidate("rank(close)"),
                    self.settings,
                ),
            ),
            created_at="2026-08-30T00:01:30+08:00",
        )[0]
        with open_database(self.database_path) as connection:
            record_submission_unknown(
                connection,
                prepared.task.task_id,
                observed_at="2026-08-30T00:01:31+08:00",
            )
        failed = fail_automated_run(
            self.database_path,
            run.run_id,
            failed_at="2026-08-30T00:02:00+08:00",
            reason="test_failed",
        )
        self.assertEqual(
            failed.stop_reason,
            "test_failed",
        )
        client = LaunchClient()
        completion = resume_automated_run(
            self.database_path,
            self.environment_path,
            run.run_id,
            clock=lambda: datetime.fromisoformat("2026-08-30T00:04:00+08:00"),
            client_factory=lambda settings: client,
        )

        self.assertEqual(completion.run.status, "failed")
        self.assertEqual(
            completion.run.stop_reason,
            "test_failed",
        )
        self.assertFalse(completion.authentication_performed)
        self.assertEqual(completion.platform_request_count, 0)
        self.assertEqual(client.calls, [])
        with open_database(self.database_path) as connection:
            expired = get_backtest_task(connection, prepared.task.task_id)
            self.assertEqual(expired.task.failure_code, "submission_outcome_timeout")
            self.assertIsNone(expired.result)

    def test_cli_resume_preserves_post_cooldown_and_reads_pending_before_retry(self):
        run = prepare_automated_run(self.database_path, self.settings_path,
            account_scope="group-account", limits=replace(self._limits(), backtest_count=2,
                max_backtests=2, max_pending_seconds=600), created_at="2026-08-30T00:00:00+08:00")
        start_automated_run(self.database_path, run.run_id, started_at="2026-08-30T00:01:00+08:00")
        prepared = prepare_automated_candidate_backtest_batch(self.database_path,
            run_id=run.run_id, candidates=tuple(AutomatedCandidateBacktest(self._candidate(f), self.settings)
                for f in ("rank(close)", "rank(open)")), created_at="2026-08-30T00:01:30+08:00")
        with open_database(self.database_path) as connection:
            record_submission_accepted(connection, prepared[0].task.task_id,
                remote_id="https://api.worldquantbrain.com/simulations/existing",
                observed_at="2026-08-30T00:01:31+08:00")
        clock = FakeTime("2026-08-30T00:01:32+08:00")
        attempts, polls = [], []
        class LimitedClient(LaunchClient):
            def submit_backtest(self, **kwargs):
                attempts.append(clock.now())
                raise WorldQuantRequestError("worldquant_submission_http_error", status_code=429,
                                             retryable=True, outcome_unknown=False)
            def poll_backtest(self, remote_id):
                polls.append(clock.now())
                return super().poll_backtest(remote_id)
        first_client = LimitedClient()
        first_client.authenticated = True
        advance_automated_cycle_backtests(self.database_path, first_client, run.run_id,
                                         observed_at=clock.now().isoformat())
        clock.wait(1)
        client = LimitedClient()
        completion = resume_automated_run(self.database_path, self.environment_path, run.run_id,
            clock=clock.now, waiter=clock.wait, client_factory=lambda settings: client)
        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(len(attempts), 4)
        self.assertLess(polls[0], attempts[0] + timedelta(seconds=120))
        self.assertGreaterEqual((attempts[1] - attempts[0]).total_seconds(), 120)
        self.assertEqual(client.calls.count("authenticate"), 1)

    def test_resume_rate_limited_run_restores_only_cancelled_unsent_tasks(self):
        self._assert_request_failure_resume("worldquant_submission_http_error", 429)

    def test_resume_network_failure_restores_only_cancelled_unsent_tasks(self):
        self._assert_request_failure_resume("worldquant_poll_request_failed:SSLEOFError", None)

    def test_resume_network_failure_before_any_request_was_sent(self):
        self._assert_request_failure_resume(
            "worldquant_authentication_request_failed:SSLEOFError", None, pending=False)

    def test_resume_network_failure_does_not_reopen_settled_cancelled_batch(self):
        self._assert_request_failure_resume(
            "worldquant_poll_request_failed:SSLEOFError", None, pending=False, settled=True)

    def _assert_request_failure_resume(self, failure_code, status_code, *, pending=True, settled=False):
        formulas = ("rank(close)", "rank(open)") if pending else ("rank(open)",)
        run = prepare_automated_run(self.database_path, self.settings_path,
            account_scope="group-account", limits=replace(self._limits(),
                backtest_count=len(formulas), max_backtests=len(formulas)),
            created_at="2026-08-30T00:00:00+08:00")
        start_automated_run(self.database_path, run.run_id, started_at="2026-08-30T00:01:00+08:00")
        prepared = prepare_automated_candidate_backtest_batch(self.database_path,
            run_id=run.run_id, candidates=tuple(AutomatedCandidateBacktest(self._candidate(f), self.settings)
                for f in formulas), created_at="2026-08-30T00:01:30+08:00")
        if pending:
            with open_database(self.database_path) as connection:
                record_submission_accepted(connection, prepared[0].task.task_id,
                    remote_id="https://api.worldquantbrain.com/simulations/existing",
                    observed_at="2026-08-30T00:01:31+08:00")
        fail_automated_run(self.database_path, run.run_id,
            failed_at="2026-08-30T00:01:40+08:00", reason="request_failure_limit_reached",
            request_failure_code=failure_code, request_status_code=status_code)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, prepared[-1].task.task_id).task.failure_code,
                             "automated_run_stopped_before_submission")
        client = LaunchClient()
        fake_time = FakeTime("2026-08-30T00:03:00+08:00")
        if settled:
            settle_failed_automated_cycle_if_terminal(
                self.database_path, run.run_id, observed_at=fake_time.now().isoformat())
            with self.assertRaisesRegex(ValueError, "automated_run_cancelled_task_not_resumable"):
                resume_automated_run(self.database_path, self.environment_path, run.run_id,
                    clock=fake_time.now, waiter=fake_time.wait, client_factory=lambda settings: client)
            self.assertEqual(client.calls, [])
            with open_database(self.database_path) as connection:
                self.assertEqual(get_automated_run(connection, run.run_id).status, "failed")
                for item in prepared:
                    snapshot = get_backtest_task(connection, item.task.task_id)
                    self.assertEqual(snapshot.task.status, "failed")
                    self.assertIsNone(snapshot.task.submission_started_at)
                    self.assertEqual(snapshot.task.failure_code, "automated_run_stopped_before_submission")
            return
        completion = resume_automated_run(self.database_path, self.environment_path, run.run_id,
            clock=fake_time.now, waiter=fake_time.wait, client_factory=lambda settings: client)
        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(completion.run.run_id, run.run_id)
        self.assertEqual(completion.run.max_cycles, run.max_cycles)
        self.assertEqual(completion.run.max_backtests, run.max_backtests)
        self.assertEqual(completion.run.settings_policy_json, run.settings_policy_json)
        self.assertEqual(completion.run.automatic_submissions_enabled, False)
        self.assertEqual(client.calls.count("submit"), 1)
        self.assertEqual(client.calls[1], "poll" if pending else "submit")
        with open_database(self.database_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM automated_runs").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM automated_run_backtests").fetchone()[0], len(formulas))
            for item in prepared:
                self.assertEqual(get_backtest_task(connection, item.task.task_id).task.status, "completed")

    def test_resume_failed_run_reads_its_pending_task_without_reposting(self) -> None:
        run = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=self._limits(),
            created_at="2026-08-30T00:00:00+08:00",
        )
        start_automated_run(
            self.database_path,
            run.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        prepared = prepare_automated_candidate_backtest_batch(
            self.database_path,
            run_id=run.run_id,
            candidates=(
                AutomatedCandidateBacktest(
                    self._candidate("rank(close)"),
                    self.settings,
                ),
            ),
            created_at="2026-08-30T00:01:30+08:00",
        )[0]
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection,
                prepared.task.task_id,
                remote_id=(
                    "https://api.worldquantbrain.com/simulations/simulation-1"
                ),
                observed_at="2026-08-30T00:01:31+08:00",
            )
        fail_automated_run(
            self.database_path,
            run.run_id,
            failed_at="2026-08-30T00:01:40+08:00",
            reason="platform_request_not_retryable:test",
        )
        client = LaunchClient()
        fake_time = FakeTime("2026-08-30T00:02:31+08:00")

        completion = resume_automated_run(
            self.database_path,
            self.environment_path,
            run.run_id,
            clock=fake_time.now,
            waiter=fake_time.wait,
            client_factory=lambda settings: client,
        )

        self.assertEqual(completion.run.status, "failed")
        self.assertEqual(
            completion.run.stop_reason,
            "platform_request_not_retryable:test",
        )
        self.assertGreaterEqual(completion.step_count, 3)
        self.assertEqual(completion.platform_request_count, 3)
        self.assertTrue(completion.authentication_performed)
        self.assertEqual(client.calls, ["authenticate", "poll", "detail", "yearly_stats"])
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, prepared.task.task_id).task.status, "completed")
        self.assertEqual(completion.run.current_cycle, 1)

    def test_resume_failed_run_settles_a_terminal_cycle_after_interruption(
        self,
    ) -> None:
        run = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=self._limits(),
            created_at="2026-08-30T00:00:00+08:00",
        )
        start_automated_run(
            self.database_path,
            run.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        prepared = prepare_automated_candidate_backtest_batch(
            self.database_path,
            run_id=run.run_id,
            candidates=(
                AutomatedCandidateBacktest(
                    self._candidate("rank(close)"),
                    self.settings,
                ),
            ),
            created_at="2026-08-30T00:01:30+08:00",
        )[0]
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection,
                prepared.task.task_id,
                remote_id=(
                    "https://api.worldquantbrain.com/simulations/simulation-1"
                ),
                observed_at="2026-08-30T00:01:31+08:00",
            )
        fail_automated_run(
            self.database_path,
            run.run_id,
            failed_at="2026-08-30T00:01:40+08:00",
            reason="platform_request_not_retryable:test",
        )
        expired = expire_real_backtest_if_pending_timeout(
            self.database_path,
            prepared.task.task_id,
            observed_at="2026-08-30T00:02:31+08:00",
            max_pending_seconds=60,
        )
        self.assertIsNotNone(expired)
        client = LaunchClient()

        completion = resume_automated_run(
            self.database_path,
            self.environment_path,
            run.run_id,
            clock=lambda: datetime.fromisoformat("2026-08-30T00:02:32+08:00"),
            client_factory=lambda settings: client,
        )

        self.assertEqual(completion.run.status, "failed")
        self.assertEqual(completion.run.current_cycle, 1)
        self.assertEqual(completion.step_count, 1)
        self.assertEqual(completion.platform_request_count, 0)
        self.assertFalse(completion.authentication_performed)
        self.assertEqual(client.calls, [])

    @staticmethod
    def _limits() -> AutomatedRunLimits:
        return AutomatedRunLimits(
            exploration_percent=30,
            self_correlation_percent=30,
            mutation_percent=40,
            direction_validation_percent=3,
            generation_count=3,
            backtest_count=1,
            max_cycles=1,
            max_backtests=1,
            max_pending_seconds=60,
            max_consecutive_failures=1,
            max_request_failures=3,
            max_in_flight_backtests=3,
            exploration_seed_attempt_multiplier=4,
            real_backtests_authorized=True,
        )

    @staticmethod
    def _candidate(formula: str) -> FormulaCandidate:
        return exploration_candidate(parse_formula(formula).expression)

    def _initialize_catalog(self) -> None:
        context = FieldCatalogContext("EQUITY", "USA", "TOP3000", 1)
        fields = tuple(
            FieldCatalogRecord(
                context=context,
                field_id=field_id,
                dataset_id="dataset",
                category="sample",
                subcategory=None,
                field_type="MATRIX",
                coverage=1.0,
                description=field_id,
                dataset_name="sample",
                category_id="sample",
                subcategory_id=None,
                raw_payload={"id": field_id},
                synced_at="2026-08-30T00:00:00+08:00",
            )
            for field_id in ("close", "open", "volume")
        )
        operators = (
            self._operator("rank", "Cross Sectional", (("x", "expr"),)),
            self._operator(
                "ts_rank",
                "Time Series",
                (("x", "expr"), ("d", "window")),
            ),
        )
        with open_database(self.database_path) as connection:
            replace_operator_roles(
                connection,
                (
                    OperatorRoleRecord("rank", "cross_sectional_normalization"),
                    OperatorRoleRecord("ts_rank", "time_series_normalization"),
                ),
            )
            replace_operator_outputs(
                connection,
                (
                    OperatorOutputRecord("rank", "signal"),
                    OperatorOutputRecord("ts_rank", "signal"),
                ),
            )
            replace_window_catalog(
                connection,
                (
                    WindowCatalogRecord(22, "month"),
                    WindowCatalogRecord(66, "quarter"),
                ),
            )
            replace_platform_catalog(
                connection,
                PlatformCatalogSyncRecord(
                    account_scope="group-account",
                    context=context,
                    field_count=len(fields),
                    operator_count=len(operators),
                    synced_at="2026-08-30T00:00:00+08:00",
                ),
                fields,
                operators,
            )

    @staticmethod
    def _operator(name, category, parameters) -> OperatorCatalogRecord:
        return OperatorCatalogRecord(
            operator_name=name,
            category=category,
            definition=name,
            description=name,
            documentation=None,
            level="ALL",
            scope=("REGULAR",),
            parameters=tuple(
                {"name": parameter_name, "kind": parameter_kind}
                for parameter_name, parameter_kind in parameters
            ),
            raw_payload={"name": name},
            synced_at="2026-08-30T00:00:00+08:00",
        )


if __name__ == "__main__":
    unittest.main()
