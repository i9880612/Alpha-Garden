from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.backtest_batches import (
    AutomatedCandidateBacktest,
    prepare_automated_candidate_backtest_batch,
)
from execution.cycle_candidates import ExplorationAttemptBudgetExhausted
from execution.driver import advance_automated_run
from execution.cycles import plan_automated_cycle
from execution.cycle_backtests import (
    settle_automated_cycle,
    synchronize_automated_cycle_submission_queue,
)
from execution.generation import ExplorationBatch, ExplorationCandidate
from execution.runner import run_automated_run
from execution.runs import AutomatedRunLimits, prepare_automated_run
from generation.candidate import FormulaCandidate, exploration_candidate
from generation.parser import parse_formula
from learning.recovery import RecoveryComparison
from persistence.backtests import (
    BacktestMutationRecord,
    create_backtest_mutation,
    get_backtest_task,
)
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
from persistence.database import open_database
from persistence.pnl import PnlSeriesRecord, list_pnl_series, save_pnl_series
from persistence.run_diagnostics import (
    CandidatePlanningDiagnostic,
    CandidatePlanningExclusionCount,
)
from persistence.schema import initialize_database_schema
from persistence.runs import get_automated_cycle_settlement
from persistence.submission_queue import list_formal_submission_queue
from persistence.submissions import (
    get_formal_submission_attempt,
    list_platform_submitted_alphas,
)
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
from worldquant.client import WorldQuantRequestError
from worldquant.submissions import (
    AlphaDetailObservation,
    FormalCheckObservation,
    FormalSubmissionObservation,
)


class DriverClient:
    authenticated = True

    def fetch_pnl(self, *, platform_alpha_id):
        from datetime import date, timedelta
        from itertools import accumulate
        from random import Random
        from worldquant.pnl import PnlObservation
        self.calls.append("pnl")
        rng = Random("synthetic:" + platform_alpha_id)
        values = accumulate([0.0, *(rng.uniform(-1, 1) for _ in range(300))])
        return PnlObservation(tuple(((date(2020, 1, 1) + timedelta(days=i)).isoformat(), v)
                                    for i, v in enumerate(values)))

    def __init__(
        self,
        *submissions,
        formal_submit_error: WorldQuantRequestError | None = None,
        confirmation_status: str = "ACTIVE",
        metrics_by_formula: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.submissions = list(submissions)
        self.calls: list[str] = []
        self.formulas: dict[str, str] = {}
        self.formal_submit_error = formal_submit_error
        self.confirmation_status = confirmation_status
        self.metrics_by_formula = metrics_by_formula or {}
        self.formal_submission_ids: list[str] = []

    def submit_backtest(self, *, formula, settings):
        self.calls.append("submit")
        result = self.submissions.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def poll_backtest(self, remote_id):
        self.calls.append("poll")
        remote_name = remote_id.rsplit("/", 1)[-1]
        return BacktestPollObservation(
            "completed",
            f"alpha-{remote_name}",
            "COMPLETE",
            None,
        )

    def fetch_backtest_detail(
        self,
        *,
        platform_alpha_id,
        expected_formula,
        expected_settings,
    ):
        self.calls.append("detail")
        self.formulas[platform_alpha_id] = expected_formula
        sharpe, fitness = self.metrics_by_formula.get(
            expected_formula,
            (1.3, 1.1),
        )
        return BacktestDetail(
            platform_alpha_id=platform_alpha_id,
            grade="SPECTACULAR",
            sharpe=sharpe,
            fitness=fitness,
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
        if self.formal_submit_error is not None:
            raise self.formal_submit_error
        return FormalSubmissionObservation(status_code=201, payload=None)

    def fetch_formal_submission_result(self, *, platform_alpha_id):
        return FormalSubmissionObservation(200, None)

    def fetch_alpha_detail(self, *, platform_alpha_id):
        self.calls.append("formal_confirmation")
        status = (
            self.confirmation_status
            if platform_alpha_id in self.formal_submission_ids
            else "UNSUBMITTED"
        )
        payload: dict[str, object] = {
            "id": platform_alpha_id,
            "grade": "SPECTACULAR",
            "status": status,
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


class BlockingFormalSubmissionClient(DriverClient):
    def __init__(
        self,
        *submissions,
        request_started: threading.Event,
        release_response: threading.Event,
    ) -> None:
        super().__init__(*submissions)
        self._request_started = request_started
        self._release_response = release_response

    def submit_formal_alpha(self, *, platform_alpha_id):
        self.calls.append("formal_submit")
        self.formal_submission_ids.append(platform_alpha_id)
        self._request_started.set()
        if not self._release_response.wait(timeout=5):
            raise TimeoutError("test_formal_submission_release_timeout")
        return FormalSubmissionObservation(status_code=201, payload=None)


class BlockingFormalConfirmationClient(DriverClient):
    def __init__(
        self,
        *submissions,
        request_started: threading.Event,
        release_response: threading.Event,
        metrics_by_formula: dict[str, tuple[float, float]],
    ) -> None:
        super().__init__(
            *submissions,
            metrics_by_formula=metrics_by_formula,
        )
        self._request_started = request_started
        self._release_response = release_response
        self._block_next_confirmation = True

    def fetch_alpha_detail(self, *, platform_alpha_id):
        if (
            self._block_next_confirmation
            and platform_alpha_id in self.formal_submission_ids
        ):
            self.calls.append("formal_confirmation")
            self._block_next_confirmation = False
            self._request_started.set()
            if not self._release_response.wait(timeout=5):
                raise TimeoutError("test_formal_confirmation_release_timeout")
            return AlphaDetailObservation(
                payload={
                    "id": platform_alpha_id,
                    "status": "ACTIVE",
                    "dateSubmitted": "2026-08-29T16:10:00+00:00",
                    "hidden": False,
                    "regular": {"code": self.formulas[platform_alpha_id]},
                }
            )
        return super().fetch_alpha_detail(platform_alpha_id=platform_alpha_id)


class FailedBacktestClient(DriverClient):
    def poll_backtest(self, remote_id):
        self.calls.append("poll")
        return BacktestPollObservation("failed", None, "FAILED", None)


class AutomatedRunDriverTests(unittest.TestCase):
    def test_unknown_keeps_one_slot_across_bounded_cycles_without_exceeding_budget(self):
        run_id = self._prepare_run(max_cycles=2, max_backtests=6, backtest_count=3)
        client = DriverClient(
            WorldQuantRequestError("lost_response", retryable=True, outcome_unknown=True),
            *(self._accepted(f"simulation-{i}") for i in range(1, 6)),
        )
        observed = datetime.fromisoformat("2026-08-30T00:01:00+08:00")
        maximum_active = 0
        unknown_id = None
        for _ in range(100):
            advanced = advance_automated_run(
                self.database_path, client, run_id, observed_at=observed.isoformat()
            )
            observed += timedelta(seconds=2)
            with open_database(self.database_path) as connection:
                rows = connection.execute(
                    "SELECT task_id, status FROM backtest_tasks"
                ).fetchall()
                queued = list_formal_submission_queue(connection)
                self.assertEqual(
                    {item.task_id for item in queued},
                    {row[0] for row in connection.execute("SELECT task_id FROM submission_checks WHERE payload_json IS NOT NULL")},
                )
            active = sum(row[1] in {"pending", "submission_unknown"} for row in rows)
            maximum_active = max(maximum_active, active)
            self.assertLessEqual(active, 3)
            unknowns = [row[0] for row in rows if row[1] == "submission_unknown"]
            if unknowns:
                unknown_id = unknown_id or unknowns[0]
                self.assertEqual(unknowns, [unknown_id])
            if advanced.run.status == "failed":
                break
        self.assertEqual(maximum_active, 3)
        self.assertEqual(advanced.run.stop_reason, "submission_reconciliation_required")
        self.assertEqual(client.calls.count("submit"), 6)
        self.assertEqual(client.calls.count("formal_submit"), 0)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(row[1] == "completed" for row in rows), 5)
        with open_database(self.database_path) as connection:
            settlement = get_automated_cycle_settlement(connection, run_id, 2)
            self.assertIsNotNone(settlement)
        repeated = settle_automated_cycle(
            self.database_path, run_id, cycle_number=2, observed_at=observed.isoformat()
        )
        self.assertTrue(repeated.already_settled)
        self.assertEqual(repeated.run.current_cycle, 1)

        # An unfinished batch has not entered its check phase; sync cannot bypass it.
        with open_database(self.database_path) as connection:
            unknown_before = get_backtest_task(connection, unknown_id)
            connection.execute("DELETE FROM submission_queue WHERE cycle_number = 1")
        recovered = synchronize_automated_cycle_submission_queue(
            self.database_path, run_id, cycle_number=1, observed_at=observed.isoformat()
        )
        self.assertEqual(recovered.enqueued_task_ids, ())
        self.assertEqual(recovered.replaced_task_ids, ())
        again = synchronize_automated_cycle_submission_queue(
            self.database_path, run_id, cycle_number=1, observed_at=observed.isoformat()
        )
        self.assertEqual(again.enqueued_task_ids, ())
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, unknown_id), unknown_before)
            self.assertIsNone(get_automated_cycle_settlement(connection, run_id, 1))
            self.assertEqual(len(list_formal_submission_queue(connection)), 3)
        with self.assertRaisesRegex(ValueError, "automated_cycle_tasks_not_terminal"):
            settle_automated_cycle(
                self.database_path, run_id, cycle_number=1, observed_at=observed.isoformat()
            )

    def test_check_and_queue_are_atomic_without_rolling_back_backtest(self):
        run_id = self._prepare_run()
        client = DriverClient(self._accepted())
        for minute in range(1, 7):
            advanced = self._advance(client, minute=minute)
            if advanced.backtest_action == "detail_captured":
                break
        self.assertEqual(advanced.backtest_action, "detail_captured")
        task_id = advanced.task_id
        with open_database(self.database_path) as connection:
            before = get_backtest_task(connection, task_id)
            self.assertEqual(list_formal_submission_queue(connection), ())
            connection.execute(
                "CREATE TRIGGER reject_queue BEFORE INSERT ON submission_queue "
                "BEGIN SELECT RAISE(ABORT, 'test_queue_write_failure'); END"
            )
        self._advance(client, minute=6)
        with open_database(self.database_path) as connection:
            before = get_backtest_task(connection, task_id)
            self.assertEqual(before.task.status, "completed")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "test_queue_write_failure"):
            self._advance(client, minute=7)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, task_id), before)
            self.assertEqual(list_formal_submission_queue(connection), ())
            connection.execute("DROP TRIGGER reject_queue")
        completed = self._advance(client, minute=8)
        self.assertEqual(completed.backtest_action, "submission_check_observed")
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, task_id).task.status, "completed")
            self.assertEqual(
                tuple(item.task_id for item in list_formal_submission_queue(connection)),
                (task_id,),
            )
            self.assertIsNone(get_automated_cycle_settlement(connection, run_id, 1))
        self.assertEqual(client.calls.count("submit"), 1)
        self.assertEqual(client.calls.count("formal_submit"), 0)

    def test_unknown_allows_later_finite_cycle_to_settle_and_submit(self):
        self._check_unknown_allows_later_cycle(max_cycles=2, max_backtests=2)

    def test_unknown_allows_continuous_cycles_without_releasing_its_slot(self):
        self._check_unknown_allows_later_cycle(max_cycles=-1, max_backtests=0)

    def _check_unknown_allows_later_cycle(self, *, max_cycles, max_backtests):
        run_id = self._prepare_run(
            max_cycles=max_cycles, max_backtests=max_backtests,
            automatic_submissions_enabled=True,
        )
        client = DriverClient(
            BacktestSubmissionObservation("unknown", None, "missing_location"),
            self._accepted("simulation-2"),
        )
        observed = datetime.fromisoformat("2026-08-30T00:01:00+08:00")
        for _ in range(60):
            advanced = advance_automated_run(
                self.database_path, client, run_id, observed_at=observed.isoformat()
            )
            observed += timedelta(seconds=2)
            if advanced.action == "cycle_settled":
                break
        self.assertEqual(advanced.action, "cycle_settled")
        self.assertEqual(advanced.cycle_number, 2)
        self.assertEqual(advanced.run.current_cycle, 1)
        self.assertEqual(advanced.run.status, "running")
        self.assertEqual(client.calls.count("submit"), 2)
        self.assertEqual(client.calls.count("formal_submit"), 1)
        with open_database(self.database_path) as connection:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM backtest_tasks WHERE status='submission_unknown'"
            ).fetchone()[0], 1)


    def test_timed_out_cycle_settles_and_next_cycle_can_submit(self):
        run_id = self._prepare_run(
            max_cycles=-1, max_backtests=0, max_pending_seconds=10,
            automatic_submissions_enabled=True,
        )
        class SlowFirstClient(DriverClient):
            def poll_backtest(self, remote_id):
                if remote_id.endswith("simulation-1"):
                    self.calls.append("slow_poll")
                    return BacktestPollObservation("pending", None, "RUNNING", None)
                return super().poll_backtest(remote_id)

        client = SlowFirstClient(*(self._accepted(f"simulation-{i}") for i in range(1, 20)))
        observed = datetime.fromisoformat("2026-08-30T00:01:00+08:00")
        for _ in range(100):
            advanced = advance_automated_run(
                self.database_path, client, run_id, observed_at=observed.isoformat(),
            )
            self.assertEqual(advanced.run.status, "running")
            observed += timedelta(seconds=2)
            if advanced.action == "cycle_settled" and advanced.cycle_number == 2:
                break
        self.assertEqual(advanced.action, "cycle_settled")
        self.assertEqual(advanced.cycle_number, 2)
        self.assertEqual(advanced.run.current_cycle, 2)
        self.assertEqual(client.calls.count("submit"), 2)
        self.assertEqual(client.calls.count("formal_submit"), 1)
        with open_database(self.database_path) as connection:
            first_task = connection.execute(
                "SELECT task_id FROM automated_run_backtests WHERE run_id=? AND cycle_number=1",
                (run_id,),
            ).fetchone()[0]
            first = get_backtest_task(connection, first_task)
            self.assertEqual(first.task.failure_code, "platform_pending_timeout")
            self.assertEqual(first.task.remote_id.rsplit("/", 1)[-1], "simulation-1")
            self.assertIsNone(first.result)
        twice = settle_automated_cycle(
            self.database_path, run_id, cycle_number=2, observed_at=observed.isoformat(),
        )
        self.assertTrue(twice.already_settled)
        self.assertEqual(twice.run.current_cycle, 2)

    def test_failed_run_remote_error_preserves_unsettled_results(self):
        run_id = self._prepare_run(
            max_cycles=-1, max_backtests=0, max_pending_seconds=60,
            backtest_count=3,
        )

        class PendingClient(DriverClient):
            reject_poll = False

            def poll_backtest(self, remote_id):
                self.calls.append("poll")
                if self.reject_poll:
                    raise WorldQuantRequestError(
                        "access_denied",
                        retryable=False,
                        outcome_unknown=False,
                    )
                return BacktestPollObservation("pending", None, "RUNNING", None)

        client = PendingClient(
            *(self._accepted(f"simulation-{i}") for i in range(1, 4))
        )
        observed = datetime.fromisoformat("2026-08-30T00:01:00+08:00")
        for _ in range(30):
            advance_automated_run(
                self.database_path, client, run_id, observed_at=observed.isoformat()
            )
            observed += timedelta(seconds=2)
            if client.calls.count("submit") == 3:
                break
        self.assertEqual(client.calls.count("submit"), 3)
        client.reject_poll = True
        for _ in range(40):
            advanced = advance_automated_run(
                self.database_path, client, run_id, observed_at=observed.isoformat()
            )
            observed += timedelta(seconds=5)
            if advanced.run.status == "failed":
                break
        self.assertEqual(advanced.run.status, "failed")
        calls_before_cleanup = tuple(client.calls)

        def wait(seconds):
            nonlocal observed
            observed += timedelta(seconds=seconds)

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=lambda: observed,
            waiter=wait,
        )
        self.assertEqual(completion.run.current_cycle, 0)
        self.assertEqual(tuple(client.calls), calls_before_cleanup + ("poll",))
        with open_database(self.database_path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM automated_cycle_settlements WHERE run_id=?",
                    (run_id,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM backtest_tasks WHERE status = 'pending'",
                ).fetchone()[0],
                3,
            )

    def test_candidate_exhaustion_waits_for_existing_slow_task(self):
        run_id = self._prepare_run(
            max_cycles=-1, max_backtests=0, max_pending_seconds=5
        )
        client = DriverClient(self._accepted())
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id, ("rank(close)",), created_at="2026-08-30T00:01:30+08:00"
        )
        self._advance(client, minute=2)
        rejected = tuple(
            ExplorationCandidate(self._candidate(formula), seed=seed)
            for seed, formula in enumerate(
                ("close/open", "open/volume", "volume/close")
            )
        )
        with patch(
            "execution.cycle_candidates.generate_exploration_batch",
            return_value=ExplorationBatch(
                candidates=rejected,
                attempted_seed_count=3,
                exclusions=(),
                shortfall=None,
            ),
        ):
            for minute in range(3, 15):
                advanced = self._advance(client, minute=minute)
                if advanced.run.status == "completed":
                    break
        self.assertEqual(advanced.run.status, "completed")
        self.assertEqual(advanced.run.current_cycle, 1)
        self.assertEqual(advanced.run.stop_reason, "selection_rejection_shortfall")
        self.assertEqual(client.calls.count("submit"), 1)
        self.assertEqual(client.calls.count("yearly_stats"), 1)
        with open_database(self.database_path) as connection:
            self.assertEqual(
                connection.execute("SELECT status FROM backtest_tasks").fetchone()[0],
                "completed",
            )

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.database_path = root / "driver.sqlite3"
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
            initialize_database_schema(connection)
        self._initialize_catalog()

    def test_connects_start_plan_requests_and_settlement(self) -> None:
        self._prepare_run()
        client = DriverClient(self._accepted())

        advances = tuple(self._advance(client, minute=minute) for minute in range(1, 9))

        self.assertEqual(
            tuple(item.action for item in advances),
            (
                "run_started",
                "cycle_planned",
                "backtest_advanced",
                "backtest_advanced",
                "backtest_advanced",
                "backtest_advanced",
                "backtest_advanced",
                "run_stopped",
            ),
        )
        self.assertEqual(
            tuple(item.backtest_action for item in advances[2:]),
            (
                "submitted",
                "detail_ready",
                "detail_captured",
                "completed",
                "submission_check_observed",
                "cycle_terminal",
            ),
        )
        self.assertEqual(client.calls, ["submit", "poll", "detail", "yearly_stats", "formal_check"])
        self.assertEqual(advances[-1].run.status, "completed")
        self.assertEqual(advances[-1].run.current_cycle, 1)

    def test_retry_wait_survives_database_reloads_and_resets_on_success(self) -> None:
        run_id = self._prepare_run(max_request_failures=3)
        client = DriverClient(
            WorldQuantRequestError(
                "worldquant_submission_http_error",
                status_code=429,
                retryable=True,
                outcome_unknown=False,
                retry_after_seconds=120,
            ),
            self._accepted(),
        )
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)",),
            created_at="2026-08-30T00:01:30+08:00",
        )

        scheduled = self._advance(client, minute=2)
        waiting = advance_automated_run(
            self.database_path,
            client,
            run_id,
            observed_at="2026-08-30T00:02:05+08:00",
        )
        submitted = self._advance(client, minute=4)

        self.assertEqual(scheduled.backtest_action, "submission_rate_limited")
        self.assertEqual(scheduled.retry_after_seconds, 120)
        self.assertEqual(scheduled.run.request_failure_count, 1)
        self.assertEqual(waiting.backtest_action, "submission_cooldown")
        self.assertEqual(waiting.retry_after_seconds, 115)
        self.assertEqual(submitted.backtest_action, "submitted")
        self.assertEqual(submitted.run.request_failure_count, 0)
        self.assertIsNone(submitted.run.retry_not_before)
        self.assertEqual(
            submitted.run.last_request_failure_code,
            "worldquant_submission_http_error",
        )
        self.assertEqual(submitted.run.last_request_status_code, 429)
        self.assertEqual(
            submitted.run.last_request_failure_at,
            "2026-08-30T00:02:00+08:00",
        )
        self.assertEqual(submitted.run.last_request_retry_after_seconds, 120)
        self.assertEqual(client.calls, ["submit", "submit"])

    def test_rate_limit_drains_peers_and_exhausts_only_the_rejected_formula(self):
        run_id = self._prepare_run(backtest_count=3, max_backtests=3, max_request_failures=3)
        error = WorldQuantRequestError("worldquant_submission_http_error", status_code=429,
                                      retryable=True, outcome_unknown=False)
        client = DriverClient(self._accepted("healthy"), error, error, error, error,
                              self._accepted("next"))
        self._advance(client, minute=1)
        self._prepare_backtests(run_id, ("rank(close)", "rank(open)", "rank(volume)"),
                                created_at="2026-08-30T00:01:30+08:00")
        first = self._advance(client, minute=2)
        limited = self._advance(client, minute=3)
        target = limited.task_id
        with patch.object(client, "poll_backtest", side_effect=WorldQuantRequestError(
                "worldquant_poll_http_error", status_code=500, retryable=True,
                outcome_unknown=False, retry_after_seconds=5)):
            deferred = advance_automated_run(self.database_path, client, run_id,
                observed_at="2026-08-30T00:03:01+08:00")
        self.assertEqual(deferred.backtest_action, "poll_retry_scheduled")
        self.assertEqual(deferred.run.request_failure_count, 1)
        self.assertEqual(deferred.run.retry_not_before, "2026-08-30T00:05:00+08:00")
        for second in (10, 20, 30):
            drained = advance_automated_run(self.database_path, client, run_id,
                observed_at=f"2026-08-30T00:03:{second}+08:00")
            self.assertEqual(drained.run.request_failure_count, 1)
            self.assertEqual(drained.run.status, "running")
        self.assertEqual(client.calls, ["submit", "submit", "poll", "detail", "yearly_stats"])
        for minute in (5, 7):
            retry = self._advance(client, minute=minute)
            self.assertEqual(retry.task_id, target)
            self.assertEqual(retry.run.status, "running")
        exhausted = self._advance(client, minute=9)
        self.assertEqual(exhausted.task_id, target)
        self.assertEqual(exhausted.backtest_action, "failed")
        self.assertEqual(exhausted.run.status, "running")
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, first.task_id).task.status, "completed")
            rejected = get_backtest_task(connection, target)
            self.assertEqual(rejected.task.failure_code, "platform_submission_rate_limit_exhausted")
            self.assertIsNone(rejected.task.submission_started_at)
            self.assertIsNone(rejected.task.remote_id)
        self.assertEqual(self._advance(client, minute=10).backtest_action, "submission_cooldown")
        self.assertEqual(self._advance(client, minute=11).backtest_action, "submitted")
        self.assertEqual(client.calls.count("submit"), 6)

    def test_last_rate_limited_formula_can_settle_the_batch(self):
        run_id = self._prepare_run()
        error = WorldQuantRequestError("worldquant_submission_http_error", status_code=429,
                                      retryable=True, outcome_unknown=False)
        client = DriverClient(error, error, error, error)
        self._advance(client, minute=1)
        self._prepare_backtests(run_id, ("rank(close)",), created_at="2026-08-30T00:01:30+08:00")
        for minute in (2, 4, 6, 8):
            self.assertEqual(self._advance(client, minute=minute).run.status, "running")
        settled = self._advance(client, minute=9)
        self.assertEqual(settled.run.current_cycle, 1)
        self.assertEqual(settled.run.stop_reason, "final_cycle_failed")
        self.assertEqual(client.calls.count("submit"), 4)

    def test_poll_rate_limit_does_not_stop_or_fail_the_task(self):
        run_id = self._prepare_run()
        client = DriverClient(self._accepted())
        self._advance(client, minute=1)
        self._prepare_backtests(run_id, ("rank(close)",), created_at="2026-08-30T00:01:30+08:00")
        self._advance(client, minute=2)
        with patch.object(client, "poll_backtest", side_effect=WorldQuantRequestError(
                "worldquant_poll_http_error", status_code=429, retryable=True,
                outcome_unknown=False, retry_after_seconds=300)):
            result = self._advance(client, minute=3)
        self.assertEqual(result.backtest_action, "poll_rate_limited")
        self.assertEqual(result.run.status, "running")
        self.assertEqual(result.run.request_failure_count, 0)
        with open_database(self.database_path) as connection:
            task = get_backtest_task(connection, result.task_id).task
            self.assertEqual(task.status, "pending")
            self.assertEqual(task.retry_not_before, "2026-08-30T00:08:00+08:00")
        self.assertEqual(self._advance(client, minute=4).backtest_action, "retry_wait")
        self.assertEqual(self._advance(client, minute=8).backtest_action, "detail_ready")

    def test_continuous_run_stops_at_the_backtest_limit(self) -> None:
        self._prepare_run(max_cycles=-1, max_backtests=2)
        client = DriverClient(
            self._accepted("simulation-1"),
            self._accepted("simulation-2"),
        )

        advances = tuple(
            self._advance(client, minute=minute) for minute in range(1, 16)
        )

        self.assertEqual(advances[7].action, "cycle_settled")
        self.assertEqual(advances[7].run.current_cycle, 1)
        self.assertEqual(advances[8].action, "cycle_planned")
        self.assertEqual(advances[-1].action, "run_stopped")
        self.assertEqual(advances[-1].run.current_cycle, 2)
        self.assertEqual(advances[-1].run.status, "completed")
        self.assertEqual(advances[-1].run.stop_reason, "backtest_limit_reached")
        self.assertEqual(client.calls.count("submit"), 2)
        self.assertEqual(client.calls.count("poll"), 2)
        self.assertEqual(client.calls.count("yearly_stats"), 2)
        self.assertEqual(client.calls.count("detail"), 2)

    def test_unlimited_run_keeps_planning_after_three_cycles(self) -> None:
        self._prepare_run(max_cycles=-1, max_backtests=0)
        client = DriverClient(
            self._accepted("simulation-1"),
            self._accepted("simulation-2"),
            self._accepted("simulation-3"),
        )

        advances = tuple(
            self._advance(client, minute=minute) for minute in range(1, 24)
        )

        self.assertEqual(advances[-2].action, "cycle_settled")
        self.assertEqual(advances[-2].run.current_cycle, 3)
        self.assertEqual(advances[-2].run.status, "running")
        self.assertIsNone(advances[-2].run.stop_reason)
        self.assertEqual(advances[-1].action, "cycle_planned")
        self.assertEqual(advances[-1].cycle_number, 4)
        self.assertEqual(client.calls.count("submit"), 3)

    def test_batch_check_retry_finishes_before_next_batch_is_planned(self):
        self._prepare_run(max_cycles=2, max_backtests=2)

        class DeferredCheckClient(DriverClient):
            check_reads = 0

            def fetch_formal_submission_check(self, *, platform_alpha_id):
                self.check_reads += 1
                if self.check_reads == 1:
                    self.calls.append("formal_check")
                    return FormalCheckObservation({}, 120)
                return super().fetch_formal_submission_check(platform_alpha_id=platform_alpha_id)

        client = DeferredCheckClient(self._accepted("simulation-1"), self._accepted("simulation-2"))
        waited = False
        second_planned = False
        for minute in range(1, 40):
            advanced = self._advance(client, minute=minute)
            if client.check_reads == 1:
                self.assertEqual(client.calls.count("submit"), 1)
                self.assertNotEqual(advanced.action, "cycle_planned")
            if advanced.backtest_action == "submission_check_wait":
                waited = True
                self.assertGreater(advanced.retry_after_seconds, 0)
            if advanced.action == "cycle_planned" and advanced.cycle_number == 2:
                second_planned = True
                self.assertEqual(client.check_reads, 2)
                with open_database(self.database_path) as connection:
                    self.assertEqual(len(list_formal_submission_queue(connection)), 1)
            if advanced.run.status == "completed":
                break
        self.assertTrue(waited)
        self.assertTrue(second_planned)
        self.assertEqual(advanced.run.status, "completed")
        self.assertEqual(client.calls.count("submit"), 2)

    def test_unlimited_run_records_failed_cycles_and_keeps_planning(self) -> None:
        self._prepare_run(max_cycles=-1, max_backtests=0)
        client = FailedBacktestClient(
            *[self._accepted(f"simulation-{i}") for i in range(3)]
        )
        for minute in range(1, 40):
            advanced = self._advance(client, minute=minute)
            self.assertEqual(advanced.run.status, "running")
            if advanced.run.current_cycle == 3 and advanced.action == "cycle_planned":
                break
        self.assertEqual(advanced.cycle_number, 4)
        self.assertEqual(advanced.run.consecutive_failures, 3)
        self.assertIsNone(advanced.run.stop_reason)

    def test_unlimited_poll_error_does_not_starve_a_healthy_peer(self) -> None:
        run_id = self._prepare_run(
            max_cycles=-1,
            max_backtests=0,
            backtest_count=2,
            max_pending_seconds=30,
        )

        class OneUnavailableSimulation(DriverClient):
            def poll_backtest(self, remote_id):
                if remote_id.endswith("simulation-1"):
                    self.calls.append("poll_failed")
                    raise WorldQuantRequestError(
                        "worldquant_poll_request_failed",
                        retryable=True,
                        outcome_unknown=False,
                    )
                return super().poll_backtest(remote_id)

        client = OneUnavailableSimulation(
            *(self._accepted(f"simulation-{i}") for i in range(1, 15)),
        )
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)", "rank(open)"),
            created_at="2026-08-30T00:01:30+08:00",
        )
        for minute in range(2, 35):
            advanced = self._advance(client, minute=minute)
            self.assertEqual(advanced.run.status, "running")
            if client.calls.count("yearly_stats"):
                break
        with open_database(self.database_path) as connection:
            statuses = [
                row[0]
                for row in connection.execute(
                    "SELECT status FROM backtest_tasks",
                )
            ]
        self.assertIn("completed", statuses)
        self.assertIn("failed", statuses)
        with open_database(self.database_path) as connection:
            self.assertEqual(connection.execute(
                "SELECT failure_code FROM backtest_tasks WHERE status='failed'",
            ).fetchone()[0], "platform_pending_timeout")
        self.assertGreaterEqual(client.calls.count("submit"), 2)
        self.assertGreaterEqual(client.calls.count("yearly_stats"), 1)

    def test_run_automatically_advances_until_max_cycles(self) -> None:
        run_id = self._prepare_run(max_cycles=2, max_backtests=2)
        client = DriverClient(
            self._accepted("simulation-1"),
            self._accepted("simulation-2"),
        )

        for minute in range(1, 20):
            advance = self._advance(client, minute=minute)
            if advance.run.status != "running":
                break

        self.assertEqual(advance.run.status, "completed")
        self.assertEqual(advance.run.current_cycle, 2)
        self.assertEqual(advance.run.stop_reason, "max_cycles_reached")
        with open_database(self.database_path) as connection:
            task_cycles = tuple(
                connection.execute(
                    """
                    SELECT cycle_number, COUNT(*)
                    FROM automated_run_backtests
                    WHERE run_id = ?
                    GROUP BY cycle_number
                    ORDER BY cycle_number
                    """,
                    (run_id,),
                ).fetchall()
            )
            settlement_cycles = tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT cycle_number
                    FROM automated_cycle_settlements
                    WHERE run_id = ?
                    ORDER BY cycle_number
                    """,
                    (run_id,),
                )
            )
        self.assertEqual(
            tuple((row[0], row[1]) for row in task_cycles),
            ((1, 1), (2, 1)),
        )
        self.assertEqual(settlement_cycles, (1, 2))

    def test_expired_task_is_read_before_timeout_decision(self) -> None:
        run_id = self._prepare_run(max_pending_seconds=60)
        client = DriverClient(self._accepted())
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)",),
            created_at="2026-08-30T00:01:30+08:00",
        )

        submitted = self._advance(client, minute=2)
        timed_out = self._advance(client, minute=3)

        self.assertEqual(submitted.backtest_action, "submitted")
        self.assertEqual(timed_out.action, "backtest_advanced")
        self.assertEqual(timed_out.backtest_action, "detail_ready")
        self.assertEqual(timed_out.run.status, "running")
        self.assertIsNone(timed_out.run.stop_reason)
        self.assertEqual(client.calls, ["submit", "poll"])
        with open_database(self.database_path) as connection:
            task = get_backtest_task(connection, timed_out.task_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.task.status, "pending")
        self.assertIsNone(task.task.failure_code)

    def test_generation_attempt_exhaustion_does_not_leave_an_active_run(
        self,
    ) -> None:
        self._prepare_run(exploration_seed_attempt_multiplier=2)
        with open_database(self.database_path) as connection:
            replace_operator_outputs(
                connection,
                (
                    OperatorOutputRecord("rank", "group"),
                    OperatorOutputRecord("ts_rank", "group"),
                ),
            )
        client = DriverClient()
        self._advance(client, minute=1)

        stopped = self._advance(client, minute=2)

        self.assertEqual(stopped.action, "run_stopped")
        self.assertEqual(stopped.run.status, "completed")
        self.assertEqual(
            stopped.run.stop_reason,
            "generation_attempt_budget_exhausted",
        )
        self.assertEqual(client.calls, [])
        diagnostic = CandidatePlanningDiagnostic.from_canonical_json(
            stopped.run.candidate_planning_stop_diagnostic_json,
            stop_reason=stopped.run.stop_reason,
        )
        self.assertEqual(diagnostic.cycle_number, 1)
        self.assertEqual(diagnostic.exploration_generation_target_count, 3)
        self.assertEqual(diagnostic.exploration_backtest_target_count, 1)
        self.assertEqual(diagnostic.seed_attempt_limit, 6)
        self.assertEqual(diagnostic.attempted_seed_count, 6)
        self.assertEqual(diagnostic.generated_candidate_count, 0)
        self.assertIsNone(diagnostic.selected_candidate_count)
        self.assertTrue(diagnostic.generation_exclusions)
        with open_database(self.database_path) as connection:
            task_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_tasks"
            ).fetchone()[0]
        self.assertEqual(task_count, 0)

    def test_selection_shortfall_stops_without_tasks_or_platform_calls(self) -> None:
        self._prepare_run()
        client = DriverClient()
        self._advance(client, minute=1)
        rejected = tuple(
            ExplorationCandidate(self._candidate(formula), seed=seed)
            for seed, formula in enumerate(
                ("close/open", "open/volume", "volume/close")
            )
        )

        with patch(
            "execution.cycle_candidates.generate_exploration_batch",
            return_value=ExplorationBatch(
                candidates=rejected,
                attempted_seed_count=3,
                exclusions=(),
                shortfall=None,
            ),
        ):
            stopped = self._advance(client, minute=2)

        self.assertEqual(stopped.action, "run_stopped")
        self.assertEqual(stopped.run.status, "completed")
        self.assertEqual(
            stopped.run.stop_reason,
            "selection_rejection_shortfall",
        )
        self.assertEqual(client.calls, [])
        diagnostic = CandidatePlanningDiagnostic.from_canonical_json(
            stopped.run.candidate_planning_stop_diagnostic_json,
            stop_reason=stopped.run.stop_reason,
        )
        self.assertEqual(diagnostic.cycle_number, 1)
        self.assertEqual(diagnostic.seed_attempt_limit, 12)
        self.assertEqual(diagnostic.attempted_seed_count, 3)
        self.assertEqual(diagnostic.generated_candidate_count, 3)
        self.assertEqual(diagnostic.selected_candidate_count, 0)
        self.assertTrue(diagnostic.selection_rejections)
        with open_database(self.database_path) as connection:
            task_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_tasks"
            ).fetchone()[0]
        self.assertEqual(task_count, 0)

    def test_missing_recovery_pnl_does_not_stop_ordinary_backtests(self) -> None:
        self._prepare_run()
        client = DriverClient(self._accepted())
        self._advance(client, minute=1)
        comparison = RecoveryComparison("child-task", "child-task", "parent-task",
            "group-account", "missing-child", "parent-alpha", "parent-alpha")
        with open_database(self.database_path) as connection:
            save_pnl_series(connection, PnlSeriesRecord("group-account", "parent-alpha",
                "2026-08-30T00:01:00+08:00", (("2020-01-01", 0.0),)))
        error = WorldQuantRequestError("worldquant_pnl_http_error", status_code=404,
                                      retryable=False, outcome_unknown=False)
        with (
            patch("execution.recovery.load_recovery_comparisons", return_value=(comparison,)),
            patch.object(client, "fetch_pnl", side_effect=error) as fetch_pnl,
        ):
            deferred = self._advance(client, minute=2)
            self.assertEqual(deferred.run.status, "running")
            self.assertEqual(deferred.run.request_failure_count, 0)
            self.assertEqual(deferred.retry_after_seconds, 0.0)
            with open_database(self.database_path) as connection:
                pending = next(r for r in list_pnl_series(connection)
                               if r.platform_alpha_id == "missing-child")
            self.assertIsNone(pending.points)
            self.assertEqual(pending.retry_not_before, "2026-08-30T01:02:00+08:00")
            planned = self._advance(client, minute=3)
            self.assertEqual(planned.action, "cycle_planned")
            advanced = self._advance(client, minute=4)
            self.assertEqual(advanced.run.status, "running")
            self.assertIn("submit", client.calls)
            fetch_pnl.assert_called_once_with(platform_alpha_id="missing-child")
            self.assertNotIn("formal_submit", client.calls)

    def test_request_failure_limit_stops_the_run(self) -> None:
        run_id = self._prepare_run(
            max_request_failures=2,
            automatic_submissions_enabled=True,
        )
        failure = WorldQuantRequestError(
            "temporary_platform_failure",
            retryable=True,
            outcome_unknown=False,
        )
        client = DriverClient(failure, failure)
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)",),
            created_at="2026-08-30T00:01:30+08:00",
        )

        first = self._advance(client, minute=2)
        stopped = self._advance(client, minute=3)

        self.assertEqual(first.action, "request_retry_scheduled")
        self.assertEqual(stopped.action, "run_stopped")
        self.assertEqual(stopped.run.status, "failed")
        self.assertEqual(stopped.run.request_failure_count, 2)
        self.assertEqual(stopped.run.stop_reason, "request_failure_limit_reached")
        self.assertEqual(client.calls, ["submit", "submit"])
        self.assertNotIn("formal_check", client.calls)
        self.assertNotIn("formal_submit", client.calls)

    def test_final_failed_cycle_does_not_start_formal_submission(self) -> None:
        run_id = self._prepare_run(automatic_submissions_enabled=True)
        client = FailedBacktestClient(self._accepted())
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)",),
            created_at="2026-08-30T00:01:30+08:00",
        )

        for minute in range(2, 8):
            advanced = self._advance(client, minute=minute)
            if advanced.run.status != "running":
                break

        self.assertEqual(advanced.run.status, "failed")
        self.assertEqual(advanced.run.stop_reason, "final_cycle_failed")
        self.assertNotIn("formal_check", client.calls)
        self.assertNotIn("formal_submit", client.calls)

    def test_authorized_formal_submission_requires_fresh_check_and_confirmation(
        self,
    ) -> None:
        run_id = self._prepare_run(automatic_submissions_enabled=True)
        client = DriverClient(self._accepted())
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)",),
            created_at="2026-08-30T00:01:30+08:00",
        )

        for minute in range(2, 20):
            advanced = self._advance(client, minute=minute)
            if advanced.run.status != "running":
                break

        self.assertEqual(advanced.run.status, "completed")
        self.assertEqual(
            client.calls,
            [
                "submit",
                "poll",
                "detail",
                "yearly_stats",
                "formal_check",
                "formal_confirmation",
                "formal_check",
                "formal_submit",
                "formal_confirmation",
            ],
        )
        with open_database(self.database_path) as connection:
            [link] = list(
                connection.execute(
                    "SELECT task_id FROM automated_run_backtests WHERE run_id = ?",
                    (run_id,),
                )
            )
            attempt = get_formal_submission_attempt(connection, link["task_id"])
            submitted = list_platform_submitted_alphas(
                connection,
                account_scope="group-account",
            )
        self.assertIsNotNone(attempt)
        assert attempt is not None
        self.assertEqual(attempt.status, "submitted")
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0].platform_alpha_id, "alpha-simulation-1")

    def test_each_cycle_submits_an_improving_family_low_to_high(
        self,
    ) -> None:
        run_id = self._prepare_run(
            max_cycles=2,
            max_backtests=2,
            automatic_submissions_enabled=True,
        )
        client = DriverClient(
            self._accepted("high"),
            self._accepted("low"),
            metrics_by_formula={
                "rank(close)": (1.3, 1.2),
                "rank(open)": (2.0, 1.1),
            },
        )
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)",),
            created_at="2026-08-30T00:01:30+08:00",
        )

        for minute in range(2, 20):
            advanced = self._advance(client, minute=minute)
            if advanced.run.current_cycle == 1:
                break

        self.assertEqual(advanced.run.status, "running")
        self.assertEqual(client.calls.count("formal_submit"), 1)
        parent_task_id = self._task_id_for_formula(run_id, "rank(close)")
        self._prepare_backtests(
            run_id,
            ("rank(open)",),
            created_at="2026-08-30T00:09:30+08:00",
        )
        child_task_id = self._task_id_for_formula(run_id, "rank(open)")
        self._record_test_lineage(
            child_task_id=child_task_id,
            parent_task_id=parent_task_id,
        )

        for minute in range(10, 35):
            advanced = self._advance(client, minute=minute)
            if advanced.run.status != "running":
                break

        self.assertEqual(advanced.run.status, "completed")
        self.assertEqual(
            tuple(
                client.formulas[alpha_id] for alpha_id in client.formal_submission_ids
            ),
            ("rank(close)", "rank(open)"),
        )
        with open_database(self.database_path) as connection:
            child_attempt = get_formal_submission_attempt(connection, child_task_id)
            parent_attempt = get_formal_submission_attempt(connection, parent_task_id)
        self.assertIsNotNone(child_attempt)
        self.assertIsNotNone(parent_attempt)
        assert child_attempt is not None
        assert parent_attempt is not None
        self.assertEqual(child_attempt.cycle_number, 2)
        self.assertEqual(parent_attempt.cycle_number, 1)

    def test_automatic_submission_processes_the_entire_family_low_to_high(
        self,
    ) -> None:
        run_id = self._prepare_run(
            backtest_count=3,
            max_backtests=3,
            automatic_submissions_enabled=True,
        )
        formulas = ("rank(close)", "rank(open)", "rank(volume)")
        client = DriverClient(
            self._accepted("one"),
            self._accepted("two"),
            self._accepted("three"),
            metrics_by_formula={
                "rank(close)": (2.0, 1.1),
                "rank(open)": (1.2, 1.2),
                "rank(volume)": (1.6, 1.3),
            },
        )
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            formulas,
            created_at="2026-08-30T00:01:30+08:00",
        )
        root_task_id = self._task_id_for_formula(run_id, "rank(close)")
        for formula in ("rank(open)", "rank(volume)"):
            self._record_test_lineage(
                child_task_id=self._task_id_for_formula(run_id, formula),
                parent_task_id=root_task_id,
            )

        for minute in range(2, 40):
            advanced = self._advance(client, minute=minute)
            if advanced.run.status != "running":
                break

        self.assertEqual(advanced.run.status, "completed")
        self.assertEqual(
            tuple(
                client.formulas[alpha_id] for alpha_id in client.formal_submission_ids
            ),
            ("rank(open)", "rank(volume)", "rank(close)"),
        )
        self.assertEqual(client.calls.count("formal_submit"), 3)

    def test_candidate_planning_stop_finishes_formal_submissions_first(
        self,
    ) -> None:
        run_id = self._prepare_run(
            max_cycles=-1,
            max_backtests=2,
            automatic_submissions_enabled=True,
        )
        client = DriverClient(self._accepted())
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)",),
            created_at="2026-08-30T00:01:30+08:00",
        )
        for minute in range(2, 10):
            advanced = self._advance(client, minute=minute)
            if advanced.run.current_cycle == 1:
                break

        diagnostic = CandidatePlanningDiagnostic(
            cycle_number=2,
            exploration_generation_target_count=3,
            exploration_backtest_target_count=1,
            seed_attempt_limit=12,
            attempted_seed_count=12,
            generated_candidate_count=0,
            selected_candidate_count=None,
            generation_exclusions=(
                CandidatePlanningExclusionCount("test_exhaustion", 12),
            ),
            selection_rejections=(),
        )
        planning_stop = ExplorationAttemptBudgetExhausted(
            "test generation exhausted",
            diagnostic=diagnostic,
        )
        with patch(
            "execution.driver.plan_automated_cycle",
            side_effect=planning_stop,
        ):
            for minute in range(10, 25):
                advanced = self._advance(client, minute=minute)
                if advanced.run.status != "running":
                    break

        self.assertEqual(advanced.run.status, "completed")
        self.assertEqual(
            advanced.run.stop_reason,
            "generation_attempt_budget_exhausted",
        )
        self.assertEqual(
            advanced.run.candidate_planning_stop_diagnostic_json,
            diagnostic.canonical_json(
                stop_reason="generation_attempt_budget_exhausted"
            ),
        )
        self.assertEqual(client.calls.count("formal_submit"), 1)

    def test_unknown_formal_submission_is_never_posted_twice(self) -> None:
        run_id = self._prepare_run(
            automatic_submissions_enabled=True,
            max_pending_seconds=3600,
        )
        client = DriverClient(
            self._accepted(),
            formal_submit_error=WorldQuantRequestError(
                "connection_lost",
                retryable=True,
                outcome_unknown=True,
            ),
            confirmation_status="UNSUBMITTED",
        )
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)",),
            created_at="2026-08-30T00:01:30+08:00",
        )

        for minute in range(2, 11):
            self._advance(client, minute=minute)

        self.assertEqual(client.calls.count("formal_submit"), 1)
        self.assertGreaterEqual(client.calls.count("formal_confirmation"), 1)
        with open_database(self.database_path) as connection:
            task_id = connection.execute(
                "SELECT task_id FROM automated_run_backtests WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            attempt = get_formal_submission_attempt(connection, task_id)
        self.assertIsNotNone(attempt)
        assert attempt is not None
        self.assertEqual(attempt.status, "submission_unknown")

    def test_concurrent_advance_does_not_reconcile_an_active_formal_post(
        self,
    ) -> None:
        run_id = self._prepare_run(automatic_submissions_enabled=True)
        request_started = threading.Event()
        release_response = threading.Event()
        client = BlockingFormalSubmissionClient(
            self._accepted(),
            request_started=request_started,
            release_response=release_response,
        )
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)",),
            created_at="2026-08-30T00:01:30+08:00",
        )
        for minute in range(2, 10):
            self._advance(client, minute=minute)

        with ThreadPoolExecutor(max_workers=1) as executor:
            posting = executor.submit(self._advance, client, minute=10)
            self.assertTrue(request_started.wait(timeout=5))
            concurrent = self._advance(client, minute=10)
            release_response.set()
            posted = posting.result(timeout=5)

        self.assertEqual(
            concurrent.formal_submission_action,
            "formal_submission_in_progress",
        )
        self.assertFalse(concurrent.platform_request_performed)
        self.assertEqual(
            posted.formal_submission_action,
            "formal_submission_confirmation_pending",
        )
        self.assertEqual(client.calls.count("formal_submit"), 1)
        self.assertEqual(client.calls.count("formal_confirmation"), 1)

    def test_late_successful_confirmation_resumes_after_concurrent_timeout(
        self,
    ) -> None:
        run_id = self._prepare_run(
            backtest_count=2,
            max_backtests=2,
            max_pending_seconds=3600,
            automatic_submissions_enabled=True,
        )
        request_started = threading.Event()
        release_response = threading.Event()
        client = BlockingFormalConfirmationClient(
            self._accepted("high"),
            self._accepted("low"),
            request_started=request_started,
            release_response=release_response,
            metrics_by_formula={
                "rank(close)": (1.6, 1.1),
                "rank(open)": (1.3, 1.2),
            },
        )
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)", "rank(open)"),
            created_at="2026-08-30T00:01:30+08:00",
        )
        high_task_id = self._task_id_for_formula(run_id, "rank(close)")
        low_task_id = self._task_id_for_formula(run_id, "rank(open)")
        self._record_test_lineage(
            child_task_id=low_task_id,
            parent_task_id=high_task_id,
        )
        for minute in range(2, 25):
            advanced = self._advance(client, minute=minute)
            if (
                advanced.formal_submission_action
                == "formal_submission_confirmation_pending"
            ):
                break

        with ThreadPoolExecutor(max_workers=1) as executor:
            confirming = executor.submit(self._advance, client, minute=25)
            try:
                self.assertTrue(request_started.wait(timeout=5))
                timed_out = advance_automated_run(
                    self.database_path,
                    client,
                    run_id,
                    observed_at="2026-08-30T02:00:00+08:00",
                )
            finally:
                release_response.set()
            confirmed = confirming.result(timeout=5)

        self.assertEqual(timed_out.run.status, "failed")
        self.assertEqual(
            timed_out.run.stop_reason,
            "submission_reconciliation_required",
        )
        self.assertEqual(confirmed.run.status, "failed")
        with open_database(self.database_path) as connection:
            low_attempt = get_formal_submission_attempt(connection, low_task_id)
        self.assertIsNotNone(low_attempt)
        assert low_attempt is not None
        self.assertEqual(low_attempt.status, "submitted")

        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=lambda: datetime.fromisoformat("2026-08-30T02:01:00+08:00"),
            waiter=lambda _seconds: None,
        )

        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(
            tuple(
                client.formulas[alpha_id] for alpha_id in client.formal_submission_ids
            ),
            ("rank(open)", "rank(close)"),
        )
        self.assertEqual(client.calls.count("formal_submit"), 2)

    def test_stale_formal_check_is_refreshed_before_any_post(self) -> None:
        run_id = self._prepare_run(automatic_submissions_enabled=True)
        client = DriverClient(self._accepted())
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)",),
            created_at="2026-08-30T00:01:30+08:00",
        )
        for minute in range(2, 10):
            self._advance(client, minute=minute)

        stale = self._advance(client, minute=15)
        refreshed = self._advance(client, minute=14)

        self.assertEqual(
            stale.formal_submission_action,
            "formal_submission_check_expired",
        )
        self.assertFalse(stale.platform_request_performed)
        self.assertEqual(
            refreshed.formal_submission_action,
            "formal_submission_check_passed",
        )
        self.assertEqual(client.calls.count("formal_check"), 3)
        self.assertEqual(client.calls.count("formal_submit"), 0)

    def test_unknown_formal_submission_blocks_new_run_and_resumes_by_read_only_check(
        self,
    ) -> None:
        run_id = self._prepare_run(
            automatic_submissions_enabled=True,
            max_pending_seconds=3600,
        )
        client = DriverClient(
            self._accepted(),
            formal_submit_error=WorldQuantRequestError(
                "connection_lost",
                retryable=True,
                outcome_unknown=True,
            ),
            confirmation_status="UNSUBMITTED",
        )
        self._advance(client, minute=1)
        self._prepare_backtests(
            run_id,
            ("rank(close)",),
            created_at="2026-08-30T00:01:30+08:00",
        )
        for minute in range(2, 11):
            self._advance(client, minute=minute)
        stopped = advance_automated_run(
            self.database_path,
            client,
            run_id,
            observed_at="2026-08-30T02:00:00+08:00",
        )

        self.assertEqual(stopped.run.status, "failed")
        self.assertEqual(
            stopped.run.stop_reason,
            "submission_reconciliation_required",
        )
        with self.assertRaisesRegex(
            ValueError,
            "automated_run_formal_submission_reconciliation_required",
        ):
            prepare_automated_run(
                self.database_path,
                self.settings_path,
                account_scope="group-account",
                limits=AutomatedRunLimits(
                    exploration_percent=30,
                    self_correlation_percent=30,
                    mutation_percent=40,
                    direction_validation_percent=3,
                    generation_count=3,
                    backtest_count=1,
                    max_cycles=1,
                    max_backtests=1,
                    max_pending_seconds=3600,
                    max_consecutive_failures=2,
                    max_request_failures=3,
                    max_in_flight_backtests=3,
                    exploration_seed_attempt_multiplier=4,
                    real_backtests_authorized=True,
                ),
                created_at="2026-08-30T01:09:00+08:00",
            )

        client.confirmation_status = "ACTIVE"
        completion = run_automated_run(
            self.database_path,
            client,
            run_id,
            clock=lambda: datetime.fromisoformat("2026-08-30T01:10:00+08:00"),
            waiter=lambda _seconds: None,
        )

        self.assertEqual(completion.run.status, "completed")
        self.assertEqual(completion.formal_submission_claimed_count, 1)
        self.assertEqual(completion.formal_submission_confirmed_count, 1)
        self.assertEqual(completion.formal_submission_unresolved_count, 0)
        self.assertEqual(client.calls.count("formal_submit"), 1)

    def _prepare_run(
        self,
        *,
        max_request_failures: int = 3,
        max_cycles: int = 1,
        max_backtests: int = 1,
        backtest_count: int = 1,
        max_pending_seconds: int = 3600,
        exploration_seed_attempt_multiplier: int = 4,
        automatic_submissions_enabled: bool = False,
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
                generation_count=max(3, backtest_count),
                backtest_count=backtest_count,
                max_cycles=max_cycles,
                max_backtests=max_backtests,
                max_pending_seconds=max_pending_seconds,
                max_consecutive_failures=2,
                max_request_failures=max_request_failures,
                max_in_flight_backtests=3,
                exploration_seed_attempt_multiplier=(
                    exploration_seed_attempt_multiplier
                ),
                real_backtests_authorized=True,
                automatic_submissions_enabled=automatic_submissions_enabled,
            ),
            created_at="2026-08-30T00:00:00+08:00",
        )
        self.run_id = run.run_id
        return run.run_id

    def _task_id_for_formula(self, run_id: str, formula: str) -> str:
        with open_database(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT tasks.task_id
                FROM backtest_tasks AS tasks
                JOIN automated_run_backtests AS links
                  ON links.task_id = tasks.task_id
                WHERE links.run_id = ? AND tasks.formula = ?
                """,
                (run_id, formula),
            ).fetchone()
        if row is None:
            raise AssertionError(f"missing test task for formula: {formula}")
        return row["task_id"]

    def _record_test_lineage(
        self,
        *,
        child_task_id: str,
        parent_task_id: str,
    ) -> None:
        with open_database(self.database_path) as connection:
            create_backtest_mutation(
                connection,
                BacktestMutationRecord(
                    child_task_id=child_task_id,
                    parent_task_id=parent_task_id,
                    action="test_mutation",
                    location="formula",
                    before="parent",
                    after="child",
                ),
            )

    def _prepare_backtests(
        self,
        run_id: str,
        formulas: tuple[str, ...],
        *,
        created_at: str,
    ) -> None:
        prepare_automated_candidate_backtest_batch(
            self.database_path,
            run_id=run_id,
            candidates=tuple(
                AutomatedCandidateBacktest(
                    self._candidate(formula),
                    self.settings,
                )
                for formula in formulas
            ),
            created_at=created_at,
        )

    @staticmethod
    def _candidate(formula: str) -> FormulaCandidate:
        return exploration_candidate(parse_formula(formula).expression)

    def _advance(self, client: DriverClient, *, minute: int):
        return advance_automated_run(
            self.database_path,
            client,
            self.run_id,
            observed_at=f"2026-08-30T00:{minute:02d}:00+08:00",
        )

    @staticmethod
    def _accepted(
        remote_name: str = "simulation-1",
    ) -> BacktestSubmissionObservation:
        return BacktestSubmissionObservation(
            "accepted",
            f"https://api.worldquantbrain.com/simulations/{remote_name}",
            None,
        )

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
