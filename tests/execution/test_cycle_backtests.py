from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.execution.catalog_fixture import initialize_test_generation_catalog
from execution.backtest_batches import (
    AutomatedCandidateBacktest,
    prepare_automated_candidate_backtest_batch,
)
from execution.cycle_backtests import (
    advance_automated_cycle_backtests,
    settle_automated_cycle,
)
from execution.real_backtests import advance_real_backtest, prepare_real_backtest
from execution.backtests import record_submission_unknown
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
from persistence.seeds import list_signal_seeds
from persistence.runs import (
    get_automated_cycle_settlement,
    get_automated_run,
    initialize_run_schema,
)
from persistence.submission_queue import initialize_submission_queue_schema
from persistence.submissions import initialize_submission_schema
from worldquant.backtests import (
    BacktestCheck,
    BacktestDetail,
    BacktestPollObservation,
    BacktestSettings,
    BacktestSubmissionObservation,
    BacktestYearlyStatsObservation,
    WorldQuantProtocolError,
)


def _checks(*items: tuple[str, str]) -> tuple[BacktestCheck, ...]:
    return tuple(
        BacktestCheck(
            name=name,
            status=status,
            threshold=None,
            actual=None,
            platform_date=None,
        )
        for name, status in sorted(items)
    )


class ScriptedClient:
    authenticated = True

    def __init__(self) -> None:
        self.submissions: list[BacktestSubmissionObservation | Exception] = []
        self.polls: dict[str, BacktestPollObservation] = {}
        self.details: dict[str, BacktestDetail] = {}
        self.calls: list[tuple[str, str]] = []

    def submit_backtest(self, *, formula, settings):
        self.calls.append(("submit", formula))
        result = self.submissions.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def poll_backtest(self, remote_id):
        self.calls.append(("poll", remote_id))
        return self.polls[remote_id]

    def fetch_backtest_detail(
        self,
        *,
        platform_alpha_id,
        expected_formula,
        expected_settings,
    ):
        self.calls.append(("detail", platform_alpha_id))
        return self.details[platform_alpha_id]

    def fetch_backtest_yearly_stats(self, *, platform_alpha_id):
        self.calls.append(("yearly_stats", platform_alpha_id))
        return BacktestYearlyStatsObservation(
            state="ready",
            stats=(),
            retry_after_seconds=None,
        )


class ConcurrentSubmissionClient(ScriptedClient):
    def __init__(
        self,
        barrier: threading.Barrier,
        calls: list[tuple[str, str]],
        calls_lock: threading.Lock,
        response: BacktestSubmissionObservation,
    ) -> None:
        super().__init__()
        self._barrier = barrier
        self._shared_calls = calls
        self._calls_lock = calls_lock
        self._response = response

    @property
    def authenticated(self) -> bool:
        self._barrier.wait(timeout=5)
        return True

    def submit_backtest(self, *, formula, settings):
        with self._calls_lock:
            self._shared_calls.append(("submit", formula))
        return self._response


class BlockingSubmissionClient(ScriptedClient):
    def __init__(
        self,
        request_started: threading.Event,
        release_response: threading.Event,
        response: BacktestSubmissionObservation,
    ) -> None:
        super().__init__()
        self._request_started = request_started
        self._release_response = release_response
        self._response = response

    def submit_backtest(self, *, formula, settings):
        self.calls.append(("submit", formula))
        self._request_started.set()
        if not self._release_response.wait(timeout=5):
            raise TimeoutError("test_submission_release_timeout")
        return self._response


class AutomatedCycleBacktestTests(unittest.TestCase):
    def test_settlement_archives_exhausted_parent_atomically_and_replay_is_idempotent(self):
        from tests.execution import test_qualified_archive as archive_fixture
        from execution.backtests import cancel_unsubmitted_backtest_task
        from persistence.qualified_archive import list_qualified_alpha_archive
        from persistence.runs import list_automated_run_backtests

        run_id = self._prepare_run(("rank(high)",))
        with open_database(self.database_path) as connection:
            parent = archive_fixture.exhausted_parent(connection)
            link = list_automated_run_backtests(connection, run_id)[0]
            cancel_unsubmitted_backtest_task(connection, link.task_id,
                                             observed_at=archive_fixture.FINISHED)
        with patch("execution.cycle_backtests.record_automated_cycle_settlement",
                   side_effect=RuntimeError("settlement failed")):
            with self.assertRaisesRegex(RuntimeError, "settlement failed"):
                settle_automated_cycle(self.database_path, run_id, cycle_number=1,
                                       observed_at=archive_fixture.ARCHIVED)
        with open_database(self.database_path) as connection:
            self.assertEqual(list_qualified_alpha_archive(connection), ())
            self.assertIsNone(get_automated_cycle_settlement(connection, run_id, 1))
        first = settle_automated_cycle(self.database_path, run_id, cycle_number=1,
                                        observed_at=archive_fixture.ARCHIVED)
        repeated = settle_automated_cycle(self.database_path, run_id, cycle_number=1,
                                           observed_at="2026-09-03T00:00:00+00:00")
        self.assertFalse(first.already_settled)
        self.assertTrue(repeated.already_settled)
        with open_database(self.database_path) as connection:
            archive = list_qualified_alpha_archive(connection)
            self.assertEqual(tuple(row.task_id for row in archive), (parent.task.task_id,))
            self.assertEqual(archive[0].archived_at, archive_fixture.ARCHIVED)

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.database_path = root / "cycle-backtests.sqlite3"
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

    def test_submits_the_batch_then_recovers_polling_and_settles(self) -> None:
        run_id = self._prepare_run(("rank(close)", "rank(open)"))
        client = ScriptedClient()
        client.submissions.extend(
            (
                self._accepted("simulation-1"),
                self._accepted("simulation-2"),
            )
        )
        client.polls.update(
            {
                self._remote("simulation-1"): BacktestPollObservation(
                    "completed", "alpha-1", "COMPLETE", None
                ),
                self._remote("simulation-2"): BacktestPollObservation(
                    "failed", None, "FAILED", None
                ),
            }
        )
        client.details["alpha-1"] = self._detail("alpha-1")

        first = self._advance(client, minute=2)
        second = self._advance(client, minute=3)
        third = self._advance(client, minute=4)
        fourth = self._advance(client, minute=5)
        fifth = self._advance(client, minute=6)
        sixth = self._advance(client, minute=7)
        terminal = self._advance(client, minute=8)
        settled = settle_automated_cycle(
            self.database_path,
            run_id,
            cycle_number=1,
            observed_at="2026-08-30T00:09:00+08:00",
        )
        repeated = settle_automated_cycle(
            self.database_path,
            run_id,
            cycle_number=1,
            observed_at="2026-08-30T00:10:00+08:00",
        )

        self.assertEqual((first.action, second.action), ("submitted", "submitted"))
        self.assertEqual(second.counts.pending, 2)
        self.assertEqual(third.action, "detail_ready")
        self.assertCountEqual(
            (fourth.action, fifth.action, sixth.action),
            ("detail_captured", "completed", "failed"),
        )
        self.assertEqual(terminal.action, "cycle_terminal")
        self.assertFalse(terminal.platform_request_performed)
        self.assertEqual(len(client.calls), 6)
        self.assertEqual(settled.outcome, "qualified")
        self.assertEqual(settled.passed_evaluations, 1)
        self.assertEqual(settled.failed_backtests, 1)
        self.assertEqual(settled.run.status, "completed")
        self.assertFalse(settled.already_settled)
        self.assertTrue(repeated.already_settled)

    def test_invalid_response_fails_one_task_and_frees_slot_for_peer(self):
        run_id = self._prepare_run(
            ("rank(close)", "rank(open)"), max_in_flight_backtests=1,
        )
        client = ScriptedClient()
        client.submissions.extend((self._accepted("bad"), self._accepted("good")))
        client.details["alpha-good"] = self._detail("alpha-good")

        def poll(remote_id):
            client.calls.append(("poll", remote_id))
            if remote_id == self._remote("bad"):
                raise WorldQuantProtocolError("worldquant_poll_status_unknown")
            return BacktestPollObservation("completed", "alpha-good", "COMPLETE", None)

        with patch.object(client, "poll_backtest", side_effect=poll):
            self.assertEqual(self._advance(client, minute=2).action, "submitted")
            failed = self._advance(client, minute=3)
            self.assertEqual(failed.action, "failed")
            self.assertEqual(failed.run_status, "running")
            self.assertEqual(failed.counts.pending, 0)
            self.assertEqual(self._advance(client, minute=4).action, "submitted")
            self.assertEqual(self._advance(client, minute=5).action, "detail_ready")
            self.assertEqual(self._advance(client, minute=6).action, "detail_captured")
            self.assertEqual(self._advance(client, minute=7).action, "completed")
        settled = settle_automated_cycle(
            self.database_path, run_id, cycle_number=1,
            observed_at="2026-08-30T00:08:00+08:00",
        )
        self.assertEqual(settled.completed_backtests, 1)
        self.assertEqual(settled.failed_backtests, 1)
        self.assertEqual(settled.run.status, "completed")
        self.assertEqual(client.calls.count(("poll", self._remote("bad"))), 3)
        self.assertEqual(len([c for c in client.calls if c[0] == "submit"]), 2)

    def test_unknown_submission_occupies_one_slot_without_blocking_peers(self) -> None:
        formulas = ("rank(close)", "rank(open)", "rank(high)")
        run_id = self._prepare_run(formulas)
        client = ScriptedClient()
        client.submissions.extend(
            (
                self._accepted("simulation-1"),
                BacktestSubmissionObservation("unknown", None, "missing_location"),
                self._accepted("simulation-3"),
            )
        )
        client.polls[self._remote("simulation-1")] = BacktestPollObservation(
            "failed", None, "FAILED", None
        )
        client.polls[self._remote("simulation-3")] = BacktestPollObservation(
            "failed", None, "FAILED", None
        )

        submitted = self._advance(client, minute=2)
        unknown = self._advance(client, minute=3)
        peer = self._advance(client, minute=4)
        self._advance(client, minute=5)
        recovered = self._advance(client, minute=6)
        stopped = self._advance(client, minute=7)

        with open_database(self.database_path) as connection:
            run = get_automated_run(connection, run_id)
            statuses = tuple(
                row["status"]
                for row in connection.execute(
                    "SELECT status FROM backtest_tasks ORDER BY task_id"
                )
            )
        self.assertEqual(submitted.action, "submitted")
        self.assertEqual(unknown.action, "reconciliation_required")
        self.assertEqual(unknown.run_status, "running")
        self.assertEqual(peer.action, "submitted")
        self.assertEqual(peer.counts.pending, 2)
        self.assertEqual(peer.counts.submission_unknown, 1)
        self.assertEqual(recovered.action, "failed")
        self.assertEqual(stopped.action, "reconciliation_required")
        self.assertFalse(stopped.platform_request_performed)
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.stop_reason, "submission_reconciliation_required")
        self.assertEqual(statuses.count("created"), 0)
        self.assertEqual(statuses.count("submission_unknown"), 1)
        self.assertEqual(statuses.count("failed"), 2)
        self.assertEqual(len(client.calls), 5)
        submitted_formulas = [formula for action, formula in client.calls if action == "submit"]
        self.assertEqual(len(set(submitted_formulas)), 3)

    def test_running_automated_run_blocks_unowned_single_submission(self) -> None:
        self._prepare_run(("rank(close)",))
        standalone = prepare_real_backtest(
            self.database_path,
            account_scope="group-account",
            formula="rank(open)",
            settings=self.settings,
            created_at="2026-08-30T00:01:31+08:00",
        )
        client = ScriptedClient()

        with self.assertRaisesRegex(
            ValueError,
            "real_backtest_automated_run_active",
        ):
            advance_real_backtest(
                self.database_path,
                client,
                standalone.task.task_id,
                observed_at="2026-08-30T00:02:00+08:00",
                allow_submission=True,
            )

        with open_database(self.database_path) as connection:
            persisted = get_backtest_task(
                connection,
                standalone.task.task_id,
            )
        self.assertEqual(persisted.task.status, "created")
        self.assertEqual(client.calls, [])

    def _historical_unknown(self, formula="-rank(close)", account_scope="group-account"):
        task = prepare_real_backtest(self.database_path, account_scope=account_scope,
            formula=formula, settings=self.settings, created_at="2026-08-29T00:00:00+08:00")
        with open_database(self.database_path) as connection:
            return record_submission_unknown(connection, task.task.task_id,
                observed_at="2026-08-29T00:01:00+08:00")

    def test_historical_unknown_uses_one_slot_and_does_not_starve_own_polls(self):
        old = self._historical_unknown()
        self._historical_unknown(account_scope="other-account")
        self._prepare_run(("rank(close)", "rank(open)", "rank(high)"))
        client = ScriptedClient()
        client.submissions.extend((self._accepted("one"), self._accepted("two")))
        client.polls[self._remote("one")] = BacktestPollObservation("pending", None, "RUNNING", 1)
        self.assertEqual(self._advance(client, minute=2).action, "submitted")
        self.assertEqual(self._advance(client, minute=3).action, "submitted")
        self.assertEqual(self._advance(client, minute=4).action, "pending")
        self.assertEqual(len([x for x in client.calls if x[0] == "submit"]), 2)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, old.task.task_id), old)

    def test_full_historical_capacity_waits_without_sending_or_failing(self):
        old = [self._historical_unknown(formula) for formula in ("-rank(close)", "-rank(open)", "-rank(high)")]
        self._prepare_run(("rank(close)",))
        client = ScriptedClient()
        for minute in (2, 3):
            advance = self._advance(client, minute=minute)
            self.assertEqual(advance.action, "capacity_wait")
            self.assertEqual(advance.retry_after_seconds, 60)
            self.assertEqual(advance.run_status, "running")
        self.assertEqual(client.calls, [])
        with open_database(self.database_path) as connection:
            self.assertEqual([get_backtest_task(connection, s.task.task_id) for s in old], old)

    def test_settlement_requires_every_task_to_be_terminal(self) -> None:
        run_id = self._prepare_run(("rank(close)",))

        with self.assertRaisesRegex(
            ValueError,
            "automated_cycle_tasks_not_terminal",
        ):
            settle_automated_cycle(
                self.database_path,
                run_id,
                cycle_number=1,
                observed_at="2026-08-30T00:02:00+08:00",
            )

    def test_pending_tasks_are_polled_in_oldest_observation_order(self) -> None:
        self._prepare_run(("rank(close)", "rank(open)"))
        client = ScriptedClient()
        client.submissions.extend(
            (self._accepted("simulation-1"), self._accepted("simulation-2"))
        )
        for name in ("simulation-1", "simulation-2"):
            client.polls[self._remote(name)] = BacktestPollObservation(
                "pending", None, "RUNNING", 0.5
            )
        self._advance(client, minute=2)
        self._advance(client, minute=3)

        self._advance(client, minute=4)
        self._advance(client, minute=5)

        self.assertEqual(
            client.calls[-2:],
            [
                ("poll", self._remote("simulation-1")),
                ("poll", self._remote("simulation-2")),
            ],
        )

    def test_in_flight_limit_polls_before_submitting_more_tasks(self) -> None:
        self._prepare_run(
            ("rank(close)", "rank(open)", "rank(high)"),
            max_in_flight_backtests=2,
        )
        client = ScriptedClient()
        client.submissions.extend(
            (self._accepted("simulation-1"), self._accepted("simulation-2"))
        )
        client.polls[self._remote("simulation-1")] = BacktestPollObservation(
            "pending", None, "RUNNING", 0.5
        )

        self._advance(client, minute=2)
        self._advance(client, minute=3)
        limited = self._advance(client, minute=4)

        self.assertEqual(limited.action, "pending")
        self.assertEqual(
            client.calls[-1],
            ("poll", self._remote("simulation-1")),
        )
        with open_database(self.database_path) as connection:
            statuses = tuple(
                row["status"]
                for row in connection.execute(
                    "SELECT status FROM backtest_tasks ORDER BY task_id"
                )
            )
        self.assertEqual(statuses.count("pending"), 2)
        self.assertEqual(statuses.count("created"), 1)

    def test_concurrent_advances_claim_one_submission_for_the_same_task(self) -> None:
        self._prepare_run(("rank(close)",), max_in_flight_backtests=1)
        with open_database(self.database_path) as connection:
            task_id = connection.execute(
                "SELECT task_id FROM backtest_tasks"
            ).fetchone()[0]
        barrier = threading.Barrier(2)
        calls: list[tuple[str, str]] = []
        calls_lock = threading.Lock()
        clients = tuple(
            ConcurrentSubmissionClient(
                barrier,
                calls,
                calls_lock,
                self._accepted(f"simulation-{index}"),
            )
            for index in (1, 2)
        )

        def advance(client: ConcurrentSubmissionClient):
            return advance_real_backtest(
                self.database_path,
                client,
                task_id,
                observed_at="2026-08-30T00:02:00+08:00",
                allow_submission=True,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(advance, clients))

        with open_database(self.database_path) as connection:
            persisted = get_backtest_task(connection, task_id)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            sum(item.platform_request_performed for item in results),
            1,
        )
        self.assertEqual(persisted.task.status, "pending")

    def test_concurrent_advances_cannot_exceed_run_in_flight_limit(self) -> None:
        self._historical_unknown()
        self._prepare_run(
            ("rank(close)", "rank(open)"),
            max_in_flight_backtests=2,
        )
        with open_database(self.database_path) as connection:
            task_ids = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT task_id FROM backtest_tasks WHERE status='created' ORDER BY task_id"
                )
            )
        barrier = threading.Barrier(2)
        calls: list[tuple[str, str]] = []
        calls_lock = threading.Lock()
        clients = tuple(
            ConcurrentSubmissionClient(
                barrier,
                calls,
                calls_lock,
                self._accepted(f"simulation-{index}"),
            )
            for index in (1, 2)
        )

        def advance(item):
            task_id, client = item
            return advance_real_backtest(
                self.database_path,
                client,
                task_id,
                observed_at="2026-08-30T00:02:00+08:00",
                allow_submission=True,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(advance, zip(task_ids, clients)))

        with open_database(self.database_path) as connection:
            statuses = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT status FROM backtest_tasks ORDER BY task_id"
                )
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            sorted(item.action for item in results),
            ["submission_deferred", "submitted"],
        )
        self.assertEqual(statuses.count("pending"), 1)
        self.assertEqual(statuses.count("created"), 1)

    def test_concurrent_cycle_advance_does_not_fail_an_active_submission(self) -> None:
        run_id = self._prepare_run(
            ("rank(close)",),
            max_in_flight_backtests=1,
        )
        request_started = threading.Event()
        release_response = threading.Event()
        self.addCleanup(release_response.set)
        submitting_client = BlockingSubmissionClient(
            request_started,
            release_response,
            self._accepted("simulation-1"),
        )

        with ThreadPoolExecutor(max_workers=1) as executor:
            owner = executor.submit(
                advance_automated_cycle_backtests,
                self.database_path,
                submitting_client,
                run_id,
                observed_at="2026-08-30T00:02:00+08:00",
            )
            self.assertTrue(request_started.wait(timeout=5))
            competing = advance_automated_cycle_backtests(
                self.database_path,
                ScriptedClient(),
                run_id,
                observed_at="2026-08-30T00:02:01+08:00",
            )
            release_response.set()
            submitted = owner.result(timeout=5)

        with open_database(self.database_path) as connection:
            run = get_automated_run(connection, run_id)
            task = connection.execute("SELECT status FROM backtest_tasks").fetchone()
        self.assertEqual(competing.action, "submission_in_progress")
        self.assertFalse(competing.platform_request_performed)
        self.assertEqual(competing.run_status, "running")
        self.assertEqual(submitted.action, "submitted")
        self.assertEqual(submitting_client.calls, [("submit", "rank(close)")])
        self.assertEqual(run.status, "running")
        self.assertEqual(task["status"], "pending")

    def test_quality_qualified_result_with_pending_sc_advances_frontier(self) -> None:
        run_id = self._prepare_run(("rank(close)",))
        client = ScriptedClient()
        client.submissions.append(self._accepted("simulation-1"))
        client.polls[self._remote("simulation-1")] = BacktestPollObservation(
            "completed", "alpha-1", "COMPLETE", None
        )
        client.details["alpha-1"] = BacktestDetail(
            platform_alpha_id="alpha-1",
            sharpe=1.3,
            fitness=1.1,
            turnover=0.12,
            returns=0.08,
            drawdown=0.04,
            margin=0.001,
            book_size=20_000_000,
            pnl=100_000,
            checks=_checks(
                ("LOW_SHARPE", "PASS"),
                ("LOW_FITNESS", "PASS"),
                ("LOW_TURNOVER", "PASS"),
                ("HIGH_TURNOVER", "PASS"),
                ("CONCENTRATED_WEIGHT", "PASS"),
                ("LOW_SUB_UNIVERSE_SHARPE", "PASS"),
                ("SELF_CORRELATION", "PENDING"),
                ("MATCHES_COMPETITION", "PASS"),
            ),
        )
        self._advance(client, minute=2)
        self._advance(client, minute=3)
        self._advance(client, minute=4)
        self._advance(client, minute=5)

        settled = settle_automated_cycle(
            self.database_path,
            run_id,
            cycle_number=1,
            observed_at="2026-08-30T00:05:00+08:00",
        )

        self.assertEqual(settled.outcome, "frontier_advanced")
        self.assertEqual(settled.passed_evaluations, 0)
        self.assertEqual(settled.pending_evaluations, 1)
        self.assertTrue(settled.frontier_advanced)

    def test_new_seed_advances_frontier_and_replays_saved_settlement(self) -> None:
        run_id = self._complete_seed_cycle()

        settled = settle_automated_cycle(
            self.database_path,
            run_id,
            cycle_number=1,
            observed_at="2026-08-30T00:05:00+08:00",
        )
        repeated = settle_automated_cycle(
            self.database_path,
            run_id,
            cycle_number=1,
            observed_at="2026-08-30T00:06:00+08:00",
        )
        with open_database(self.database_path) as connection:
            seeds = list_signal_seeds(connection)

        self.assertEqual(settled.outcome, "frontier_advanced")
        self.assertTrue(settled.frontier_advanced)
        self.assertEqual(settled.run.stop_reason, "max_cycles_reached")
        self.assertTrue(repeated.frontier_advanced)
        self.assertEqual(repeated.outcome, "frontier_advanced")
        self.assertTrue(repeated.already_settled)
        self.assertEqual(len(seeds), 1)

    def test_terminal_run_settles_late_cycle_without_changing_stop_reason(
        self,
    ) -> None:
        run_id = self._complete_seed_cycle()
        failed = fail_automated_run(
            self.database_path,
            run_id,
            failed_at="2026-08-30T00:05:00+08:00",
            reason="manual_stop",
        )

        settled = settle_automated_cycle(
            self.database_path,
            run_id,
            cycle_number=1,
            observed_at="2026-08-30T00:06:00+08:00",
        )
        repeated = settle_automated_cycle(
            self.database_path,
            run_id,
            cycle_number=1,
            observed_at="2026-08-30T00:07:00+08:00",
        )
        with open_database(self.database_path) as connection:
            seeds = list_signal_seeds(connection)

        self.assertEqual(failed.current_cycle, 0)
        self.assertEqual(settled.run.status, "failed")
        self.assertEqual(settled.run.stop_reason, "manual_stop")
        self.assertEqual(settled.run.current_cycle, 1)
        self.assertEqual(settled.outcome, "frontier_advanced")
        self.assertTrue(settled.frontier_advanced)
        self.assertEqual(len(seeds), 1)
        self.assertTrue(repeated.already_settled)
        self.assertEqual(repeated.run.stop_reason, "manual_stop")

    def test_late_cycle_settlement_rolls_back_seed_when_state_update_fails(
        self,
    ) -> None:
        run_id = self._complete_seed_cycle()
        fail_automated_run(
            self.database_path,
            run_id,
            failed_at="2026-08-30T00:05:00+08:00",
            reason="manual_stop",
        )

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_observed_at_invalid",
        ):
            settle_automated_cycle(
                self.database_path,
                run_id,
                cycle_number=1,
                observed_at="2026-08-30T00:00:30+08:00",
            )

        with open_database(self.database_path) as connection:
            run = get_automated_run(connection, run_id)
            settlement = get_automated_cycle_settlement(connection, run_id, 1)
            seeds = list_signal_seeds(connection)
        self.assertEqual(run.current_cycle, 0)
        self.assertIsNone(settlement)
        self.assertEqual(seeds, ())

    def test_all_platform_failures_mark_the_final_cycle_failed(self) -> None:
        for max_cycles in (1, -1):
            with self.subTest(max_cycles=max_cycles):
                formula = "rank(close)" if max_cycles == 1 else "rank(open)"
                run_id = self._prepare_run(
                    (formula,),
                    max_cycles=max_cycles,
                    max_consecutive_failures=2,
                )
                remote_name = (
                    "simulation-fixed"
                    if max_cycles == 1
                    else "simulation-continuous"
                )
                client = ScriptedClient()
                client.submissions.append(self._accepted(remote_name))
                client.polls[self._remote(remote_name)] = BacktestPollObservation(
                    "failed", None, "FAILED", None
                )
                self._advance(client, minute=2)
                self._advance(client, minute=3)

                settled = settle_automated_cycle(
                    self.database_path,
                    run_id,
                    cycle_number=1,
                    observed_at="2026-08-30T00:04:00+08:00",
                )

                self.assertEqual(settled.outcome, "failed")
                self.assertEqual(settled.completed_backtests, 0)
                self.assertEqual(settled.failed_backtests, 1)
                self.assertEqual(settled.run.status, "failed")
                self.assertEqual(settled.run.stop_reason, "final_cycle_failed")

    def test_pending_timeout_queries_remote_and_keeps_unsent_peer(self):
        run_id = self._prepare_run(
            ("rank(close)", "rank(open)"), max_in_flight_backtests=1, max_pending_seconds=60,
        )
        client = ScriptedClient()
        client.submissions.extend((self._accepted("one"), self._accepted("two")))
        client.polls[self._remote("one")] = BacktestPollObservation("pending", None, "RUNNING", 0.5)
        self.assertEqual(self._advance(client, minute=2).action, "submitted")
        timed_out = self._advance(client, minute=3)
        self.assertEqual(timed_out.action, "pending_timeout")
        self.assertTrue(timed_out.platform_request_performed)
        self.assertEqual(timed_out.run_status, "running")
        self.assertEqual(timed_out.counts.failed, 1)
        self.assertEqual(timed_out.counts.created, 1)
        self.assertEqual(self._advance(client, minute=4).action, "submitted")
        with open_database(self.database_path) as connection:
            self.assertEqual(get_automated_run(connection, run_id).status, "running")

    def test_completed_remote_is_collected_after_deadline(self):
        self._prepare_run(("rank(close)",), max_pending_seconds=60)
        client = ScriptedClient()
        client.submissions.append(self._accepted("one"))
        client.polls[self._remote("one")] = BacktestPollObservation("completed", "alpha-1", "COMPLETE", None)
        client.details["alpha-1"] = self._detail("alpha-1")
        self.assertEqual(self._advance(client, minute=2).action, "submitted")
        self.assertEqual(self._advance(client, minute=30).action, "detail_ready")
        self.assertEqual(self._advance(client, minute=31).action, "detail_captured")
        self.assertEqual(self._advance(client, minute=32).action, "completed")
        self.assertEqual(len([call for call in client.calls if call[0] == "submit"]), 1)

    def test_completed_and_timed_out_tasks_settle_one_mixed_cycle(self) -> None:
        run_id = self._prepare_run(
            ("rank(close)", "rank(open)"),
            max_in_flight_backtests=2,
            max_pending_seconds=120,
        )
        client = ScriptedClient()
        client.submissions.extend(
            (self._accepted("simulation-1"), self._accepted("simulation-2"))
        )
        client.polls[self._remote("simulation-1")] = BacktestPollObservation(
            "pending", None, "RUNNING", 0.5
        )
        client.polls[self._remote("simulation-2")] = BacktestPollObservation(
            "completed", "alpha-2", "COMPLETE", None
        )
        client.details["alpha-2"] = self._detail("alpha-2")

        self._advance(client, minute=2)
        self._advance(client, minute=2)
        for second in range(10):
            advanced = advance_automated_cycle_backtests(
                self.database_path,
                client,
                run_id,
                observed_at=f"2026-08-30T00:03:{second:02d}+08:00",
            )
            if advanced.counts.completed == 1:
                break
        self.assertEqual(advanced.counts.completed, 1)
        timed_out = self._advance(client, minute=4)
        settled = settle_automated_cycle(
            self.database_path,
            run_id,
            cycle_number=1,
            observed_at="2026-08-30T00:04:00+08:00",
        )

        self.assertEqual(timed_out.action, "pending_timeout")
        self.assertEqual(settled.completed_backtests, 1)
        self.assertEqual(settled.failed_backtests, 1)
        self.assertEqual(settled.outcome, "qualified")
        self.assertEqual(settled.run.status, "completed")
        self.assertEqual(settled.run.stop_reason, "max_cycles_reached")

    def _complete_seed_cycle(self) -> str:
        run_id = self._prepare_run(("rank(close)",))
        client = ScriptedClient()
        client.submissions.append(self._accepted("simulation-1"))
        client.polls[self._remote("simulation-1")] = BacktestPollObservation(
            "completed", "alpha-1", "COMPLETE", None
        )
        client.details["alpha-1"] = self._seed_detail("alpha-1")
        self._advance(client, minute=2)
        self._advance(client, minute=3)
        self._advance(client, minute=4)
        self._advance(client, minute=5)
        return run_id

    def test_all_slow_slots_expire_and_unsent_peer_completes(self) -> None:
        run_id = self._prepare_run(
            ("rank(close)", "rank(open)", "rank(high)", "rank(low)"),
            unlimited=True, max_pending_seconds=15, max_in_flight_backtests=3,
        )
        client = ScriptedClient()
        client.submissions = [self._accepted(f"simulation-{i}") for i in (1, 2, 3, 4)]
        for i in (1, 2, 3):
            client.polls[self._remote(f"simulation-{i}")] = BacktestPollObservation(
                "pending", None, "RUNNING", 0.1,
            )
        client.polls[self._remote("simulation-4")] = BacktestPollObservation(
            "completed", "alpha-4", "COMPLETE", None,
        )
        client.details["alpha-4"] = self._detail("alpha-4")
        observed = datetime.fromisoformat("2026-08-30T00:02:00+08:00")
        for _ in range(50):
            result = advance_automated_cycle_backtests(
                self.database_path, client, run_id, observed_at=observed.isoformat(),
            )
            self.assertLessEqual(result.counts.pending, 3)
            self.assertEqual(result.run_status, "running")
            observed += timedelta(seconds=1)
            if result.action == "cycle_terminal":
                break
        self.assertEqual(result.counts.completed, 1)
        self.assertEqual(result.counts.failed, 3)
        self.assertEqual(result.counts.pending, 0)
        self.assertEqual(len([call for call in client.calls if call[0] == "submit"]), 4)
        with open_database(self.database_path) as connection:
            failed = connection.execute(
                "SELECT failure_code,remote_id FROM backtest_tasks WHERE status='failed'",
            ).fetchall()
        self.assertEqual({row[0] for row in failed}, {"platform_pending_timeout"})
        self.assertEqual({row[1] for row in failed},
                         {self._remote(f"simulation-{i}") for i in (1, 2, 3)})
        settled = settle_automated_cycle(
            self.database_path, run_id, cycle_number=1, observed_at=observed.isoformat(),
        )
        self.assertEqual(settled.run.status, "running")
        self.assertEqual(settled.completed_backtests, 1)

    def _prepare_run(
        self,
        formulas: tuple[str, ...],
        *,
        max_cycles: int = 1,
        max_in_flight_backtests: int = 3,
        max_pending_seconds: int = 3600,
        max_consecutive_failures: int = 1,
        unlimited: bool = False,
    ) -> str:
        prepared = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=max(5, len(formulas)),
                backtest_count=len(formulas),
                max_cycles=-1 if unlimited else max_cycles,
                max_backtests=0 if unlimited else len(formulas),
                max_pending_seconds=max_pending_seconds,
                max_consecutive_failures=max_consecutive_failures,
                max_request_failures=3,
                max_in_flight_backtests=max_in_flight_backtests,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ),
            created_at="2026-08-30T00:00:00+08:00",
        )
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        prepare_automated_candidate_backtest_batch(
            self.database_path,
            run_id=prepared.run_id,
            candidates=tuple(
                AutomatedCandidateBacktest(
                    self._candidate(formula),
                    self.settings,
                )
                for formula in formulas
            ),
            created_at="2026-08-30T00:01:30+08:00",
        )
        self.run_id = prepared.run_id
        return prepared.run_id

    @staticmethod
    def _candidate(formula: str) -> FormulaCandidate:
        return exploration_candidate(parse_formula(formula).expression)

    def _advance(self, client: ScriptedClient, *, minute: int):
        return advance_automated_cycle_backtests(
            self.database_path,
            client,
            self.run_id,
            observed_at=f"2026-08-30T00:{minute:02d}:00+08:00",
        )

    @staticmethod
    def _accepted(remote_name: str) -> BacktestSubmissionObservation:
        return BacktestSubmissionObservation(
            "accepted",
            f"https://api.worldquantbrain.com/simulations/{remote_name}",
            None,
        )

    @staticmethod
    def _remote(remote_name: str) -> str:
        return f"https://api.worldquantbrain.com/simulations/{remote_name}"

    @staticmethod
    def _seed_detail(alpha_id: str) -> BacktestDetail:
        return BacktestDetail(
            platform_alpha_id=alpha_id,
            sharpe=1.0,
            fitness=0.7,
            turnover=0.12,
            returns=0.04,
            drawdown=0.08,
            margin=0.001,
            book_size=20_000_000,
            pnl=100_000,
            checks=_checks(
                ("LOW_SHARPE", "FAIL"),
                ("LOW_FITNESS", "FAIL"),
                ("LOW_SUB_UNIVERSE_SHARPE", "PASS"),
                ("CONCENTRATED_WEIGHT", "PASS"),
                ("LOW_TURNOVER", "PASS"),
                ("HIGH_TURNOVER", "PASS"),
                ("MATCHES_COMPETITION", "PASS"),
                ("SELF_CORRELATION", "PENDING"),
            ),
        )

    @staticmethod
    def _detail(alpha_id: str) -> BacktestDetail:
        return BacktestDetail(
            platform_alpha_id=alpha_id,
            sharpe=1.3,
            fitness=1.1,
            turnover=0.12,
            returns=0.08,
            drawdown=0.04,
            margin=0.001,
            book_size=20_000_000,
            pnl=100_000,
            checks=_checks(
                ("LOW_SHARPE", "PASS"),
                ("LOW_FITNESS", "PASS"),
                ("LOW_TURNOVER", "PASS"),
                ("HIGH_TURNOVER", "PASS"),
                ("CONCENTRATED_WEIGHT", "PASS"),
                ("LOW_SUB_UNIVERSE_SHARPE", "PASS"),
                ("SELF_CORRELATION", "PASS"),
                ("MATCHES_COMPETITION", "PASS"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
