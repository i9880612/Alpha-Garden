from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.backtests import (
    apply_backtest_detail,
    apply_backtest_poll_observation,
    apply_backtest_submission_observation,
    fail_backtest_task,
    cancel_unsubmitted_backtest_task,
    prepare_backtest_task,
    record_backtest_yearly_stats,
    record_pending_observation,
    record_submission_accepted,
    record_submission_not_accepted,
    record_submission_unknown,
)
from persistence.backtests import get_backtest_task, initialize_backtest_schema
from persistence.database import open_database
from worldquant.backtests import (
    BacktestCheck,
    BacktestDetail,
    BacktestPollObservation,
    BacktestSubmissionObservation,
    STANDARD_REGULAR_CHECK_NAMES,
    parse_poll_response,
)


class BacktestExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "backtests.sqlite3"
        with open_database(self.database_path) as connection:
            initialize_backtest_schema(connection)

    def test_same_normalized_formula_and_settings_reuse_one_task(self) -> None:
        with open_database(self.database_path) as connection:
            first = self._prepare(connection, formula=" rank(close) ")
            replayed = self._prepare(
                connection,
                settings={"universe": "TOP3000", "delay": 1},
                created_at="2026-08-29T00:02:00+00:00",
            )

        self.assertEqual(first, replayed)
        self.assertEqual(first.task.formula, "rank(close)")

    def test_same_formula_with_changed_settings_creates_a_distinct_task(self) -> None:
        with open_database(self.database_path) as connection:
            first = self._prepare(connection)
            changed = self._prepare(
                connection,
                settings={"delay": 1, "universe": "TOP500"},
            )

        self.assertNotEqual(first.task.task_id, changed.task.task_id)
        self.assertEqual(first.task.formula_fingerprint, changed.task.formula_fingerprint)

    def test_unsent_cancellation_keeps_history_and_allows_one_new_request(self):
        with open_database(self.database_path) as connection:
            original = self._prepare(connection)
            cancelled = cancel_unsubmitted_backtest_task(connection, original.task.task_id,
                observed_at="2026-08-29T00:01:00+00:00")
            retry = self._prepare(connection, created_at="2026-08-29T00:02:00+00:00")
            self.assertNotEqual(retry.task.task_id, original.task.task_id)
            self.assertEqual(retry.task.request_fingerprint, original.task.request_fingerprint)
            self.assertEqual(get_backtest_task(connection, original.task.task_id), cancelled)
            self.assertEqual(self._prepare(connection, created_at="2026-08-29T00:03:00+00:00"), retry)
            record_submission_unknown(connection, retry.task.task_id, observed_at="2026-08-29T00:04:00+00:00")
            unknown = get_backtest_task(connection, retry.task.task_id)
            self.assertEqual(self._prepare(connection, created_at="2026-08-29T00:05:00+00:00"), unknown)
            self.assertEqual(connection.execute("SELECT count(*) FROM backtest_tasks").fetchone()[0], 2)

    def test_maximum_controls_are_part_of_the_persisted_request_identity(self) -> None:
        baseline = {
            "delay": 1,
            "maxTrade": "OFF",
            "maxPosition": "OFF",
        }
        with open_database(self.database_path) as connection:
            first = self._prepare(connection, settings=baseline)
            changed_trade = self._prepare(
                connection,
                settings={**baseline, "maxTrade": "ON"},
            )
            changed_position = self._prepare(
                connection,
                settings={**baseline, "maxPosition": "ON"},
            )

        self.assertEqual(
            len(
                {
                    first.task.task_id,
                    changed_trade.task.task_id,
                    changed_position.task.task_id,
                }
            ),
            3,
        )
        self.assertEqual(
            len(
                {
                    first.task.request_fingerprint,
                    changed_trade.task.request_fingerprint,
                    changed_position.task.request_fingerprint,
                }
            ),
            3,
        )
        self.assertIn('"maxTrade":"OFF"', first.task.settings_json)
        self.assertIn('"maxPosition":"OFF"', first.task.settings_json)

    def test_same_formula_and_settings_are_distinct_between_accounts(self) -> None:
        with open_database(self.database_path) as connection:
            first = self._prepare(connection)
            other_account = self._prepare(connection, account_scope="research-account")

        self.assertNotEqual(first.task.task_id, other_account.task.task_id)
        self.assertNotEqual(
            first.task.request_fingerprint,
            other_account.task.request_fingerprint,
        )

    def test_deterministic_invalid_formula_is_not_persisted(self) -> None:
        with self.assertRaisesRegex(ValueError, "backtest_formula_logic_invalid"):
            with open_database(self.database_path) as connection:
                self._prepare(connection, formula="rank(close/0)")
        with open_database(self.database_path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM backtest_tasks").fetchone()[0]
        self.assertEqual(count, 0)

    def test_unknown_submission_blocks_progress_until_reconciled(self) -> None:
        with open_database(self.database_path) as connection:
            task = self._prepare(connection)
            unknown = record_submission_unknown(
                connection,
                task.task.task_id,
                observed_at="2026-08-29T00:02:00+00:00",
            )
            replayed = record_submission_unknown(
                connection,
                task.task.task_id,
                observed_at="2026-08-29T00:03:00+00:00",
            )

        self.assertEqual(unknown, replayed)
        with self.assertRaisesRegex(ValueError, "backtest_status_invalid"):
            with open_database(self.database_path) as connection:
                record_pending_observation(
                    connection,
                    task.task.task_id,
                    observed_at="2026-08-29T00:04:00+00:00",
                )

    def test_unknown_can_be_resolved_to_not_accepted_or_remote_pending(self) -> None:
        with open_database(self.database_path) as connection:
            task = self._prepare(connection)
            record_submission_unknown(
                connection,
                task.task.task_id,
                observed_at="2026-08-29T00:02:00+00:00",
            )
            created = record_submission_not_accepted(connection, task.task.task_id)
            record_submission_unknown(
                connection,
                task.task.task_id,
                observed_at="2026-08-29T00:03:00+00:00",
            )
            pending = record_submission_accepted(
                connection,
                task.task.task_id,
                remote_id="simulation_1",
                observed_at="2026-08-29T00:04:00+00:00",
            )

        self.assertEqual(created.task.status, "created")
        self.assertEqual(pending.task.status, "pending")

    def test_older_poll_response_cannot_overwrite_newer_observation(self) -> None:
        with open_database(self.database_path) as connection:
            task = self._prepare(connection)
            record_submission_accepted(
                connection,
                task.task.task_id,
                remote_id="simulation_1",
                observed_at="2026-08-29T00:03:00+00:00",
            )

        with self.assertRaisesRegex(ValueError, "backtest_observation_out_of_order"):
            with open_database(self.database_path) as connection:
                record_pending_observation(
                    connection,
                    task.task.task_id,
                    observed_at="2026-08-29T00:02:00+00:00",
                )

    def test_completion_is_atomic_and_repeated_response_does_not_duplicate(self) -> None:
        with open_database(self.database_path) as connection:
            task = self._prepare(connection)
            record_submission_accepted(
                connection,
                task.task.task_id,
                remote_id="simulation_1",
                observed_at="2026-08-29T00:02:00+00:00",
                platform_alpha_id="alpha_1",
            )
            first = self._complete(connection, task.task.task_id)
        with open_database(self.database_path) as connection:
            replayed = self._complete(
                connection,
                task.task.task_id,
                observed_at="2026-08-29T00:04:00+00:00",
            )
            result_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_results"
            ).fetchone()[0]
            check_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_checks"
            ).fetchone()[0]

        self.assertEqual(first, replayed)
        self.assertEqual(result_count, 1)
        self.assertEqual(check_count, 1)
        self.assertTrue(first.result.check_details_captured)
        self.assertEqual(first.result.checks[0].threshold, 1.25)
        self.assertEqual(first.result.checks[0].actual, 1.2)

        with self.assertRaisesRegex(ValueError, "backtest_result_conflict"):
            with open_database(self.database_path) as connection:
                self._complete(
                    connection,
                    task.task.task_id,
                    checks=(
                        BacktestCheck(
                            name="LOW_SHARPE",
                            status="PASS",
                            threshold=1.25,
                            actual=9.0,
                            platform_date=None,
                        ),
                    ),
                )

    def test_check_failure_rolls_back_task_result_and_all_check_facts(self) -> None:
        with open_database(self.database_path) as connection:
            task = self._prepare(connection)
            record_submission_accepted(
                connection,
                task.task.task_id,
                remote_id="simulation_1",
                observed_at="2026-08-29T00:02:00+00:00",
                platform_alpha_id="alpha_1",
            )
        with self.assertRaisesRegex(ValueError, "backtest_check_actual_invalid"):
            with open_database(self.database_path) as connection:
                self._complete(
                    connection,
                    task.task.task_id,
                    checks=(
                        BacktestCheck(
                            name="LOW_SHARPE",
                            status="PASS",
                            threshold=1.25,
                            actual=float("nan"),
                            platform_date=None,
                        ),
                    ),
                )

        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, task.task.task_id)
            result_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_results"
            ).fetchone()[0]
            check_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_checks"
            ).fetchone()[0]

        self.assertEqual(snapshot.task.status, "pending")
        self.assertIsNone(snapshot.result)
        self.assertEqual(result_count, 0)
        self.assertEqual(check_count, 0)

    def test_platform_failure_after_submission_has_no_success_result(self) -> None:
        with open_database(self.database_path) as connection:
            task = self._prepare(connection)
            record_submission_unknown(
                connection,
                task.task.task_id,
                observed_at="2026-08-29T00:02:00+00:00",
            )
            failed = fail_backtest_task(
                connection,
                task.task.task_id,
                failure_code="platform_rejected",
                failure_message="平台明确拒绝该请求",
                observed_at="2026-08-29T00:03:00+00:00",
            )

        self.assertEqual(failed.task.status, "failed")
        self.assertIsNone(failed.result)

    def test_platform_observations_advance_one_task_to_one_result(self) -> None:
        with open_database(self.database_path) as connection:
            task = self._prepare(connection)
            apply_backtest_submission_observation(
                connection,
                task.task.task_id,
                BacktestSubmissionObservation(
                    state="accepted",
                    remote_id="https://api.worldquantbrain.com/simulations/1",
                    reason=None,
                ),
                observed_at="2026-08-29T00:02:00+00:00",
            )
            apply_backtest_poll_observation(
                connection,
                task.task.task_id,
                BacktestPollObservation(
                    state="pending",
                    platform_alpha_id=None,
                    platform_status="RUNNING",
                    progress=0.35,
                ),
                observed_at="2026-08-29T00:03:00+00:00",
            )
            apply_backtest_poll_observation(
                connection,
                task.task.task_id,
                BacktestPollObservation(
                    state="completed",
                    platform_alpha_id="alpha-1",
                    platform_status="COMPLETE",
                    progress=None,
                ),
                observed_at="2026-08-29T00:04:00+00:00",
            )
            apply_backtest_detail(
                connection,
                task.task.task_id,
                BacktestDetail(
                    platform_alpha_id="alpha-1",
                    sharpe=1.3,
                    fitness=1.1,
                    turnover=0.12,
                    returns=0.08,
                    drawdown=0.04,
                    margin=0.001,
                    book_size=20_000_000,
                    pnl=100_000,
                    long_count=1200,
                    short_count=1100,
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
                ),
                observed_at="2026-08-29T00:05:00+00:00",
            )
            completed = record_backtest_yearly_stats(
                connection,
                task.task.task_id,
                (),
                observed_at="2026-08-29T00:05:30+00:00",
            )

        self.assertEqual(completed.task.status, "completed")
        self.assertEqual(completed.result.sharpe, 1.3)
        self.assertEqual(completed.result.long_count, 1200)
        self.assertEqual(completed.result.short_count, 1100)
        self.assertEqual(completed.yearly_stats, ())

    def test_platform_warning_preserves_alpha_identity_and_message(self) -> None:
        with open_database(self.database_path) as connection:
            task = self._prepare(connection)
            apply_backtest_submission_observation(
                connection,
                task.task.task_id,
                BacktestSubmissionObservation(
                    state="accepted",
                    remote_id="https://api.worldquantbrain.com/simulations/1",
                    reason=None,
                ),
                observed_at="2026-08-29T00:02:00+00:00",
            )
            failed = apply_backtest_poll_observation(
                connection,
                task.task.task_id,
                BacktestPollObservation(
                    state="failed",
                    platform_alpha_id="alpha-warning",
                    platform_status="WARNING",
                    progress=None,
                    failure_message="Incompatible unit",
                ),
                observed_at="2026-08-29T00:03:00+00:00",
            )

        self.assertEqual(failed.task.status, "failed")
        self.assertEqual(failed.task.platform_alpha_id, "alpha-warning")
        self.assertEqual(failed.task.failure_code, "platform_warning")
        self.assertEqual(failed.task.failure_message, "Incompatible unit")
        self.assertIsNone(failed.result)

    def test_platform_error_message_survives_poll_and_persistence(self) -> None:
        message = "Unexpected keyword argument 'd'"
        with open_database(self.database_path) as connection:
            task = self._prepare(connection)
            record_submission_unknown(connection, task.task.task_id,
                                      observed_at="2026-08-29T00:01:00+00:00")
            record_submission_accepted(connection, task.task.task_id,
                                       remote_id="https://api.worldquantbrain.com/simulations/1",
                                       observed_at="2026-08-29T00:02:00+00:00")
            apply_backtest_poll_observation(
                connection, task.task.task_id,
                parse_poll_response({"status": "ERROR", "message": message}),
                observed_at="2026-08-29T00:03:00+00:00",
            )
        with open_database(self.database_path) as connection:
            failed = get_backtest_task(connection, task.task.task_id)
        self.assertEqual(failed.task.status, "failed")
        self.assertEqual(failed.task.failure_code, "platform_error")
        self.assertEqual(failed.task.failure_message, message)
        self.assertIsNone(failed.result)
        self.assertIsNone(failed.task.platform_alpha_id)

    def _prepare(self, connection, **overrides):
        arguments = {
            "account_scope": "group-account",
            "formula": "rank(close)",
            "settings": {"delay": 1, "universe": "TOP3000"},
            "created_at": "2026-08-29T00:01:00+00:00",
        }
        arguments.update(overrides)
        return prepare_backtest_task(connection, **arguments)

    @staticmethod
    def _complete(connection, task_id: str, **overrides):
        arguments = {
            "platform_alpha_id": "alpha_1",
            "observed_at": "2026-08-29T00:03:00+00:00",
            "sharpe": 1.2,
            "fitness": 0.9,
            "turnover": 0.2,
            "returns": 0.1,
            "drawdown": 0.05,
            "margin": 0.001,
            "book_size": None,
            "pnl": None,
            "checks": (
                BacktestCheck(
                    name="LOW_SHARPE",
                    status="PASS",
                    threshold=1.25,
                    actual=1.2,
                    platform_date=None,
                ),
            ),
        }
        arguments.update(overrides)
        observed_at = arguments.pop("observed_at")
        apply_backtest_detail(
            connection,
            task_id,
            BacktestDetail(**arguments),
            observed_at=observed_at,
        )
        return record_backtest_yearly_stats(
            connection,
            task_id,
            (),
            observed_at=observed_at,
        )


if __name__ == "__main__":
    unittest.main()
