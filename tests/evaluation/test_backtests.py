from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.backtests import CheckFailure, evaluate_backtest
from persistence.backtests import (
    BacktestCheckRecord,
    BacktestResultRecord,
    BacktestSnapshot,
    BacktestTaskRecord,
    BacktestYearlyStatRecord,
)


class BacktestEvaluationTests(unittest.TestCase):
    def test_classifies_failed_pending_and_negative_evidence_without_sc_failure(self) -> None:
        evaluation = evaluate_backtest(self._snapshot())

        self.assertEqual(evaluation.state, "failed")
        self.assertEqual(
            evaluation.failed_checks,
            ("LOW_FITNESS", "LOW_SHARPE", "LOW_SUB_UNIVERSE_SHARPE"),
        )
        self.assertEqual(evaluation.pending_checks, ("SELF_CORRELATION",))
        self.assertNotIn(
            CheckFailure("SELF_CORRELATION", "self_correlation"),
            evaluation.failures,
        )
        self.assertEqual(
            evaluation.failures,
            (
                CheckFailure("LOW_FITNESS", "signal_quality"),
                CheckFailure("LOW_SHARPE", "signal_quality"),
                CheckFailure("LOW_SUB_UNIVERSE_SHARPE", "robustness"),
            ),
        )
        self.assertEqual(
            evaluation.negative_metrics,
            ("sharpe", "fitness", "returns", "margin", "pnl"),
        )

    def test_pending_without_failure_is_not_reported_as_passed(self) -> None:
        snapshot = self._snapshot(
            checks={"LOW_SHARPE": "PASS", "SELF_CORRELATION": "PENDING"},
        )

        evaluation = evaluate_backtest(snapshot)

        self.assertEqual(evaluation.state, "pending")
        self.assertEqual(evaluation.failures, ())

    def test_platform_check_error_is_unresolved_and_not_passed(self) -> None:
        snapshot = self._snapshot(
            checks={
                "LOW_SHARPE": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "ERROR",
            },
        )

        evaluation = evaluate_backtest(snapshot)

        self.assertEqual(evaluation.state, "pending")
        self.assertEqual(evaluation.passed_checks, ("LOW_SHARPE",))
        self.assertEqual(
            evaluation.pending_checks,
            ("LOW_SUB_UNIVERSE_SHARPE",),
        )
        self.assertEqual(evaluation.failed_checks, ())
        self.assertEqual(evaluation.failures, ())

    def test_platform_status_remains_authoritative_over_threshold_and_actual(self) -> None:
        snapshot = self._snapshot(
            checks={"LOW_SHARPE": "FAIL"},
            check_details={"LOW_SHARPE": (1.25, 1.3, None)},
        )

        evaluation = evaluate_backtest(snapshot)

        self.assertEqual(evaluation.failed_checks, ("LOW_SHARPE",))
        self.assertEqual(evaluation.checks[0].threshold, 1.25)
        self.assertEqual(evaluation.checks[0].actual, 1.3)
        self.assertTrue(evaluation.check_details_captured)

    def test_platform_warning_is_unresolved_and_not_passed(self) -> None:
        evaluation = evaluate_backtest(
            self._snapshot(checks={"UNITS": "WARNING"})
        )

        self.assertEqual(evaluation.state, "pending")
        self.assertEqual(evaluation.pending_checks, ("UNITS",))

    def test_all_passed_checks_are_clear(self) -> None:
        snapshot = self._snapshot(
            checks={
                "LOW_SHARPE": "PASS",
                "LOW_FITNESS": "PASS",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "SELF_CORRELATION": "PASS",
                "MATCHES_COMPETITION": "PASS",
            },
        )

        evaluation = evaluate_backtest(snapshot)

        self.assertEqual(evaluation.state, "passed")
        self.assertEqual(evaluation.failed_checks, ())
        self.assertEqual(evaluation.pending_checks, ())
        self.assertEqual(evaluation.missing_checks, ())
        self.assertEqual(evaluation.unexpected_checks, ())
        self.assertTrue(evaluation.check_set_complete)
        self.assertTrue(evaluation.non_sc_check_set_complete)

    def test_exposes_position_counts_and_yearly_facts_without_new_quality_judgment(
        self,
    ) -> None:
        snapshot = self._snapshot(
            checks={
                "LOW_SHARPE": "PASS",
                "LOW_FITNESS": "PASS",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "SELF_CORRELATION": "PASS",
                "MATCHES_COMPETITION": "PASS",
            },
            long_count=731,
            short_count=694,
        )
        assert snapshot.result is not None
        snapshot = replace(
            snapshot,
            result=replace(
                snapshot.result,
                sharpe=1.1,
                fitness=0.8,
                returns=0.05,
                margin=0.0002,
                pnl=100_000,
            ),
        )
        yearly_stat = BacktestYearlyStatRecord(
            task_id=snapshot.task.task_id,
            year=2024,
            pnl=-25_000,
            book_size=20_000_000,
            long_count=705,
            short_count=None,
            turnover=0.12,
            sharpe=-0.2,
            returns=-0.01,
            drawdown=0.2,
            margin=-0.0001,
            fitness=-0.05,
            stage="IS",
        )
        snapshot = replace(snapshot, yearly_stats=(yearly_stat,))

        evaluation = evaluate_backtest(snapshot)

        self.assertEqual(evaluation.long_count, 731)
        self.assertEqual(evaluation.short_count, 694)
        self.assertEqual(evaluation.yearly_stats, (yearly_stat,))
        self.assertEqual(evaluation.state, "passed")
        self.assertEqual(evaluation.failures, ())
        self.assertEqual(evaluation.negative_metrics, ())

    def test_requires_explicitly_captured_yearly_facts(self) -> None:
        snapshot = replace(self._snapshot(), yearly_stats=None)

        with self.assertRaisesRegex(
            ValueError,
            "evaluation_yearly_stats_not_captured",
        ):
            evaluate_backtest(snapshot)

    def test_partial_passed_checks_are_not_reported_as_complete(self) -> None:
        evaluation = evaluate_backtest(
            self._snapshot(
                checks={
                    "LOW_SHARPE": "PASS",
                    "SELF_CORRELATION": "PASS",
                }
            )
        )

        self.assertEqual(evaluation.state, "pending")
        self.assertFalse(evaluation.check_set_complete)
        self.assertFalse(evaluation.non_sc_check_set_complete)
        self.assertIn("LOW_FITNESS", evaluation.missing_checks)
        self.assertEqual(evaluation.unexpected_checks, ())

    def test_self_correlation_may_be_missing_from_base_quality_evidence(self) -> None:
        evaluation = evaluate_backtest(
            self._snapshot(
                checks={
                    "LOW_SHARPE": "PASS",
                    "LOW_FITNESS": "PASS",
                    "LOW_TURNOVER": "PASS",
                    "HIGH_TURNOVER": "PASS",
                    "CONCENTRATED_WEIGHT": "PASS",
                    "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                    "MATCHES_COMPETITION": "PASS",
                }
            )
        )

        self.assertEqual(evaluation.state, "pending")
        self.assertFalse(evaluation.check_set_complete)
        self.assertTrue(evaluation.non_sc_check_set_complete)

    def test_incomplete_backtest_stops_and_unknown_platform_status_stays_pending(
        self,
    ) -> None:
        completed = self._snapshot()
        with self.assertRaisesRegex(ValueError, "evaluation_completed_backtest_required"):
            evaluate_backtest(
                BacktestSnapshot(
                    task=replace(completed.task, status="pending"),
                    result=None,
                    yearly_stats=None,
                )
            )
        evaluation = evaluate_backtest(
            self._snapshot(checks={"LOW_SHARPE": "UNKNOWN"})
        )
        self.assertEqual(evaluation.state, "pending")
        self.assertEqual(evaluation.pending_checks, ("LOW_SHARPE",))

    def _snapshot(
        self,
        *,
        checks=None,
        check_details=None,
        long_count: int | None = None,
        short_count: int | None = None,
    ) -> BacktestSnapshot:
        formula = "rank(close)"
        identity = sha256(formula.encode("utf-8")).hexdigest()
        task_id = f"backtest_{identity}"
        task = BacktestTaskRecord(
            task_id=task_id,
            account_scope="group-account",
            formula=formula,
            formula_fingerprint=identity,
            settings_json='{"delay":1}',
            request_fingerprint="request",
            status="completed",
            remote_id="simulation",
            platform_alpha_id="alpha",
            created_at="2026-08-29T00:00:00+00:00",
            submission_started_at="2026-08-29T00:01:00+00:00",
            last_observed_at="2026-08-29T00:02:00+00:00",
            retry_not_before=None,
            finished_at="2026-08-29T00:02:00+00:00",
            failure_code=None,
            failure_message=None,
        )
        statuses = checks or {
            "LOW_FITNESS": "FAIL",
            "LOW_SHARPE": "FAIL",
            "LOW_SUB_UNIVERSE_SHARPE": "FAIL",
            "SELF_CORRELATION": "PENDING",
            "CONCENTRATED_WEIGHT": "PASS",
        }
        details = check_details or {}
        result = BacktestResultRecord(
            task_id=task_id,
            sharpe=-0.08,
            fitness=-0.03,
            turnover=0.0832,
            returns=-0.0136,
            drawdown=0.3539,
            margin=-0.000328,
            book_size=20_000_000,
            pnl=-674_669,
            long_count=long_count,
            short_count=short_count,
            check_details_captured=True,
            checks=tuple(
                BacktestCheckRecord(
                    name=name,
                    status=status,
                    threshold=details.get(name, (None, None, None))[0],
                    actual=details.get(name, (None, None, None))[1],
                    platform_date=details.get(name, (None, None, None))[2],
                )
                for name, status in sorted(statuses.items())
            ),
        )
        return BacktestSnapshot(task=task, result=result, yearly_stats=())


if __name__ == "__main__":
    unittest.main()
