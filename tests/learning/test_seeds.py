from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from learning.seeds import assess_signal_seed
from persistence.backtests import (
    BacktestCheckRecord,
    BacktestResultRecord,
    BacktestSnapshot,
    BacktestTaskRecord,
)


class SignalSeedAssessmentTests(unittest.TestCase):
    def test_sc_state_is_recorded_but_does_not_block_seed(self) -> None:
        for status, expected in (
            ("PASS", "passed"),
            ("FAIL", "failed"),
            ("PENDING", "pending"),
        ):
            with self.subTest(status=status):
                assessment = assess_signal_seed(
                    self._snapshot(
                        checks={
                            **self._positive_signal_checks(),
                            "SELF_CORRELATION": status,
                        }
                    )
                )

                self.assertTrue(assessment.eligible)
                self.assertEqual(assessment.self_correlation_status, expected)

    def test_positive_signal_floor_and_safety_checks_are_required(self) -> None:
        checks = self._positive_signal_checks()
        del checks["CONCENTRATED_WEIGHT"]
        assessment = assess_signal_seed(self._snapshot(fitness=0.69, checks=checks))

        self.assertFalse(assessment.eligible)
        self.assertIn(
            "signal_seed_fitness_below_positive_threshold",
            assessment.rejection_reasons,
        )
        self.assertIn(
            "signal_seed_check_not_passed:CONCENTRATED_WEIGHT",
            assessment.rejection_reasons,
        )

    def test_new_seed_numeric_admission_boundaries_are_inclusive(self) -> None:
        checks = self._positive_signal_checks()

        below_sharpe = assess_signal_seed(
            self._snapshot(sharpe=0.999, fitness=0.7, checks=checks)
        )
        below_fitness = assess_signal_seed(
            self._snapshot(sharpe=1.0, fitness=0.699, checks=checks)
        )
        at_boundary = assess_signal_seed(
            self._snapshot(sharpe=1.0, fitness=0.7, checks=checks)
        )

        self.assertFalse(below_sharpe.eligible)
        self.assertIn(
            "signal_seed_sharpe_below_positive_threshold",
            below_sharpe.rejection_reasons,
        )
        self.assertFalse(below_fitness.eligible)
        self.assertIn(
            "signal_seed_fitness_below_positive_threshold",
            below_fitness.rejection_reasons,
        )
        self.assertTrue(at_boundary.eligible)
        self.assertEqual(at_boundary.rejection_reasons, ())

    def test_low_sub_universe_failure_is_a_repairable_seed_target(self) -> None:
        checks = self._positive_signal_checks()
        checks["LOW_FITNESS"] = "PASS"
        checks["LOW_SUB_UNIVERSE_SHARPE"] = "FAIL"

        assessment = assess_signal_seed(self._snapshot(checks=checks))

        self.assertTrue(assessment.eligible)
        self.assertEqual(assessment.rejection_reasons, ())

    def test_low_sub_universe_target_must_be_explicitly_resolved(self) -> None:
        for status in (None, "PENDING", "ERROR"):
            checks = self._positive_signal_checks()
            if status is None:
                del checks["LOW_SUB_UNIVERSE_SHARPE"]
            else:
                checks["LOW_SUB_UNIVERSE_SHARPE"] = status
            with self.subTest(status=status):
                assessment = assess_signal_seed(self._snapshot(checks=checks))

                self.assertFalse(assessment.eligible)
                self.assertIn(
                    "signal_seed_check_not_resolved:LOW_SUB_UNIVERSE_SHARPE",
                    assessment.rejection_reasons,
                )

    def test_concentrated_weight_failure_remains_a_safety_rejection(self) -> None:
        checks = self._positive_signal_checks()
        checks["CONCENTRATED_WEIGHT"] = "FAIL"

        assessment = assess_signal_seed(self._snapshot(checks=checks))

        self.assertFalse(assessment.eligible)
        self.assertIn(
            "signal_seed_check_not_passed:CONCENTRATED_WEIGHT",
            assessment.rejection_reasons,
        )

    def test_fully_passing_formula_can_start_qualified_evolution(self) -> None:
        assessment = assess_signal_seed(
            self._snapshot(
                fitness=1.1,
                checks={
                    **self._positive_signal_checks(),
                    "LOW_FITNESS": "PASS",
                },
            )
        )

        self.assertTrue(assessment.eligible)
        self.assertEqual(assessment.rejection_reasons, ())

    def test_turnover_bounds_are_strict(self) -> None:
        for turnover in (0.01, 0.70):
            with self.subTest(turnover=turnover):
                assessment = assess_signal_seed(
                    self._snapshot(
                        turnover=turnover,
                        checks=self._positive_signal_checks(),
                    )
                )

                self.assertFalse(assessment.eligible)
                self.assertIn(
                    "signal_seed_turnover_out_of_range",
                    assessment.rejection_reasons,
                )

    def test_incomplete_check_set_cannot_become_a_seed(self) -> None:
        assessment = assess_signal_seed(
            self._snapshot(
                checks={
                    "LOW_SHARPE": "PASS",
                    "LOW_FITNESS": "FAIL",
                    "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                    "CONCENTRATED_WEIGHT": "PASS",
                }
            )
        )

        self.assertFalse(assessment.eligible)
        self.assertIn(
            "signal_seed_check_set_incomplete",
            assessment.rejection_reasons,
        )

    def test_self_correlation_may_be_missing_but_unexpected_checks_are_rejected(
        self,
    ) -> None:
        without_sc = self._positive_signal_checks()
        del without_sc["SELF_CORRELATION"]
        self.assertTrue(assess_signal_seed(self._snapshot(checks=without_sc)).eligible)

        with_unexpected = self._positive_signal_checks()
        with_unexpected["NEW_PLATFORM_CHECK"] = "PASS"
        assessment = assess_signal_seed(self._snapshot(checks=with_unexpected))
        self.assertFalse(assessment.eligible)
        self.assertIn(
            "signal_seed_check_set_incomplete",
            assessment.rejection_reasons,
        )

    @staticmethod
    def _positive_signal_checks() -> dict[str, str]:
        return {
            "CONCENTRATED_WEIGHT": "PASS",
            "HIGH_TURNOVER": "PASS",
            "LOW_SHARPE": "PASS",
            "LOW_FITNESS": "FAIL",
            "LOW_SUB_UNIVERSE_SHARPE": "PASS",
            "LOW_TURNOVER": "PASS",
            "MATCHES_COMPETITION": "PASS",
            "SELF_CORRELATION": "PENDING",
        }

    @staticmethod
    def _snapshot(
        *,
        fitness: float = 0.9,
        sharpe: float = 1.3,
        turnover: float = 0.12,
        checks: dict[str, str],
    ) -> BacktestSnapshot:
        task = BacktestTaskRecord(
            task_id="backtest_seed",
            account_scope="group-account",
            formula="rank(close)",
            formula_fingerprint="formula_seed",
            settings_json='{"delay":1}',
            request_fingerprint="request_seed",
            status="completed",
            remote_id="simulation_seed",
            platform_alpha_id="alpha_seed",
            created_at="2026-08-31T00:00:00+00:00",
            submission_started_at="2026-08-31T00:01:00+00:00",
            last_observed_at="2026-08-31T00:02:00+00:00",
            retry_not_before=None,
            finished_at="2026-08-31T00:02:00+00:00",
            failure_code=None,
            failure_message=None,
        )
        return BacktestSnapshot(
            task=task,
            result=BacktestResultRecord(
                task_id=task.task_id,
                sharpe=sharpe,
                fitness=fitness,
                turnover=turnover,
                returns=0.08,
                drawdown=0.04,
                margin=0.001,
                book_size=20_000_000,
                pnl=100_000,
                long_count=None,
                short_count=None,
                check_details_captured=True,
                checks=tuple(
                    BacktestCheckRecord(
                        name=name,
                        status=status,
                        threshold=None,
                        actual=None,
                        platform_date=None,
                    )
                    for name, status in sorted(checks.items())
                ),
            ),
            yearly_stats=(),
        )


if __name__ == "__main__":
    unittest.main()
