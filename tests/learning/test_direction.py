import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from learning.direction import negative_direction_is_testable
from tests.learning import test_seeds as seed_tests


class NegativeDirectionTests(unittest.TestCase):
    def snapshot(self, sharpe=-1.0, fitness=-0.7, turnover=0.12, **changes):
        checks = seed_tests.SignalSeedAssessmentTests._positive_signal_checks()
        checks.update(LOW_SHARPE="FAIL", LOW_FITNESS="FAIL")
        checks.update(changes)
        return seed_tests.SignalSeedAssessmentTests._snapshot(
            sharpe=sharpe, fitness=fitness, turnover=turnover, checks=checks
        )

    def test_only_negative_floors_not_absolute_values_or_mixed_signs(self):
        for sharpe, fitness, expected in (
            (-1, -0.7, True),
            (-0.999, -0.7, False),
            (-1, -0.699, False),
            (1, 0.7, False),
            (-1, 0.7, False),
            (1, -0.7, False),
        ):
            with self.subTest(sharpe=sharpe, fitness=fitness):
                self.assertEqual(
                    negative_direction_is_testable(self.snapshot(sharpe, fitness)),
                    expected,
                )

    def test_safety_and_unknown_checks_still_block(self):
        for changes in (
            {"CONCENTRATED_WEIGHT": "FAIL"},
            {"CONCENTRATED_WEIGHT": "PENDING"},
            {"LOW_SUB_UNIVERSE_SHARPE": "PENDING"},
            {"MATCHES_COMPETITION": "FAIL"},
            {"UNEXPECTED": "PASS"},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(
                    negative_direction_is_testable(self.snapshot(**changes))
                )
        for turnover in (0.01, 0.70):
            self.assertFalse(
                negative_direction_is_testable(self.snapshot(turnover=turnover))
            )

    def test_sc_pending_is_not_converted_to_pass(self):
        snapshot = self.snapshot(SELF_CORRELATION="PENDING")
        before = snapshot
        self.assertTrue(negative_direction_is_testable(snapshot))
        self.assertEqual(snapshot, before)
        self.assertEqual(
            next(
                c.status for c in snapshot.result.checks if c.name == "SELF_CORRELATION"
            ),
            "PENDING",
        )

    def test_incomplete_tasks_and_missing_platform_identity_do_not_qualify(self):
        snapshot = self.snapshot()
        for task in (
            replace(snapshot.task, status="pending"),
            replace(snapshot.task, status="failed"),
            replace(snapshot.task, platform_alpha_id=None),
            replace(snapshot.task, submission_started_at=None),
        ):
            self.assertFalse(
                negative_direction_is_testable(replace(snapshot, task=task))
            )
