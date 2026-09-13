from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.run_config import load_automated_run_limits
from execution.runs import AutomatedRunLimits


class AutomatedRunConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "run.json"

    def test_default_config_combines_with_the_requested_cycle_count(self) -> None:
        path = Path(__file__).resolve().parents[2] / "config" / "run.default.json"

        limits = load_automated_run_limits(path, cycles=1)

        self.assertEqual(
            limits,
            AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=500,
                backtest_count=100,
                max_cycles=1,
                max_backtests=100,
                max_pending_seconds=600,
                max_consecutive_failures=2,
                max_request_failures=3,
                max_in_flight_backtests=3,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
                automatic_submissions_enabled=False,
            ),
        )

    def test_positive_cycle_count_directly_sets_the_run_size(self) -> None:
        payload = self._payload()
        payload["maxPendingSeconds"] = 900
        self.path.write_text(json.dumps(payload), encoding="utf-8")

        limits = load_automated_run_limits(self.path, cycles=5)

        self.assertEqual(limits.max_cycles, 5)
        self.assertEqual(limits.max_backtests, 100)
        self.assertEqual(limits.max_pending_seconds, 900)
        self.assertTrue(limits.real_backtests_authorized)
        self.assertFalse(limits.automatic_submissions_enabled)

    def test_percentages_are_required_and_validate_sum_and_direction_subset(self) -> None:
        for key, value in (
            ("explorationPercent", 0), ("selfCorrelationPercent", -1),
            ("mutationPercent", 41), ("directionValidationPercent", 41),
            ("directionValidationPercent", True), ("explorationPercent", 30.5),
        ):
            with self.subTest(key=key, value=value):
                payload = self._payload()
                payload[key] = value
                self.path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "automated_run_config_file_invalid"):
                    load_automated_run_limits(self.path, cycles=1)
        payload = self._payload()
        payload.update(explorationPercent=20, selfCorrelationPercent=50,
                       mutationPercent=30, directionValidationPercent=5)
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        limits = load_automated_run_limits(self.path, cycles=1)
        self.assertEqual((limits.exploration_percent, limits.self_correlation_percent,
                          limits.mutation_percent, limits.direction_validation_percent),
                         (20, 50, 30, 5))

    def test_continuous_mode_has_no_total_backtest_limit(self) -> None:
        self.path.write_text(json.dumps(self._payload()), encoding="utf-8")

        limits = load_automated_run_limits(self.path, cycles=-1)

        self.assertEqual(limits.max_cycles, -1)
        self.assertEqual(limits.max_backtests, 0)
        self.assertTrue(limits.real_backtests_authorized)

    def test_explicit_auto_submit_switch_is_frozen(self) -> None:
        self.path.write_text(json.dumps(self._payload()), encoding="utf-8")

        limits = load_automated_run_limits(
            self.path,
            cycles=1,
            automatic_submissions_enabled=True,
        )

        self.assertTrue(limits.automatic_submissions_enabled)

    def test_auto_submit_switch_rejects_non_boolean_values(self) -> None:
        self.path.write_text(json.dumps(self._payload()), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_config_file_invalid",
        ):
            load_automated_run_limits(
                self.path,
                cycles=1,
                automatic_submissions_enabled=1,
            )

    def test_missing_extra_boolean_and_invalid_values_are_rejected(self) -> None:
        invalid_payloads = []
        missing = self._payload()
        del missing["maxPendingSeconds"]
        invalid_payloads.append(missing)
        extra = self._payload()
        extra["unknown"] = 1
        invalid_payloads.append(extra)
        boolean = self._payload()
        boolean["backtestCount"] = True
        invalid_payloads.append(boolean)
        invalid_limit = self._payload()
        invalid_limit["continuousBacktestLimit"] = 60
        invalid_payloads.append(invalid_limit)

        for index, payload in enumerate(invalid_payloads):
            with self.subTest(index=index):
                self.path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError,
                    "automated_run_config_file_invalid",
                ):
                    load_automated_run_limits(self.path, cycles=1)

    def test_undefined_cycle_values_are_rejected(self) -> None:
        self.path.write_text(json.dumps(self._payload()), encoding="utf-8")

        for cycles in (0, -2, True):
            with self.subTest(cycles=cycles):
                with self.assertRaisesRegex(
                    ValueError,
                    "automated_run_config_file_invalid",
                ):
                    load_automated_run_limits(self.path, cycles=cycles)

    def test_missing_and_malformed_files_are_rejected(self) -> None:
        for path in (self.path, self.path.with_name("missing.json")):
            with self.subTest(path=path.name):
                if path == self.path:
                    path.write_text("{", encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError,
                    "automated_run_config_file_invalid",
                ):
                    load_automated_run_limits(path, cycles=1)

    @staticmethod
    def _payload() -> dict[str, int]:
        return {
            "generationCount": 500,
            "backtestCount": 20,
            "explorationPercent": 30,
            "selfCorrelationPercent": 30,
            "mutationPercent": 40,
            "directionValidationPercent": 3,
            "maxPendingSeconds": 600,
            "maxConsecutiveFailures": 2,
            "maxRequestFailures": 3,
            "maxInFlightBacktests": 3,
            "explorationSeedAttemptMultiplier": 4,
        }


if __name__ == "__main__":
    unittest.main()
