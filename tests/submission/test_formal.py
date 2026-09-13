from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from submission.formal import (
    FormalSubmissionCandidate,
    assess_formal_check_payload,
    confirm_formal_submission,
    select_next_family_candidate,
)


class FormalSubmissionDecisionTests(unittest.TestCase):
    def test_requires_every_current_standard_check_to_pass(self) -> None:
        names = (
            "CONCENTRATED_WEIGHT",
            "HIGH_TURNOVER",
            "LOW_FITNESS",
            "LOW_SHARPE",
            "LOW_SUB_UNIVERSE_SHARPE",
            "LOW_TURNOVER",
            "MATCHES_COMPETITION",
            "SELF_CORRELATION",
        )
        passed = assess_formal_check_payload(
            {"is": {"checks": [{"name": name, "result": "PASS"} for name in names]}}
        )
        pending = assess_formal_check_payload(
            {
                "is": {
                    "checks": [
                        {"name": name, "result": "PASS"}
                        for name in names
                        if name != "SELF_CORRELATION"
                    ]
                }
            }
        )
        failed = assess_formal_check_payload(
            {
                "is": {
                    "checks": [
                        {
                            "name": name,
                            "result": "FAIL" if name == "SELF_CORRELATION" else "PASS",
                        }
                        for name in names
                    ]
                }
            }
        )

        self.assertEqual(passed.state, "passed")
        self.assertEqual(pending.state, "pending")
        self.assertEqual(pending.unresolved_checks, ("SELF_CORRELATION",))
        self.assertEqual(failed.state, "failed")
        self.assertEqual(failed.failed_checks, ("SELF_CORRELATION",))

    def test_new_platform_check_cannot_be_ignored(self) -> None:
        names = (
            "CONCENTRATED_WEIGHT",
            "HIGH_TURNOVER",
            "LOW_FITNESS",
            "LOW_SHARPE",
            "LOW_SUB_UNIVERSE_SHARPE",
            "LOW_TURNOVER",
            "MATCHES_COMPETITION",
            "SELF_CORRELATION",
        )

        failed = assess_formal_check_payload(
            {
                "is": {
                    "checks": [
                        *({"name": name, "result": "PASS"} for name in names),
                        {"name": "NEW_PLATFORM_GATE", "result": "FAIL"},
                    ]
                }
            }
        )
        pending = assess_formal_check_payload(
            {
                "is": {
                    "checks": [
                        *({"name": name, "result": "PASS"} for name in names),
                        {"name": "NEW_PLATFORM_GATE", "result": "PENDING"},
                    ]
                }
            }
        )

        self.assertEqual(failed.state, "failed")
        self.assertEqual(failed.failed_checks, ("NEW_PLATFORM_GATE",))
        self.assertEqual(pending.state, "pending")
        self.assertEqual(pending.unresolved_checks, ("NEW_PLATFORM_GATE",))

    def test_selects_lowest_sharpe_inside_highest_potential_family(self) -> None:
        candidates = (
            self._candidate("a-low", "family-a", 1.4, 1.8),
            self._candidate("a-high", "family-a", 2.1, 1.0),
            self._candidate("b", "family-b", 2.0, 2.0),
        )

        selected = select_next_family_candidate(candidates)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.task_id, "a-low")

    def test_family_priority_uses_its_highest_sharpe_candidate(self) -> None:
        candidates = (
            self._candidate("a-low", "family-a", 1.2, 1.0),
            self._candidate("a-high", "family-a", 2.2, 1.0),
            self._candidate("b-low", "family-b", 1.5, 2.0),
            self._candidate("b-high", "family-b", 2.0, 2.0),
        )

        selected = select_next_family_candidate(candidates)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.task_id, "a-low")

    def test_confirmation_rejects_unsubmitted_or_mismatched_formula(self) -> None:
        unsubmitted = {
            "id": "alpha-1",
            "status": "UNSUBMITTED",
            "regular": {"code": "rank(close)"},
        }
        self.assertIsNone(
            confirm_formal_submission(
                unsubmitted,
                expected_alpha_id="alpha-1",
                expected_normalized_formula="rank(close)",
                normalize_formula=lambda value: value.replace(" ", "").lower(),
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "formal_submission_confirmation_formula_mismatch",
        ):
            confirm_formal_submission(
                {
                    "id": "alpha-1",
                    "status": "ACTIVE",
                    "dateSubmitted": "2026-09-04T01:00:00+00:00",
                    "hidden": False,
                    "regular": {"code": "rank(open)"},
                },
                expected_alpha_id="alpha-1",
                expected_normalized_formula="rank(close)",
                normalize_formula=lambda value: value.replace(" ", "").lower(),
            )

    @staticmethod
    def _candidate(
        task_id: str,
        family_root_task_id: str,
        sharpe: float,
        fitness: float,
    ) -> FormalSubmissionCandidate:
        return FormalSubmissionCandidate(
            task_id=task_id,
            cycle_number=1,
            family_root_task_id=family_root_task_id,
            platform_alpha_id=f"alpha-{task_id}",
            formula="rank(close)",
            normalized_formula="rank(close)",
            sharpe=sharpe,
            fitness=fitness,
            finished_at="2026-09-04T01:00:00+00:00",
        )


if __name__ == "__main__":
    unittest.main()
