from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persistence.run_diagnostics import (
    CandidatePlanningDiagnostic,
    CandidatePlanningExclusionCount,
)


class CandidatePlanningDiagnosticTests(unittest.TestCase):
    def test_generation_stop_round_trips_as_canonical_json(self) -> None:
        diagnostic = self._generation_stop()

        encoded = diagnostic.canonical_json(
            stop_reason="generation_attempt_budget_exhausted"
        )
        decoded = CandidatePlanningDiagnostic.from_canonical_json(
            encoded,
            stop_reason="generation_attempt_budget_exhausted",
        )

        self.assertEqual(decoded, diagnostic)
        self.assertEqual(
            encoded,
            '{"attemptedSeedCount":12,"cycleNumber":2,'
            '"explorationBacktestTargetCount":2,'
            '"explorationGenerationTargetCount":3,'
            '"generatedCandidateCount":1,'
            '"generationExclusionCounts":{"duplicate":4,"invalid":7},'
            '"seedAttemptLimit":12,"selectedCandidateCount":null,'
            '"selectionRejectionCounts":{}}',
        )

    def test_selection_stop_keeps_generation_and_selection_reasons_distinct(
        self,
    ) -> None:
        diagnostic = CandidatePlanningDiagnostic(
            cycle_number=1,
            exploration_generation_target_count=3,
            exploration_backtest_target_count=2,
            seed_attempt_limit=12,
            attempted_seed_count=4,
            generated_candidate_count=3,
            selected_candidate_count=1,
            generation_exclusions=(
                CandidatePlanningExclusionCount("duplicate", 1),
            ),
            selection_rejections=(
                CandidatePlanningExclusionCount("low_coverage", 1),
                CandidatePlanningExclusionCount("uncontrolled_tail", 2),
            ),
        )

        encoded = diagnostic.canonical_json(
            stop_reason="selection_rejection_shortfall"
        )

        self.assertEqual(
            CandidatePlanningDiagnostic.from_canonical_json(
                encoded,
                stop_reason="selection_rejection_shortfall",
            ),
            diagnostic,
        )

    def test_noncanonical_or_inconsistent_diagnostic_is_rejected(self) -> None:
        diagnostic = self._generation_stop()
        encoded = diagnostic.canonical_json(
            stop_reason="generation_attempt_budget_exhausted"
        )
        cases = (
            (
                lambda: CandidatePlanningDiagnostic.from_canonical_json(
                    encoded + " ",
                    stop_reason="generation_attempt_budget_exhausted",
                ),
                "candidate_planning_diagnostic_invalid",
            ),
            (
                lambda: replace(
                    diagnostic,
                    attempted_seed_count=11,
                    generation_exclusions=(
                        CandidatePlanningExclusionCount("duplicate", 4),
                        CandidatePlanningExclusionCount("invalid", 6),
                    ),
                ).canonical_json(
                    stop_reason="generation_attempt_budget_exhausted"
                ),
                "candidate_planning_generation_stop_invalid",
            ),
            (
                lambda: replace(
                    diagnostic,
                    generation_exclusions=(
                        CandidatePlanningExclusionCount("invalid", 10),
                    ),
                ).canonical_json(
                    stop_reason="generation_attempt_budget_exhausted"
                ),
                "candidate_planning_generation_exclusions_invalid",
            ),
            (
                lambda: CandidatePlanningDiagnostic(
                    cycle_number=1,
                    exploration_generation_target_count=3,
                    exploration_backtest_target_count=3,
                    seed_attempt_limit=12,
                    attempted_seed_count=3,
                    generated_candidate_count=3,
                    selected_candidate_count=2,
                    generation_exclusions=(),
                    selection_rejections=(
                        CandidatePlanningExclusionCount(
                            "impossible_duplicate_reason",
                            2,
                        ),
                    ),
                ).canonical_json(
                    stop_reason="selection_rejection_shortfall"
                ),
                "candidate_planning_selection_rejections_invalid",
            ),
        )
        for action, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(
                ValueError,
                error,
            ):
                action()

    @staticmethod
    def _generation_stop() -> CandidatePlanningDiagnostic:
        return CandidatePlanningDiagnostic(
            cycle_number=2,
            exploration_generation_target_count=3,
            exploration_backtest_target_count=2,
            seed_attempt_limit=12,
            attempted_seed_count=12,
            generated_candidate_count=1,
            selected_candidate_count=None,
            generation_exclusions=(
                CandidatePlanningExclusionCount("duplicate", 4),
                CandidatePlanningExclusionCount("invalid", 7),
            ),
            selection_rejections=(),
        )


if __name__ == "__main__":
    unittest.main()
