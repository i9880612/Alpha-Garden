from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from selection.candidates import (
    DiversityDimension,
    SelectionCandidate,
    select_candidates,
)


class CandidateSelectionTests(unittest.TestCase):
    def test_selects_one_candidate_per_family_in_stable_identity_order(self) -> None:
        result = select_candidates(
            (
                SelectionCandidate("candidate_c", "family_c"),
                SelectionCandidate("candidate_a", "family_a"),
                SelectionCandidate("candidate_b", "family_b"),
            ),
            requested_count=2,
        )

        self.assertEqual(
            tuple(item.candidate_id for item in result.selected),
            ("candidate_a", "candidate_b"),
        )
        self.assertEqual(result.missing_count, 0)
        self.assertTrue(
            all(
                item.representative_reason == "only_family_member"
                for item in result.selected
            )
        )

    def test_multiple_family_members_require_an_explicit_representative(self) -> None:
        candidates = (
            SelectionCandidate("candidate_root", "candidate_root"),
            SelectionCandidate("candidate_child", "candidate_root"),
        )

        with self.assertRaisesRegex(
            ValueError,
            "selection_family_representative_required",
        ):
            select_candidates(candidates, requested_count=1)

    def test_explicit_representative_is_used_without_same_family_backfill(self) -> None:
        result = select_candidates(
            (
                SelectionCandidate("candidate_root", "candidate_root"),
                SelectionCandidate("candidate_child", "candidate_root"),
                SelectionCandidate("candidate_other", "candidate_other"),
            ),
            requested_count=3,
            family_representatives=(("candidate_root", "candidate_child"),),
        )

        self.assertEqual(
            {item.candidate_id for item in result.selected},
            {"candidate_child", "candidate_other"},
        )
        self.assertEqual(result.missing_count, 1)
        selected_child = next(
            item
            for item in result.selected
            if item.candidate_id == "candidate_child"
        )
        self.assertEqual(
            selected_child.representative_reason,
            "explicit_family_representative",
        )

    def test_representative_must_belong_to_the_declared_family(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "selection_family_representative_mismatch",
        ):
            select_candidates(
                (
                    SelectionCandidate("candidate_a", "family_a"),
                    SelectionCandidate("candidate_b", "family_b"),
                ),
                requested_count=1,
                family_representatives=(("family_a", "candidate_b"),),
            )

    def test_prefers_new_dimensions_without_claiming_quality(self) -> None:
        repeated = (
            DiversityDimension("binary_operator", ("*",)),
            DiversityDimension("cross_sectional_operator", ("rank",)),
        )
        result = select_candidates(
            (
                SelectionCandidate("candidate_a", "family_a", repeated),
                SelectionCandidate("candidate_b", "family_b", repeated),
                SelectionCandidate(
                    "candidate_c",
                    "family_c",
                    (
                        DiversityDimension("binary_operator", ("+",)),
                        DiversityDimension(
                            "cross_sectional_operator",
                            ("zscore",),
                        ),
                    ),
                ),
            ),
            requested_count=2,
        )

        self.assertEqual(
            tuple(item.candidate_id for item in result.selected),
            ("candidate_a", "candidate_c"),
        )

    def test_rejected_candidates_never_consume_selection_quota(self) -> None:
        result = select_candidates(
            (
                SelectionCandidate(
                    "unsafe",
                    "unsafe",
                    rejection_reasons=("uncontrolled_tail",),
                ),
                SelectionCandidate("safe", "safe"),
            ),
            requested_count=2,
        )

        self.assertEqual(
            tuple(item.candidate_id for item in result.selected),
            ("safe",),
        )
        self.assertEqual(result.missing_count, 1)
        self.assertEqual(result.rejected[0].candidate_id, "unsafe")

    def test_counts_dimensions_equally_instead_of_raw_value_count(self) -> None:
        result = select_candidates(
            (
                SelectionCandidate(
                    "candidate_a",
                    "family_a",
                    (
                        DiversityDimension("window", ("5",)),
                        DiversityDimension("operator", ("rank",)),
                    ),
                ),
                SelectionCandidate(
                    "candidate_b",
                    "family_b",
                    (
                        DiversityDimension("window", ("22", "66", "120")),
                    ),
                ),
                SelectionCandidate(
                    "candidate_c",
                    "family_c",
                    (
                        DiversityDimension("window", ("22",)),
                        DiversityDimension("operator", ("zscore",)),
                    ),
                ),
            ),
            requested_count=2,
        )

        self.assertEqual(
            tuple(item.candidate_id for item in result.selected),
            ("candidate_a", "candidate_c"),
        )

    def test_continues_balancing_structure_after_initial_coverage(self) -> None:
        candidates = tuple(
            SelectionCandidate(
                candidate_id,
                candidate_id,
                (
                    DiversityDimension("binary_operator", (operator,)),
                    DiversityDimension(
                        "cross_sectional_operator",
                        ("rank",),
                    ),
                ),
            )
            for candidate_id, operator in (
                ("candidate_a", "*"),
                ("candidate_b", "*"),
                ("candidate_c", "*"),
                ("candidate_d", "+"),
                ("candidate_e", "+"),
                ("candidate_f", "+"),
                ("candidate_g", "-"),
                ("candidate_h", "-"),
                ("candidate_i", "-"),
            )
        )

        result = select_candidates(candidates, requested_count=6)

        operators_by_candidate = {
            candidate.candidate_id: candidate.diversity[0].values[0]
            for candidate in candidates
        }
        selected_operators = tuple(
            operators_by_candidate[item.candidate_id]
            for item in result.selected
        )
        self.assertEqual(selected_operators.count("*"), 2)
        self.assertEqual(selected_operators.count("+"), 2)
        self.assertEqual(selected_operators.count("-"), 2)


if __name__ == "__main__":
    unittest.main()
