from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generation.candidate import (
    CandidateChange,
    FormulaCandidate,
    exploration_candidate,
    mutation_candidate,
)
from generation.formula import Name
from generation.parser import parse_formula


class FormulaCandidateTests(unittest.TestCase):
    def test_exploration_candidate_derives_one_normalized_formula_identity(self) -> None:
        candidate = exploration_candidate(Name("close"), strategy_label="directory")

        self.assertEqual(candidate.formula, "close")
        self.assertEqual(len(candidate.fingerprint), 64)
        self.assertEqual(candidate.generation_action, "exploration")
        self.assertIsNone(candidate.parent_task_id)
        self.assertIsNone(candidate.change)

    def test_changed_candidate_requires_direct_parent_and_change_together(self) -> None:
        change = CandidateChange(
            action="group_neutralize",
            location="$",
            before="rank(close)",
            after="group_neutralize(rank(close),subindustry)",
            parameters=(("group", "subindustry"),),
        )
        candidate = FormulaCandidate(
            expression=Name("close"),
            generation_action="repair",
            parent_task_id="backtest_parent",
            parent_formula_fingerprint="parent_formula",
            change=change,
        )

        self.assertEqual(candidate.parent_task_id, "backtest_parent")
        self.assertEqual(candidate.change.parameters, (("group", "subindustry"),))

        with self.assertRaisesRegex(ValueError, "candidate_lineage_incomplete"):
            FormulaCandidate(
                expression=Name("close"),
                generation_action="mutation",
                parent_task_id="backtest_parent",
            )

    def test_change_rejects_duplicate_parameter_names(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "candidate_change_parameter_duplicate",
        ):
            CandidateChange(
                action="repair",
                location="$",
                before="close",
                after="rank(close)",
                parameters=(("group", "industry"), ("group", "subindustry")),
            )

    def test_mutation_candidate_records_direct_lineage(self) -> None:
        change = CandidateChange(
            action="field_swap",
            location="formula.arguments[0]",
            before="close",
            after="open",
        )

        candidate = mutation_candidate(
            Name("open"),
            parent_task_id="backtest_parent",
            parent_formula_fingerprint="parent_formula",
            change=change,
        )

        self.assertEqual(candidate.generation_action, "mutation")
        self.assertEqual(candidate.parent_task_id, "backtest_parent")
        self.assertEqual(candidate.parent_formula_fingerprint, "parent_formula")
        self.assertEqual(candidate.change, change)

    def test_candidate_rejects_deterministic_logic_error(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "candidate_logic_invalid:complete_cancellation",
        ):
            exploration_candidate(parse_formula("close-close").expression)

    def test_candidate_stores_safely_simplified_expression(self) -> None:
        candidate = exploration_candidate(
            parse_formula("(close+0)*1").expression
        )

        self.assertEqual(candidate.formula, "close")


if __name__ == "__main__":
    unittest.main()
