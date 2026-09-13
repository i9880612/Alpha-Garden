import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from generation.direction import reverse_direction_candidate
from generation.parser import parse_formula


class DirectionGenerationTests(unittest.TestCase):
    def test_whole_formula_and_lineage_are_preserved(self):
        for source, expected in (
            ("rank(close)", "-rank(close)"),
            ("close - open", "-(close-open)"),
            ("-rank(close)", "rank(close)"),
        ):
            with self.subTest(source=source):
                parsed = parse_formula(source)
                candidate = reverse_direction_candidate(
                    parsed.expression, parent_task_id="parent"
                )
                self.assertEqual(candidate.formula, expected)
                self.assertEqual(candidate.parent_task_id, "parent")
                self.assertEqual(
                    candidate.parent_formula_fingerprint, parsed.fingerprint
                )
                self.assertEqual(candidate.change.before, parsed.normalized)
                self.assertEqual(candidate.change.after, expected)

    def test_no_constant_candidate(self):
        with self.assertRaises(ValueError):
            reverse_direction_candidate(
                parse_formula("1").expression, parent_task_id="parent"
            )
