from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generation.project_catalog import (
    OPERATOR_OUTPUTS,
    OPERATOR_ROLES,
    STANDARD_WINDOWS,
)


class ProjectCatalogTests(unittest.TestCase):
    def test_project_generation_semantics_are_complete_and_deterministic(self) -> None:
        self.assertEqual(
            STANDARD_WINDOWS,
            (
                (5, "week"),
                (22, "month"),
                (66, "quarter"),
                (120, "half_year"),
                (250, "year"),
            ),
        )
        self.assertEqual(len(OPERATOR_ROLES), 28)
        self.assertEqual(OPERATOR_ROLES, tuple(sorted(OPERATOR_ROLES)))
        self.assertEqual(len(set(OPERATOR_ROLES)), len(OPERATOR_ROLES))
        self.assertEqual(len(OPERATOR_OUTPUTS), 67)
        self.assertEqual(OPERATOR_OUTPUTS, tuple(sorted(OPERATOR_OUTPUTS)))
        self.assertEqual(
            len({operator_name for operator_name, _ in OPERATOR_OUTPUTS}),
            len(OPERATOR_OUTPUTS),
        )
        self.assertEqual(
            Counter(output_kind for _, output_kind in OPERATOR_OUTPUTS),
            {"condition": 10, "group": 2, "signal": 55},
        )
        output_operators = {operator_name for operator_name, _ in OPERATOR_OUTPUTS}
        self.assertTrue(
            all(operator_name in output_operators for operator_name, _ in OPERATOR_ROLES)
        )


if __name__ == "__main__":
    unittest.main()
