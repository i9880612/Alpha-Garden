from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.combo_enhancement import MAX_LINEAGES_PER_COMPANION, load_combo_enhancement_leaves
from generation.combo_enhancement import COMBO_ENHANCEMENT, companion_marker
from generation.parser import parse_formula
from persistence.backtests import BacktestMutationRecord, create_backtest_mutation, list_backtest_mutations
from persistence.database import open_database
from persistence.formula_ideas import FormulaIdeaRecord, record_formula_idea
from persistence.pnl import save_pnl_series
from persistence.schema import initialize_database_schema
from tests.execution.test_qualified_archive import complete_candidate, prepare_candidate
from tests.learning.test_seed_correlation import series


class ComboEnhancementCompanionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database_path = Path(directory.name) / "enhance.sqlite3"
        with open_database(self.database_path) as connection:
            initialize_database_schema(connection)

    def _companion(self, connection, sharpe=1.0):
        snapshot = complete_candidate(connection, prepare_candidate(connection), sharpe=sharpe, qualified=False)
        record_formula_idea(connection, FormulaIdeaRecord(snapshot.task.task_id, "reversal", "price_action",
                                                          ("close",), "test"))
        return snapshot

    def _offered(self, connection, parents):
        seen = {}

        def capture(expression, parent, companions, catalog, **_):
            seen[parent.alpha_id] = [companion.alpha_id for companion in companions]
            return iter(())

        with patch("execution.combo_enhancement.iter_combo_enhancement_leaves", side_effect=capture):
            load_combo_enhancement_leaves(connection, None, account_scope="group-account",
                                          parents=parents, mutations=list_backtest_mutations(connection))
        return seen

    def test_a_companion_used_by_enough_lineages_is_no_longer_offered(self):
        with open_database(self.database_path) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection), grade="AVERAGE")
            companion = self._companion(connection)
            other = self._companion(connection, sharpe=0.9)
            marker = companion_marker(parse_formula(companion.task.formula).expression)
            for _ in range(MAX_LINEAGES_PER_COMPANION):
                user = complete_candidate(connection, prepare_candidate(connection), grade="AVERAGE")
                child = prepare_candidate(connection)
                create_backtest_mutation(connection, BacktestMutationRecord(
                    child.task.task_id, user.task.task_id, COMBO_ENHANCEMENT, "formula",
                    user.task.formula, f"zscore({user.task.formula})+{marker}"))
            offered = self._offered(connection, (parent,))
        self.assertEqual(offered[parent.task.platform_alpha_id], [other.task.platform_alpha_id])

    def test_near_copies_of_one_companion_idea_count_once(self):
        with open_database(self.database_path) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection), grade="AVERAGE")
            strong = self._companion(connection, sharpe=1.2)
            copy = self._companion(connection, sharpe=1.1)
            # Overwrite the copy's curve with the strong companion's.
            connection.execute("DELETE FROM platform_pnl_series WHERE platform_alpha_id=?",
                               (copy.task.platform_alpha_id,))
            points = connection.execute("SELECT points_json FROM platform_pnl_series WHERE platform_alpha_id=?",
                                        (strong.task.platform_alpha_id,)).fetchone()[0]
            increments = [b - a for (_, a), (_, b) in zip(json.loads(points), json.loads(points)[1:])]
            save_pnl_series(connection, series(copy.task.platform_alpha_id, increments, "group-account"))
            offered = self._offered(connection, (parent,))
        self.assertEqual(offered[parent.task.platform_alpha_id], [strong.task.platform_alpha_id])

    def test_a_concentrated_signal_is_never_a_companion(self):
        with open_database(self.database_path) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection), grade="AVERAGE")
            concentrated = self._companion(connection, sharpe=1.3)
            clean = self._companion(connection, sharpe=0.9)
            connection.execute("UPDATE backtest_checks SET status='FAIL' WHERE task_id=? "
                               "AND check_name='CONCENTRATED_WEIGHT'", (concentrated.task.task_id,))
            offered = self._offered(connection, (parent,))
        self.assertEqual(offered[parent.task.platform_alpha_id], [clean.task.platform_alpha_id])

    def _accept(self, connection, parent):
        captured = {}

        def capture(expression, parent_signal, companions, catalog, **kwargs):
            captured["accept"] = kwargs["accept"]
            captured["companions"] = {c.alpha_id: c for c in companions}
            return iter(())

        with patch("execution.combo_enhancement.iter_combo_enhancement_leaves", side_effect=capture):
            load_combo_enhancement_leaves(connection, None, account_scope="group-account",
                                          parents=(parent,), mutations=list_backtest_mutations(connection))
        return captured

    def _replace_curve(self, connection, snapshot, increments):
        connection.execute("DELETE FROM platform_pnl_series WHERE platform_alpha_id=?",
                           (snapshot.task.platform_alpha_id,))
        save_pnl_series(connection, series(snapshot.task.platform_alpha_id, increments, "group-account"))

    def _increments(self, connection, snapshot):
        points = json.loads(connection.execute(
            "SELECT points_json FROM platform_pnl_series WHERE platform_alpha_id=?",
            (snapshot.task.platform_alpha_id,)).fetchone()[0])
        return [b - a for (_, a), (_, b) in zip(points, points[1:])]

    def test_a_child_predicted_to_copy_another_lineage_is_skipped_before_backtest(self):
        from generation.combo_enhancement import EnhancementPlan
        with open_database(self.database_path) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection), grade="AVERAGE")
            companion = self._companion(connection)
            other = complete_candidate(connection, prepare_candidate(connection), grade="GOOD")
            # The other lineage already holds parent + companion.
            mixed = [p + c for p, c in zip(self._increments(connection, parent), self._increments(connection, companion))]
            self._replace_curve(connection, other, mixed)
            captured = self._accept(connection, parent)
            plan = EnhancementPlan(sign=1, weight=1.0, expected_gain=1.2, correlation=0.0, sharpe_gain=1.2)
            self.assertFalse(captured["accept"](plan, captured["companions"][companion.task.platform_alpha_id]))
            # A different mix of the same two signals is still a different alpha.
            far = EnhancementPlan(sign=-1, weight=1.0, expected_gain=1.2, correlation=0.0, sharpe_gain=1.2)
            self.assertTrue(captured["accept"](far, captured["companions"][companion.task.platform_alpha_id]))

    def test_resembling_its_own_lineage_is_allowed(self):
        from generation.combo_enhancement import EnhancementPlan
        with open_database(self.database_path) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection), grade="AVERAGE")
            companion = self._companion(connection)
            sibling = complete_candidate(connection, prepare_candidate(connection, parent=parent), grade="GOOD")
            mixed = [p + c for p, c in zip(self._increments(connection, parent), self._increments(connection, companion))]
            self._replace_curve(connection, sibling, mixed)
            captured = self._accept(connection, parent)
            plan = EnhancementPlan(sign=1, weight=1.0, expected_gain=1.2, correlation=0.0, sharpe_gain=1.2)
            self.assertTrue(captured["accept"](plan, captured["companions"][companion.task.platform_alpha_id]))
