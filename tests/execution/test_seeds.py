from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.backtests import (
    apply_backtest_detail,
    prepare_backtest_task,
    record_backtest_yearly_stats,
    record_submission_accepted,
)
from execution.seeds import (
    batch_advances_signal_frontier,
    load_signal_frontiers,
    synchronize_signal_seeds,
)
from generation.direction import reverse_direction_candidate
from generation.parser import parse_formula
from persistence.backtests import (
    BacktestMutationRecord,
    create_backtest_mutation,
    get_backtest_task,
    initialize_backtest_schema,
)
from persistence.database import open_database
from persistence.runs import initialize_run_schema
from persistence.submissions import initialize_submission_schema
from persistence.seeds import list_signal_seeds
from persistence.pnl import save_pnl_series
from persistence.submissions import PlatformSubmittedAlphaRecord, record_platform_submitted_alphas
from worldquant.backtests import BacktestCheck, BacktestDetail


def _checks(statuses: dict[str, str]) -> tuple[BacktestCheck, ...]:
    return tuple(
        BacktestCheck(
            name=name,
            status=status,
            threshold=None,
            actual=None,
            platform_date=None,
        )
        for name, status in sorted(statuses.items())
    )


class SignalSeedExecutionTests(unittest.TestCase):
    def test_optimization_excludes_unqualified_research_branch(self):
        with open_database(self.database_path) as connection:
            self._completed(connection, "rank(close)", "1", 1.3, 0.9)
            synchronize_signal_seeds(connection)
            self.assertTrue(load_signal_frontiers(connection).active_branch_task_ids)
            self.assertEqual(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids, ())

    def test_optimization_requires_known_below_target_grade_and_complete_checks(self):
        from tests.execution.test_qualified_archive import prepare_candidate, complete_candidate
        from worldquant.backtests import STANDARD_REGULAR_CHECK_NAMES

        with open_database(self.database_path) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection))
            synchronize_signal_seeds(connection)
            task_id = parent.task.task_id
            for grade in ("INFERIOR", "AVERAGE", "GOOD", "EXCELLENT", "SPECTACULAR", None):
                with self.subTest(grade=grade):
                    connection.execute("UPDATE backtest_results SET grade=? WHERE task_id=?", (grade, task_id))
                    expected = () if grade in (None, "SPECTACULAR") else (task_id,)
                    self.assertEqual(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids, expected)
                    self.assertEqual(load_signal_frontiers(connection).active_branch_task_ids, (task_id,))
            connection.execute("UPDATE backtest_results SET grade='GOOD' WHERE task_id=?", (task_id,))
            for state in ("FAIL", "PENDING", "MISSING", "ERROR"):
                with self.subTest(sc=state):
                    checks = [{"name": name, "result": state if name == "SELF_CORRELATION" else "PASS"}
                              for name in STANDARD_REGULAR_CHECK_NAMES
                              if not (state == "MISSING" and name == "SELF_CORRELATION")]
                    payload = None if state == "ERROR" else json.dumps({"is": {"checks": checks}})
                    connection.execute("UPDATE submission_checks SET payload_json=?, error_code=? WHERE task_id=?",
                                       (payload, "request_failed" if state == "ERROR" else None, task_id))
                    self.assertEqual(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids, ())
                    self.assertEqual(load_signal_frontiers(connection).active_branch_task_ids, (task_id,))
            passed = json.dumps({"is": {"checks": [{"name": name, "result": "PASS"}
                                                     for name in STANDARD_REGULAR_CHECK_NAMES]}})
            connection.execute("UPDATE submission_checks SET payload_json=?, error_code=NULL WHERE task_id=?", (passed, task_id))
            connection.execute("DELETE FROM backtest_yearly_stats WHERE task_id=?", (task_id,))
            self.assertEqual(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids, ())

    def test_optimization_follows_takeover_without_reviving_replaced_parent(self):
        from tests.execution.test_qualified_archive import prepare_candidate, complete_candidate

        with open_database(self.database_path) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection))
            synchronize_signal_seeds(connection)
            child = complete_candidate(connection, prepare_candidate(connection, parent=parent),
                                       sharpe=1.8, grade="EXCELLENT")
            frontier = load_signal_frontiers(connection, optimization_only=True)
            self.assertEqual(frontier.active_branch_task_ids, (child.task.task_id,))
            self.assertEqual(frontier.records[0].qualified_parent_remaining_attempts, 20)
            connection.execute("UPDATE backtest_results SET grade='SPECTACULAR' WHERE task_id=?", (child.task.task_id,))
            self.assertEqual(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids, ())
            self.assertEqual(load_signal_frontiers(connection).active_branch_task_ids, (child.task.task_id,))

    def test_optimization_excludes_submitted_formula_without_retiring_normal_research(self):
        from tests.execution.test_qualified_archive import prepare_candidate, complete_candidate

        with open_database(self.database_path) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection))
            synchronize_signal_seeds(connection)
            record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
                parent.task.account_scope, parent.task.platform_alpha_id, parent.task.formula, "ACTIVE",
                "2026-09-01T00:00:00+00:00", False,
                {"id": parent.task.platform_alpha_id, "status": "ACTIVE", "hidden": False,
                 "dateSubmitted": "2026-09-01T00:00:00+00:00", "regular": {"code": parent.task.formula}},
                "2026-09-09T00:00:00+00:00"),))
            self.assertEqual(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids, ())
            self.assertEqual(load_signal_frontiers(connection).active_branch_task_ids, (parent.task.task_id,))

    def test_optimization_keeps_checked_descendant_of_submitted_higher_sharpe_parent(self):
        from tests.execution.test_qualified_archive import prepare_candidate, complete_candidate

        with open_database(self.database_path) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection), sharpe=2.1)
            synchronize_signal_seeds(connection)
            child = complete_candidate(connection, prepare_candidate(connection, parent=parent), sharpe=1.6)
            record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
                parent.task.account_scope, parent.task.platform_alpha_id, parent.task.formula, "ACTIVE",
                "2026-09-01T00:00:00+00:00", False,
                {"id": parent.task.platform_alpha_id, "status": "ACTIVE", "hidden": False,
                 "dateSubmitted": "2026-09-01T00:00:00+00:00", "regular": {"code": parent.task.formula}},
                "2026-09-09T00:00:00+00:00"),))
            self.assertEqual(load_signal_frontiers(connection).active_branch_task_ids, (parent.task.task_id,))
            special = load_signal_frontiers(connection, optimization_only=True)
            self.assertEqual(special.active_branch_task_ids, (child.task.task_id,))
            self.assertEqual(special.records[0].branches[0].remaining_attempts, 20)
            self.assertEqual(special.records[0].root_task_id, parent.task.task_id)

    def test_optimization_does_not_replace_checked_parent_with_sc_failed_better_child(self):
        from tests.execution.test_qualified_archive import prepare_candidate, complete_candidate

        with open_database(self.database_path) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection), sharpe=1.5)
            synchronize_signal_seeds(connection)
            child = complete_candidate(connection, prepare_candidate(connection, parent=parent), sharpe=1.8)
            payload = json.loads(connection.execute('SELECT payload_json FROM submission_checks WHERE task_id=?',
                                                    (child.task.task_id,)).fetchone()[0])
            for check in payload['is']['checks']:
                if check['name'] == 'SELF_CORRELATION':
                    check['result'] = 'FAIL'
            connection.execute('UPDATE submission_checks SET payload_json=? WHERE task_id=?',
                               (json.dumps(payload), child.task.task_id))
            self.assertEqual(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids,
                             (parent.task.task_id,))

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "seeds.sqlite3"
        with open_database(self.database_path) as connection:
            initialize_backtest_schema(connection)
            initialize_run_schema(connection)
            initialize_submission_schema(connection)

    def test_correlated_lineage_keeps_research_budget_without_creating_child_roots(self):
        import math
        from tests.learning.test_seed_correlation import series
        from persistence.backtests import list_backtest_mutations
        from learning.seed_correlation import assess_seed_correlation
        from persistence.submissions import list_platform_submitted_alphas
        from persistence.pnl import list_pnl_series
        with open_database(self.database_path) as connection:
            root = self._completed(connection, "rank(close)", "1", 1.3, 0.9)
            synchronize_signal_seeds(connection)
            child = self._child(connection, root.task.task_id, "rank(open)", "2",
                                sharpe=2.0, fitness=1.1, base_passed=True)
            self._child(connection, child.task.task_id, "rank(volume)", "3", sharpe=0.5, fitness=0.3)
            record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
                "group-account", "reference", "rank(reference)", "ACTIVE",
                "2026-09-01T00:00:00+00:00", False,
                {"id": "reference", "status": "ACTIVE", "hidden": False,
                 "dateSubmitted": "2026-09-01T00:00:00+00:00", "regular": {"code": "rank(reference)"}},
                "2026-09-09T00:00:00+00:00"),))
            for alpha in (root.task.platform_alpha_id, child.task.platform_alpha_id, "reference"):
                save_pnl_series(connection, series(alpha, [math.sin(i) for i in range(300)], "group-account"))
            refs = list_platform_submitted_alphas(connection, account_scope="group-account")
            self.assertEqual(assess_seed_correlation(child, refs, list_pnl_series(connection)).state, "failed")
            mutations = list_backtest_mutations(connection)
            before = load_signal_frontiers(connection)
            self.assertEqual(before.active_branch_task_ids, (child.task.task_id,))
            self.assertEqual(before.records[0].branches[0].remaining_attempts, 19)
            self.assertEqual(synchronize_signal_seeds(connection), ())
            self.assertEqual(tuple(x.root_task_id for x in list_signal_seeds(connection)), (root.task.task_id,))
            self.assertEqual(load_signal_frontiers(connection), before)
            self.assertEqual(list_backtest_mutations(connection), mutations)

    def test_missing_pnl_blocks_new_roots_but_allows_existing_lineage_research(self):
        with open_database(self.database_path) as connection:
            root = self._completed(connection, "rank(close)", "1", 1.3, 0.9)
            synchronize_signal_seeds(connection)
            child = self._child(connection, root.task.task_id, "rank(open)", "2", sharpe=2.0, fitness=1.1, base_passed=True)
            record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
                "group-account", "reference", "rank(reference)", "ACTIVE",
                "2026-09-01T00:00:00+00:00", False,
                {"id": "reference", "status": "ACTIVE", "hidden": False,
                 "dateSubmitted": "2026-09-01T00:00:00+00:00", "regular": {"code": "rank(reference)"}},
                "2026-09-09T00:00:00+00:00"),))
            self._completed(connection, "rank(volume)", "3", 1.3, 0.9)
            self.assertEqual(synchronize_signal_seeds(connection), ())
            self.assertEqual(len(list_signal_seeds(connection)), 1)
            self.assertEqual(load_signal_frontiers(connection).active_branch_task_ids, (child.task.task_id,))

    def test_confirmed_direction_result_starts_positive_root_with_source_lineage(self):
        with open_database(self.database_path) as connection:
            source = self._completed(connection, "rank(close)", "1", -1.4, -1.1)
            child = self._completed(connection, "-rank(close)", "2", 1.3, 0.9)
            candidate = reverse_direction_candidate(
                parse_formula(source.task.formula).expression,
                parent_task_id=source.task.task_id,
            )
            change = candidate.change
            create_backtest_mutation(
                connection,
                BacktestMutationRecord(
                    child_task_id=child.task.task_id,
                    parent_task_id=source.task.task_id,
                    action=change.action,
                    location=change.location,
                    before=change.before,
                    after=change.after,
                ),
            )
            created = synchronize_signal_seeds(connection)
            self.assertEqual(
                tuple(s.root_task_id for s in created), (child.task.task_id,)
            )
            self.assertEqual(synchronize_signal_seeds(connection), ())
            frontiers = load_signal_frontiers(connection)
            self.assertEqual(frontiers.active_branch_task_ids, (child.task.task_id,))
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM backtest_mutations"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(source.result.sharpe, -1.4)

    def test_direction_pending_or_low_quality_result_is_not_a_seed(self):
        with open_database(self.database_path) as connection:
            for index, complete in enumerate((False, True)):
                formula = ("rank(close)", "rank(open)")[index]
                source = self._completed(
                    connection, formula, str(index * 2 + 1), -1.4, -1.1
                )
                candidate = reverse_direction_candidate(
                    parse_formula(formula).expression,
                    parent_task_id=source.task.task_id,
                )
                child = self._completed(
                    connection,
                    candidate.formula,
                    str(index * 2 + 2),
                    0.8,
                    0.5,
                    complete=complete,
                )
                change = candidate.change
                create_backtest_mutation(
                    connection,
                    BacktestMutationRecord(
                        child_task_id=child.task.task_id,
                        parent_task_id=source.task.task_id,
                        action=change.action,
                        location=change.location,
                        before=change.before,
                        after=change.after,
                    ),
                )
            self.assertEqual(synchronize_signal_seeds(connection), ())

    def test_positive_exploration_result_starts_a_seed_lineage(self) -> None:
        with open_database(self.database_path) as connection:
            root = self._completed(connection, "rank(close)", "1", 1.0, 0.7)
            created = synchronize_signal_seeds(
                connection,
                candidate_task_ids=(root.task.task_id,),
            )
            replayed = synchronize_signal_seeds(
                connection,
                candidate_task_ids=(root.task.task_id,),
            )

            seeds = list_signal_seeds(connection)
            advanced = batch_advances_signal_frontier(
                connection,
                (root.task.task_id,),
            )

        self.assertEqual(created, seeds)
        self.assertEqual(replayed, ())
        self.assertTrue(advanced)
        self.assertEqual(seeds[0].root_task_id, root.task.task_id)

    def test_exploration_below_new_admission_floor_does_not_start_seed(self) -> None:
        with open_database(self.database_path) as connection:
            low_sharpe = self._completed(
                connection,
                "rank(close)",
                "1",
                0.999,
                0.7,
            )
            low_fitness = self._completed(
                connection,
                "rank(open)",
                "2",
                1.0,
                0.699,
            )

            created = synchronize_signal_seeds(
                connection,
                candidate_task_ids=(
                    low_sharpe.task.task_id,
                    low_fitness.task.task_id,
                ),
            )

        self.assertEqual(created, ())

    def test_low_sub_universe_failure_starts_a_repair_lineage(self) -> None:
        with open_database(self.database_path) as connection:
            root = self._completed(
                connection,
                "rank(close)",
                "1",
                1.0,
                0.7,
                checks={
                    "LOW_SHARPE": "PASS",
                    "LOW_FITNESS": "PASS",
                    "LOW_SUB_UNIVERSE_SHARPE": "FAIL",
                    "CONCENTRATED_WEIGHT": "PASS",
                    "LOW_TURNOVER": "PASS",
                    "HIGH_TURNOVER": "PASS",
                    "MATCHES_COMPETITION": "PASS",
                    "SELF_CORRELATION": "PENDING",
                },
            )

            created = synchronize_signal_seeds(
                connection,
                candidate_task_ids=(root.task.task_id,),
            )
            frontiers = load_signal_frontiers(connection)

        self.assertEqual(len(created), 1)
        self.assertEqual(frontiers.active_branch_task_ids, (root.task.task_id,))

    def test_fully_passing_exploration_result_starts_qualified_lineage(self) -> None:
        with open_database(self.database_path) as connection:
            root = self._completed(
                connection,
                "rank(close)",
                "1",
                1.4,
                1.1,
                checks={
                    "LOW_SHARPE": "PASS",
                    "LOW_FITNESS": "PASS",
                    "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                    "CONCENTRATED_WEIGHT": "PASS",
                    "LOW_TURNOVER": "PASS",
                    "HIGH_TURNOVER": "PASS",
                    "MATCHES_COMPETITION": "PASS",
                    "SELF_CORRELATION": "PASS",
                },
            )

            created = synchronize_signal_seeds(
                connection,
                candidate_task_ids=(root.task.task_id,),
            )
            frontiers = load_signal_frontiers(connection)

        self.assertEqual(len(created), 1)
        self.assertTrue(frontiers.records[0].qualified)
        self.assertEqual(frontiers.active_branch_task_ids, (root.task.task_id,))

    def test_partial_failed_result_cannot_start_a_seed_lineage(self) -> None:
        with open_database(self.database_path) as connection:
            root = self._completed(
                connection,
                "rank(close)",
                "1",
                1.1,
                0.8,
                checks={
                    "LOW_SHARPE": "FAIL",
                    "LOW_FITNESS": "PASS",
                    "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                    "CONCENTRATED_WEIGHT": "PASS",
                },
            )
            created = synchronize_signal_seeds(
                connection,
                candidate_task_ids=(root.task.task_id,),
            )

            self.assertEqual(created, ())
            self.assertEqual(list_signal_seeds(connection), ())

    def test_tradeoff_children_remain_active_together(self) -> None:
        with open_database(self.database_path) as connection:
            root = self._completed(connection, "rank(close)", "1", 1.1, 0.8)
            synchronize_signal_seeds(
                connection,
                candidate_task_ids=(root.task.task_id,),
            )
            sharpe = self._child(
                connection,
                root.task.task_id,
                "rank(open)",
                "2",
                sharpe=1.3,
                fitness=0.75,
            )
            fitness = self._child(
                connection,
                root.task.task_id,
                "rank(returns)",
                "3",
                sharpe=1.05,
                fitness=0.9,
            )

            advanced = batch_advances_signal_frontier(
                connection,
                (sharpe.task.task_id, fitness.task.task_id),
            )
            frontiers = load_signal_frontiers(connection)

        self.assertTrue(advanced)
        self.assertEqual(
            {branch.task_id for branch in frontiers.records[0].branches},
            {root.task.task_id, sharpe.task.task_id, fitness.task.task_id},
        )

    def test_dominated_or_incomparable_child_does_not_advance_frontier(self) -> None:
        with open_database(self.database_path) as connection:
            root = self._completed(connection, "rank(close)", "1", 1.1, 0.8)
            synchronize_signal_seeds(
                connection,
                candidate_task_ids=(root.task.task_id,),
            )
            dominated = self._child(
                connection,
                root.task.task_id,
                "rank(open)",
                "2",
                sharpe=1.0,
                fitness=0.7,
            )
            different = self._child(
                connection,
                root.task.task_id,
                "rank(returns)",
                "3",
                sharpe=1.2,
                fitness=0.9,
                settings={"delay": 0},
            )

            advanced = batch_advances_signal_frontier(
                connection,
                (dominated.task.task_id, different.task.task_id),
            )

        self.assertFalse(advanced)

    def test_qualified_child_becomes_active_without_persisted_status(self) -> None:
        with open_database(self.database_path) as connection:
            root = self._completed(connection, "rank(close)", "1", 1.0, 0.7)
            synchronize_signal_seeds(
                connection,
                candidate_task_ids=(root.task.task_id,),
            )
            qualified = self._child(
                connection,
                root.task.task_id,
                "rank(open)",
                "2",
                sharpe=1.3,
                fitness=1.1,
                base_passed=True,
            )

            advanced = batch_advances_signal_frontier(
                connection,
                (qualified.task.task_id,),
            )
            frontiers = load_signal_frontiers(connection)

        self.assertTrue(advanced)
        self.assertTrue(frontiers.records[0].qualified)
        self.assertEqual(
            frontiers.active_branch_task_ids,
            (qualified.task.task_id,),
        )

    def test_only_submitted_children_consume_qualified_parent_budget(self) -> None:
        passing_checks = {
            "LOW_SHARPE": "PASS",
            "LOW_FITNESS": "PASS",
            "LOW_SUB_UNIVERSE_SHARPE": "PASS",
            "CONCENTRATED_WEIGHT": "PASS",
            "LOW_TURNOVER": "PASS",
            "HIGH_TURNOVER": "PASS",
            "MATCHES_COMPETITION": "PASS",
            "SELF_CORRELATION": "PENDING",
        }
        with open_database(self.database_path) as connection:
            root = self._completed(
                connection,
                "rank(close)",
                "1",
                1.4,
                1.1,
                checks=passing_checks,
            )
            synchronize_signal_seeds(
                connection,
                candidate_task_ids=(root.task.task_id,),
            )
            unsubmitted = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula="rank(open)",
                settings={"delay": 1},
                created_at="2026-08-31T00:02:00+00:00",
            )
            create_backtest_mutation(
                connection,
                BacktestMutationRecord(
                    child_task_id=unsubmitted.task.task_id,
                    parent_task_id=root.task.task_id,
                    action="structural",
                    location="formula",
                    before="rank(close)",
                    after="rank(open)",
                ),
            )
            self._child(
                connection,
                root.task.task_id,
                "rank(returns)",
                "3",
                sharpe=1.3,
                fitness=1.1,
                base_passed=True,
            )
            frontier = load_signal_frontiers(connection).records[0]

        self.assertEqual(
            tuple(branch.task_id for branch in frontier.branches),
            (root.task.task_id,),
        )
        self.assertEqual(frontier.qualified_parent_attempt_count, 1)
        self.assertEqual(frontier.qualified_parent_remaining_attempts, 19)

    def test_unqualified_expiry_preserves_history_and_accepts_late_improvement(self):
        with open_database(self.database_path) as connection:
            root = self._completed(connection, "rank(close)", "1", 1.1, 0.8)
            synchronize_signal_seeds(connection)
            children = []
            for index in range(20):
                child = prepare_backtest_task(
                    connection, account_scope="group-account",
                    formula=f"ts_mean(close,{index + 2})", settings={"delay": 1},
                    created_at="2026-08-31T00:02:00+00:00",
                )
                create_backtest_mutation(connection, BacktestMutationRecord(
                    child_task_id=child.task.task_id, parent_task_id=root.task.task_id,
                    action="structural", location="formula", before=root.task.formula,
                    after=child.task.formula,
                ))
                children.append(child.task.task_id)
                if index < 19:
                    record_submission_accepted(connection, child.task.task_id,
                        remote_id=f"pending-{index}", observed_at="2026-08-31T00:02:10+00:00")
            frontier = load_signal_frontiers(connection).records[0]
            self.assertEqual(frontier.branches[0].remaining_attempts, 1)
            record_submission_accepted(connection, children[-1], remote_id="last",
                                       observed_at="2026-08-31T00:02:10+00:00")
            self.assertFalse(load_signal_frontiers(connection).active_branch_task_ids)
        with open_database(self.database_path) as connection:
            self.assertFalse(load_signal_frontiers(connection).active_branch_task_ids)
            self.assertEqual(len(list_signal_seeds(connection)), 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM backtest_tasks WHERE status='pending'").fetchone()[0], 20)
            self._finish(connection, children[-1], "3", 1.2, 0.9)
            self.assertEqual(synchronize_signal_seeds(connection), ())
            frontier = load_signal_frontiers(connection).records[0]
            self.assertEqual(tuple(branch.task_id for branch in frontier.branches), (children[-1],))
            self.assertEqual(frontier.branches[0].remaining_attempts, 20)
            self.assertEqual(len(list_signal_seeds(connection)), 1)

    def test_below_spectacular_qualified_parent_has_twenty_attempts_and_retains_history(self):
        checks = {name: "PASS" for name in (
            "LOW_SHARPE", "LOW_FITNESS", "LOW_SUB_UNIVERSE_SHARPE", "CONCENTRATED_WEIGHT",
            "LOW_TURNOVER", "HIGH_TURNOVER", "MATCHES_COMPETITION", "SELF_CORRELATION",
        )}
        with open_database(self.database_path) as connection:
            root = self._completed(connection, "rank(close)", "1", 1.5, 1.6,
                                   checks=checks, grade="GOOD")
            synchronize_signal_seeds(connection)
            frontier = load_signal_frontiers(connection).records[0]
            self.assertTrue(frontier.qualified_evolution_active)
            self.assertEqual(frontier.qualified_parent_remaining_attempts, 20)
            children = []
            for index in range(20):
                child = prepare_backtest_task(connection, account_scope="group-account",
                    formula=f"ts_mean(close,{index+2})", settings={"delay": 1},
                    created_at="2026-08-31T00:02:00+00:00")
                create_backtest_mutation(connection, BacktestMutationRecord(
                    child.task.task_id, root.task.task_id, "structural", "formula",
                    root.task.formula, child.task.formula))
                record_submission_accepted(connection, child.task.task_id,
                    remote_id=f"attempt-{index}", observed_at="2026-08-31T00:02:10+00:00")
                children.append(child.task.task_id)
                frontier = load_signal_frontiers(connection).records[0]
                self.assertEqual(frontier.qualified_parent_remaining_attempts, 19-index)
                self.assertEqual(frontier.qualified_evolution_active, index < 19)
        with open_database(self.database_path) as connection:
            self.assertFalse(load_signal_frontiers(connection).active_branch_task_ids)
            self.assertEqual(get_backtest_task(connection, root.task.task_id), root)
            self.assertEqual(len(list_signal_seeds(connection)), 1)
            improved = self._finish(connection, children[-1], "3", 1.8, 2.2,
                                    checks=checks, grade="EXCELLENT")
            frontier = load_signal_frontiers(connection).records[0]
            self.assertEqual(frontier.branches[0].task_id, improved.task.task_id)
            self.assertEqual(frontier.branches[0].remaining_attempts, 20)
            self.assertEqual(get_backtest_task(connection, root.task.task_id), root)

    def _child(
        self,
        connection,
        parent_task_id: str,
        formula: str,
        identity: str,
        *,
        sharpe: float,
        fitness: float,
        settings: dict[str, int] | None = None,
        base_passed: bool = False,
    ):
        child = self._completed(
            connection,
            formula,
            identity,
            sharpe,
            fitness,
            settings=settings,
            complete=False,
        )
        create_backtest_mutation(
            connection,
            BacktestMutationRecord(
                child_task_id=child.task.task_id,
                parent_task_id=parent_task_id,
                action="structural",
                location="formula",
                before="parent",
                after=formula,
            ),
        )
        return self._finish(
            connection,
            child.task.task_id,
            identity,
            sharpe,
            fitness,
            base_passed=base_passed,
        )

    def _completed(
        self,
        connection,
        formula: str,
        identity: str,
        sharpe: float,
        fitness: float,
        *,
        settings: dict[str, int] | None = None,
        complete: bool = True,
        checks: dict[str, str] | None = None,
        grade: str | None = None,
    ):
        task = prepare_backtest_task(
            connection,
            account_scope="group-account",
            formula=formula,
            settings=settings or {"delay": 1},
            created_at=f"2026-08-31T00:0{identity}:00+00:00",
        )
        pending = record_submission_accepted(
            connection,
            task.task.task_id,
            remote_id=f"simulation_{identity}",
            observed_at=f"2026-08-31T00:0{identity}:10+00:00",
        )
        if not complete:
            return pending
        return self._finish(
            connection,
            task.task.task_id,
            identity,
            sharpe,
            fitness,
            checks=checks,
            grade=grade,
        )

    @staticmethod
    def _finish(
        connection,
        task_id: str,
        identity: str,
        sharpe: float,
        fitness: float,
        *,
        base_passed: bool = False,
        checks: dict[str, str] | None = None,
        grade: str | None = None,
    ):
        platform_alpha_id = f"alpha_{identity}"
        observed_at = f"2026-08-31T00:0{identity}:20+00:00"
        apply_backtest_detail(
            connection,
            task_id,
            BacktestDetail(
                platform_alpha_id=platform_alpha_id,
                grade=grade,
                sharpe=sharpe,
                fitness=fitness,
                turnover=0.12,
                returns=0.08,
                drawdown=0.04,
                margin=0.001,
                book_size=20_000_000,
                pnl=100_000,
                checks=_checks(
                    checks
                    or {
                        "LOW_SHARPE": "PASS" if base_passed else "FAIL",
                        "LOW_FITNESS": "PASS" if base_passed else "FAIL",
                        "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                        "CONCENTRATED_WEIGHT": "PASS",
                        "LOW_TURNOVER": "PASS",
                        "HIGH_TURNOVER": "PASS",
                        "MATCHES_COMPETITION": "PASS",
                        "SELF_CORRELATION": "PENDING",
                    }
                ),
            ),
            observed_at=observed_at,
        )
        return record_backtest_yearly_stats(
            connection,
            task_id,
            (),
            observed_at=observed_at,
        )


if __name__ == "__main__":
    unittest.main()
