import math
import unittest
from dataclasses import replace
from datetime import datetime

from execution.cycles import plan_automated_cycle
from execution.seeds import load_signal_frontiers
from execution.signal_duplicates import load_ordinary_signal_duplicates
from learning.action_effects import ExplicitSelfCorrelationEvidence
from persistence.backtests import BacktestCheckRecord, BacktestMutationRecord, create_backtest_mutation
from persistence.database import open_database
from persistence.pnl import save_pnl_series
from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
from persistence.submissions import PlatformSubmittedAlphaRecord
from tests.execution import test_cycles as fixtures
from tests.learning.test_qualified_evolution import checked
from tests.learning.test_seed_correlation import series


class OrdinarySignalDuplicateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AutomatedCyclePlanningTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.database_path = self.fixture.database_path

    def _parents(self):
        first = self.fixture._signal_seed("rank(ts_rank(close,22)+ts_zscore(open,66))")
        second = self.fixture._signal_seed("rank(ts_rank(open,22)+ts_zscore(close,66))")
        distinct = self.fixture._signal_seed("rank(ts_rank(volume,22)+ts_zscore(returns,66))")
        with open_database(self.database_path) as connection:
            for task, phase in ((first, 0), (second, 0), (distinct, 1)):
                curve = series(f"alpha-{task}", [math.sin(i + phase) for i in range(300)], "group-account")
                save_pnl_series(connection, replace(curve, observed_at="2026-08-30T00:03:00+08:00"))
        representative, duplicate = sorted((first, second))
        return representative, duplicate, distinct

    def test_ordinary_batch_uses_one_representative_across_roots_and_keeps_frozen_choice(self):
        representative, duplicate, distinct = self._parents()
        run = self.fixture._start_run(generation_count=20, backtest_count=10, max_backtests=20)
        plan = plan_automated_cycle(self.database_path, run_id=run, created_at="2026-08-30T00:04:00+08:00")
        self.assertEqual([(d.task_id, d.representative_task_id) for d in plan.signal_duplicates], [(duplicate, representative)])
        allocation = plan.planned_source_allocation
        self.assertEqual({a.parent_task_id for a in allocation.signal_improvements}, {representative, distinct})
        self.assertEqual(allocation.minimum_exploration_count, 3)
        self.assertGreaterEqual(plan.exploration_backtest_count, 3)
        self.assertEqual(len(plan.backtests), 10)
        with open_database(self.database_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM signal_seeds").fetchone()[0], 3)
            self.assertTrue(all(branch.remaining_attempts == 20 for frontier in load_signal_frontiers(connection).records
                                for branch in frontier.branches))
            # Later uncertainty changes comparability, but cannot rewrite this batch.
            save_submission_check(connection, SubmissionCheckRecord(representative,
                "2026-08-30T00:05:00+08:00", None, "request_error"))
        restored = plan_automated_cycle(self.database_path, run_id=run, created_at="2026-08-30T00:06:00+08:00")
        self.assertTrue(restored.recovered)
        self.assertEqual([s.task.task_id for s in restored.backtests], [s.task.task_id for s in plan.backtests])

    def test_exhausted_representative_does_not_reactivate_an_equivalent_root(self):
        representative, duplicate, distinct = self._parents()
        for i in range(20):
            child = self.fixture._completed_backtest(f"rank(ts_rank(volume,{300 + i}))",
                sharpe=0.5, fitness=0.4, sharpe_status="FAIL", fitness_status="FAIL", time_offset=i + 4)
            with open_database(self.database_path) as connection:
                create_backtest_mutation(connection, BacktestMutationRecord(child, representative,
                    "single_window_mutation", "formula", "22", str(300 + i)))
        run = self.fixture._start_run(generation_count=20, backtest_count=10, max_backtests=20)
        plan = plan_automated_cycle(self.database_path, run_id=run, created_at="2026-08-30T00:30:00+08:00")
        self.assertEqual([(d.task_id, d.representative_task_id) for d in plan.signal_duplicates], [(duplicate, representative)])
        self.assertEqual({a.parent_task_id for a in plan.planned_source_allocation.signal_improvements}, {distinct})
        with open_database(self.database_path) as connection:
            frontiers = {f.root_task_id: f for f in load_signal_frontiers(connection).records}
            self.assertEqual(frontiers[representative].branches, ())
            self.assertEqual(frontiers[duplicate].branches[0].remaining_attempts, 20)
        restored = plan_automated_cycle(self.database_path, run_id=run, created_at="2026-08-30T00:31:00+08:00")
        self.assertEqual([s.task.task_id for s in restored.backtests], [s.task.task_id for s in plan.backtests])

    def test_missing_pnl_keeps_both_roots_without_fetching_or_writing_evidence(self):
        parents = {self.fixture._signal_seed(formula) for formula in (
            "rank(ts_rank(close,22)+ts_zscore(open,66))", "rank(ts_rank(open,22)+ts_zscore(close,66))")}
        run = self.fixture._start_run(generation_count=20, backtest_count=10, max_backtests=20)
        plan = plan_automated_cycle(self.database_path, run_id=run, created_at="2026-08-30T00:04:00+08:00")
        self.assertEqual(plan.signal_duplicates, ())
        self.assertEqual({a.parent_task_id for a in plan.planned_source_allocation.signal_improvements}, parents)
        with open_database(self.database_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM platform_pnl_series").fetchone()[0], 0)

    def test_curve_and_result_cutoffs_do_not_leak_future_evidence(self):
        candidates = (checked("a", 1.5), checked("b", 1.5))
        with open_database(self.database_path) as connection:
            for name in ("a", "b"):
                save_pnl_series(connection, series(name, [math.sin(i) for i in range(300)]))
            def assess(at, snapshots=candidates):
                return load_ordinary_signal_duplicates(connection, snapshots, account_scope="account",
                    observed_at=datetime.fromisoformat(at), explicit_checks=(), submitted_alphas=())
            self.assertEqual(assess("2026-09-08T00:00:00+00:00"), ())
            self.assertEqual(len(assess("2026-09-09T00:00:00+00:00")), 1)
            future = replace(candidates[1], task=replace(candidates[1].task, finished_at="2026-09-10T00:00:00+00:00"))
            self.assertEqual(assess("2026-09-09T00:00:00+00:00", (candidates[0], future)), ())
            self.assertEqual(load_ordinary_signal_duplicates(connection, candidates, account_scope="other",
                observed_at=datetime.fromisoformat("2026-09-09T00:00:00+00:00"), explicit_checks=(), submitted_alphas=()), ())

    def test_current_check_states_and_submitted_status_keep_repair_routes_separate(self):
        candidates = tuple(replace(checked(name, 1.5), task=replace(checked(name, 1.5).task, formula=f"rank({name})"),
            result=replace(checked(name, 1.5).result,
                checks=(BacktestCheckRecord("SELF_CORRELATION", state, None, None, None),)))
            for name, state in (("a", "PASS"), ("b", "PENDING")))
        stamp = "2026-09-09T00:00:00+00:00"
        with open_database(self.database_path) as connection:
            for name in ("a", "b"):
                save_pnl_series(connection, series(name, [math.sin(i) for i in range(300)]))
            def assess(checks=(), submitted=()):
                before = connection.total_changes
                result = load_ordinary_signal_duplicates(connection, candidates, account_scope="account",
                    observed_at=datetime.fromisoformat(stamp), explicit_checks=checks, submitted_alphas=submitted)
                self.assertEqual(connection.total_changes, before)
                return result
            self.assertEqual(assess(), ())
            checks = tuple(ExplicitSelfCorrelationEvidence(name, "FAIL", "failed", stamp) for name in ("a", "b"))
            self.assertEqual(len(assess(checks)), 1)
            self.assertEqual(assess((checks[0], replace(checks[1], status="PENDING", formal_check_state="pending"))), ())
            self.assertEqual(assess(tuple(replace(c, observed_at="2026-09-10T00:00:00+00:00") for c in checks)), ())
            self.assertEqual(assess(tuple(replace(c, observed_at="2026-09-01T00:00:00+00:00") for c in checks)), ())
            submitted = PlatformSubmittedAlphaRecord("account", "a", "rank(a)", "ACTIVE", stamp, False, {}, stamp)
            self.assertEqual(assess(checks, (submitted,)), ())
