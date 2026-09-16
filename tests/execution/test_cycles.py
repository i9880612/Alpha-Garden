from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.backtest_batches import (
    AutomatedCandidateBacktest,
    prepare_automated_candidate_backtest_batch,
)
from execution.cycle_backtests import settle_automated_cycle
from execution.backtests import (
    apply_backtest_detail,
    cancel_unsubmitted_backtest_task,
    prepare_backtest_task,
    record_backtest_yearly_stats,
    record_submission_accepted,
)
from execution.cycles import AutomatedCyclePlan, plan_automated_cycle
from execution.generation import BatchShortfall, ExclusionCount, ExplorationBatch
from execution.seeds import load_signal_frontiers, synchronize_signal_seeds
from execution.runs import (
    AutomatedRunLimits,
    prepare_automated_run,
    record_automated_cycle_settlement,
    start_automated_run,
)
from generation.candidate import exploration_candidate
from generation.direction import DIRECTION_REVERSAL
from generation.formula import Call, analyze_formula, render_formula
from generation.parser import parse_formula
from generation.polishing import SINGLE_WINDOW_MUTATION
from generation.internal_edits import INTERNAL_EDIT_FAMILIES
from generation.self_correlation import SELF_CORRELATION_REPAIR, SELF_CORRELATION_REPAIR_FAMILIES
from generation.transformations import COMPLEMENTARY_SIGNAL_REFRAME
from learning.action_effects import build_defect_action_strategies
from persistence.backtests import (
    BacktestMutationRecord,
    create_backtest_mutation,
    get_backtest_mutation,
    get_backtest_task,
)
from persistence.catalog import (
    FieldCatalogContext,
    FieldCatalogRecord,
    OperatorCatalogRecord,
    OperatorOutputRecord,
    OperatorRoleRecord,
    PlatformCatalogSyncRecord,
    WindowCatalogRecord,
    replace_operator_outputs,
    replace_operator_roles,
    replace_platform_catalog,
    replace_window_catalog,
)
from persistence.database import open_database
from persistence.schema import initialize_database_schema
from persistence.submissions import (
    FormalSubmissionAttemptRecord,
    PlatformSubmittedAlphaRecord,
    canonical_submission_json,
    create_formal_submission_attempt,
    record_platform_submitted_alphas,
)
from selection.allocations import LOW_SUB_UNIVERSE_SHARPE_STRUCTURAL_FAMILIES
from worldquant.backtests import (
    BacktestCheck,
    BacktestDetail,
    BacktestSettings,
    STANDARD_REGULAR_CHECK_NAMES,
)


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


class AutomatedCyclePlanningTests(unittest.TestCase):
    def test_optimization_empty_pool_finishes_authorized_late_submission(self):
        from execution.runner import run_automated_run
        from execution.submission_queue import synchronize_submission_queue
        from tests.execution.test_runner import RunnerClient, FakeTime
        from tests.execution.test_qualified_archive import complete_candidate
        from persistence.submission_checks import SubmissionCheckRecord, save_submission_check

        self._optimization_parent()
        run_id = self._start_run(generation_count=6, backtest_count=4, max_cycles=2,
                                 max_backtests=8, optimization_only=True, automatic_submissions_enabled=True)
        plan = plan_automated_cycle(self.database_path, run_id=run_id,
                                    created_at="2026-08-30T00:04:00+08:00")
        with open_database(self.database_path) as connection:
            for index, task in enumerate(plan.backtests):
                task = record_submission_accepted(connection, task.task.task_id,
                    remote_id=f"simulation-opt-{index}", observed_at="2026-09-01T00:01:00+00:00")
                complete_candidate(connection, task, sharpe=1.8 if index == 0 else 0.5,
                                   grade="SPECTACULAR" if index == 0 else "GOOD", qualified=index == 0)
            target_id = plan.backtests[0].task.task_id
            connection.execute("UPDATE submission_checks SET payload_json='null',attempt_count=3 WHERE task_id=?", (target_id,))
        settle_automated_cycle(self.database_path, run_id, cycle_number=1,
                               observed_at="2026-09-01T00:03:00+00:00")
        with open_database(self.database_path) as connection:
            save_submission_check(connection, SubmissionCheckRecord(target_id, "2026-09-02T00:00:00+00:00",
                canonical_submission_json({"is": {"checks": [{"name": name, "result": "PASS"}
                                                               for name in STANDARD_REGULAR_CHECK_NAMES]}}),
                None, attempt_count=4))
            synchronize_submission_queue(connection, candidate_task_ids=(target_id,), enqueued_at="2026-09-02T00:00:00+00:00")
            target = get_backtest_task(connection, target_id)
            self.assertEqual(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids, ())
        client = RunnerClient()
        client.formulas[target.task.platform_alpha_id] = target.task.formula
        clock = FakeTime("2026-09-02T00:01:00+00:00")
        result = run_automated_run(self.database_path, client, run_id,
                                   clock=clock.now, waiter=clock.wait)
        self.assertEqual(result.run.stop_reason, "optimization_candidates_exhausted")
        self.assertEqual(client.formal_submission_ids, [target.task.platform_alpha_id])
        self.assertNotIn("submit", client.calls)

    def test_optimization_result_checks_queue_spectacular_without_automatic_submission(self):
        from execution.runner import run_automated_run
        from tests.execution.test_runner import RunnerClient, FakeTime, AutomatedRunnerTests
        from worldquant.backtests import BacktestPollObservation

        self._optimization_parent()
        self._record_submitted_alpha("rank(open)")
        run_id = self._start_run(generation_count=6, backtest_count=4, max_cycles=1,
                                 max_backtests=4, optimization_only=True)
        client = RunnerClient(
            AutomatedRunnerTests._accepted("opt-1"), AutomatedRunnerTests._accepted("opt-2"),
            polls=(BacktestPollObservation("completed", "opt-alpha-1", "COMPLETE", None),
                   BacktestPollObservation("completed", "opt-alpha-2", "COMPLETE", None)),
        )
        clock = FakeTime("2026-08-30T00:04:00+08:00")
        result = run_automated_run(self.database_path, client, run_id,
                                   clock=clock.now, waiter=clock.wait)
        self.assertEqual((result.run.status, result.run.current_cycle), ("completed", 1))
        self.assertEqual(client.calls.count("submit"), 2)
        self.assertNotIn("formal_submit", client.calls)
        with open_database(self.database_path) as connection:
            grades = connection.execute("SELECT r.grade FROM submission_queue q JOIN backtest_results r USING(task_id)").fetchall()
            self.assertEqual([row[0] for row in grades], ["SPECTACULAR", "SPECTACULAR"])

    def test_optimization_partial_batch_recovers_and_settles_without_exploration(self):
        parent_id = self._optimization_parent()
        run_id = self._start_run(generation_count=6, backtest_count=4, max_cycles=1,
                                 max_backtests=4, optimization_only=True)
        with patch("execution.cycle_candidates.generate_exploration_batch",
                   side_effect=AssertionError("optimization must not generate exploration")):
            plan = plan_automated_cycle(self.database_path, run_id=run_id,
                                        created_at="2026-08-30T00:04:00+08:00")
        self.assertEqual(len(plan.backtests), 2)
        self.assertEqual(plan.exploration_backtest_count, 0)
        self.assertEqual(plan.direction_validation_backtest_count, 0)
        self.assertTrue(all(item.parent_task_id == parent_id
                            and item.candidate_family not in SELF_CORRELATION_REPAIR_FAMILIES
                            for item in plan.planned_source_allocation.signal_improvements))
        recovered = plan_automated_cycle(self.database_path, run_id=run_id,
                                         created_at="2026-08-30T00:05:00+08:00")
        self.assertTrue(recovered.recovered)
        self.assertEqual(recovered.backtests, plan.backtests)
        self._complete_plan(plan, observed_at="2026-08-30T00:06:00+08:00")
        from execution.driver import advance_automated_run
        result = advance_automated_run(self.database_path, Mock(), run_id,
                                      observed_at="2026-08-30T00:07:00+08:00")
        self.assertEqual(result.run.status, "completed")
        self.assertEqual(result.run.current_cycle, 1)
        with open_database(self.database_path) as connection:
            frontier = load_signal_frontiers(connection).records[0]
            self.assertEqual(frontier.qualified_parent_remaining_attempts, 18)
            self.assertEqual(connection.execute("SELECT count(*) FROM formal_submission_attempts").fetchone()[0], 0)

    def test_optimization_plans_both_checked_siblings_and_preserves_their_budgets(self):
        from tests.execution.test_qualified_archive import complete_candidate

        root_id = self._optimization_parent()
        children = []
        with open_database(self.database_path) as connection:
            root = get_backtest_task(connection, root_id)
            for index, formula in enumerate((
                "rank(ts_rank(close,44)+ts_zscore(open,66))",
                "rank(ts_rank(open,22)+ts_zscore(close,66))",
            )):
                child = prepare_backtest_task(connection, account_scope="group-account",
                    formula=formula, settings=self.settings, created_at="2026-08-30T00:03:00+08:00")
                create_backtest_mutation(connection, BacktestMutationRecord(
                    child.task.task_id, root_id, "structural", "formula", root.task.formula, formula))
                child = record_submission_accepted(connection, child.task.task_id,
                    remote_id=f"simulation-sibling-{index}", observed_at="2026-08-30T00:03:10+08:00")
                children.append(complete_candidate(connection, child, sharpe=1.6 + index / 10).task.task_id)
        run_id = self._start_run(generation_count=20, backtest_count=4,
                                 max_cycles=1, max_backtests=4, optimization_only=True)
        plan = plan_automated_cycle(self.database_path, run_id=run_id,
                                    created_at="2026-09-02T00:04:00+00:00")
        self.assertEqual(len(plan.backtests), 2)
        self.assertEqual(plan.exploration_backtest_count, 0)
        self.assertEqual({item.parent_task_id for item in plan.planned_source_allocation.signal_improvements},
                         set(children))
        self.assertEqual({item.root_task_id for item in plan.planned_source_allocation.signal_improvements},
                         {root_id})
        recovered = plan_automated_cycle(self.database_path, run_id=run_id,
                                         created_at="2026-09-02T00:05:00+00:00")
        self.assertTrue(recovered.recovered)
        self.assertEqual(recovered.backtests, plan.backtests)
        self._complete_plan(plan, observed_at="2026-09-02T00:06:00+00:00")
        with open_database(self.database_path) as connection:
            branches = tuple(branch for frontier in load_signal_frontiers(
                connection, optimization_only=True).records for branch in frontier.branches)
            self.assertEqual({branch.task_id: branch.remaining_attempts for branch in branches},
                             {task_id: 19 for task_id in children})
            self.assertEqual(connection.execute("SELECT count(*) FROM formal_submission_attempts").fetchone()[0], 0)

    def test_optimization_uses_last_shared_attempt_then_archives_parent(self):
        parent_id = self._optimization_parent()
        with open_database(self.database_path) as connection:
            parent = get_backtest_task(connection, parent_id)
        for index in range(19):
            formula = f"ts_mean(close,{index + 2})"
            child_id = self._completed_backtest(formula, sharpe=0.5, fitness=0.2,
                sharpe_status="FAIL", fitness_status="FAIL", time_offset=2)
            with open_database(self.database_path) as connection:
                create_backtest_mutation(connection, BacktestMutationRecord(
                    child_id, parent_id, "structural", "formula", parent.task.formula, formula))
        run_id = self._start_run(generation_count=6, backtest_count=4, optimization_only=True,
                                 max_cycles=2, max_backtests=8)
        plan = plan_automated_cycle(self.database_path, run_id=run_id,
                                    created_at="2026-08-30T00:04:00+08:00")
        self.assertEqual(len(plan.backtests), 1)
        self._complete_plan(plan, observed_at="2026-08-30T00:06:00+08:00")
        settle_automated_cycle(self.database_path, run_id, cycle_number=1,
                               observed_at="2026-08-30T00:07:00+08:00")
        from execution.driver import advance_automated_run
        result = advance_automated_run(self.database_path, Mock(), run_id,
                                      observed_at="2026-08-30T00:08:00+08:00")
        self.assertEqual(result.run.status, "completed")
        self.assertEqual(result.run.stop_reason, "optimization_candidates_exhausted")
        with open_database(self.database_path) as connection:
            self.assertFalse(load_signal_frontiers(connection).active_branch_task_ids)
            self.assertEqual(connection.execute("SELECT task_id FROM qualified_alpha_archive").fetchone()[0], parent_id)

    def test_optimization_without_eligible_parents_finishes_without_backtests(self):
        from execution.driver import advance_automated_run
        run_id = self._start_run(optimization_only=True)
        client = Mock()
        result = advance_automated_run(self.database_path, client, run_id,
                                      observed_at="2026-08-30T00:04:00+08:00")
        self.assertEqual(result.run.status, "completed")
        self.assertEqual(result.run.stop_reason, "optimization_candidates_exhausted")
        self.assertEqual(client.mock_calls, [])
        with open_database(self.database_path) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM automated_run_backtests").fetchone()[0], 0)

    def _optimization_parent(self):
        from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
        parent_id = self._signal_seed("rank(ts_rank(close,22)+ts_zscore(open,66))",
                                      sharpe_status="PASS", fitness_status="PASS")
        with open_database(self.database_path) as connection:
            connection.execute("UPDATE backtest_results SET grade='GOOD', sharpe=1.5, fitness=1.2 WHERE task_id=?", (parent_id,))
            connection.execute("INSERT INTO backtest_yearly_stats(task_id,stage,year,sharpe) VALUES (?, 'IS', 2023, 1.5)", (parent_id,))
            save_submission_check(connection, SubmissionCheckRecord(parent_id, "2026-08-30T00:03:00+08:00",
                canonical_submission_json({"is": {"checks": [{"name": name, "result": "PASS"}
                                                               for name in STANDARD_REGULAR_CHECK_NAMES]}}), None))
        return parent_id

    def test_missing_historical_policy_does_not_reselect_a_frozen_batch(self):
        run_id = self._start_run()
        first = plan_automated_cycle(self.database_path, run_id=run_id,
                                    created_at="2026-08-30T00:04:00+08:00")
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM automated_run_allocations WHERE run_id=?", (run_id,))
        recovered = plan_automated_cycle(self.database_path, run_id=run_id,
                                        created_at="2026-08-30T00:05:00+08:00")
        self.assertEqual({s.task.task_id for s in first.backtests},
                         {s.task.task_id for s in recovered.backtests})
        self.assertTrue(recovered.recovered)

    def test_missing_policy_blocks_new_batch_before_creating_tasks(self):
        run_id = self._start_run()
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM automated_run_allocations WHERE run_id=?", (run_id,))
        with self.assertRaisesRegex(ValueError, "automated_run_allocation_missing"):
            plan_automated_cycle(self.database_path, run_id=run_id,
                                 created_at="2026-08-30T00:04:00+08:00")
        with open_database(self.database_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM backtest_tasks").fetchone()[0], 0)

    def test_default_sized_cold_start_freezes_one_hundred_unique_tasks_from_five_hundred(self):
        run_id = self._start_run(generation_count=500, backtest_count=100, max_cycles=1, max_backtests=100)
        plan = plan_automated_cycle(self.database_path, run_id=run_id, created_at="2026-08-30T00:04:00+08:00")
        self.assertEqual(len(plan.backtests), 100)
        self.assertEqual(plan.requested_generation_count, 500)
        self.assertEqual(plan.exploration_backtest_count, 100)
        self.assertEqual(len({s.task.request_fingerprint for s in plan.backtests}), 100)
        with open_database(self.database_path) as connection:
            run = connection.execute("SELECT minimum_exploration_backtests,max_backtests,max_in_flight_backtests FROM automated_runs WHERE run_id=?", (run_id,)).fetchone()
            self.assertEqual(tuple(run), (30, 100, 3))
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM formal_submission_attempts").fetchone()[0], 0)

    def test_negative_direction_is_automatically_frozen_and_becomes_usable_seed(self):
        source_id = self._completed_backtest(
            "rank(close)", sharpe=-1.4, fitness=-1.1,
            sharpe_status="FAIL", fitness_status="FAIL", time_offset=2,
        )
        run_id = self._start_run()
        plan = plan_automated_cycle(
            self.database_path, run_id=run_id, created_at="2026-08-30T00:04:00+08:00",
        )
        directions = [item for item in plan.backtests if item.task.formula == "-rank(close)"]
        self.assertEqual(len(directions), 1)
        child = directions[0]
        self.assertEqual(child.task.settings_json, json.dumps(self.settings, sort_keys=True, separators=(",", ":")))
        self.assertEqual(plan.direction_validation_backtest_count, 1)
        self.assertEqual(plan.exploration_backtest_count, 1)
        self.assertEqual(plan.qualified_evolution_backtest_count, 0)
        self.assertEqual(self._mutation(child.task.task_id).parent_task_id, source_id)
        self.assertEqual(self._mutation(child.task.task_id).action, DIRECTION_REVERSAL)
        self._completed_backtest(
            "rank(open)", sharpe=-2, fitness=-1.5,
            sharpe_status="FAIL", fitness_status="FAIL", time_offset=5,
        )
        recovered = plan_automated_cycle(
            self.database_path, run_id=run_id, created_at="2026-08-30T00:06:00+08:00",
        )
        self.assertTrue(recovered.recovered)
        self.assertEqual(recovered.backtests, plan.backtests)
        self._complete_plan(plan, observed_at="2026-08-30T00:07:00+08:00", qualified_task_id=child.task.task_id)
        with open_database(self.database_path) as connection:
            seeds = synchronize_signal_seeds(connection)
            self.assertEqual(tuple(seed.root_task_id for seed in seeds), (child.task.task_id,))
            self.assertIn(child.task.task_id, load_signal_frontiers(connection).active_branch_task_ids)
            record_automated_cycle_settlement(
                connection, run_id, cycle_number=1, outcome="qualified",
                frontier_advanced=True, observed_at="2026-08-30T00:08:00+08:00",
            )
        second = plan_automated_cycle(
            self.database_path, run_id=run_id, created_at="2026-08-30T00:09:00+08:00",
        )
        self.assertEqual(len(second.backtests), 2)
        self.assertNotIn(child.task.task_id, tuple(item.task.task_id for item in second.backtests))
        self.assertIn("-rank(open)", tuple(item.task.formula for item in second.backtests))

    def test_single_task_batch_keeps_exploration_reserve(self):
        self._completed_backtest(
            "rank(close)", sharpe=-1.4, fitness=-1.1,
            sharpe_status="FAIL", fitness_status="FAIL", time_offset=2,
        )
        run_id = self._start_run(backtest_count=1, max_backtests=1)
        plan = plan_automated_cycle(
            self.database_path, run_id=run_id, created_at="2026-08-30T00:04:00+08:00",
        )
        self.assertEqual(plan.direction_validation_backtest_count, 0)
        self.assertEqual(plan.exploration_backtest_count, 1)
        self.assertIsNone(self._mutation(plan.backtests[0].task.task_id))

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.database_path = root / "cycles.sqlite3"
        self.policy_path = root / "backtest.json"
        self.settings = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 4,
            "neutralization": "SECTOR",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "OFF",
            "language": "FASTEXPR",
            "visualization": False,
            "maxTrade": "OFF",
            "maxPosition": "OFF",
        }
        self.policy_path.write_text(
            json.dumps(
                {
                    "catalogContext": {
                        "instrumentType": "EQUITY",
                        "region": "USA",
                        "universe": "TOP3000",
                        "delay": 1,
                    },
                    "decay": 4,
                    "neutralization": {
                        "default": "SECTOR",
                        "rootGroupNeutralize": "NONE",
                        "byFieldCategory": {
                            "Option": "SUBINDUSTRY",
                            "sample": "SECTOR",
                        },
                    },
                    "truncation": {"default": 0.08, "tailRisk": 0.05},
                    "pasteurization": "ON",
                    "unitHandling": "VERIFY",
                    "nanHandling": "OFF",
                    "language": "FASTEXPR",
                    "visualization": False,
                    "maxTrade": "OFF",
                    "maxPosition": "OFF",
                }
            ),
            encoding="utf-8",
        )
        with open_database(self.database_path) as connection:
            initialize_database_schema(connection)
        self._replace_catalog(
            fields=("close", "open", "returns", "volume"),
            cross_sectional=("rank", "zscore"),
            time_series=("ts_rank", "ts_zscore"),
            windows=(5, 22, 66, 120, 250),
        )

    def test_plans_one_exploration_cycle_without_platform_input(self) -> None:
        run_id = self._start_run()

        plan = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:02:00+08:00",
        )

        self.assertEqual(plan.cycle_number, 1)
        self.assertEqual(plan.exploration_backtest_count, 2)
        self.assertEqual(plan.structural_evolution_backtest_count, 0)
        self.assertEqual(plan.local_polishing_backtest_count, 0)
        self.assertEqual(plan.requested_generation_count, 5)
        self.assertEqual(plan.requested_backtest_count, 2)
        self.assertEqual(len(plan.backtests), 2)
        self.assertFalse(plan.recovered)
        self.assertTrue(
            all(snapshot.task.status == "created" for snapshot in plan.backtests)
        )
        self.assertEqual(self._counts(), (2, 2))

    def test_unparseable_submitted_formula_does_not_block_cycle_planning(
        self,
    ) -> None:
        self._record_submitted_alpha(
            "rank(close) > 0.8 ? rank(open) : nan",
        )
        run_id = self._start_run(
            generation_count=5,
            backtest_count=2,
            max_cycles=1,
            max_backtests=2,
        )

        plan = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:02:00+08:00",
        )

        self.assertEqual(plan.exploration_backtest_count, 2)
        self.assertEqual(len(plan.backtests), 2)

    def test_cold_start_keeps_rare_signal_categories_in_final_plan(self) -> None:
        dominant_fields = tuple(f"dominant_{index}" for index in range(100))
        fields = dominant_fields + ("rare_b", "rare_c")
        field_categories = {
            **{field_id: "category_a" for field_id in dominant_fields},
            "rare_b": "category_b",
            "rare_c": "category_c",
        }
        self._replace_catalog(
            fields=fields,
            field_categories=field_categories,
            cross_sectional=("rank",),
            time_series=("ts_rank",),
            windows=(5, 22, 66),
        )
        run_id = self._start_run(
            generation_count=60,
            backtest_count=6,
            max_backtests=6,
        )

        plan = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:02:00+08:00",
        )
        selected_categories = {
            field_categories[field_id]
            for snapshot in plan.backtests
            for field_id in analyze_formula(
                parse_formula(snapshot.task.formula).expression
            ).referenced_names
        }

        self.assertEqual(
            selected_categories,
            {"category_a", "category_b", "category_c"},
        )

    def test_cycle_persists_settings_derived_from_field_category(self) -> None:
        self._replace_catalog(
            fields=("option_signal",),
            field_categories={"option_signal": "Option"},
            cross_sectional=("rank",),
            time_series=("ts_rank",),
            windows=(22, 66),
        )
        run_id = self._start_run(
            generation_count=5,
            backtest_count=1,
            max_backtests=1,
        )

        plan = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:02:00+08:00",
        )

        settings = json.loads(plan.backtests[0].task.settings_json)
        self.assertEqual(settings["neutralization"], "SUBINDUSTRY")

    def test_open_cycle_uses_vector_and_group_catalog_fields_and_recovers(self) -> None:
        self._replace_catalog(
            fields=("close", "open"),
            vector_fields=("news",),
            group_fields=("sector",),
            cross_sectional=("rank",),
            time_series=("ts_mean",),
            vector=("vec_avg",),
            group=("group_rank",),
            windows=(5, 22, 66),
        )
        run_id = self._start_run(
            generation_count=50,
            backtest_count=10,
            max_backtests=10,
        )

        first = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:02:00+08:00",
        )
        recovered = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:03:00+08:00",
        )
        formulas = tuple(item.task.formula for item in first.backtests)

        self.assertEqual(len(formulas), 10)
        self.assertTrue(any("vec_avg(news)" in formula for formula in formulas))
        self.assertTrue(
            any(
                "group_rank(" in formula and "sector" in formula for formula in formulas
            )
        )
        self.assertTrue(recovered.recovered)
        self.assertEqual(
            tuple(item.task.task_id for item in recovered.backtests),
            tuple(item.task.task_id for item in first.backtests),
        )
        self.assertEqual(self._counts(), (10, 10))

    def test_retry_recovers_the_same_cycle_without_new_tasks(self) -> None:
        run_id = self._start_run()
        first = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:02:00+08:00",
        )

        recovered = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:09:00+08:00",
        )

        self.assertTrue(recovered.recovered)
        self.assertEqual(
            tuple(item.task.task_id for item in recovered.backtests),
            tuple(item.task.task_id for item in first.backtests),
        )
        self.assertEqual(self._counts(), (2, 2))

    def test_recovery_uses_formal_checks_observed_before_cycle_cutoff(self) -> None:
        root_task_id = self._signal_seed("rank(ts_rank(close,22)+ts_zscore(open,66))")
        run_id = self._start_run(max_cycles=2, max_backtests=4)
        first = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:04:00+08:00",
        )
        self._complete_plan(first, observed_at="2026-08-30T00:05:00+08:00")
        with open_database(self.database_path) as connection:
            record_automated_cycle_settlement(
                connection,
                run_id,
                cycle_number=1,
                outcome="not_qualified",
                frontier_advanced=False,
                observed_at="2026-08-30T00:06:00+08:00",
            )
        mutation_task_id = next(
            snapshot.task.task_id
            for snapshot in first.backtests
            if self._mutation(snapshot.task.task_id) is not None
        )
        exploration_task_id = next(
            snapshot.task.task_id
            for snapshot in first.backtests
            if snapshot.task.task_id != mutation_task_id
        )
        self._record_formal_attempt(
            task_id=mutation_task_id,
            run_id=run_id,
            family_root_task_id=root_task_id,
            observed_at="2026-08-30T00:07:00+08:00",
            checks={"LOW_SHARPE": "FAIL", "SELF_CORRELATION": "PASS"},
        )
        second = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:10:00+08:00",
        )
        self._record_formal_attempt(
            task_id=exploration_task_id,
            run_id=run_id,
            family_root_task_id=root_task_id,
            observed_at="2026-08-30T00:11:00+08:00",
            checks={"SELF_CORRELATION": "FAIL"},
        )

        with patch(
            "execution.cycles.build_defect_action_strategies",
            wraps=build_defect_action_strategies,
        ) as strategy_builder:
            recovered = plan_automated_cycle(
                self.database_path,
                run_id=run_id,
                created_at="2026-08-30T00:20:00+08:00",
            )

        strategy_builder.assert_not_called()
        self.assertTrue(recovered.recovered)
        self.assertEqual(
            tuple(item.task.task_id for item in recovered.backtests),
            tuple(item.task.task_id for item in second.backtests),
        )

    def test_final_cycle_uses_only_the_remaining_total_quota(self) -> None:
        run_id = self._start_run(max_backtests=3)
        first = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:02:00+08:00",
        )
        with open_database(self.database_path) as connection:
            from execution.backtests import cancel_unsubmitted_backtest_task

            for snapshot in first.backtests:
                cancel_unsubmitted_backtest_task(
                    connection, snapshot.task.task_id,
                    observed_at="2026-08-30T00:03:00+08:00",
                )
            record_automated_cycle_settlement(
                connection,
                run_id,
                cycle_number=1,
                outcome="failed",
                frontier_advanced=False,
                observed_at="2026-08-30T00:03:00+08:00",
            )

        second = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:04:00+08:00",
        )

        self.assertEqual(second.cycle_number, 2)
        self.assertEqual(second.requested_backtest_count, 1)
        self.assertEqual(len(second.backtests), 1)
        self.assertEqual(self._counts(), (3, 3))

    def test_generation_shortfall_creates_no_backtest_tasks(self) -> None:
        self._replace_catalog(
            fields=("close", "open"),
            cross_sectional=("rank",),
            time_series=(),
            windows=(22,),
        )
        with open_database(self.database_path) as connection:
            replace_operator_outputs(
                connection,
                (OperatorOutputRecord("rank", "group"),),
            )
        run_id = self._start_run(generation_count=20)

        with self.assertRaisesRegex(
            ValueError,
            "automated_cycle_generation_shortfall",
        ):
            plan_automated_cycle(
                self.database_path,
                run_id=run_id,
                created_at="2026-08-30T00:02:00+08:00",
            )

        self.assertEqual(self._counts(), (0, 0))

    def test_mixed_cycle_shortfall_persists_neither_tasks_nor_lineage(self) -> None:
        self._signal_seed("rank(close)")
        run_id = self._start_run()
        shortfall = ExplorationBatch(
            candidates=(),
            attempted_seed_count=16,
            exclusions=(ExclusionCount("test_rejection", 16),),
            shortfall=BatchShortfall(
                missing_count=4,
            ),
        )

        with (
            patch(
                "execution.cycle_candidates.generate_exploration_batch",
                return_value=shortfall,
            ),
            self.assertRaisesRegex(
                ValueError,
                "automated_cycle_generation_shortfall:4",
            ),
        ):
            plan_automated_cycle(
                self.database_path,
                run_id=run_id,
                created_at="2026-08-30T00:04:00+08:00",
            )

        self.assertEqual(self._counts(), (1, 0))
        with open_database(self.database_path) as connection:
            mutation_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_mutations"
            ).fetchone()[0]
        self.assertEqual(mutation_count, 0)

    def test_partial_cycle_tasks_are_not_silently_filled(self) -> None:
        run_id = self._start_run()
        prepare_automated_candidate_backtest_batch(
            self.database_path,
            run_id=run_id,
            candidates=(
                AutomatedCandidateBacktest(
                    candidate=exploration_candidate(
                        parse_formula("rank(ts_rank(close,22)*open)").expression
                    ),
                    settings=BacktestSettings.from_platform_dict(self.settings),
                ),
            ),
            created_at="2026-08-30T00:02:00+08:00",
        )

        with self.assertRaisesRegex(
            ValueError,
            "automated_cycle_partial_tasks_detected",
        ):
            plan_automated_cycle(
                self.database_path,
                run_id=run_id,
                created_at="2026-08-30T00:03:00+08:00",
            )

        self.assertEqual(self._counts(), (1, 1))

    def test_recovery_rejects_current_cycle_task_from_other_account(self) -> None:
        run_id = self._start_run(
            generation_count=5,
            backtest_count=1,
            max_backtests=1,
        )
        first = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:02:00+08:00",
        )
        with open_database(self.database_path) as connection:
            cursor = connection.execute(
                "UPDATE backtest_tasks SET account_scope = ? WHERE task_id = ?",
                ("other-account", first.backtests[0].task.task_id),
            )
        self.assertEqual(cursor.rowcount, 1)

        with self.assertRaisesRegex(
            ValueError,
            "automated_cycle_recovery_identity_conflict",
        ):
            plan_automated_cycle(
                self.database_path,
                run_id=run_id,
                created_at="2026-08-30T00:03:00+08:00",
            )

    def test_cycle_prepares_exploration_and_one_parent_mutation_atomically(
        self,
    ) -> None:
        parent_task_id = self._signal_seed("rank(ts_rank(close,22)+ts_zscore(open,66))")
        run_id = self._start_run()

        first = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:04:00+08:00",
        )
        recovered = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:05:00+08:00",
        )

        self.assertEqual(first.planned_source_allocation.improvement_count, 1)
        self.assertEqual(first.exploration_backtest_count, 1)
        self.assertEqual(first.structural_evolution_backtest_count, 1)
        self.assertEqual(first.local_polishing_backtest_count, 0)
        self.assertTrue(recovered.recovered)
        self.assertEqual(
            tuple(item.task.task_id for item in recovered.backtests),
            tuple(item.task.task_id for item in first.backtests),
        )
        self.assertIsNone(recovered.planned_source_allocation)
        with open_database(self.database_path) as connection:
            mutations = tuple(
                mutation
                for snapshot in first.backtests
                if (
                    mutation := get_backtest_mutation(
                        connection,
                        snapshot.task.task_id,
                    )
                )
                is not None
            )
        self.assertEqual(len(mutations), 1)
        self.assertEqual(mutations[0].parent_task_id, parent_task_id)
        self.assertEqual(self._counts(), (3, 2))

        with open_database(self.database_path) as connection:
            for snapshot in first.backtests:
                task_id = snapshot.task.task_id
                record_submission_accepted(
                    connection,
                    task_id,
                    remote_id=f"simulation-{task_id}",
                    observed_at="2026-08-30T00:06:00+08:00",
                )
                self._complete_with_empty_yearly_stats(
                    connection,
                    task_id,
                    platform_alpha_id=f"alpha-{task_id}",
                    observed_at="2026-08-30T00:07:00+08:00",
                    sharpe=1.4,
                    fitness=1.2,
                    turnover=0.08,
                    returns=0.02,
                    drawdown=0.08,
                    margin=0.0003,
                    checks=_checks(
                        {
                            "LOW_SHARPE": "PASS",
                            "LOW_FITNESS": "PASS",
                            "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                            "CONCENTRATED_WEIGHT": "PASS",
                            "LOW_TURNOVER": "PASS",
                            "HIGH_TURNOVER": "PASS",
                            "MATCHES_COMPETITION": "PASS",
                            "SELF_CORRELATION": "PASS",
                        }
                    ),
                )
        completed_recovery = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:08:00+08:00",
        )

        self.assertTrue(completed_recovery.recovered)
        self.assertEqual(
            tuple(item.task.task_id for item in completed_recovery.backtests),
            tuple(item.task.task_id for item in first.backtests),
        )
        self.assertEqual(
            tuple(self._mutation(item.child_task_id) for item in mutations), mutations
        )

    def test_cycle_prioritizes_unsubmitted_parent_and_freezes_that_choice(self):
        import math
        from persistence.pnl import save_pnl_series
        from tests.learning.test_seed_correlation import series

        submitted_formula = "rank(ts_rank(close,22)+ts_zscore(open,66))"
        submitted_id = self._signal_seed(submitted_formula)
        unsubmitted_id = self._signal_seed("rank(ts_rank(open,22)+ts_zscore(close,66))")
        # A different remote ID with the same formula is still already submitted.
        self._record_submitted_alpha(submitted_formula,
            observed_at="2026-08-30T00:03:00+08:00", settings=self.settings)
        with open_database(self.database_path) as connection:
            save_pnl_series(connection, series("submitted-alpha", [math.sin(i) for i in range(300)], "group-account"))
            save_pnl_series(connection, series(f"alpha-{unsubmitted_id}", [math.cos(i) for i in range(300)], "group-account"))
        run_id = self._start_run()
        plan = plan_automated_cycle(self.database_path, run_id=run_id,
                                   created_at="2026-08-30T00:04:00+08:00")
        self.assertEqual([a.parent_task_id for a in plan.planned_source_allocation.signal_improvements],
                         [unsubmitted_id])
        self.assertNotEqual(submitted_id, unsubmitted_id)
        # Later manual submission must not rewrite an already frozen batch.
        self._record_submitted_alpha("rank(ts_rank(open,22)+ts_zscore(close,66))",
            observed_at="2026-08-30T00:05:00+08:00", settings=self.settings, alpha_id="later-submitted")
        recovered = plan_automated_cycle(self.database_path, run_id=run_id,
                                        created_at="2026-08-30T00:06:00+08:00")
        self.assertTrue(recovered.recovered)
        self.assertEqual([s.task.task_id for s in recovered.backtests], [s.task.task_id for s in plan.backtests])

    def test_qualified_lineage_ignores_unsubmitted_cycle_and_remains_active(
        self,
    ) -> None:
        self._signal_seed("rank(ts_rank(close,22)+ts_zscore(open,66))")
        run_id = self._start_run(
            generation_count=5,
            backtest_count=5,
            max_cycles=6,
            max_backtests=30,
        )
        first = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:04:00+08:00",
        )
        first_mutations = tuple(
            snapshot
            for snapshot in first.backtests
            if self._mutation(snapshot.task.task_id) is not None
        )
        self.assertTrue(first_mutations)
        qualified_task_id = first_mutations[0].task.task_id
        self._complete_plan(
            first,
            qualified_task_id=qualified_task_id,
            qualified_sharpe=1.3,
            observed_at="2026-08-30T00:10:00+08:00",
        )
        first_settlement = settle_automated_cycle(
            self.database_path,
            run_id,
            cycle_number=1,
            observed_at="2026-08-30T00:11:00+08:00",
        )
        self.assertEqual(first_settlement.outcome, "frontier_advanced")
        self.assertTrue(first_settlement.frontier_advanced)

        second = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:12:00+08:00",
        )

        self.assertEqual(second.qualified_evolution_backtest_count, 2)
        self.assertEqual(second.structural_evolution_backtest_count, 0)
        self.assertEqual(second.local_polishing_backtest_count, 0)
        self.assertEqual(
            {
                allocation.parent_task_id
                for allocation in second.planned_source_allocation.signal_improvements
            },
            {qualified_task_id},
        )
        self.assertTrue(
            all(
                allocation.target is None
                for allocation in second.planned_source_allocation.signal_improvements
            )
        )

        self._complete_plan(
            second,
            observed_at="2026-08-30T00:20:00+08:00",
            cancel_mutations=True,
        )
        recovered = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:21:00+08:00",
        )
        self.assertTrue(recovered.recovered)
        self.assertEqual(
            tuple(item.task.task_id for item in recovered.backtests),
            tuple(item.task.task_id for item in second.backtests),
        )
        second_settlement = settle_automated_cycle(
            self.database_path,
            run_id,
            cycle_number=2,
            observed_at="2026-08-30T00:22:00+08:00",
        )
        self.assertEqual(second_settlement.outcome, "not_qualified")
        self.assertFalse(second_settlement.frontier_advanced)

        for cycle_number, minute in ((3, 30), (4, 40), (5, 50)):
            plan = plan_automated_cycle(
                self.database_path,
                run_id=run_id,
                created_at=(f"2026-08-30T00:{minute - 2:02d}:00+08:00"),
            )
            self.assertEqual(plan.qualified_evolution_backtest_count, 2)
            self._complete_plan(
                plan,
                observed_at=f"2026-08-30T00:{minute:02d}:00+08:00",
            )
            settlement = settle_automated_cycle(
                self.database_path,
                run_id,
                cycle_number=cycle_number,
                observed_at=(f"2026-08-30T00:{minute + 1:02d}:00+08:00"),
            )
            self.assertEqual(settlement.outcome, "not_qualified")
            self.assertFalse(settlement.frontier_advanced)

        sixth = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:52:00+08:00",
        )

        with open_database(self.database_path) as connection:
            frontier = load_signal_frontiers(connection).records[0]

        self.assertEqual(frontier.qualified_parent_attempt_count, 6)
        self.assertEqual(frontier.qualified_parent_remaining_attempts, 14)
        self.assertEqual(sixth.qualified_evolution_backtest_count, 2)
        self.assertEqual(sixth.planned_source_allocation.improvement_count, 2)
        self.assertEqual(sixth.exploration_backtest_count, 3)

    def test_unqualified_parent_last_slot_then_exit_survives_cycle_recovery(self):
        formula = "rank(ts_rank(close,22)+ts_zscore(open,66))"
        parent_id = self._signal_seed(formula)
        for index in range(19):
            child_formula = f"ts_mean(close,{index + 2})"
            child_id = self._completed_backtest(child_formula, sharpe=0.5, fitness=0.2,
                sharpe_status="FAIL", fitness_status="FAIL", time_offset=2)
            with open_database(self.database_path) as connection:
                create_backtest_mutation(connection, BacktestMutationRecord(
                    child_task_id=child_id, parent_task_id=parent_id,
                    action="structural", location="formula", before=formula, after=child_formula))
        run_id = self._start_run(generation_count=5, backtest_count=5, max_cycles=2, max_backtests=10)
        first = plan_automated_cycle(self.database_path, run_id=run_id,
                                     created_at="2026-08-30T00:04:00+08:00")
        self.assertEqual(first.planned_source_allocation.improvement_count, 1)
        self.assertEqual(first.exploration_backtest_count, 4)
        self.assertEqual(first.planned_source_allocation.signal_improvements[0].parent_task_id, parent_id)
        recovered = plan_automated_cycle(self.database_path, run_id=run_id,
                                        created_at="2026-08-30T00:05:00+08:00")
        self.assertTrue(recovered.recovered)
        self.assertEqual(tuple(s.task.task_id for s in first.backtests),
                         tuple(s.task.task_id for s in recovered.backtests))
        self._complete_plan(first, observed_at="2026-08-30T00:06:00+08:00")
        settle_automated_cycle(self.database_path, run_id, cycle_number=1,
                               observed_at="2026-08-30T00:07:00+08:00")
        second = plan_automated_cycle(self.database_path, run_id=run_id,
                                      created_at="2026-08-30T00:08:00+08:00")
        self.assertEqual(second.planned_source_allocation.improvement_count, 0)
        self.assertEqual(second.exploration_backtest_count, 5)
        with open_database(self.database_path) as connection:
            self.assertFalse(load_signal_frontiers(connection).active_branch_task_ids)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM signal_seeds").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM formal_submission_attempts").fetchone()[0], 0)

    def test_structural_evolution_inherits_allowed_parent_settings(self) -> None:
        parent_settings = dict(self.settings)
        parent_settings["neutralization"] = "SUBINDUSTRY"
        parent_task_id = self._signal_seed(
            "rank(ts_rank(close,22)+ts_zscore(open,66))",
            settings=parent_settings,
        )
        run_id = self._start_run()

        plan = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:04:00+08:00",
        )

        evolved = next(
            snapshot
            for snapshot in plan.backtests
            if (mutation := self._mutation(snapshot.task.task_id)) is not None
            and mutation.parent_task_id == parent_task_id
        )
        self.assertEqual(json.loads(evolved.task.settings_json), parent_settings)
        self.assertEqual(plan.structural_evolution_backtest_count, 1)
        self.assertEqual(plan.local_polishing_backtest_count, 0)

    def test_low_sub_universe_seed_receives_only_targeted_structural_action(
        self,
    ) -> None:
        self._replace_catalog(
            fields=("close", "open"),
            group_fields=("industry",),
            cross_sectional=("rank",),
            time_series=("ts_rank",),
            group=("group_rank",),
            windows=(5, 22, 66),
        )
        parent_task_id = self._signal_seed(
            "rank(close)",
            sharpe_status="PASS",
            fitness_status="PASS",
            low_sub_status="FAIL",
        )
        run_id = self._start_run()

        plan = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:04:00+08:00",
        )

        self.assertEqual(plan.structural_evolution_backtest_count, 1)
        allocation = plan.planned_source_allocation.signal_improvements[0]
        mutation = next(
            item
            for snapshot in plan.backtests
            if (item := self._mutation(snapshot.task.task_id)) is not None
        )
        self.assertEqual(allocation.parent_task_id, parent_task_id)
        self.assertEqual(
            allocation.target.check_name,
            "LOW_SUB_UNIVERSE_SHARPE",
        )
        self.assertEqual(mutation.action, allocation.candidate_family)
        self.assertIn(
            mutation.action,
            (*LOW_SUB_UNIVERSE_SHARPE_STRUCTURAL_FAMILIES, *INTERNAL_EDIT_FAMILIES),
        )

    def test_near_single_target_branch_uses_local_polishing_exclusively(
        self,
    ) -> None:
        failed_formulas = (
            "rank(ts_zscore(open,5))",
            "rank(ts_zscore(open,66))",
            "rank(ts_zscore(open,120))",
            "rank(ts_zscore(open,250))",
        )
        passed_formulas = tuple(
            f"zscore(ts_rank(open,{window}))" for window in (5, 22, 66, 120, 250)
        )
        for index, formula in enumerate(failed_formulas):
            self._completed_backtest(
                formula,
                sharpe=1.3,
                fitness=0.8 + index / 100,
                sharpe_status="PASS",
                fitness_status="FAIL",
                time_offset=index,
            )
        for index, formula in enumerate(passed_formulas, start=10):
            self._completed_backtest(
                formula,
                sharpe=1.3,
                fitness=1.0 + (index - 10) / 100,
                sharpe_status="PASS",
                fitness_status="PASS",
                time_offset=index,
            )
        parent_task_id = self._completed_backtest(
            "rank(ts_rank(close,22))",
            sharpe=1.3,
            fitness=0.98,
            sharpe_status="PASS",
            fitness_status="FAIL",
            time_offset=20,
        )
        self._mark_check_details_not_captured(parent_task_id)
        with open_database(self.database_path) as connection:
            synchronize_signal_seeds(
                connection,
                candidate_task_ids=(parent_task_id,),
            )
        run_id = self._start_run()

        first = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:30:00+08:00",
        )
        self.assertEqual(first.exploration_backtest_count, 1)
        self.assertEqual(first.structural_evolution_backtest_count, 0)
        self.assertEqual(first.local_polishing_backtest_count, 1)
        polished = next(
            snapshot
            for snapshot in first.backtests
            if self._mutation(snapshot.task.task_id) is not None
        )
        mutation = self._mutation(polished.task.task_id)
        self.assertEqual(mutation.parent_task_id, parent_task_id)
        self.assertIn(mutation.action, (SINGLE_WINDOW_MUTATION, *INTERNAL_EDIT_FAMILIES))

        with open_database(self.database_path) as connection:
            for snapshot in first.backtests:
                task_id = snapshot.task.task_id
                record_submission_accepted(
                    connection,
                    task_id,
                    remote_id=f"simulation-{task_id}",
                    observed_at="2026-08-30T00:31:00+08:00",
                )
                self._complete_with_empty_yearly_stats(
                    connection,
                    task_id,
                    platform_alpha_id=f"alpha-{task_id}",
                    observed_at="2026-08-30T00:32:00+08:00",
                    sharpe=1.3,
                    fitness=0.99,
                    turnover=0.08,
                    returns=0.02,
                    drawdown=0.08,
                    margin=0.0003,
                    checks=_checks(
                        {
                            "LOW_SHARPE": "PASS",
                            "LOW_FITNESS": "FAIL",
                            "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                            "CONCENTRATED_WEIGHT": "PASS",
                            "LOW_TURNOVER": "PASS",
                            "HIGH_TURNOVER": "PASS",
                            "MATCHES_COMPETITION": "PASS",
                            "SELF_CORRELATION": "PENDING",
                        }
                    ),
                )
        recovered = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:33:00+08:00",
        )

        self.assertTrue(recovered.recovered)
        self.assertEqual(
            tuple(item.task.task_id for item in recovered.backtests),
            tuple(item.task.task_id for item in first.backtests),
        )
        self.assertEqual(self._mutation(polished.task.task_id), mutation)

        self._mark_check_details_not_captured(polished.task.task_id)
        with open_database(self.database_path) as connection:
            record_automated_cycle_settlement(
                connection,
                run_id,
                cycle_number=1,
                outcome="frontier_advanced",
                frontier_advanced=True,
                observed_at="2026-08-30T00:34:00+08:00",
            )

        second = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:35:00+08:00",
        )
        second_mutation = next(
            mutation
            for snapshot in second.backtests
            if (mutation := self._mutation(snapshot.task.task_id)) is not None
        )

        self.assertEqual(second.local_polishing_backtest_count, 1)
        self.assertEqual(second_mutation.parent_task_id, polished.task.task_id)
        self.assertIn(
            second_mutation.action, (SINGLE_WINDOW_MUTATION, *INTERNAL_EDIT_FAMILIES)
        )
        self.assertNotEqual(second_mutation.before, second_mutation.after)
        if second_mutation.action == SINGLE_WINDOW_MUTATION:
            self.assertIn(second_mutation.after, {"66", "120", "250"})

    def test_unfilled_mutation_quota_returns_to_exploration(self) -> None:
        self._replace_catalog(
            fields=("close",),
            cross_sectional=("rank", "zscore"),
            time_series=("ts_rank", "ts_zscore"),
            windows=(5, 22, 66, 120, 250),
        )
        self._signal_seed("close")
        run_id = self._start_run()

        plan = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:04:00+08:00",
        )

        self.assertEqual(plan.exploration_backtest_count, 2)
        self.assertEqual(plan.structural_evolution_backtest_count, 0)
        self.assertEqual(plan.local_polishing_backtest_count, 0)
        self.assertEqual(
            plan.planned_source_allocation.reason,
            "signal_improvement_neighborhood_exhausted",
        )
        self.assertEqual(self._counts(), (3, 2))

    def test_last_bounded_structural_action_can_fill_the_slot(self) -> None:
        parent_formula = "rank(close)"
        parent_task_id = self._signal_seed(parent_formula)
        run_id = self._start_run()

        plan = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:04:00+08:00",
        )

        self.assertEqual(plan.structural_evolution_backtest_count, 1)
        self.assertEqual(plan.local_polishing_backtest_count, 0)
        mutation = next(
            item
            for snapshot in plan.backtests
            if (item := self._mutation(snapshot.task.task_id)) is not None
        )
        self.assertEqual(mutation.parent_task_id, parent_task_id)
        self.assertEqual(mutation.action, COMPLEMENTARY_SIGNAL_REFRAME)

    def test_next_cycle_consumes_a_different_leaf_for_the_same_branch(self) -> None:
        parent_task_id = self._signal_seed("rank(close)")
        run_id = self._start_run()
        first = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:04:00+08:00",
        )
        first_mutation = next(
            (snapshot, mutation)
            for snapshot in first.backtests
            if (mutation := self._mutation(snapshot.task.task_id)) is not None
        )
        with open_database(self.database_path) as connection:
            for snapshot in first.backtests:
                task_id = snapshot.task.task_id
                record_submission_accepted(
                    connection,
                    task_id,
                    remote_id=f"simulation-{task_id}",
                    observed_at="2026-08-30T00:05:00+08:00",
                )
                self._complete_with_empty_yearly_stats(
                    connection,
                    task_id,
                    platform_alpha_id=f"alpha-{task_id}",
                    observed_at="2026-08-30T00:06:00+08:00",
                    sharpe=0.8,
                    fitness=0.7,
                    turnover=0.08,
                    returns=0.01,
                    drawdown=0.1,
                    margin=0.0002,
                    checks=_checks(
                        {
                            "LOW_SHARPE": "FAIL",
                            "LOW_FITNESS": "FAIL",
                            "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                            "CONCENTRATED_WEIGHT": "PASS",
                            "LOW_TURNOVER": "PASS",
                            "HIGH_TURNOVER": "PASS",
                            "MATCHES_COMPETITION": "PASS",
                            "SELF_CORRELATION": "PASS",
                        }
                    ),
                )
        with open_database(self.database_path) as connection:
            record_automated_cycle_settlement(
                connection,
                run_id,
                cycle_number=1,
                outcome="not_qualified",
                frontier_advanced=False,
                observed_at="2026-08-30T00:07:00+08:00",
            )

        second = plan_automated_cycle(
            self.database_path,
            run_id=run_id,
            created_at="2026-08-30T00:08:00+08:00",
        )
        second_mutation = next(
            (snapshot, mutation)
            for snapshot in second.backtests
            if (mutation := self._mutation(snapshot.task.task_id)) is not None
        )

        self.assertEqual(first_mutation[1].parent_task_id, parent_task_id)
        self.assertEqual(second_mutation[1].parent_task_id, parent_task_id)
        self.assertNotEqual(
            first_mutation[0].task.formula_fingerprint,
            second_mutation[0].task.formula_fingerprint,
        )

    def test_sc_repairs_share_frozen_plans_original_lineage_and_attempt_budget(self):
        self._replace_catalog(fields=("close", "open", "returns", "volume"),
            cross_sectional=("rank", "zscore"), time_series=("ts_rank", "ts_zscore"),
            windows=(5, 22, 66, 120, 250), pairwise=("vector_neut",))
        run_id = self._start_run(generation_count=6, backtest_count=4, max_cycles=6, max_backtests=24)
        first = plan_automated_cycle(self.database_path, run_id=run_id,
            created_at="2026-08-30T00:04:00+08:00")
        parent = next(s for s in first.backtests
                      if isinstance(parse_formula(s.task.formula).expression, Call)
                      and isinstance(parse_formula(s.task.formula).expression.arguments[0].value, Call))
        parent_id = parent.task.task_id
        self._complete_plan(first, observed_at="2026-08-30T00:05:00+08:00",
                            qualified_task_id=parent_id, qualified_sharpe=1.5)
        with open_database(self.database_path) as connection:
            connection.execute("INSERT INTO backtest_yearly_stats(task_id,stage,year,sharpe) VALUES (?, 'IS', 2023, 1.5)", (parent_id,))
            synchronize_signal_seeds(connection)
            record_automated_cycle_settlement(connection, run_id, cycle_number=1,
                outcome="qualified", frontier_advanced=True, observed_at="2026-08-30T00:06:00+08:00")
        reference_formula = render_formula(parse_formula(parent.task.formula).expression.arguments[0].value)
        self._record_submitted_alpha(reference_formula, observed_at="2026-08-30T00:06:00+08:00", settings=self.settings)
        # Submitted and unsubmitted research share the same frozen batch and parent budget.
        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, parent_id)
            record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
                account_scope=snapshot.task.account_scope,
                platform_alpha_id=snapshot.task.platform_alpha_id,
                formula=snapshot.task.formula, status="ACTIVE",
                date_submitted="2026-08-30T00:06:00+08:00", hidden=False,
                raw_payload={"settings": self.settings, "id": snapshot.task.platform_alpha_id,
                             "status": "ACTIVE", "hidden": False,
                             "dateSubmitted": "2026-08-30T00:06:00+08:00",
                             "regular": {"code": snapshot.task.formula}},
                observed_at="2026-08-30T00:06:00+08:00",
            ),))
        self._record_formal_attempt(task_id=parent_id, run_id=run_id, family_root_task_id=parent_id,
            observed_at="2026-08-30T00:07:00+08:00",
            checks={name: "FAIL" if name == "SELF_CORRELATION" else "PASS" for name in STANDARD_REGULAR_CHECK_NAMES},
            correlation_detail={"max": 0.8, "schema": {"properties": [{"name": "id"}, {"name": "correlation"}]},
                "records": [["submitted-alpha", 0.8]]})
        repair_families = set()
        repair_fingerprints = set()
        attempted_children = 0
        for cycle in range(2, 7):
            minute = 8 + cycle * 4
            plan = plan_automated_cycle(self.database_path, run_id=run_id,
                created_at=f"2026-08-30T00:{minute:02d}:00+08:00")
            recovered = plan_automated_cycle(self.database_path, run_id=run_id,
                created_at=f"2026-08-30T00:{minute:02d}:30+08:00")
            self.assertTrue(recovered.recovered)
            self.assertEqual(tuple(s.task.task_id for s in recovered.backtests), tuple(s.task.task_id for s in plan.backtests))
            self.assertGreaterEqual(plan.exploration_backtest_count, 2)
            for snapshot in plan.backtests:
                mutation = self._mutation(snapshot.task.task_id)
                if mutation and mutation.parent_task_id == parent_id:
                    attempted_children += 1
                if mutation and mutation.action in SELF_CORRELATION_REPAIR_FAMILIES:
                    repair_families.add(mutation.action)
                    self.assertNotIn(snapshot.task.formula_fingerprint, repair_fingerprints)
                    repair_fingerprints.add(snapshot.task.formula_fingerprint)
                    self.assertEqual(mutation.parent_task_id, parent_id)
                    self.assertEqual(snapshot.task.settings_json, parent.task.settings_json)
            if cycle == 2:
                self.assertTrue(repair_families, "SC actions must enter the next unfrozen batch")
            self._complete_plan(plan, observed_at=f"2026-08-30T00:{minute + 1:02d}:00+08:00")
            with open_database(self.database_path) as connection:
                frontier = next(item for item in load_signal_frontiers(connection).records if item.root_task_id == parent_id)
                self.assertEqual(frontier.qualified_parent_remaining_attempts, 20 - attempted_children)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM formal_submission_attempts").fetchone()[0], 1)
                record_automated_cycle_settlement(connection, run_id, cycle_number=cycle,
                    outcome="not_qualified", frontier_advanced=False,
                    observed_at=f"2026-08-30T00:{minute + 2:02d}:00+08:00")
        self.assertTrue(repair_families)
        self.assertTrue(repair_families <= {"self_correlation_shared_field_replacement", "self_correlation_shared_field_operator_replacement"})

    def test_submitted_parent_without_failed_check_uses_ordinary_quota_and_recovers(self):
        formula = "rank(ts_rank(close,22)+ts_zscore(open,66))"
        parent_id = self._signal_seed(formula, sharpe_status="PASS", fitness_status="PASS")
        with open_database(self.database_path) as connection:
            connection.execute("INSERT INTO backtest_yearly_stats(task_id,stage,year,sharpe) VALUES (?, 'IS', 2023, 1.5)", (parent_id,))
        self._record_submitted_alpha(formula, observed_at="2026-08-30T00:03:00+08:00",
                                     settings=self.settings, alpha_id=f"alpha-{parent_id}")
        run_id = self._start_run(generation_count=6, backtest_count=4)
        plan = plan_automated_cycle(self.database_path, run_id=run_id,
                                   created_at="2026-08-30T00:04:00+08:00")
        allocations = plan.planned_source_allocation.signal_improvements
        self.assertEqual(len(allocations), 2)
        self.assertTrue(all(a.candidate_family not in SELF_CORRELATION_REPAIR_FAMILIES for a in allocations))
        self.assertTrue(all(a.parent_task_id == parent_id for a in allocations))
        self.assertFalse(any(a.candidate_family == SELF_CORRELATION_REPAIR for a in allocations))
        recovered = plan_automated_cycle(self.database_path, run_id=run_id,
                                        created_at="2026-08-30T00:05:00+08:00")
        self.assertTrue(recovered.recovered)
        self.assertEqual([s.task.task_id for s in recovered.backtests], [s.task.task_id for s in plan.backtests])
        with open_database(self.database_path) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM formal_submission_attempts").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT count(*) FROM submission_checks").fetchone()[0], 0)
            self.assertTrue(all(s.task.submission_started_at is None for s in plan.backtests))

    def _record_submitted_alpha(self, formula: str, *, observed_at="2026-09-03T00:00:00+00:00", settings=None, alpha_id="submitted-alpha", sharpe=None) -> None:
        raw_payload = {
            "id": alpha_id,
            "status": "ACTIVE",
            "dateSubmitted": "2026-08-01T00:00:00+00:00",
            "hidden": False,
            "regular": {"code": formula},
        }
        if settings is not None:
            raw_payload["settings"] = settings
        if sharpe is not None:
            raw_payload["is"] = {"sharpe": sharpe}
        with open_database(self.database_path) as connection:
            record_platform_submitted_alphas(
                connection,
                (
                    PlatformSubmittedAlphaRecord(
                        account_scope="group-account",
                        platform_alpha_id=alpha_id,
                        formula=formula,
                        status="ACTIVE",
                        date_submitted="2026-08-01T00:00:00+00:00",
                        hidden=False,
                        raw_payload=raw_payload,
                        observed_at=observed_at,
                    ),
                ),
            )

    def _record_formal_attempt(
        self,
        *,
        task_id: str,
        run_id: str,
        family_root_task_id: str,
        observed_at: str,
        checks: dict[str, str],
        correlation_detail: dict | None = None,
    ) -> None:
        payload = {
            "is": {
                "checks": [
                    {"name": name, "result": status}
                    for name, status in sorted(checks.items())
                ]
            }
        }
        if correlation_detail is not None:
            payload["is"]["selfCorrelated"] = correlation_detail
            for check in payload["is"]["checks"]:
                if check["name"] == "SELF_CORRELATION":
                    check.update(value=correlation_detail["max"], limit=0.7)
        with open_database(self.database_path) as connection:
            create_formal_submission_attempt(
                connection,
                FormalSubmissionAttemptRecord(
                    task_id=task_id,
                    run_id=run_id,
                    cycle_number=1,
                    family_root_task_id=family_root_task_id,
                    submission_mode="manual",
                    status="ineligible",
                    check_attempt_count=1,
                    check_payload_json=canonical_submission_json(payload),
                    check_observed_at=observed_at,
                    retry_not_before=None,
                    submission_claimed_at=None,
                    submit_http_status=None,
                    submit_response_json=None,
                    confirmation_observed_at=None,
                    failure_code="formal_checks_failed",
                    created_at=observed_at,
                    updated_at=observed_at,
                ),
            )

    def _start_run(
        self,
        *,
        generation_count: int = 5,
        backtest_count: int = 2,
        max_cycles: int = 2,
        max_backtests: int = 4,
        created_at: str = "2026-08-30T00:00:00+08:00",
        started_at: str = "2026-08-30T00:01:00+08:00",
        optimization_only: bool = False,
        automatic_submissions_enabled: bool = False,
    ) -> str:
        prepared = prepare_automated_run(
            self.database_path,
            self.policy_path,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=generation_count,
                backtest_count=backtest_count,
                max_cycles=max_cycles,
                max_backtests=max_backtests,
                max_pending_seconds=3600,
                max_consecutive_failures=2,
                max_request_failures=3,
                max_in_flight_backtests=3,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
                optimization_only=optimization_only,
                automatic_submissions_enabled=automatic_submissions_enabled,
            ),
            created_at=created_at,
        )
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at=started_at,
        )
        return prepared.run_id

    def _complete_plan(
        self,
        plan: AutomatedCyclePlan,
        *,
        observed_at: str,
        qualified_task_id: str | None = None,
        qualified_sharpe: float = 1.3,
        cancel_mutations: bool = False,
    ) -> None:
        with open_database(self.database_path) as connection:
            for snapshot in plan.backtests:
                task_id = snapshot.task.task_id
                if (
                    cancel_mutations
                    and get_backtest_mutation(connection, task_id) is not None
                ):
                    cancel_unsubmitted_backtest_task(
                        connection,
                        task_id,
                        observed_at=observed_at,
                    )
                    continue
                qualified = task_id == qualified_task_id
                record_submission_accepted(
                    connection,
                    task_id,
                    remote_id=f"simulation-{task_id}",
                    observed_at=observed_at,
                )
                self._complete_with_empty_yearly_stats(
                    connection,
                    task_id,
                    platform_alpha_id=f"alpha-{task_id}",
                    observed_at=observed_at,
                    sharpe=qualified_sharpe if qualified else 0.5,
                    fitness=1.1 if qualified else 0.2,
                    turnover=0.08,
                    returns=0.02,
                    drawdown=0.08,
                    margin=0.0003,
                    checks=_checks(
                        {
                            "LOW_SHARPE": "PASS" if qualified else "FAIL",
                            "LOW_FITNESS": "PASS" if qualified else "FAIL",
                            "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                            "CONCENTRATED_WEIGHT": "PASS",
                            "LOW_TURNOVER": "PASS",
                            "HIGH_TURNOVER": "PASS",
                            "MATCHES_COMPETITION": "PASS",
                            "SELF_CORRELATION": "PENDING",
                        }
                    ),
                )

    def _signal_seed(
        self,
        formula: str,
        *,
        sc_status: str = "PASS",
        sharpe_status: str = "FAIL",
        fitness_status: str = "FAIL",
        low_sub_status: str = "PASS",
        settings: dict[str, object] | None = None,
    ) -> str:
        with open_database(self.database_path) as connection:
            parent = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula=formula,
                settings=self.settings if settings is None else settings,
                created_at="2026-08-30T00:00:00+08:00",
            )
            record_submission_accepted(
                connection,
                parent.task.task_id,
                remote_id=f"simulation-{parent.task.task_id}",
                observed_at="2026-08-30T00:01:00+08:00",
            )
            completed = self._complete_with_empty_yearly_stats(
                connection,
                parent.task.task_id,
                platform_alpha_id=f"alpha-{parent.task.task_id}",
                observed_at="2026-08-30T00:02:00+08:00",
                sharpe=1.1,
                fitness=0.9,
                turnover=0.08,
                returns=0.01,
                drawdown=0.1,
                margin=0.0002,
                checks=_checks(
                    {
                        "LOW_SHARPE": sharpe_status,
                        "LOW_FITNESS": fitness_status,
                        "LOW_SUB_UNIVERSE_SHARPE": low_sub_status,
                        "CONCENTRATED_WEIGHT": "PASS",
                        "LOW_TURNOVER": "PASS",
                        "HIGH_TURNOVER": "PASS",
                        "MATCHES_COMPETITION": "PASS",
                        "SELF_CORRELATION": sc_status,
                    }
                ),
            )
            synchronize_signal_seeds(
                connection,
                candidate_task_ids=(completed.task.task_id,),
            )
        return completed.task.task_id

    def _completed_backtest(
        self,
        formula: str,
        *,
        sharpe: float,
        fitness: float,
        sharpe_status: str,
        fitness_status: str,
        time_offset: int,
    ) -> str:
        with open_database(self.database_path) as connection:
            prepared = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula=formula,
                settings=self.settings,
                created_at=f"2026-08-30T00:{time_offset:02d}:00+08:00",
            )
            record_submission_accepted(
                connection,
                prepared.task.task_id,
                remote_id=f"simulation-{prepared.task.task_id}",
                observed_at=f"2026-08-30T00:{time_offset:02d}:10+08:00",
            )
            completed = self._complete_with_empty_yearly_stats(
                connection,
                prepared.task.task_id,
                platform_alpha_id=f"alpha-{prepared.task.task_id}",
                observed_at=f"2026-08-30T00:{time_offset:02d}:20+08:00",
                sharpe=sharpe,
                fitness=fitness,
                turnover=0.08,
                returns=0.02,
                drawdown=0.08,
                margin=0.0003,
                checks=_checks(
                    {
                        "LOW_SHARPE": sharpe_status,
                        "LOW_FITNESS": fitness_status,
                        "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                        "CONCENTRATED_WEIGHT": "PASS",
                        "LOW_TURNOVER": "PASS",
                        "HIGH_TURNOVER": "PASS",
                        "MATCHES_COMPETITION": "PASS",
                        "SELF_CORRELATION": "PENDING",
                    }
                ),
            )
        return completed.task.task_id

    @staticmethod
    def _complete_with_empty_yearly_stats(
        connection,
        task_id: str,
        *,
        platform_alpha_id: str,
        observed_at: str,
        sharpe: float,
        fitness: float,
        turnover: float,
        returns: float,
        drawdown: float,
        margin: float,
        checks: tuple[BacktestCheck, ...],
    ):
        apply_backtest_detail(
            connection,
            task_id,
            BacktestDetail(
                platform_alpha_id=platform_alpha_id,
                sharpe=sharpe,
                fitness=fitness,
                turnover=turnover,
                returns=returns,
                drawdown=drawdown,
                margin=margin,
                book_size=None,
                pnl=None,
                checks=checks,
            ),
            observed_at=observed_at,
        )
        return record_backtest_yearly_stats(
            connection,
            task_id,
            (),
            observed_at=observed_at,
        )

    def _mutation(self, task_id: str):
        with open_database(self.database_path) as connection:
            return get_backtest_mutation(connection, task_id)

    def _mark_check_details_not_captured(self, task_id: str) -> None:
        with open_database(self.database_path) as connection:
            updated = connection.execute(
                """
                UPDATE backtest_results
                SET check_details_captured = 0
                WHERE task_id = ?
                """,
                (task_id,),
            )
        self.assertEqual(updated.rowcount, 1)

    def _counts(self) -> tuple[int, int]:
        with open_database(self.database_path) as connection:
            task_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_tasks"
            ).fetchone()[0]
            link_count = connection.execute(
                "SELECT COUNT(*) FROM automated_run_backtests"
            ).fetchone()[0]
        return task_count, link_count

    def _replace_catalog(
        self,
        *,
        fields: tuple[str, ...],
        pairwise: tuple[str, ...] = (),
        field_categories: dict[str, str] | None = None,
        vector_fields: tuple[str, ...] = (),
        group_fields: tuple[str, ...] = (),
        cross_sectional: tuple[str, ...],
        time_series: tuple[str, ...],
        vector: tuple[str, ...] = (),
        group: tuple[str, ...] = (),
        windows: tuple[int, ...],
    ) -> None:
        categories = field_categories or {}
        context = FieldCatalogContext("EQUITY", "USA", "TOP3000", 1)
        field_records = tuple(
            FieldCatalogRecord(
                context=context,
                field_id=field_id,
                dataset_id="dataset",
                category=categories.get(field_id, "sample"),
                subcategory=None,
                field_type=field_type,
                coverage=1.0,
                description=field_id,
                dataset_name="sample",
                category_id="sample",
                subcategory_id=None,
                raw_payload={"id": field_id},
                synced_at="2026-08-30T00:00:00+08:00",
            )
            for field_id, field_type in (
                *((field_id, "MATRIX") for field_id in fields),
                *((field_id, "VECTOR") for field_id in vector_fields),
                *((field_id, "GROUP") for field_id in group_fields),
            )
        )
        operators = (
            tuple(
                self._operator(name, "Cross Sectional", (("x", "expr"),))
                for name in cross_sectional
            )
            + tuple(
                self._operator(
                    name,
                    "Time Series",
                    (("x", "expr"), ("d", "window")),
                )
                for name in time_series
            )
            + tuple(self._operator(name, "Vector", (("x", "expr"),)) for name in vector)
            + tuple(self._operator(name, "Cross Sectional", (("x", "expr"), ("y", "expr"))) for name in pairwise)
            + tuple(
                self._operator(
                    name,
                    "Group",
                    (("x", "expr"), ("group", "group")),
                )
                for name in group
            )
        )
        with open_database(self.database_path) as connection:
            replace_operator_roles(
                connection,
                tuple(
                    OperatorRoleRecord(
                        record.operator_name,
                        (
                            "cross_sectional_normalization"
                            if record.category == "Cross Sectional"
                            else "time_series_normalization"
                        ),
                    )
                    for record in operators
                    if record.category in {"Cross Sectional", "Time Series"}
                ),
            )
            replace_operator_outputs(
                connection,
                tuple(
                    OperatorOutputRecord(record.operator_name, "signal")
                    for record in operators
                ),
            )
            replace_window_catalog(
                connection,
                tuple(
                    WindowCatalogRecord(value, f"window_{value}") for value in windows
                ),
            )
            replace_platform_catalog(
                connection,
                PlatformCatalogSyncRecord(
                    account_scope="group-account",
                    context=context,
                    field_count=len(field_records),
                    operator_count=len(operators),
                    synced_at="2026-08-30T00:00:00+08:00",
                ),
                field_records,
                operators,
            )

    @staticmethod
    def _operator(
        name: str,
        category: str,
        parameters: tuple[tuple[str, str], ...],
    ) -> OperatorCatalogRecord:
        return OperatorCatalogRecord(
            operator_name=name,
            category=category,
            definition=name,
            description=name,
            documentation=None,
            level="ALL",
            scope=("REGULAR",),
            parameters=tuple(
                {"name": parameter_name, "kind": parameter_kind}
                for parameter_name, parameter_kind in parameters
            ),
            raw_payload={"name": name},
            synced_at="2026-08-30T00:00:00+08:00",
        )


if __name__ == "__main__":
    unittest.main()
