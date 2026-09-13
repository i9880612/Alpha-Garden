from __future__ import annotations

# ruff: noqa: E402

import json
import sys
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.cycle_candidates import (
    CandidateSelectionShortfall,
    ExplorationAttemptBudgetExhausted,
    ImprovementCandidatePool,
    ImprovementPoolCandidate,
    build_cycle_candidates,
    build_improvement_candidate_pools,
)
from execution.generation import (
    BatchShortfall,
    ExclusionCount,
    ExplorationBatch,
    ExplorationCandidate,
)
from generation.candidate import (
    CandidateChange,
    exploration_candidate,
    mutation_candidate,
)
from generation.catalog import (
    CatalogContext,
    FieldDefinition,
    GenerationCatalog,
    OperatorDefinition,
    OperatorParameter,
    WindowDefinition,
)
from generation.parser import parse_formula
from generation.polishing import SINGLE_WINDOW_MUTATION
from generation.internal_edits import INTERNAL_EDIT_FAMILIES
from generation.self_correlation import (
    SELF_CORRELATION_REPAIR, SELF_CORRELATION_INTERNAL_FAMILIES,
    SELF_CORRELATION_LIGHT_FAMILIES,
)
from learning.self_correlation import SelfCorrelationReference
from generation.transformations import (
    STRUCTURAL_TRANSFORMATION_FAMILIES,
    TEMPORAL_CHANGE_REFRAME,
    TEMPORAL_PERSISTENCE_REFRAME,
    iter_transformation_leaves,
)
from learning.optimization_targets import OptimizationTarget
from learning.quality_proximity import (
    LOCAL_POLISHING_STAGE,
    QUALIFIED_EVOLUTION_STAGE,
    STRUCTURAL_EVOLUTION_STAGE,
    ParentImprovementStage,
    ParentImprovementStageSet,
)
from persistence.backtests import (
    BacktestCheckRecord,
    BacktestResultRecord,
    BacktestSnapshot,
    BacktestTaskRecord,
)
from persistence.run_diagnostics import (
    CandidatePlanningDiagnostic,
    CandidatePlanningExclusionCount,
)
from selection.allocations import (
    BacktestSourceAllocation,
    SignalImprovementAllocation,
)
from selection.settings import (
    BacktestSettingsPolicy,
    CategoryNeutralization,
)
from worldquant.backtests import BacktestSettings


class ImprovementCandidatePoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            fields=(self._field("close"), self._field("open")),
            operators=(
                self._operator(
                    "rank",
                    ("cross_sectional_normalization",),
                    OperatorParameter("x", "expr"),
                ),
                self._operator(
                    "ts_delta",
                    ("time_series_change",),
                    OperatorParameter("x", "expr"),
                    OperatorParameter("d", "window"),
                ),
                self._operator(
                    "ts_mean",
                    ("time_series_smoothing",),
                    OperatorParameter("x", "expr"),
                    OperatorParameter("d", "window"),
                ),
            ),
            windows=(
                WindowDefinition(5, "week"),
                WindowDefinition(22, "month"),
            ),
        )
        self.settings = BacktestSettings(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
            decay=4,
            neutralization="SECTOR",
            truncation=0.08,
            pasteurization="ON",
            unit_handling="VERIFY",
            nan_handling="OFF",
            language="FASTEXPR",
            visualization=False,
            max_trade="OFF",
            max_position="OFF",
        )
        self.settings_json = self._settings_json(self.settings)
        self.policy = BacktestSettingsPolicy(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
            decay=4,
            default_neutralization="SECTOR",
            root_group_neutralization="NONE",
            category_neutralizations=(CategoryNeutralization("sample", "SECTOR"),),
            default_truncation=0.08,
            tail_risk_truncation=0.05,
            pasteurization="ON",
            unit_handling="VERIFY",
            nan_handling="OFF",
            language="FASTEXPR",
            visualization=False,
            max_trade="OFF",
            max_position="OFF",
        )
        self.parent = self._parent_snapshot(
            "parent-a",
            "rank(close)",
        )

    def test_completed_and_failed_requests_are_consumed(self) -> None:
        baseline = self._pool()
        self.assertTrue(baseline.candidates)
        leaf = baseline.candidates[0]

        for status in ("completed", "failed"):
            with self.subTest(status=status):
                reserved = self._reserved_task(leaf, status=status)
                pool = self._pool(reserved_tasks=(reserved,))

                self.assertEqual(pool.consumed_request_count, 1)
                self.assertEqual(pool.blocked_request_count, 0)
                self.assertEqual(
                    pool.attempted_request_count,
                    1 if status == "completed" else 0,
                )
                self.assertEqual(
                    len(pool.candidates),
                    len(baseline.candidates) - 1,
                )
                self.assertNotIn(
                    leaf.leaf_id,
                    {item.leaf_id for item in pool.candidates},
                )

    def test_active_requests_are_blocked(self) -> None:
        baseline = self._pool()
        leaf = baseline.candidates[0]

        for status in ("created", "pending", "submission_unknown"):
            with self.subTest(status=status):
                reserved = self._reserved_task(leaf, status=status)
                pool = self._pool(reserved_tasks=(reserved,))

                self.assertEqual(pool.blocked_request_count, 1)
                self.assertEqual(pool.consumed_request_count, 0)
                self.assertEqual(
                    pool.attempted_request_count,
                    0 if status == "created" else 1,
                )
                self.assertEqual(
                    len(pool.candidates),
                    len(baseline.candidates) - 1,
                )
                self.assertNotIn(
                    leaf.leaf_id,
                    {item.leaf_id for item in pool.candidates},
                )

    def test_only_unsent_cancellation_releases_candidate_and_cannot_hide_new_attempt(self):
        leaf = self._pool().candidates[0]
        cancelled = replace(self._reserved_task(leaf, status="failed"),
                            failure_code="automated_run_stopped_before_submission")
        pool = self._pool(reserved_tasks=(cancelled,))
        self.assertIn(leaf.leaf_id, {c.leaf_id for c in pool.candidates})
        active = replace(self._reserved_task(leaf, status="pending"), task_id="retried-task")
        for history in ((active, cancelled), (cancelled, active)):
            pool = self._pool(reserved_tasks=history)
            self.assertNotIn(leaf.leaf_id, {c.leaf_id for c in pool.candidates})
            self.assertEqual(pool.blocked_request_count, 1)

    def test_explicitly_excluded_formula_is_consumed(self) -> None:
        baseline = self._pool()
        leaf = baseline.candidates[0]

        pool = self._pool(
            excluded_formula_fingerprints=frozenset(
                {leaf.candidate.fingerprint}
            ),
        )

        self.assertEqual(pool.consumed_request_count, 1)
        self.assertEqual(pool.blocked_request_count, 0)
        self.assertEqual(pool.attempted_request_count, 0)
        self.assertEqual(len(pool.candidates), len(baseline.candidates) - 1)
        self.assertNotIn(
            leaf.leaf_id,
            {item.leaf_id for item in pool.candidates},
        )

    def test_failed_request_counts_only_after_platform_attempt_started(self) -> None:
        baseline = self._pool()
        leaf = baseline.candidates[0]
        failed_before_submission = self._reserved_task(leaf, status="failed")
        failed_after_submission = replace(
            failed_before_submission,
            submission_started_at="2026-09-01T00:01:00+00:00",
        )

        before_pool = self._pool(reserved_tasks=(failed_before_submission,))
        after_pool = self._pool(reserved_tasks=(failed_after_submission,))

        self.assertEqual(before_pool.consumed_request_count, 1)
        self.assertEqual(before_pool.attempted_request_count, 0)
        self.assertEqual(after_pool.consumed_request_count, 1)
        self.assertEqual(after_pool.attempted_request_count, 1)

    def test_same_formula_under_different_settings_does_not_block(self) -> None:
        baseline = self._pool()
        leaf = baseline.candidates[0]
        other_settings = replace(self.settings, neutralization="NONE")
        reserved = self._reserved_task(
            leaf,
            status="completed",
            settings_json=self._settings_json(other_settings),
        )

        pool = self._pool(reserved_tasks=(reserved,))

        self.assertEqual(pool.candidates, baseline.candidates)
        self.assertEqual(pool.blocked_request_count, 0)
        self.assertEqual(pool.consumed_request_count, 0)

    def test_same_formula_under_different_account_does_not_block(self) -> None:
        baseline = self._pool()
        leaf = baseline.candidates[0]
        reserved = self._reserved_task(
            leaf,
            status="completed",
            account_scope="other-account",
        )

        pool = self._pool(reserved_tasks=(reserved,))

        self.assertEqual(pool.candidates, baseline.candidates)
        self.assertEqual(pool.blocked_request_count, 0)
        self.assertEqual(pool.consumed_request_count, 0)

    def test_shared_request_is_claimed_by_one_stable_parent(self) -> None:
        parents = {
            "parent-z": self._parent_snapshot("parent-z", "rank(close)"),
            "parent-a": self._parent_snapshot("parent-a", "rank(open)"),
        }
        shared_leaf = next(
            iter_transformation_leaves(
                parse_formula("rank(close)").expression,
                self.catalog,
            )
        )

        with patch(
            "execution.cycle_candidates.iter_transformation_leaves",
            return_value=(shared_leaf,),
        ):
            reversed_input = self._pools(
                eligible_parent_task_ids=("parent-z", "parent-a"),
                completed_by_task_id=parents,
            )
            forward_input = self._pools(
                eligible_parent_task_ids=("parent-a", "parent-z"),
                completed_by_task_id=parents,
            )

        expected = (("parent-a", 1), ("parent-z", 0))
        self.assertEqual(
            tuple(
                (
                    pool.parent_task_id,
                    sum(
                        c.candidate.fingerprint == shared_leaf.formula_fingerprint
                        for c in pool.candidates
                    ),
                )
                for pool in reversed_input
            ),
            expected,
        )
        self.assertEqual(
            tuple(
                (
                    pool.parent_task_id,
                    sum(
                        c.candidate.fingerprint == shared_leaf.formula_fingerprint
                        for c in pool.candidates
                    ),
                )
                for pool in forward_input
            ),
            expected,
        )

    def test_cycle_uses_allocated_families_without_reusing_a_leaf(self) -> None:
        pool = ImprovementCandidatePool(
            parent_task_id=self.parent.task.task_id,
            stage=STRUCTURAL_EVOLUTION_STAGE,
            settings=self.settings,
            candidates=(
                self._pool_candidate(
                    "rank(ts_delta(rank(close),5))",
                    TEMPORAL_CHANGE_REFRAME,
                ),
                self._pool_candidate(
                    "rank(ts_delta(rank(close),22))",
                    TEMPORAL_CHANGE_REFRAME,
                ),
                self._pool_candidate(
                    "rank(ts_mean(rank(close),5))",
                    TEMPORAL_PERSISTENCE_REFRAME,
                ),
            ),
            family_usage=(),
        )
        target = OptimizationTarget(
            check_name="LOW_FITNESS",
            details_captured=True,
            threshold=1.0,
            actual=0.8,
            platform_date="2026-09-01",
            normalized_gap=0.2,
            gap_state="available",
            actions=(),
        )
        persistence = SignalImprovementAllocation(
            root_task_id="root-a",
            parent_task_id=self.parent.task.task_id,
            stage=STRUCTURAL_EVOLUTION_STAGE,
            sc_risk_state="acceptable",
            target=target,
            candidate_family=TEMPORAL_PERSISTENCE_REFRAME,
        )
        change = SignalImprovementAllocation(
            root_task_id="root-a",
            parent_task_id=self.parent.task.task_id,
            stage=STRUCTURAL_EVOLUTION_STAGE,
            sc_risk_state="acceptable",
            target=target,
            candidate_family=TEMPORAL_CHANGE_REFRAME,
        )
        allocation = BacktestSourceAllocation(
            requested_count=3,
            minimum_exploration_count=1,
            exploration_count=1,
            structural_evolution_count=2,
            local_polishing_count=0,
            signal_improvements=(persistence, change),
            reason="signal_improvement_attempts",
        )
        exploration = exploration_candidate(parse_formula("rank(open)").expression)
        exploration_batch = ExplorationBatch(
            candidates=(ExplorationCandidate(exploration, seed=0),),
            attempted_seed_count=1,
            exclusions=(),
            shortfall=None,
        )

        with patch(
            "execution.cycle_candidates.generate_exploration_batch",
            return_value=exploration_batch,
        ):
            plan = build_cycle_candidates(
                self.catalog,
                self.policy,
                cycle_number=1,
                requested_generation_count=3,
                requested_backtest_count=3,
                exploration_seed_attempt_multiplier=4,
                source_allocation=allocation,
                improvement_pools=(pool,),
                exploration_field_candidates=("close", "open"),
                group_candidates=(),
                existing_formulas=(),
                seed_start=0,
            )

        evolved = plan.candidates[: allocation.improvement_count]
        self.assertEqual(
            plan.structural_evolution_backtest_count,
            allocation.structural_evolution_count,
        )
        self.assertEqual(plan.local_polishing_backtest_count, 0)
        self.assertEqual(plan.exploration_backtest_count, 1)
        self.assertEqual(len(plan.candidates), allocation.requested_count)
        self.assertEqual(
            tuple(item.candidate.change.action for item in evolved),
            (TEMPORAL_PERSISTENCE_REFRAME, TEMPORAL_CHANGE_REFRAME),
        )
        self.assertEqual(
            len({item.candidate.fingerprint for item in evolved}),
            allocation.improvement_count,
        )

    def test_selection_rejection_shortfall_has_its_own_stop_type(self) -> None:
        allocation = BacktestSourceAllocation(
            requested_count=1,
            minimum_exploration_count=1,
            exploration_count=1,
            structural_evolution_count=0,
            local_polishing_count=0,
            signal_improvements=(),
            reason="exploration_reserve_uses_full_batch",
        )
        rejected = exploration_candidate(parse_formula("close/open").expression)
        exploration_batch = ExplorationBatch(
            candidates=(ExplorationCandidate(rejected, seed=0),),
            attempted_seed_count=1,
            exclusions=(),
            shortfall=None,
        )

        with (
            patch(
                "execution.cycle_candidates.generate_exploration_batch",
                return_value=exploration_batch,
            ),
            self.assertRaisesRegex(
                CandidateSelectionShortfall,
                "automated_cycle_selection_shortfall:1",
            ) as raised,
        ):
            build_cycle_candidates(
                self.catalog,
                self.policy,
                cycle_number=1,
                requested_generation_count=1,
                requested_backtest_count=1,
                exploration_seed_attempt_multiplier=4,
                source_allocation=allocation,
                improvement_pools=(),
                exploration_field_candidates=("close", "open"),
                group_candidates=(),
                existing_formulas=(),
                seed_start=0,
            )

        self.assertEqual(
            raised.exception.diagnostic,
            CandidatePlanningDiagnostic(
                cycle_number=1,
                exploration_generation_target_count=1,
                exploration_backtest_target_count=1,
                seed_attempt_limit=4,
                attempted_seed_count=1,
                generated_candidate_count=1,
                selected_candidate_count=0,
                generation_exclusions=(),
                selection_rejections=(
                    CandidatePlanningExclusionCount("uncontrolled_tail", 1),
                ),
            ),
        )

    def test_frozen_multiplier_sets_attempt_limit_and_generation_diagnostic(
        self,
    ) -> None:
        allocation = BacktestSourceAllocation(
            requested_count=1,
            minimum_exploration_count=1,
            exploration_count=1,
            structural_evolution_count=0,
            local_polishing_count=0,
            signal_improvements=(),
            reason="exploration_reserve_uses_full_batch",
        )
        shortfall = ExplorationBatch(
            candidates=(),
            attempted_seed_count=6,
            exclusions=(ExclusionCount("test_rejection", 6),),
            shortfall=BatchShortfall(missing_count=2),
        )

        with (
            patch(
                "execution.cycle_candidates.generate_exploration_batch",
                return_value=shortfall,
            ) as generate,
            self.assertRaisesRegex(
                ExplorationAttemptBudgetExhausted,
                "automated_cycle_generation_shortfall:2",
            ) as raised,
        ):
            build_cycle_candidates(
                self.catalog,
                self.policy,
                cycle_number=2,
                requested_generation_count=2,
                requested_backtest_count=1,
                exploration_seed_attempt_multiplier=3,
                source_allocation=allocation,
                improvement_pools=(),
                exploration_field_candidates=("close", "open"),
                group_candidates=(),
                existing_formulas=(),
                seed_start=10,
            )

        self.assertEqual(generate.call_args.kwargs["seeds"], tuple(range(10, 16)))
        self.assertEqual(
            raised.exception.diagnostic,
            CandidatePlanningDiagnostic(
                cycle_number=2,
                exploration_generation_target_count=2,
                exploration_backtest_target_count=1,
                seed_attempt_limit=6,
                attempted_seed_count=6,
                generated_candidate_count=0,
                selected_candidate_count=None,
                generation_exclusions=(
                    CandidatePlanningExclusionCount("test_rejection", 6),
                ),
                selection_rejections=(),
            ),
        )

    def test_local_polishing_pool_uses_single_window_mutation(self) -> None:
        parent = self._parent_snapshot(
            "parent-polish",
            "rank(ts_delta(close,5))",
        )
        pool = self._pools(
            eligible_parent_task_ids=(parent.task.task_id,),
            completed_by_task_id={parent.task.task_id: parent},
            stage=LOCAL_POLISHING_STAGE,
        )[0]

        self.assertEqual(pool.stage, LOCAL_POLISHING_STAGE)
        self.assertTrue(
            any(c.family in INTERNAL_EDIT_FAMILIES for c in pool.candidates)
        )
        self.assertTrue(
            all(
                c.family in (SINGLE_WINDOW_MUTATION, *INTERNAL_EDIT_FAMILIES)
                for c in pool.candidates
            )
        )
        self.assertEqual(
            pool.candidates[0].candidate.formula,
            "rank(ts_delta(close,22))",
        )
        self.assertEqual(
            pool.candidates[0].candidate.change.action,
            SINGLE_WINDOW_MUTATION,
        )

    def test_local_pool_accepts_only_proven_inherited_tail_structure(self) -> None:
        parent = self._parent_snapshot(
            "parent-polish-tail",
            "ts_delta(close / open,5)",
        )

        pool = self._pools(
            eligible_parent_task_ids=(parent.task.task_id,),
            completed_by_task_id={parent.task.task_id: parent},
            stage=LOCAL_POLISHING_STAGE,
        )[0]

        self.assertEqual(len(pool.candidates), 1)
        self.assertEqual(
            pool.candidates[0].candidate.formula,
            "ts_delta(close/open,22)",
        )

    def test_qualified_evolution_pool_contains_window_and_structural_changes(
        self,
    ) -> None:
        parent = self._parent_snapshot(
            "parent-qualified",
            "rank(ts_delta(close,5))",
        )

        pool = self._pools(
            eligible_parent_task_ids=(parent.task.task_id,),
            completed_by_task_id={parent.task.task_id: parent},
            stage=QUALIFIED_EVOLUTION_STAGE,
        )[0]

        families = {candidate.family for candidate in pool.candidates}
        self.assertIn(SINGLE_WINDOW_MUTATION, families)
        self.assertTrue(families.intersection(STRUCTURAL_TRANSFORMATION_FAMILIES))

    def test_high_or_unknown_sc_keeps_ordinary_candidates_before_deduplication(self):
        parent = self._parent_snapshot("parent-sc", "rank(ts_mean(close,5))")
        def pool(correlation):
            return self._pools(eligible_parent_task_ids=(parent.task.task_id,),
                completed_by_task_id={parent.task.task_id: parent}, stage=QUALIFIED_EVOLUTION_STAGE,
                self_correlation_references=(SelfCorrelationReference(parent.task.task_id,
                    "ts_mean(close,22)", correlation=correlation),))[0]
        for correlation in (0.85, 0.99, None):
            result = pool(correlation)
            self.assertFalse(any(x.family.startswith("self_correlation_") for x in result.candidates))
            ordinary = next(x for x in result.candidates if x.candidate.formula == "rank(ts_mean(open,5))")
            self.assertEqual(ordinary.family, "internal_field_replacement")
        middle = pool(0.8)
        same = [x for x in middle.candidates if x.candidate.formula == "rank(ts_mean(open,5))"]
        self.assertEqual(len(same), 1)
        self.assertIn(same[0].family, SELF_CORRELATION_INTERNAL_FAMILIES)

    def test_targeted_internal_edits_keep_sc_identity_when_ordinary_edits_overlap(self):
        parent = self._parent_snapshot("parent-sc", "rank(ts_mean(close,5))")
        references = (SelfCorrelationReference(parent.task.task_id, "ts_mean(close,22)", correlation=0.8),)
        def pool(**kwargs):
            return self._pools(
                eligible_parent_task_ids=(parent.task.task_id,),
                completed_by_task_id={parent.task.task_id: parent},
                stage=QUALIFIED_EVOLUTION_STAGE,
                self_correlation_references=references, **kwargs,
            )[0]
        baseline = pool()
        repairs = [item for item in baseline.candidates
                   if item.family in SELF_CORRELATION_INTERNAL_FAMILIES]
        self.assertEqual({item.family for item in repairs}, set(SELF_CORRELATION_INTERNAL_FAMILIES))
        self.assertEqual(len({item.candidate.fingerprint for item in baseline.candidates}), len(baseline.candidates))
        for repair in repairs:
            self.assertEqual(repair.candidate.parent_task_id, parent.task.task_id)
            self.assertEqual(repair.candidate.parent_formula_fingerprint, parent.task.formula_fingerprint)
            self.assertEqual(baseline.settings, self.settings)
            for status in ("created", "pending", "submission_unknown", "completed", "failed"):
                repeated = pool(reserved_tasks=(self._reserved_task(repair, status=status),))
                self.assertNotIn(repair.candidate.fingerprint, {item.candidate.fingerprint for item in repeated.candidates})
            submitted = pool(excluded_formula_fingerprints=frozenset({repair.candidate.fingerprint}))
            self.assertNotIn(repair.candidate.fingerprint, {item.candidate.fingerprint for item in submitted.candidates})
        for stage in (LOCAL_POLISHING_STAGE, STRUCTURAL_EVOLUTION_STAGE):
            ordinary = self._pools(
                eligible_parent_task_ids=(parent.task.task_id,),
                completed_by_task_id={parent.task.task_id: parent}, stage=stage,
                self_correlation_references=references,
            )[0]
            self.assertFalse(any(item.family in SELF_CORRELATION_INTERNAL_FAMILIES for item in ordinary.candidates))

    def test_light_sc_candidates_keep_parent_settings_and_request_deduplication(self):
        self.catalog = replace(self.catalog,
            fields=self.catalog.fields + (
                FieldDefinition("industry", "groups", "sample", "sample", "GROUP", 1.0),),
            operators=self.catalog.operators + (
                self._operator("group_rank", (), OperatorParameter("x", "expr"), OperatorParameter("group", "group")),
                self._operator("group_neutralize", (), OperatorParameter("x", "expr"), OperatorParameter("group", "group")),
                self._operator("ts_decay_linear", ("time_series_smoothing",), OperatorParameter("x", "expr"), OperatorParameter("d", "window")),))
        parent = self.parent
        def pool(**kwargs):
            return self._pools(eligible_parent_task_ids=(parent.task.task_id,),
                completed_by_task_id={parent.task.task_id: parent}, stage=QUALIFIED_EVOLUTION_STAGE,
                self_correlation_references=(SelfCorrelationReference(parent.task.task_id, "rank(open)", correlation=0.74),),
                **kwargs)[0]
        baseline = pool()
        light = [item for item in baseline.candidates if item.family in SELF_CORRELATION_LIGHT_FAMILIES]
        self.assertEqual({item.family for item in light}, set(SELF_CORRELATION_LIGHT_FAMILIES))
        self.assertEqual(len({item.candidate.fingerprint for item in baseline.candidates}), len(baseline.candidates))
        self.assertEqual(baseline.settings, self.settings)
        for item in light:
            self.assertEqual(item.candidate.parent_task_id, parent.task.task_id)
            for status in ("created", "pending", "submission_unknown", "completed", "failed"):
                repeated = pool(reserved_tasks=(self._reserved_task(item, status=status),))
                self.assertNotIn(item.candidate.fingerprint, {c.candidate.fingerprint for c in repeated.candidates})
            excluded = pool(excluded_formula_fingerprints=frozenset({item.candidate.fingerprint}))
            self.assertNotIn(item.candidate.fingerprint, {c.candidate.fingerprint for c in excluded.candidates})

    def _pool(
        self,
        *,
        reserved_tasks: tuple[BacktestTaskRecord, ...] = (),
        excluded_formula_fingerprints: frozenset[str] = frozenset(),
    ) -> ImprovementCandidatePool:
        return self._pools(
            eligible_parent_task_ids=(self.parent.task.task_id,),
            completed_by_task_id={self.parent.task.task_id: self.parent},
            reserved_tasks=reserved_tasks,
            excluded_formula_fingerprints=excluded_formula_fingerprints,
        )[0]

    def _pools(
        self,
        *,
        eligible_parent_task_ids: tuple[str, ...],
        completed_by_task_id: dict[str, BacktestSnapshot],
        reserved_tasks: tuple[BacktestTaskRecord, ...] = (),
        excluded_formula_fingerprints: frozenset[str] = frozenset(),
        stage: str = STRUCTURAL_EVOLUTION_STAGE,
        self_correlation_references: tuple[SelfCorrelationReference, ...] = (),
    ) -> tuple[ImprovementCandidatePool, ...]:
        return build_improvement_candidate_pools(
            self.catalog,
            self.policy,
            eligible_parent_task_ids=eligible_parent_task_ids,
            improvement_stages=ParentImprovementStageSet(
                records=tuple(
                    ParentImprovementStage(
                        parent_task_id=task_id,
                        stage=stage,
                        proximity=None,
                    )
                    for task_id in eligible_parent_task_ids
                )
            ),
            completed_by_task_id=completed_by_task_id,
            field_candidates=("close", "open"),
            group_candidates=(),
            reserved_tasks=reserved_tasks,
            excluded_formula_fingerprints=excluded_formula_fingerprints,
            self_correlation_references=self_correlation_references,
        )

    def _parent_snapshot(self, task_id: str, formula: str) -> BacktestSnapshot:
        parsed = parse_formula(formula)
        task = self._task_record(
            task_id=task_id,
            formula=parsed.normalized,
            formula_fingerprint=parsed.fingerprint,
            account_scope="group-account",
            settings_json=self.settings_json,
            status="completed",
        )
        return BacktestSnapshot(
            task=task,
            result=BacktestResultRecord(
                task_id=task.task_id,
                sharpe=1.2,
                fitness=0.9,
                turnover=0.1,
                returns=0.02,
                drawdown=0.05,
                margin=0.001,
                book_size=None,
                pnl=None,
                long_count=None,
                short_count=None,
                check_details_captured=True,
                checks=(
                    BacktestCheckRecord(
                        name="CONCENTRATED_WEIGHT",
                        status="PASS",
                        threshold=None,
                        actual=None,
                        platform_date=None,
                    ),
                    BacktestCheckRecord(
                        name="LOW_FITNESS",
                        status="FAIL",
                        threshold=1.0,
                        actual=0.9,
                        platform_date="2026-09-01",
                    ),
                    BacktestCheckRecord(
                        name="LOW_SUB_UNIVERSE_SHARPE",
                        status="PASS",
                        threshold=None,
                        actual=None,
                        platform_date=None,
                    ),
                ),
            ),
            yearly_stats=(),
        )

    def _reserved_task(
        self,
        leaf: ImprovementPoolCandidate,
        *,
        status: str,
        account_scope: str = "group-account",
        settings_json: str | None = None,
    ) -> BacktestTaskRecord:
        return self._task_record(
            task_id=f"reserved-{status}-{leaf.candidate.fingerprint}",
            formula=leaf.candidate.formula,
            formula_fingerprint=leaf.candidate.fingerprint,
            account_scope=account_scope,
            settings_json=settings_json or self.settings_json,
            status=status,
        )

    def _pool_candidate(
        self,
        formula: str,
        family: str,
    ) -> ImprovementPoolCandidate:
        parsed = parse_formula(formula)
        parent = parse_formula(self.parent.task.formula)
        candidate = mutation_candidate(
            parsed.expression,
            parent_task_id=self.parent.task.task_id,
            parent_formula_fingerprint=parent.fingerprint,
            change=CandidateChange(
                action=family,
                location="formula",
                before=parent.normalized,
                after=parsed.normalized,
            ),
        )
        return ImprovementPoolCandidate(
            candidate=candidate,
            leaf_id=f"{family}:{candidate.fingerprint}",
            family=family,
        )

    @staticmethod
    def _task_record(
        *,
        task_id: str,
        formula: str,
        formula_fingerprint: str,
        account_scope: str,
        settings_json: str,
        status: str,
    ) -> BacktestTaskRecord:
        remote_id = None
        platform_alpha_id = None
        submission_started_at = None
        last_observed_at = None
        finished_at = None
        failure_code = None
        failure_message = None
        if status == "submission_unknown":
            submission_started_at = "2026-09-01T00:01:00+00:00"
            last_observed_at = "2026-09-01T00:01:00+00:00"
        elif status in {"pending", "completed"}:
            remote_id = f"remote-{task_id}"
            submission_started_at = "2026-09-01T00:01:00+00:00"
            last_observed_at = "2026-09-01T00:02:00+00:00"
            if status == "completed":
                platform_alpha_id = f"alpha-{task_id}"
                finished_at = "2026-09-01T00:02:00+00:00"
        elif status == "failed":
            last_observed_at = "2026-09-01T00:02:00+00:00"
            finished_at = "2026-09-01T00:02:00+00:00"
            failure_code = "platform_zero_capital"
        failure_message = None
        request_fingerprint = sha256(
            f"{account_scope}|{formula_fingerprint}|{settings_json}".encode()
        ).hexdigest()
        return BacktestTaskRecord(
            task_id=task_id,
            account_scope=account_scope,
            formula=formula,
            formula_fingerprint=formula_fingerprint,
            settings_json=settings_json,
            request_fingerprint=request_fingerprint,
            status=status,
            remote_id=remote_id,
            platform_alpha_id=platform_alpha_id,
            created_at="2026-09-01T00:00:00+00:00",
            submission_started_at=submission_started_at,
            last_observed_at=last_observed_at,
            retry_not_before=None,
            finished_at=finished_at,
            failure_code=failure_code,
            failure_message=failure_message,
        )

    @staticmethod
    def _settings_json(settings: BacktestSettings) -> str:
        return json.dumps(
            settings.as_platform_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _field(field_id: str) -> FieldDefinition:
        return FieldDefinition(
            field_id=field_id,
            dataset_id="sample-dataset",
            category="sample",
            subcategory="sample",
            field_type="MATRIX",
            coverage=1.0,
        )

    @staticmethod
    def _operator(
        name: str,
        roles: tuple[str, ...],
        *parameters: OperatorParameter,
    ) -> OperatorDefinition:
        return OperatorDefinition(
            name=name,
            category="sample",
            scope=("REGULAR",),
            parameters=tuple(parameters),
            roles=roles,
            output_kind="signal",
        )


if __name__ == "__main__":
    unittest.main()
