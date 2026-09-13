from __future__ import annotations

import sys
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generation.polishing import SINGLE_WINDOW_MUTATION
from generation.internal_edits import INTERNAL_EDIT_FAMILIES
from generation.self_correlation import (
    SELF_CORRELATION_FIELD_REPLACEMENT, SELF_CORRELATION_REPAIR_FAMILIES,
    SELF_CORRELATION_LIGHT_FAMILIES, SELF_CORRELATION_INTERNAL_FAMILIES,
    SELF_CORRELATION_HALF_FAMILIES,
)
from generation.transformations import (
    STRUCTURAL_TRANSFORMATION_FAMILIES,
    TEMPORAL_CHANGE_REFRAME,
)
from learning.action_effects import (
    ACTION_STATE_DEPRIORITIZED,
    ACTION_STATE_INSUFFICIENT,
    ACTION_STATE_PREFERRED,
    ACTION_STRATEGY_EXPLOITATION,
    ACTION_STRATEGY_EXPLORATION,
    DefectActionEvidence,
    DefectActionStrategy,
    DefectActionStrategySet,
)
from learning.optimization_targets import (
    OptimizationActionEvidence,
    OptimizationTarget,
    ParentOptimizationTargetSet,
    ParentOptimizationTargets,
)
from learning.parent_settings import ParentSettingsEvidence, ParentSettingsEvidenceSet
from learning.quality_proximity import (
    LOCAL_POLISHING_STAGE,
    QUALIFIED_EVOLUTION_STAGE,
    STRUCTURAL_EVOLUTION_STAGE,
)
from selection.allocations import (
    LOW_SUB_UNIVERSE_SHARPE_STRUCTURAL_FAMILIES,
    QUALIFIED_EVOLUTION_FAMILIES,
    SignalImprovementFamilyCapacity,
    SignalImprovementSource,
    allocate_backtest_sources,
    direction_validation_limit,
)


class BacktestSourceAllocationTests(unittest.TestCase):
    def test_submitted_self_reference_without_sc_uses_only_ordinary_slots(self):
        source = self._source("root", "parent", 3, is_submitted=True, self_correlation=None,
            stage=QUALIFIED_EVOLUTION_STAGE, families=QUALIFIED_EVOLUTION_FAMILIES,
            family_counts={SELF_CORRELATION_HALF_FAMILIES[0]: 2, SINGLE_WINDOW_MUTATION: 1})
        result = self._allocate(requested_count=4, minimum_exploration_count=2,
            sources=(source,), settings=(self._settings("parent", risk=None),),
            targets=(self._branch_targets("parent", ()),))
        self.assertEqual([a.candidate_family for a in result.signal_improvements], [SINGLE_WINDOW_MUTATION])
        self.assertEqual(result.exploration_count, 3)

    def test_ordinary_slots_exhaust_unsubmitted_capacity_before_submitted_fallback(self):
        for shared_root in (False, True):
            for capacity, budget, expected_unsubmitted in ((4, 20, 4), (1, 20, 1), (4, 1, 1), (0, 20, 0)):
                with self.subTest(shared_root=shared_root, capacity=capacity, budget=budget):
                    sources = (
                        self._source("submitted-root", "submitted", 6, is_submitted=True),
                        self._source("submitted-root" if shared_root else "other-root",
                                     "unsubmitted", capacity, remaining_attempts=budget),
                    )
                    result = self._allocate(requested_count=5, minimum_exploration_count=1,
                        sources=sources,
                        settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
                        targets=tuple(self._branch_targets(s.branch_task_id, (self._target("LOW_FITNESS"),)) for s in sources))
                    self.assertEqual([a.parent_task_id for a in result.signal_improvements],
                        ["unsubmitted"] * expected_unsubmitted + ["submitted"] * (4 - expected_unsubmitted))
                    self.assertEqual(result.exploration_count, 1)

    def test_unsubmitted_priority_respects_qualified_lineage_cap(self):
        sources = (
            self._source("submitted-root", "submitted", 5, is_submitted=True),
            self._source("other-root", "unsubmitted", 5, stage=QUALIFIED_EVOLUTION_STAGE,
                         families=(SINGLE_WINDOW_MUTATION,)),
        )
        result = self._allocate(requested_count=5, minimum_exploration_count=1,
            sources=sources, settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
            targets=(self._branch_targets("submitted", (self._target("LOW_FITNESS"),)),
                     self._branch_targets("unsubmitted", ())))
        self.assertEqual([a.parent_task_id for a in result.signal_improvements],
                         ["unsubmitted", "unsubmitted", "submitted", "submitted"])

    def test_unusable_target_families_do_not_block_submitted_fallback(self):
        sources = (
            self._source("submitted-root", "submitted", 2, is_submitted=True),
            self._source("other-root", "unsubmitted", 3,
                         family_counts={TEMPORAL_CHANGE_REFRAME: 3}),
        )
        result = self._allocate(requested_count=3, minimum_exploration_count=1,
            sources=sources, settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
            targets=(self._branch_targets("submitted", (self._target("LOW_FITNESS"),)),
                     self._branch_targets("unsubmitted", (self._target("LOW_SUB_UNIVERSE_SHARPE"),))))
        self.assertEqual([a.parent_task_id for a in result.signal_improvements], ["submitted"] * 2)

    def test_unsubmitted_priority_leaves_sc_and_direction_reserves_intact(self):
        sources = (
            self._source("sc-root", "submitted", 8, is_submitted=True,
                         stage=QUALIFIED_EVOLUTION_STAGE, families=QUALIFIED_EVOLUTION_FAMILIES,
                         family_counts={SELF_CORRELATION_FIELD_REPLACEMENT: 4, SINGLE_WINDOW_MUTATION: 4}),
            self._source("ordinary-root", "unsubmitted", 8),
        )
        result = self._allocate(requested_count=10, minimum_exploration_count=3,
            direction_validation_count=1, direction_validation_percent=10,
            self_correlation_percent=20, sources=sources,
            settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
            targets=(self._branch_targets("submitted", ()),
                     self._branch_targets("unsubmitted", (self._target("LOW_FITNESS"),))))
        self.assertEqual([a.candidate_family for a in result.signal_improvements[:2]],
                         [SELF_CORRELATION_FIELD_REPLACEMENT] * 2)
        self.assertEqual([a.parent_task_id for a in result.signal_improvements[2:]], ["unsubmitted"] * 4)
        self.assertEqual((result.exploration_count, result.direction_validation_count), (3, 1))

    def test_sc_bands_only_allocate_allowed_actions_and_high_band_returns_to_ordinary(self):
        for correlation, expected in (
            (0.74, SELF_CORRELATION_LIGHT_FAMILIES),
            (0.75, SELF_CORRELATION_INTERNAL_FAMILIES),
            (0.849999, SELF_CORRELATION_INTERNAL_FAMILIES),
            (0.85, ()), (1.0, ()), (None, ()),
        ):
            with self.subTest(correlation=correlation):
                source = self._source("root", "parent", len(SELF_CORRELATION_REPAIR_FAMILIES)+2,
                    self_correlation=correlation, stage=QUALIFIED_EVOLUTION_STAGE,
                    families=QUALIFIED_EVOLUTION_FAMILIES,
                    family_counts={**{f: 1 for f in SELF_CORRELATION_REPAIR_FAMILIES}, SINGLE_WINDOW_MUTATION: 2},
                    attempted_by_family={f: 5 if f in expected else 0 for f in SELF_CORRELATION_REPAIR_FAMILIES})
                result = self._allocate(requested_count=4, minimum_exploration_count=2,
                    sources=(source,), settings=(self._settings("parent", risk=0.9),),
                    targets=(self._branch_targets("parent", ()),))
                actions = [a.candidate_family for a in result.signal_improvements]
                self.assertEqual(len(actions), 2)
                if expected:
                    self.assertEqual(len(set(actions)), 2)
                    self.assertTrue(set(actions) <= set(expected))
                else:
                    self.assertEqual(actions, [SINGLE_WINDOW_MUTATION]*2)
                self.assertEqual(result.exploration_count, 2)

    def test_band_exhaustion_does_not_borrow_other_sc_actions_or_reset_parent_budget(self):
        for count in (0, 1):
            source = self._source("root", "parent", 3+count,
                self_correlation=0.74, stage=QUALIFIED_EVOLUTION_STAGE,
                families=QUALIFIED_EVOLUTION_FAMILIES,
                family_counts={SELF_CORRELATION_LIGHT_FAMILIES[0]: count,
                               SELF_CORRELATION_FIELD_REPLACEMENT: 2, SINGLE_WINDOW_MUTATION: 1})
            result = self._allocate(requested_count=4, minimum_exploration_count=2,
                sources=(source,), settings=(self._settings("parent", risk=None),),
                targets=(self._branch_targets("parent", ()),))
            self.assertEqual([x.candidate_family for x in result.signal_improvements],
                             ([SELF_CORRELATION_LIGHT_FAMILIES[0]] if count else []) + [SINGLE_WINDOW_MUTATION])
            last = self._allocate(requested_count=4, minimum_exploration_count=2,
                sources=(replace(source, parent_remaining_attempts=1),),
                settings=(self._settings("parent", risk=None),),
                targets=(self._branch_targets("parent", ()),))
            self.assertEqual(len(last.signal_improvements), 1)

    def test_invalid_sc_is_not_silently_classified_as_a_band(self):
        source = self._source("root", "parent", 1, stage=QUALIFIED_EVOLUTION_STAGE,
                              families=(SELF_CORRELATION_FIELD_REPLACEMENT,))
        for value in (True, float("nan"), float("inf"), -0.1, 1.1, "0.8"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._allocate(requested_count=2, minimum_exploration_count=1,
                    sources=(replace(source, self_correlation=value),),
                    settings=(self._settings("parent", risk=None),),
                    targets=(self._branch_targets("parent", ()),))

    def test_custom_percentages_change_actual_allocation(self):
        sources = tuple(self._source(f"sc-{i}", f"sc-{i}", 1,
            stage=QUALIFIED_EVOLUTION_STAGE, families=(SELF_CORRELATION_FIELD_REPLACEMENT,),
            remaining_attempts=1) for i in range(60)) + tuple(
            self._source(f"ordinary-{i}", f"ordinary-{i}", 20) for i in range(4))
        result = self._allocate(requested_count=100, minimum_exploration_count=20,
            self_correlation_percent=50, direction_validation_percent=5,
            direction_validation_count=5, sources=sources,
            settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
            targets=tuple(self._branch_targets(s.branch_task_id, () if s.stage == QUALIFIED_EVOLUTION_STAGE
                else (self._target("LOW_FITNESS"),)) for s in sources))
        sc = sum(x.candidate_family in SELF_CORRELATION_REPAIR_FAMILIES for x in result.signal_improvements)
        self.assertEqual((result.exploration_count, sc, result.improvement_count-sc,
                          result.direction_validation_count), (20, 50, 25, 5))

    def test_hundred_slots_keep_sc_and_direction_caps_and_fill_shortfalls(self):
        for sc_count, ordinary_roots, directions, expected in (
            (50, 4, 3, (30, 30, 37, 3)),
            (50, 4, 0, (30, 30, 40, 0)),
            (50, 4, 1, (30, 30, 39, 1)),
            (10, 4, 3, (30, 10, 57, 3)),
            (50, 0, 3, (67, 30, 0, 3)),
            (0, 0, 0, (100, 0, 0, 0)),
        ):
            with self.subTest(sc=sc_count, ordinary=ordinary_roots, directions=directions):
                sources = tuple(self._source(f"sc-{i}", f"sc-{i}", 1,
                    stage=QUALIFIED_EVOLUTION_STAGE, families=(SELF_CORRELATION_FIELD_REPLACEMENT,),
                    remaining_attempts=1) for i in range(sc_count)) + tuple(
                    self._source(f"ordinary-{i}", f"ordinary-{i}", 20)
                    for i in range(ordinary_roots))
                result = self._allocate(requested_count=100, minimum_exploration_count=30,
                    sources=sources, direction_validation_count=directions,
                    settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
                    targets=tuple(self._branch_targets(s.branch_task_id, () if s.stage == QUALIFIED_EVOLUTION_STAGE
                        else (self._target("LOW_FITNESS"),)) for s in sources))
                sc = sum(x.candidate_family in SELF_CORRELATION_REPAIR_FAMILIES for x in result.signal_improvements)
                actual=(result.exploration_count, sc, len(result.signal_improvements)-sc, result.direction_validation_count)
                self.assertEqual(actual, expected)
                self.assertEqual(sum(actual), 100)
                counts=Counter(x.parent_task_id for x in result.signal_improvements)
                self.assertTrue(all(counts[s.branch_task_id] <= s.parent_remaining_attempts for s in sources))

    def test_active_sc_actions_share_quota_and_last_parent_attempt(self):
        families = (*SELF_CORRELATION_LIGHT_FAMILIES, *SELF_CORRELATION_INTERNAL_FAMILIES)
        sources = tuple(self._source(f"root-{i}", f"parent-{i}", 10,
            stage=QUALIFIED_EVOLUTION_STAGE, families=(family,), remaining_attempts=1,
            self_correlation=0.74 if family in SELF_CORRELATION_LIGHT_FAMILIES else 0.8)
            for i, family in enumerate(families)) + (self._source("ordinary", "ordinary", 20),)
        result = self._allocate(requested_count=20, minimum_exploration_count=10,
            direction_validation_count=1, sources=sources, self_correlation_percent=40,
            settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
            targets=tuple(self._branch_targets(s.branch_task_id,
                () if s.stage == QUALIFIED_EVOLUTION_STAGE else (self._target("LOW_FITNESS"),)) for s in sources))
        actions = result.signal_improvements
        self.assertEqual({a.candidate_family for a in actions[:len(families)]}, set(families))
        self.assertEqual(Counter(a.parent_task_id for a in actions if a.parent_task_id != "ordinary"),
                         {f"parent-{i}": 1 for i in range(len(families))})
        self.assertEqual((result.exploration_count, result.direction_validation_count, len(actions)), (10, 1, 9))

    def test_sc_actions_explore_less_tried_allowed_family_first(self):
        source = self._source("root", "parent", 20, stage=QUALIFIED_EVOLUTION_STAGE,
            families=SELF_CORRELATION_INTERNAL_FAMILIES,
            family_counts={f: 10 for f in SELF_CORRELATION_INTERNAL_FAMILIES},
            attempted_by_family={SELF_CORRELATION_FIELD_REPLACEMENT: 5})
        result = self._allocate(requested_count=4, minimum_exploration_count=2,
            sources=(source,), settings=(self._settings("parent", risk=None),),
            targets=(self._branch_targets("parent", ()),))
        self.assertEqual(result.signal_improvements[0].candidate_family, SELF_CORRELATION_INTERNAL_FAMILIES[1])

    def test_sc_repairs_fill_improvement_capacity_before_ordinary_mutations(self):
        for sc_count in (0, 6, 12):
            with self.subTest(sc_count=sc_count):
                sources = tuple(
                    self._source(f"sc-{i}", f"sc-{i}", 1,
                                 stage=QUALIFIED_EVOLUTION_STAGE,
                                 families=(SELF_CORRELATION_FIELD_REPLACEMENT,), remaining_attempts=1)
                    for i in range(sc_count)
                ) + (self._source("ordinary", "ordinary", 20),)
                result = self._allocate(
                    requested_count=20, minimum_exploration_count=10,
                    direction_validation_count=1, sources=sources,
                    settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
                    targets=tuple(self._branch_targets(s.branch_task_id,
                                  () if s.stage == QUALIFIED_EVOLUTION_STAGE else (self._target("LOW_FITNESS"),))
                                  for s in sources),
                )
                actions = [i.candidate_family for i in result.signal_improvements]
                expected_sc = min(sc_count, 6)
                self.assertEqual(actions[:expected_sc], [SELF_CORRELATION_FIELD_REPLACEMENT] * expected_sc)
                self.assertEqual(actions.count(SELF_CORRELATION_FIELD_REPLACEMENT), expected_sc)
                self.assertEqual(len(actions), 9)
                self.assertEqual(result.exploration_count, 10)
                self.assertEqual(result.direction_validation_count, 1)

    def test_sc_priority_rotates_roots_before_reusing_wide_lineage(self):
        sources = tuple(
            self._source(root, parent, 1, stage=QUALIFIED_EVOLUTION_STAGE,
                         families=(SELF_CORRELATION_FIELD_REPLACEMENT,))
            for root, parent in (("wide", "a"), ("wide", "b"), ("wide", "c"), ("small", "d"))
        )
        result = self._allocate(
            requested_count=6, minimum_exploration_count=4, sources=sources,
            settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
            targets=tuple(self._branch_targets(s.branch_task_id, ()) for s in sources),
        )
        self.assertEqual({i.root_task_id for i in result.signal_improvements}, {"wide", "small"})
        self.assertTrue(all(i.candidate_family == SELF_CORRELATION_FIELD_REPLACEMENT for i in result.signal_improvements))

    def test_sc_priority_returns_to_ordinary_work_when_lineage_or_parent_budget_is_spent(self):
        for branch_count in (1, 3):
            with self.subTest(branch_count=branch_count):
                sources = tuple(
                    self._source("sc-root", f"sc-{i}", 3, stage=QUALIFIED_EVOLUTION_STAGE,
                                 families=(SELF_CORRELATION_FIELD_REPLACEMENT,), remaining_attempts=1)
                    for i in range(branch_count)
                ) + (self._source("ordinary", "ordinary", 20),)
                result = self._allocate(
                    requested_count=20, minimum_exploration_count=10, sources=sources,
                    settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
                    targets=tuple(self._branch_targets(s.branch_task_id,
                                  () if s.stage == QUALIFIED_EVOLUTION_STAGE else (self._target("LOW_FITNESS"),))
                                  for s in sources),
                )
                self.assertEqual(result.qualified_evolution_count, min(branch_count, 2))
                self.assertEqual(result.improvement_count, 10)
                self.assertTrue(all(count == 1 for parent, count in
                    Counter(i.parent_task_id for i in result.signal_improvements).items() if parent != "ordinary"))

    def test_all_stages_rank_roots_by_used_parent_budget_not_current_leaves(self):
        for stage, families in (
            (STRUCTURAL_EVOLUTION_STAGE, STRUCTURAL_TRANSFORMATION_FAMILIES),
            (LOCAL_POLISHING_STAGE, (SINGLE_WINDOW_MUTATION,)),
        ):
            with self.subTest(stage=stage):
                sources = (
                    self._source("busy", "busy", 10, stage=stage,
                                 families=families, remaining_attempts=4, attempted=0),
                    self._source("less-used", "less-used", 10,
                                 stage=QUALIFIED_EVOLUTION_STAGE,
                                 families=(SINGLE_WINDOW_MUTATION,), remaining_attempts=11),
                )
                result = self._allocate(
                    requested_count=2, minimum_exploration_count=1, sources=sources,
                    settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
                    targets=(self._branch_targets("busy", (self._target("LOW_FITNESS"),)),
                             self._branch_targets("less-used", ())),
                )
                self.assertEqual(result.signal_improvements[0].parent_task_id, "less-used")

    def test_unusable_fresh_sibling_does_not_promote_busy_root(self):
        for blocked in (0, 1):
            with self.subTest(blocked=blocked):
                sources = (
                    self._source("busy-root", "empty-fresh", 0, blocked=blocked),
                    self._source("busy-root", "busy", 10, remaining_attempts=4),
                    self._source("other-root", "other", 10, remaining_attempts=16),
                )
                result = self._allocate(
                    requested_count=2, minimum_exploration_count=1, sources=sources,
                    settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
                    targets=tuple(self._branch_targets(s.branch_task_id, (self._target("LOW_FITNESS"),))
                                  for s in sources),
                )
                self.assertEqual(result.signal_improvements[0].parent_task_id, "other")

    def test_available_sc_repair_gets_coverage_among_many_fresh_roots(self):
        sources = tuple(self._source(f"root-{i}", f"parent-{i}", 20) for i in range(30)) + (
            self._source("sc-root", "sc-parent", 11, stage=QUALIFIED_EVOLUTION_STAGE,
                         families=QUALIFIED_EVOLUTION_FAMILIES, remaining_attempts=1,
                         family_counts={SELF_CORRELATION_FIELD_REPLACEMENT: 1, SINGLE_WINDOW_MUTATION: 10}),
        )
        result = self._allocate(
            requested_count=20, minimum_exploration_count=10, sources=sources,
            direction_validation_count=1,
            settings=tuple(self._settings(s.branch_task_id, risk=0.9) for s in sources),
            targets=tuple(self._branch_targets(s.branch_task_id,
                          () if s.stage == QUALIFIED_EVOLUTION_STAGE else (self._target("LOW_FITNESS"),))
                          for s in sources),
        )
        self.assertEqual(result.signal_improvements[0].candidate_family, SELF_CORRELATION_FIELD_REPLACEMENT)
        self.assertEqual(sum(i.parent_task_id == "sc-parent" for i in result.signal_improvements), 1)
        self.assertEqual(result.improvement_count, 9)
        self.assertEqual(result.exploration_count, 10)
        self.assertEqual(result.direction_validation_count, 1)

    def test_sc_coverage_moves_to_unreserved_candidates_without_extra_slots(self):
        sources = tuple(
            self._source(f"sc-root-{i}", f"sc-parent-{i}", 11,
                         stage=QUALIFIED_EVOLUTION_STAGE, families=QUALIFIED_EVOLUTION_FAMILIES,
                         remaining_attempts=10,
                         family_counts={SELF_CORRELATION_FIELD_REPLACEMENT: 1, SINGLE_WINDOW_MUTATION: 10})
            for i in range(6)
        ) + (self._source("ordinary", "ordinary", 20),)
        seen = set()
        for cycle in range(7):
            result = self._allocate(
                requested_count=2, minimum_exploration_count=1, sources=sources,
                settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
                targets=tuple(self._branch_targets(s.branch_task_id,
                              () if s.stage == QUALIFIED_EVOLUTION_STAGE else (self._target("LOW_FITNESS"),))
                              for s in sources),
            )
            item, = result.signal_improvements
            self.assertEqual(result.exploration_count, 1)
            if cycle == 6:
                self.assertNotEqual(item.candidate_family, SELF_CORRELATION_FIELD_REPLACEMENT)
                continue
            self.assertEqual(item.candidate_family, SELF_CORRELATION_FIELD_REPLACEMENT)
            self.assertNotIn(item.parent_task_id, seen)
            seen.add(item.parent_task_id)
            sources = tuple(replace(s, family_capacities=tuple(
                replace(f, available_leaf_count=0, blocked_leaf_count=1)
                if s.branch_task_id == item.parent_task_id and f.family == SELF_CORRELATION_FIELD_REPLACEMENT
                else f for f in s.family_capacities)) for s in sources)
        self.assertEqual(len(seen), 6)

    def test_sc_coverage_respects_shared_qualified_lineage_limit(self):
        sources = (
            self._source("root", "sc", 3, stage=QUALIFIED_EVOLUTION_STAGE,
                         families=(SELF_CORRELATION_FIELD_REPLACEMENT, SINGLE_WINDOW_MUTATION),
                         family_counts={SELF_CORRELATION_FIELD_REPLACEMENT: 1, SINGLE_WINDOW_MUTATION: 2}),
            self._source("root", "sibling", 5, stage=QUALIFIED_EVOLUTION_STAGE,
                         families=(SINGLE_WINDOW_MUTATION,)),
        )
        result = self._allocate(
            requested_count=8, minimum_exploration_count=2, sources=sources,
            settings=tuple(self._settings(s.branch_task_id, risk=None) for s in sources),
            targets=tuple(self._branch_targets(s.branch_task_id, ()) for s in sources),
        )
        self.assertEqual(result.signal_improvements[0].candidate_family, SELF_CORRELATION_FIELD_REPLACEMENT)
        self.assertEqual(result.qualified_evolution_count, 2)
        self.assertEqual(result.exploration_count, 6)

    def test_sc_coverage_cannot_take_exploration_or_direction_capacity(self):
        for requested, directions in ((10, 0), (11, 1)):
            with self.subTest(requested=requested):
                result = self._allocate(
                    requested_count=requested, minimum_exploration_count=10,
                    direction_validation_count=directions,
                    sources=(self._source("sc", "sc", 1, stage=QUALIFIED_EVOLUTION_STAGE,
                                          families=(SELF_CORRELATION_FIELD_REPLACEMENT,)),),
                    settings=(self._settings("sc", risk=None),),
                    targets=(self._branch_targets("sc", ()),),
                )
                self.assertEqual(result.improvement_count, 0)
                self.assertEqual(result.exploration_count, 10)
                self.assertEqual(result.direction_validation_count, directions)

    def test_unqualified_parent_last_attempt_is_shared_across_targets_and_families(
        self,
    ):
        for stage, families in (
            (
                STRUCTURAL_EVOLUTION_STAGE,
                (*STRUCTURAL_TRANSFORMATION_FAMILIES, *INTERNAL_EDIT_FAMILIES),
            ),
            (LOCAL_POLISHING_STAGE, (SINGLE_WINDOW_MUTATION, *INTERNAL_EDIT_FAMILIES)),
        ):
            with self.subTest(stage=stage):
                allocation = self._allocate(
                    requested_count=8,
                    minimum_exploration_count=2,
                    sources=(
                        self._source(
                            "root",
                            "parent",
                            len(families) * 2,
                            stage=stage,
                            families=families,
                            family_counts={family: 2 for family in families},
                            remaining_attempts=1,
                        ),
                    ),
                    settings=(self._settings("parent", risk=None),),
                    targets=(
                        self._branch_targets(
                            "parent",
                            (self._target("LOW_SHARPE"), self._target("LOW_FITNESS")),
                        ),
                    ),
                )
                self.assertEqual(allocation.improvement_count, 1)
                self.assertEqual(allocation.exploration_count, 7)

    def test_unqualified_sibling_branches_cannot_spend_each_others_budget(self):
        allocation = self._allocate(
            requested_count=8, minimum_exploration_count=2,
            sources=tuple(self._source("root", name, 8, remaining_attempts=remaining)
                          for name, remaining in (("near-limit", 1), ("other", 2))),
            settings=tuple(self._settings(name, risk=None) for name in ("near-limit", "other")),
            targets=tuple(self._branch_targets(name, (self._target("LOW_FITNESS"),))
                          for name in ("near-limit", "other")),
        )
        self.assertEqual(Counter(item.parent_task_id for item in allocation.signal_improvements),
                         {"near-limit": 1, "other": 2})
        self.assertEqual(allocation.exploration_count, 5)

    def test_sc_repair_shares_last_parent_attempt_and_exploration_reserve(self):
        allocation = self._allocate(
            requested_count=8, minimum_exploration_count=4,
            sources=(self._source("root", "parent", 1, stage=QUALIFIED_EVOLUTION_STAGE,
                families=(SELF_CORRELATION_FIELD_REPLACEMENT,), remaining_attempts=1),),
            settings=(self._settings("parent", risk=None),),
            targets=(self._branch_targets("parent", ()),),
        )
        self.assertEqual(len(allocation.signal_improvements), 1)
        self.assertEqual(allocation.signal_improvements[0].candidate_family, SELF_CORRELATION_FIELD_REPLACEMENT)
        self.assertEqual(allocation.qualified_evolution_count, 1)
        self.assertEqual(allocation.exploration_count, 7)

    def test_direction_uses_one_improvement_slot_without_reducing_exploration(self):
        allocation = self._allocate(
            requested_count=20,
            minimum_exploration_count=10,
            sources=(self._source("root-a", "branch-a", 20),),
            settings=(self._settings("branch-a", risk=None),),
            targets=(self._branch_targets("branch-a", (self._target("LOW_FITNESS"),)),),
            direction_validation_count=1,
        )
        self.assertEqual(allocation.direction_validation_count, 1)
        self.assertEqual(allocation.improvement_count, 9)
        self.assertEqual(allocation.exploration_count, 10)
        self.assertEqual(allocation.requested_count, 20)

    def test_direction_without_positive_seeds_returns_unused_slots_to_exploration(self):
        allocation = self._allocate(
            requested_count=20, minimum_exploration_count=10,
            sources=(), settings=(), targets=(), direction_validation_count=1,
        )
        self.assertEqual(allocation.exploration_count, 19)
        self.assertEqual(allocation.improvement_count, 0)
        self.assertEqual(allocation.direction_validation_count, 1)

    def test_direction_limit_cannot_use_reserved_exploration_or_expand_batch(self):
        self.assertEqual(direction_validation_limit(1, 1, 3), 0)
        self.assertEqual(direction_validation_limit(20, 10, 3), 1)
        self.assertEqual(direction_validation_limit(100, 30, 3), 3)
        self.assertEqual(direction_validation_limit(100, 30, 8), 8)
        for total, reserve, directions in ((1, 1, 1), (20, 10, 4), (2, 1, -1)):
            with self.subTest(total=total, reserve=reserve, directions=directions):
                with self.assertRaisesRegex(ValueError, "direction_count_invalid"):
                    self._allocate(
                        requested_count=total, minimum_exploration_count=reserve,
                        sources=(), settings=(), targets=(),
                        direction_validation_count=directions,
                    )

    def test_missing_seeds_returns_every_unused_slot_to_exploration(self) -> None:
        allocation = self._allocate(
            requested_count=20,
            minimum_exploration_count=15,
            sources=(),
            settings=(),
            targets=(),
        )

        self.assertEqual(allocation.exploration_count, 20)
        self.assertEqual(allocation.improvement_count, 0)
        self.assertEqual(allocation.signal_improvements, ())
        self.assertEqual(allocation.reason, "signal_seed_missing")

    def test_branch_without_explicit_failure_target_returns_to_exploration(
        self,
    ) -> None:
        allocation = self._allocate(
            requested_count=4,
            minimum_exploration_count=2,
            sources=(self._source("root-a", "branch-a", 2),),
            settings=(self._settings("branch-a", risk=None),),
            targets=(self._branch_targets("branch-a", ()),),
        )

        self.assertEqual(allocation.exploration_count, 4)
        self.assertEqual(allocation.improvement_count, 0)
        self.assertEqual(
            allocation.reason,
            "signal_optimization_target_missing",
        )

    def test_allocation_balances_roots_before_reusing_their_branches(self) -> None:
        sources = (
            self._source("root-wide", "wide-a", 4),
            self._source("root-wide", "wide-b", 4),
            self._source("root-wide", "wide-c", 4),
            self._source("root-small", "small-a", 4),
        )
        allocation = self._allocate(
            requested_count=8,
            minimum_exploration_count=4,
            sources=sources,
            settings=tuple(
                self._settings(source.branch_task_id, risk=None) for source in sources
            ),
            targets=tuple(
                self._branch_targets(
                    source.branch_task_id,
                    (self._target("LOW_FITNESS"),),
                )
                for source in sources
            ),
        )

        self.assertEqual(allocation.exploration_count, 4)
        self.assertEqual(allocation.improvement_count, 4)
        self.assertEqual(
            Counter(item.root_task_id for item in allocation.signal_improvements),
            {"root-wide": 2, "root-small": 2},
        )
        self.assertTrue(
            all(
                item.candidate_family in STRUCTURAL_TRANSFORMATION_FAMILIES
                for item in allocation.signal_improvements
            )
        )

    def test_transformations_with_less_evidence_are_attempted_first(self) -> None:
        frequent = OptimizationActionEvidence(
            action=STRUCTURAL_TRANSFORMATION_FAMILIES[0],
            comparable_count=101,
            repaired_count=1,
            safe_repaired_count=1,
            remaining_failed_count=100,
            unresolved_count=0,
        )
        observed = OptimizationActionEvidence(
            action=STRUCTURAL_TRANSFORMATION_FAMILIES[1],
            comparable_count=1,
            repaired_count=0,
            safe_repaired_count=0,
            remaining_failed_count=1,
            unresolved_count=0,
        )
        allocation = self._allocate(
            requested_count=2,
            minimum_exploration_count=1,
            sources=(
                self._source(
                    "root-a",
                    "branch-a",
                    len(STRUCTURAL_TRANSFORMATION_FAMILIES),
                    family_counts={
                        family: 1 for family in STRUCTURAL_TRANSFORMATION_FAMILIES
                    },
                ),
            ),
            settings=(self._settings("branch-a", risk=None),),
            targets=(
                self._branch_targets(
                    "branch-a",
                    (
                        self._target(
                            "LOW_SHARPE",
                            actions=(frequent, observed),
                        ),
                    ),
                ),
            ),
        )

        self.assertIn(
            allocation.signal_improvements[0].candidate_family,
            STRUCTURAL_TRANSFORMATION_FAMILIES[2:],
        )

    def test_preferred_action_gets_only_one_exploitation_slot_per_context(
        self,
    ) -> None:
        preferred, neutral = STRUCTURAL_TRANSFORMATION_FAMILIES[:2]
        frequent = OptimizationActionEvidence(
            action=preferred,
            comparable_count=100,
            repaired_count=80,
            safe_repaired_count=80,
            remaining_failed_count=20,
            unresolved_count=0,
        )
        allocation = self._allocate(
            requested_count=4,
            minimum_exploration_count=1,
            sources=(
                self._source(
                    "root-a",
                    "branch-a",
                    3,
                    families=(preferred, neutral),
                    family_counts={preferred: 2, neutral: 1},
                ),
            ),
            settings=(self._settings("branch-a", risk=None),),
            targets=(
                self._branch_targets(
                    "branch-a",
                    (self._target("LOW_FITNESS", actions=(frequent,)),),
                ),
            ),
            action_strategies=(
                self._action_strategy(
                    "branch-a",
                    "LOW_FITNESS",
                    preferred=(preferred,),
                    action_states={
                        preferred: ACTION_STATE_PREFERRED,
                        neutral: ACTION_STATE_INSUFFICIENT,
                    },
                ),
            ),
        )

        self.assertEqual(
            tuple(item.candidate_family for item in allocation.signal_improvements),
            (preferred, neutral, preferred),
        )

    def test_deprioritized_action_remains_available_for_capacity_recovery(
        self,
    ) -> None:
        deprioritized, neutral = STRUCTURAL_TRANSFORMATION_FAMILIES[:2]
        frequent_neutral = OptimizationActionEvidence(
            action=neutral,
            comparable_count=100,
            repaired_count=0,
            safe_repaired_count=0,
            remaining_failed_count=100,
            unresolved_count=0,
        )
        allocation = self._allocate(
            requested_count=3,
            minimum_exploration_count=1,
            sources=(
                self._source(
                    "root-a",
                    "branch-a",
                    2,
                    families=(deprioritized, neutral),
                    family_counts={deprioritized: 1, neutral: 1},
                ),
            ),
            settings=(self._settings("branch-a", risk=None),),
            targets=(
                self._branch_targets(
                    "branch-a",
                    (
                        self._target(
                            "LOW_FITNESS",
                            actions=(frequent_neutral,),
                        ),
                    ),
                ),
            ),
            action_strategies=(
                self._action_strategy(
                    "branch-a",
                    "LOW_FITNESS",
                    deprioritized=(deprioritized,),
                    action_states={
                        deprioritized: ACTION_STATE_DEPRIORITIZED,
                        neutral: ACTION_STATE_INSUFFICIENT,
                    },
                ),
            ),
        )

        self.assertEqual(
            tuple(item.candidate_family for item in allocation.signal_improvements),
            (neutral, deprioritized),
        )

    def test_sc_risk_orders_branches_inside_one_root_without_removing_them(
        self,
    ) -> None:
        arguments = dict(
            sources=(
                self._source("root", "branch-high", 2),
                self._source("root", "branch-low", 2),
            ),
            settings=(
                self._settings("branch-high", risk=0.7),
                self._settings("branch-low", risk=0.3),
            ),
            targets=(
                self._branch_targets(
                    "branch-high",
                    (self._target("LOW_FITNESS"),),
                ),
                self._branch_targets(
                    "branch-low",
                    (self._target("LOW_FITNESS"),),
                ),
            ),
        )
        limited = self._allocate(
            requested_count=2,
            minimum_exploration_count=1,
            **arguments,
        )
        expanded = self._allocate(
            requested_count=3,
            minimum_exploration_count=1,
            **arguments,
        )

        self.assertEqual(
            limited.signal_improvements[0].parent_task_id,
            "branch-low",
        )
        self.assertEqual(
            {item.parent_task_id for item in expanded.signal_improvements},
            {"branch-high", "branch-low"},
        )

    def test_branch_settings_and_target_settings_must_match(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "source_allocation_parent_evidence_mismatch",
        ):
            self._allocate(
                requested_count=2,
                minimum_exploration_count=1,
                sources=(self._source("root-a", "branch-a", 1),),
                settings=(self._settings("branch-a", risk=None),),
                targets=(
                    self._branch_targets(
                        "branch-a",
                        (self._target("LOW_FITNESS"),),
                        settings_key="settings-other",
                    ),
                ),
            )

    def test_source_identity_must_be_non_empty_text(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "source_allocation_improvement_sources_invalid",
        ):
            self._allocate(
                requested_count=2,
                minimum_exploration_count=1,
                sources=(self._source(None, "branch-a", 1),),
                settings=(),
                targets=(),
            )

    def test_attempted_family_capacity_cannot_be_negative(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "source_allocation_improvement_sources_invalid",
        ):
            self._allocate(
                requested_count=2,
                minimum_exploration_count=1,
                sources=(self._source("root-a", "branch-a", 1, attempted=-1),),
                settings=(),
                targets=(),
            )

    def test_qualified_remaining_attempts_cannot_exceed_budget(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "source_allocation_improvement_sources_invalid",
        ):
            self._allocate(
                requested_count=2,
                minimum_exploration_count=1,
                sources=(
                    self._source(
                        "root-a",
                        "branch-a",
                        1,
                        stage=QUALIFIED_EVOLUTION_STAGE,
                        remaining_attempts=21,
                    ),
                ),
                settings=(),
                targets=(),
            )

    def test_branch_capacity_returns_unused_slots_to_exploration(self) -> None:
        allocation = self._allocate(
            requested_count=6,
            minimum_exploration_count=2,
            sources=(self._source("root-a", "branch-a", 2),),
            settings=(self._settings("branch-a", risk=None),),
            targets=(
                self._branch_targets(
                    "branch-a",
                    (self._target("LOW_FITNESS"),),
                ),
            ),
        )

        self.assertEqual(allocation.improvement_count, 2)
        self.assertEqual(allocation.exploration_count, 4)
        self.assertEqual(len(allocation.signal_improvements), 2)
        self.assertEqual(
            allocation.reason,
            "signal_improvement_capacity_limited",
        )

    def test_one_leaf_is_not_allocated_twice_for_two_targets(self) -> None:
        allocation = self._allocate(
            requested_count=4,
            minimum_exploration_count=2,
            sources=(self._source("root-a", "branch-a", 1),),
            settings=(self._settings("branch-a", risk=None),),
            targets=(
                self._branch_targets(
                    "branch-a",
                    (
                        self._target("LOW_SHARPE"),
                        self._target("LOW_FITNESS"),
                    ),
                ),
            ),
        )

        self.assertEqual(allocation.improvement_count, 1)
        self.assertEqual(allocation.exploration_count, 3)
        self.assertEqual(len(allocation.signal_improvements), 1)
        self.assertEqual(
            allocation.signal_improvements[0].parent_task_id,
            "branch-a",
        )

    def test_branch_is_covered_before_reusing_its_second_target(self) -> None:
        allocation = self._allocate(
            requested_count=4,
            minimum_exploration_count=2,
            sources=(
                self._source("root-a", "branch-dual", 2),
                self._source("root-a", "branch-single", 2),
            ),
            settings=(
                self._settings("branch-dual", risk=None),
                self._settings("branch-single", risk=None),
            ),
            targets=(
                self._branch_targets(
                    "branch-dual",
                    (
                        self._target("LOW_SHARPE"),
                        self._target("LOW_FITNESS"),
                    ),
                ),
                self._branch_targets(
                    "branch-single",
                    (self._target("LOW_FITNESS"),),
                ),
            ),
        )

        self.assertEqual(
            {item.parent_task_id for item in allocation.signal_improvements},
            {"branch-dual", "branch-single"},
        )

    def test_exhausted_branch_does_not_block_an_available_sibling(self) -> None:
        sources = (
            self._source("root-a", "branch-exhausted", 0),
            self._source("root-a", "branch-ready", 2),
        )
        allocation = self._allocate(
            requested_count=4,
            minimum_exploration_count=2,
            sources=sources,
            settings=tuple(
                self._settings(source.branch_task_id, risk=None) for source in sources
            ),
            targets=tuple(
                self._branch_targets(
                    source.branch_task_id,
                    (self._target("LOW_FITNESS"),),
                )
                for source in sources
            ),
        )

        self.assertEqual(allocation.improvement_count, 2)
        self.assertEqual(allocation.exploration_count, 2)
        self.assertEqual(
            {item.parent_task_id for item in allocation.signal_improvements},
            {"branch-ready"},
        )
        self.assertEqual(allocation.reason, "signal_improvement_attempts")

    def test_all_exhausted_neighborhoods_return_to_exploration(self) -> None:
        allocation = self._allocate(
            requested_count=4,
            minimum_exploration_count=2,
            sources=(self._source("root-a", "branch-a", 0),),
            settings=(self._settings("branch-a", risk=None),),
            targets=(
                self._branch_targets(
                    "branch-a",
                    (self._target("LOW_FITNESS"),),
                ),
            ),
        )

        self.assertEqual(allocation.improvement_count, 0)
        self.assertEqual(allocation.exploration_count, 4)
        self.assertEqual(allocation.signal_improvements, ())
        self.assertEqual(
            allocation.reason,
            "signal_improvement_neighborhood_exhausted",
        )

    def test_pending_transformation_requests_do_not_claim_exhaustion(self) -> None:
        allocation = self._allocate(
            requested_count=4,
            minimum_exploration_count=2,
            sources=(self._source("root-a", "branch-a", 0, blocked=1),),
            settings=(self._settings("branch-a", risk=None),),
            targets=(
                self._branch_targets(
                    "branch-a",
                    (self._target("LOW_FITNESS"),),
                ),
            ),
        )

        self.assertEqual(allocation.improvement_count, 0)
        self.assertEqual(allocation.exploration_count, 4)
        self.assertEqual(allocation.signal_improvements, ())
        self.assertEqual(
            allocation.reason,
            "signal_improvement_requests_pending",
        )

    def test_blocked_branch_without_target_does_not_mask_exhaustion(self) -> None:
        allocation = self._allocate(
            requested_count=4,
            minimum_exploration_count=2,
            sources=(
                self._source("root-a", "branch-target", 0),
                self._source(
                    "root-a",
                    "branch-without-target",
                    0,
                    blocked=1,
                ),
            ),
            settings=(
                self._settings("branch-target", risk=None),
                self._settings("branch-without-target", risk=None),
            ),
            targets=(
                self._branch_targets(
                    "branch-target",
                    (self._target("LOW_FITNESS"),),
                ),
                self._branch_targets("branch-without-target", ()),
            ),
        )

        self.assertEqual(
            allocation.reason,
            "signal_improvement_neighborhood_exhausted",
        )

    def test_structural_and_polishing_sources_share_one_improvement_budget(
        self,
    ) -> None:
        sources = (
            self._source("root-structure", "branch-structure", 1),
            self._source(
                "root-polish",
                "branch-polish",
                1,
                stage=LOCAL_POLISHING_STAGE,
                families=(SINGLE_WINDOW_MUTATION,),
            ),
        )
        allocation = self._allocate(
            requested_count=4,
            minimum_exploration_count=2,
            sources=sources,
            settings=tuple(
                self._settings(source.branch_task_id, risk=None) for source in sources
            ),
            targets=tuple(
                self._branch_targets(
                    source.branch_task_id,
                    (self._target("LOW_FITNESS"),),
                )
                for source in sources
            ),
        )

        self.assertEqual(allocation.exploration_count, 2)
        self.assertEqual(allocation.structural_evolution_count, 1)
        self.assertEqual(allocation.local_polishing_count, 1)
        self.assertEqual(
            {item.stage for item in allocation.signal_improvements},
            {STRUCTURAL_EVOLUTION_STAGE, LOCAL_POLISHING_STAGE},
        )
        polishing = next(
            item
            for item in allocation.signal_improvements
            if item.stage == LOCAL_POLISHING_STAGE
        )
        self.assertEqual(
            polishing.candidate_family,
            SINGLE_WINDOW_MUTATION,
        )

    def test_qualified_lineage_uses_at_most_two_slots_per_cycle(self) -> None:
        allocation = self._allocate(
            requested_count=8,
            minimum_exploration_count=2,
            sources=(
                self._source(
                    "root-qualified",
                    "branch-qualified",
                    6,
                    stage=QUALIFIED_EVOLUTION_STAGE,
                    families=QUALIFIED_EVOLUTION_FAMILIES,
                ),
            ),
            settings=(self._settings("branch-qualified", risk=None),),
            targets=(self._branch_targets("branch-qualified", ()),),
        )

        self.assertEqual(allocation.improvement_count, 2)
        self.assertEqual(allocation.qualified_evolution_count, 2)
        self.assertEqual(allocation.structural_evolution_count, 0)
        self.assertEqual(allocation.local_polishing_count, 0)
        self.assertEqual(allocation.exploration_count, 6)
        self.assertTrue(
            all(item.target is None for item in allocation.signal_improvements)
        )

    def test_qualified_lineage_cannot_exceed_its_last_remaining_attempt(self) -> None:
        allocation = self._allocate(
            requested_count=8,
            minimum_exploration_count=2,
            sources=(
                self._source(
                    "root-qualified",
                    "branch-qualified",
                    6,
                    stage=QUALIFIED_EVOLUTION_STAGE,
                    families=QUALIFIED_EVOLUTION_FAMILIES,
                    remaining_attempts=1,
                ),
            ),
            settings=(self._settings("branch-qualified", risk=None),),
            targets=(self._branch_targets("branch-qualified", ()),),
        )

        self.assertEqual(allocation.qualified_evolution_count, 1)
        self.assertEqual(allocation.improvement_count, 1)
        self.assertEqual(allocation.exploration_count, 7)

    def test_each_qualified_lineage_can_receive_two_slots(self) -> None:
        sources = tuple(
            self._source(
                root_task_id,
                branch_task_id,
                4,
                stage=QUALIFIED_EVOLUTION_STAGE,
                families=QUALIFIED_EVOLUTION_FAMILIES,
            )
            for root_task_id, branch_task_id in (
                ("root-a", "branch-a"),
                ("root-b", "branch-b"),
            )
        )
        allocation = self._allocate(
            requested_count=6,
            minimum_exploration_count=2,
            sources=sources,
            settings=tuple(
                self._settings(source.branch_task_id, risk=None)
                for source in sources
            ),
            targets=tuple(
                self._branch_targets(source.branch_task_id, ())
                for source in sources
            ),
        )

        self.assertEqual(
            Counter(item.root_task_id for item in allocation.signal_improvements),
            {"root-a": 2, "root-b": 2},
        )
        self.assertEqual(allocation.qualified_evolution_count, 4)

    def test_less_attempted_qualified_parent_is_selected_first(self) -> None:
        sources = (
            self._source(
                "root-used",
                "branch-used",
                1,
                stage=QUALIFIED_EVOLUTION_STAGE,
                families=QUALIFIED_EVOLUTION_FAMILIES,
                remaining_attempts=15,
            ),
            self._source(
                "root-fresh",
                "branch-fresh",
                1,
                stage=QUALIFIED_EVOLUTION_STAGE,
                families=QUALIFIED_EVOLUTION_FAMILIES,
            ),
        )
        allocation = self._allocate(
            requested_count=2,
            minimum_exploration_count=1,
            sources=sources,
            settings=tuple(
                self._settings(source.branch_task_id, risk=None)
                for source in sources
            ),
            targets=tuple(
                self._branch_targets(source.branch_task_id, ())
                for source in sources
            ),
        )

        self.assertEqual(
            allocation.signal_improvements[0].root_task_id,
            "root-fresh",
        )

    def test_qualified_parent_uses_less_attempted_family_first(self) -> None:
        family_counts = {family: 1 for family in QUALIFIED_EVOLUTION_FAMILIES}
        least_used_family = SELF_CORRELATION_INTERNAL_FAMILIES[-1]
        attempted_by_family = {
            family: 0 if family == least_used_family else 1
            for family in QUALIFIED_EVOLUTION_FAMILIES
        }
        allocation = self._allocate(
            requested_count=2,
            minimum_exploration_count=1,
            sources=(
                self._source(
                    "root-qualified",
                    "branch-qualified",
                    len(family_counts),
                    stage=QUALIFIED_EVOLUTION_STAGE,
                    families=QUALIFIED_EVOLUTION_FAMILIES,
                    family_counts=family_counts,
                    attempted_by_family=attempted_by_family,
                ),
            ),
            settings=(self._settings("branch-qualified", risk=None),),
            targets=(self._branch_targets("branch-qualified", ()),),
        )

        self.assertEqual(
            allocation.signal_improvements[0].candidate_family,
            least_used_family,
        )

    def test_qualified_parent_covers_two_families_before_repeating(self) -> None:
        family_counts = {family: 2 for family in QUALIFIED_EVOLUTION_FAMILIES}
        family_counts[SELF_CORRELATION_FIELD_REPLACEMENT] = 1
        allocation = self._allocate(
            requested_count=3,
            minimum_exploration_count=1,
            sources=(
                self._source(
                    "root-qualified",
                    "branch-qualified",
                    sum(family_counts.values()),
                    stage=QUALIFIED_EVOLUTION_STAGE,
                    families=QUALIFIED_EVOLUTION_FAMILIES,
                    family_counts=family_counts,
                ),
            ),
            settings=(self._settings("branch-qualified", risk=None),),
            targets=(self._branch_targets("branch-qualified", ()),),
        )

        self.assertEqual(allocation.qualified_evolution_count, 2)
        self.assertEqual(
            len(
                {
                    item.candidate_family
                    for item in allocation.signal_improvements
                }
            ),
            2,
        )

    def test_low_sub_universe_target_uses_only_robustness_families(self) -> None:
        family_counts = {family: 1 for family in STRUCTURAL_TRANSFORMATION_FAMILIES}
        allocation = self._allocate(
            requested_count=6,
            minimum_exploration_count=1,
            sources=(
                self._source(
                    "root-low-sub",
                    "branch-low-sub",
                    len(family_counts),
                    family_counts=family_counts,
                ),
            ),
            settings=(self._settings("branch-low-sub", risk=None),),
            targets=(
                self._branch_targets(
                    "branch-low-sub",
                    (self._target("LOW_SUB_UNIVERSE_SHARPE"),),
                ),
            ),
        )

        self.assertEqual(allocation.improvement_count, 3)
        self.assertEqual(allocation.exploration_count, 3)
        self.assertEqual(
            {item.candidate_family for item in allocation.signal_improvements},
            set(LOW_SUB_UNIVERSE_SHARPE_STRUCTURAL_FAMILIES),
        )

    def test_low_sub_universe_disallowed_capacity_returns_to_exploration(
        self,
    ) -> None:
        allocation = self._allocate(
            requested_count=2,
            minimum_exploration_count=1,
            sources=(
                self._source(
                    "root-low-sub",
                    "branch-low-sub",
                    1,
                    blocked=1,
                    family_counts={TEMPORAL_CHANGE_REFRAME: 1},
                    blocked_family=TEMPORAL_CHANGE_REFRAME,
                ),
            ),
            settings=(self._settings("branch-low-sub", risk=None),),
            targets=(
                self._branch_targets(
                    "branch-low-sub",
                    (self._target("LOW_SUB_UNIVERSE_SHARPE"),),
                ),
            ),
        )

        self.assertEqual(allocation.improvement_count, 0)
        self.assertEqual(allocation.exploration_count, 2)
        self.assertEqual(
            allocation.reason,
            "signal_improvement_neighborhood_exhausted",
        )

    def test_constrained_target_does_not_double_allocate_shared_families(
        self,
    ) -> None:
        allocation = self._allocate(
            requested_count=4,
            minimum_exploration_count=2,
            sources=(
                self._source(
                    "root-shared",
                    "branch-shared",
                    2,
                    family_counts={
                        TEMPORAL_CHANGE_REFRAME: 1,
                        LOW_SUB_UNIVERSE_SHARPE_STRUCTURAL_FAMILIES[0]: 1,
                    },
                ),
            ),
            settings=(self._settings("branch-shared", risk=None),),
            targets=(
                self._branch_targets(
                    "branch-shared",
                    (
                        self._target("LOW_SHARPE"),
                        self._target("LOW_SUB_UNIVERSE_SHARPE"),
                    ),
                ),
            ),
        )

        self.assertEqual(allocation.improvement_count, 2)
        self.assertEqual(
            {item.candidate_family for item in allocation.signal_improvements},
            {
                TEMPORAL_CHANGE_REFRAME,
                LOW_SUB_UNIVERSE_SHARPE_STRUCTURAL_FAMILIES[0],
            },
        )
        low_sub = next(
            item
            for item in allocation.signal_improvements
            if item.target.check_name == "LOW_SUB_UNIVERSE_SHARPE"
        )
        self.assertEqual(
            low_sub.candidate_family,
            LOW_SUB_UNIVERSE_SHARPE_STRUCTURAL_FAMILIES[0],
        )

    def _allocate(
        self,
        *,
        requested_count: int,
        minimum_exploration_count: int,
        sources: tuple[SignalImprovementSource, ...],
        settings: tuple[ParentSettingsEvidence, ...],
        targets: tuple[ParentOptimizationTargets, ...],
        action_strategies: tuple[DefectActionStrategy, ...] = (),
        direction_validation_count: int = 0,
        self_correlation_percent: int = 30,
        direction_validation_percent: int = 3,
    ):
        return allocate_backtest_sources(
            requested_count=requested_count,
            minimum_exploration_count=minimum_exploration_count,
            improvement_sources=sources,
            settings_evidence=ParentSettingsEvidenceSet(records=settings),
            optimization_targets=ParentOptimizationTargetSet(records=targets),
            action_strategies=DefectActionStrategySet(
                records=action_strategies,
                observations=(),
            ),
            account_scope="group-account",
            rotation_key="run-a:1",
            direction_validation_count=direction_validation_count,
            self_correlation_percent=self_correlation_percent,
            direction_validation_percent=direction_validation_percent,
        )

    @staticmethod
    def _source(
        root_task_id: str,
        branch_task_id: str,
        available: int,
        *,
        blocked: int = 0,
        stage: str = STRUCTURAL_EVOLUTION_STAGE,
        families: tuple[str, ...] = STRUCTURAL_TRANSFORMATION_FAMILIES,
        family_counts: dict[str, int] | None = None,
        blocked_family: str | None = None,
        attempted: int = 0,
        attempted_by_family: dict[str, int] | None = None,
        remaining_attempts: int = 20,
        is_submitted: bool = False,
        self_correlation: float | None = 0.8,
    ) -> SignalImprovementSource:
        counts = {family: 0 for family in families}
        if family_counts is None:
            counts[families[0]] = available
        else:
            counts.update(family_counts)
            assert sum(counts.values()) == available
        blocked_target = blocked_family or families[0]
        attempted_counts = {family: 0 for family in families}
        if attempted_by_family is None:
            attempted_counts[families[0]] = attempted
        else:
            attempted_counts.update(attempted_by_family)
        return SignalImprovementSource(
            root_task_id=root_task_id,
            branch_task_id=branch_task_id,
            stage=stage,
            parent_remaining_attempts=remaining_attempts,
            is_submitted=is_submitted,
            self_correlation=self_correlation,
            family_capacities=tuple(
                SignalImprovementFamilyCapacity(
                    family=family,
                    available_leaf_count=counts[family],
                    blocked_leaf_count=blocked if family == blocked_target else 0,
                    attempted_leaf_count=attempted_counts[family],
                )
                for family in families
            ),
        )

    @staticmethod
    def _target(
        check_name: str,
        *,
        actions: tuple[OptimizationActionEvidence, ...] = (),
    ) -> OptimizationTarget:
        return OptimizationTarget(
            check_name=check_name,
            details_captured=False,
            threshold=None,
            actual=None,
            platform_date=None,
            normalized_gap=None,
            gap_state="historical_not_captured",
            actions=actions,
        )

    @staticmethod
    def _branch_targets(
        parent_task_id: str,
        targets: tuple[OptimizationTarget, ...],
        *,
        settings_key: str = "settings-own",
    ) -> ParentOptimizationTargets:
        return ParentOptimizationTargets(
            parent_task_id=parent_task_id,
            account_scope="group-account",
            settings_key=settings_key,
            targets=targets,
        )

    @staticmethod
    def _action_strategy(
        parent_task_id: str,
        target_check_name: str,
        *,
        preferred: tuple[str, ...] = (),
        deprioritized: tuple[str, ...] = (),
        action_states: dict[str, str],
    ) -> DefectActionStrategy:
        return DefectActionStrategy(
            parent_task_id=parent_task_id,
            account_scope="group-account",
            settings_key="settings-own",
            defect_checks=(target_check_name,),
            target_check_name=target_check_name,
            mode=(
                ACTION_STRATEGY_EXPLOITATION
                if preferred
                else ACTION_STRATEGY_EXPLORATION
            ),
            reason=(
                "preferred_action_supported"
                if preferred
                else "no_preferred_action"
            ),
            preferred_actions=preferred,
            deprioritized_actions=deprioritized,
            actions=tuple(
                DefectActionEvidence(
                    action=action,
                    state=state,
                    resolved_count=10,
                    safe_progress_count=(
                        10 if state == ACTION_STATE_PREFERRED else 0
                    ),
                    no_progress_count=(
                        10 if state == ACTION_STATE_DEPRIORITIZED else 0
                    ),
                    conflict_count=0,
                    unresolved_count=0,
                    root_lineage_count=5,
                    run_count=1,
                )
                for action, state in sorted(action_states.items())
            ),
        )

    @staticmethod
    def _settings(
        parent_task_id: str,
        *,
        risk: float | None,
    ) -> ParentSettingsEvidence:
        return ParentSettingsEvidence(
            parent_task_id=parent_task_id,
            account_scope="group-account",
            settings_key="settings-own",
            child_count=4,
            child_quality_passed_count=1,
            child_quality_failed_count=3,
            child_quality_pending_count=0,
            sc_passed_count=3 if risk is not None else 1,
            sc_failed_count=2 if risk is not None else 0,
            sc_pending_count=0,
            sc_missing_count=0,
            sc_explicit_count=5 if risk is not None else 1,
            sc_risk_available=risk is not None,
            sc_smoothed_risk=risk,
        )


if __name__ == "__main__":
    unittest.main()
