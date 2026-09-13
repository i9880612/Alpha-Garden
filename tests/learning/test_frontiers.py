from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from learning.evidence import LearningEvidenceRecord, LearningEvidenceSet
from learning.frontiers import (
    PARENT_ATTEMPT_BUDGET,
    build_signal_frontiers,
)
from persistence.backtests import BacktestCheckRecord, BacktestMutationRecord


class SignalFrontierTests(unittest.TestCase):
    def test_decorrelated_quality_branch_shares_root_and_expires_without_revival(self):
        root = self._record(
            "root",
            sharpe=2.0,
            fitness=1.5,
            passed_checks=(
                "LOW_SHARPE",
                "LOW_FITNESS",
                "LOW_SUB_UNIVERSE_SHARPE",
                "CONCENTRATED_WEIGHT",
            ),
            failed_checks=(),
        )
        recovery = self._record("recovery", sharpe=1.1, fitness=0.9)
        worse = self._record("worse", sharpe=1.0, fitness=0.8)
        better = self._record("better", sharpe=1.2, fitness=1.0)
        attempts = tuple(f"attempt-{i}" for i in range(20))
        mutations = (
            self._mutation("root", "recovery"),
            self._mutation("recovery", "worse"),
            *(self._mutation("recovery", child) for child in attempts),
        )
        for count in (0, 19, 20):
            with self.subTest(count=count):
                frontier = build_signal_frontiers(
                    LearningEvidenceSet((root, recovery, worse), ()),
                    mutations,
                    ("root",),
                    attempts[:count],
                    recovery_task_ids=frozenset({"recovery", "worse"}),
                ).records[0]
                self.assertEqual(
                    {b.task_id for b in frontier.branches},
                    {"root", "recovery"} if count < 20 else {"root"},
                )
                self.assertTrue(
                    all(b.root_task_id == "root" for b in frontier.branches)
                )
                self.assertEqual(frontier.qualified_task_ids, ("root",))
        late = build_signal_frontiers(
            LearningEvidenceSet((root, recovery, worse, better), ()),
            (*mutations, self._mutation("recovery", "better")),
            ("root",),
            attempts,
            recovery_task_ids=frozenset({"recovery", "worse", "better"}),
        )
        self.assertEqual(set(late.active_branch_task_ids), {"root", "better"})
        unchanged = build_signal_frontiers(
            LearningEvidenceSet((root, recovery), ()), mutations, ("root",)
        )
        self.assertEqual(unchanged.active_branch_task_ids, ("root",))

    def test_unqualified_parent_exits_at_twenty_actual_attempts(self):
        root = self._record("root", sharpe=1.1, fitness=0.8)
        child_ids = tuple(f"attempt-{index}" for index in range(21))
        mutations = tuple(self._mutation("root", child) for child in child_ids)
        for count in (0, 19, 20, 21):
            with self.subTest(count=count):
                frontier = build_signal_frontiers(
                    LearningEvidenceSet(records=(root,), settings=()),
                    mutations, ("root",), child_ids[:count],
                ).records[0]
                self.assertFalse(frontier.qualified)
                self.assertEqual(len(frontier.branches), int(count < 20))
                if frontier.branches:
                    self.assertEqual(frontier.branches[0].remaining_attempts, 20 - count)

    def test_expiry_does_not_revive_dominated_or_equivalent_candidates(self):
        records = (
            self._record("root", sharpe=1.0, fitness=0.7),
            self._record("best", sharpe=1.2, fitness=0.9),
            self._record("worse", sharpe=1.1, fitness=0.8),
            self._record("equal", sharpe=1.2, fitness=0.9,
                         finished_at="2026-08-31T00:03:00+00:00"),
        )
        attempted = tuple(f"attempt-{index}" for index in range(20))
        mutations = (
            self._mutation("root", "best"),
            self._mutation("best", "worse"), self._mutation("best", "equal"),
            *(self._mutation("best", child) for child in attempted),
        )
        for ordered in (records, tuple(reversed(records))):
            frontier = build_signal_frontiers(
                LearningEvidenceSet(records=ordered, settings=()),
                mutations, ("root",), attempted,
            ).records[0]
            self.assertFalse(frontier.branches)

    def test_expired_parent_does_not_block_late_improvement_or_other_tradeoff(self):
        root = self._record("root", sharpe=1.1, fitness=0.8)
        tradeoff = self._record("tradeoff", sharpe=1.0, fitness=0.9)
        late = self._record("late", sharpe=1.2, fitness=0.85)
        attempted = tuple(f"attempt-{index}" for index in range(20))
        mutations = (
            self._mutation("root", "tradeoff"),
            *(self._mutation("root", child) for child in attempted),
        )
        before = build_signal_frontiers(
            LearningEvidenceSet(records=(root, tradeoff), settings=()),
            mutations, ("root",), attempted,
        )
        self.assertEqual(before.active_branch_task_ids, ("tradeoff",))
        after = build_signal_frontiers(
            LearningEvidenceSet(records=(root, tradeoff, late), settings=()),
            (*mutations, self._mutation("root", "late")), ("root",), attempted,
        )
        self.assertEqual(set(after.active_branch_task_ids), {"tradeoff", "late"})
        self.assertTrue(all(branch.remaining_attempts == 20 for branch in after.records[0].branches))

    def test_tradeoff_branches_survive_and_dominated_nodes_do_not(self) -> None:
        records = (
            self._record("root", sharpe=1.1, fitness=0.8),
            self._record("fitness", sharpe=1.15, fitness=1.0),
            self._record("sharpe", sharpe=1.3, fitness=0.85),
            self._record("dominated", sharpe=1.09, fitness=0.79),
        )

        frontier = build_signal_frontiers(
            LearningEvidenceSet(records=records, settings=()),
            (
                self._mutation("root", "fitness"),
                self._mutation("root", "sharpe"),
                self._mutation("root", "dominated"),
            ),
            ("root",),
        ).records[0]

        self.assertEqual(
            tuple(branch.task_id for branch in frontier.branches),
            ("fitness", "sharpe"),
        )
        self.assertFalse(frontier.qualified)

    def test_frontier_is_bounded_to_two_extremes_and_fixed_middle(self) -> None:
        records = tuple(
            self._record(
                f"point-{index}",
                sharpe=1.0 + index / 10,
                fitness=1.1 - index / 10,
            )
            for index in range(5)
        )

        frontier = build_signal_frontiers(
            LearningEvidenceSet(records=tuple(reversed(records)), settings=()),
            tuple(self._mutation("point-0", f"point-{index}") for index in range(1, 5)),
            ("point-0",),
        ).records[0]

        self.assertEqual(
            tuple(branch.task_id for branch in frontier.branches),
            ("point-0", "point-2", "point-4"),
        )

    def test_equal_metrics_keep_the_earliest_stable_fact(self) -> None:
        root = self._record("root", sharpe=1.0, fitness=0.7)
        later = self._record(
            "later",
            sharpe=1.1,
            fitness=0.8,
            finished_at="2026-08-31T00:03:00+00:00",
        )
        earlier = self._record(
            "earlier",
            sharpe=1.1,
            fitness=0.8,
            finished_at="2026-08-31T00:02:00+00:00",
        )

        frontier = build_signal_frontiers(
            LearningEvidenceSet(records=(root, later, earlier), settings=()),
            (
                self._mutation("root", "later"),
                self._mutation("root", "earlier"),
            ),
            ("root",),
        ).records[0]

        self.assertEqual(
            tuple(branch.task_id for branch in frontier.branches),
            ("earlier",),
        )

    def test_incomparable_path_or_unsafe_child_cannot_reenter_lineage(self) -> None:
        records = (
            self._record("root", sharpe=1.0, fitness=0.7),
            self._record(
                "different-settings",
                sharpe=1.0,
                fitness=0.8,
                settings_key="other",
            ),
            self._record("returned", sharpe=1.1, fitness=0.9),
            self._record(
                "unsafe",
                sharpe=1.0,
                fitness=0.8,
                passed_checks=(
                    "LOW_SUB_UNIVERSE_SHARPE",
                    "LOW_FITNESS",
                ),
                failed_checks=("LOW_SHARPE", "CONCENTRATED_WEIGHT"),
            ),
            self._record("after-unsafe", sharpe=1.2, fitness=1.0),
        )

        frontier = build_signal_frontiers(
            LearningEvidenceSet(records=records, settings=()),
            (
                self._mutation("root", "different-settings"),
                self._mutation("different-settings", "returned"),
                self._mutation("root", "unsafe"),
                self._mutation("unsafe", "after-unsafe"),
            ),
            ("root",),
        ).records[0]

        self.assertEqual(
            tuple(branch.task_id for branch in frontier.branches),
            ("root",),
        )

    def test_branches_below_seed_quality_floor_leave_the_lineage(self) -> None:
        records = (
            self._record("root", sharpe=1.1, fitness=0.8),
            self._record("low-sharpe", sharpe=0.999, fitness=1.0),
            self._record("after-low-sharpe", sharpe=2.0, fitness=2.0),
            self._record("low-fitness", sharpe=1.2, fitness=0.699),
            self._record("after-low-fitness", sharpe=2.0, fitness=2.0),
        )

        frontier = build_signal_frontiers(
            LearningEvidenceSet(records=records, settings=()),
            (
                self._mutation("root", "low-sharpe"),
                self._mutation("low-sharpe", "after-low-sharpe"),
                self._mutation("root", "low-fitness"),
                self._mutation("low-fitness", "after-low-fitness"),
            ),
            ("root",),
        ).records[0]

        self.assertEqual(
            tuple(branch.task_id for branch in frontier.branches),
            ("root",),
        )

    def test_root_below_seed_quality_floor_fails_closed(self) -> None:
        for root in (
            self._record("low-sharpe-root", sharpe=0.999, fitness=0.8),
            self._record("low-fitness-root", sharpe=1.1, fitness=0.699),
        ):
            with self.subTest(task_id=root.task_id):
                with self.assertRaisesRegex(
                    ValueError,
                    "signal_frontier_root_not_safe",
                ):
                    build_signal_frontiers(
                        LearningEvidenceSet(records=(root,), settings=()),
                        (),
                        (root.task_id,),
                    )

    def test_fully_passing_descendant_becomes_the_evolution_parent(self) -> None:
        root = self._record("root", sharpe=1.0, fitness=0.7)
        qualified = self._record(
            "qualified",
            sharpe=1.3,
            fitness=1.1,
            passed_checks=(
                "LOW_SHARPE",
                "LOW_FITNESS",
                "LOW_SUB_UNIVERSE_SHARPE",
                "CONCENTRATED_WEIGHT",
            ),
            failed_checks=(),
        )

        built = build_signal_frontiers(
            LearningEvidenceSet(records=(root, qualified), settings=()),
            (self._mutation("root", "qualified"),),
            ("root",),
        )

        self.assertEqual(built.records[0].qualified_task_ids, ("qualified",))
        self.assertEqual(built.active_branch_task_ids, ("qualified",))
        self.assertTrue(built.records[0].qualified_evolution_active)

    def test_higher_qualified_child_promotes_and_resets_stagnation(self) -> None:
        root = self._record("root", sharpe=1.0, fitness=0.7)
        qualified = self._record(
            "qualified",
            sharpe=1.3,
            fitness=1.1,
            passed_checks=(
                "LOW_SHARPE",
                "LOW_FITNESS",
                "LOW_SUB_UNIVERSE_SHARPE",
                "CONCENTRATED_WEIGHT",
            ),
            failed_checks=(),
        )
        improved = self._record(
            "improved",
            sharpe=3.0,
            fitness=1.0,
            finished_at="2026-08-31T00:03:00+00:00",
            passed_checks=qualified.passed_checks,
            failed_checks=(),
        )

        built = build_signal_frontiers(
            LearningEvidenceSet(records=(root, qualified, improved), settings=()),
            (
                self._mutation("root", "qualified"),
                self._mutation("qualified", "improved"),
            ),
            ("root",),
            ("improved",),
        )

        frontier = built.records[0]
        self.assertEqual(
            frontier.qualified_task_ids,
            ("qualified", "improved"),
        )
        self.assertEqual(built.active_branch_task_ids, ("improved",))
        self.assertEqual(frontier.qualified_parent_attempt_count, 0)
        self.assertEqual(
            frontier.qualified_parent_remaining_attempts,
            20,
        )

    def test_equal_sharpe_child_does_not_replace_qualified_parent(self) -> None:
        root = self._record(
            "root",
            sharpe=1.3,
            fitness=1.1,
            passed_checks=(
                "LOW_SHARPE",
                "LOW_FITNESS",
                "LOW_SUB_UNIVERSE_SHARPE",
                "CONCENTRATED_WEIGHT",
            ),
            failed_checks=(),
        )
        equal = self._record(
            "equal",
            sharpe=1.3,
            fitness=1.3,
            finished_at="2026-08-31T00:03:00+00:00",
            passed_checks=root.passed_checks,
            failed_checks=(),
        )

        built = build_signal_frontiers(
            LearningEvidenceSet(records=(root, equal), settings=()),
            (self._mutation("root", "equal"),),
            ("root",),
            ("equal",),
        )

        self.assertEqual(built.active_branch_task_ids, ("root",))
        self.assertEqual(built.records[0].qualified_parent_attempt_count, 1)
        self.assertEqual(
            built.records[0].qualified_parent_remaining_attempts,
            PARENT_ATTEMPT_BUDGET - 1,
        )

    def test_qualified_parent_remains_active_until_attempt_budget_is_used(
        self,
    ) -> None:
        root = self._record(
            "root",
            sharpe=1.3,
            fitness=1.1,
            passed_checks=(
                "LOW_SHARPE",
                "LOW_FITNESS",
                "LOW_SUB_UNIVERSE_SHARPE",
                "CONCENTRATED_WEIGHT",
            ),
            failed_checks=(),
        )
        child_ids = tuple(
            f"attempt-{index}"
            for index in range(1, 21)
        )
        mutations = tuple(self._mutation("root", child_id) for child_id in child_ids)

        after_three = build_signal_frontiers(
            LearningEvidenceSet(records=(root,), settings=()),
            mutations,
            ("root",),
            child_ids[:3],
        ).records[0]
        last_attempt_remaining = build_signal_frontiers(
            LearningEvidenceSet(records=(root,), settings=()),
            mutations,
            ("root",),
            child_ids[:-1],
        ).records[0]
        exhausted = build_signal_frontiers(
            LearningEvidenceSet(records=(root,), settings=()),
            mutations,
            ("root",),
            child_ids,
        ).records[0]

        self.assertTrue(after_three.qualified_evolution_active)
        self.assertEqual(after_three.qualified_parent_attempt_count, 3)
        self.assertEqual(after_three.qualified_parent_remaining_attempts, 17)
        self.assertTrue(last_attempt_remaining.qualified_evolution_active)
        self.assertEqual(last_attempt_remaining.qualified_parent_remaining_attempts, 1)
        self.assertFalse(exhausted.qualified_evolution_active)
        self.assertEqual(exhausted.qualified_parent_remaining_attempts, 0)

        historical = build_signal_frontiers(
            LearningEvidenceSet(records=(root,), settings=()),
            (*mutations, self._mutation("root", "historical-extra")),
            ("root",),
            (*child_ids, "historical-extra"),
        ).records[0]
        self.assertEqual(historical.qualified_parent_attempt_count, 21)
        self.assertFalse(historical.qualified_evolution_active)
        self.assertEqual(historical.qualified_parent_remaining_attempts, 0)

    def test_attempted_child_ids_must_be_unique_mutation_children(self) -> None:
        root = self._record(
            "root",
            sharpe=1.3,
            fitness=1.1,
            passed_checks=(
                "LOW_SHARPE",
                "LOW_FITNESS",
                "LOW_SUB_UNIVERSE_SHARPE",
                "CONCENTRATED_WEIGHT",
            ),
            failed_checks=(),
        )
        mutation = self._mutation("root", "child")

        with self.assertRaisesRegex(
            ValueError,
            "signal_frontier_attempted_child_id_duplicated",
        ):
            build_signal_frontiers(
                LearningEvidenceSet(records=(root,), settings=()),
                (mutation,),
                ("root",),
                ("child", "child"),
            )
        with self.assertRaisesRegex(
            ValueError,
            "signal_frontier_attempted_child_missing",
        ):
            build_signal_frontiers(
                LearningEvidenceSet(records=(root,), settings=()),
                (mutation,),
                ("root",),
                ("other",),
            )

    def test_low_sub_universe_failure_is_an_active_frontier_target(self) -> None:
        root = self._record(
            "low-sub-root",
            sharpe=1.0,
            fitness=0.7,
            passed_checks=(
                "LOW_SHARPE",
                "LOW_FITNESS",
                "CONCENTRATED_WEIGHT",
            ),
            failed_checks=("LOW_SUB_UNIVERSE_SHARPE",),
            low_sub_detail=(0.36, 0.25),
        )

        frontier = build_signal_frontiers(
            LearningEvidenceSet(records=(root,), settings=()),
            (),
            (root.task_id,),
        ).records[0]

        self.assertEqual(
            tuple(branch.task_id for branch in frontier.branches),
            (root.task_id,),
        )
        self.assertFalse(frontier.qualified)

    def test_low_sub_universe_gap_improvement_survives_lower_base_metrics(
        self,
    ) -> None:
        passed = (
            "LOW_SHARPE",
            "LOW_FITNESS",
            "CONCENTRATED_WEIGHT",
        )
        root = self._record(
            "root",
            sharpe=1.1,
            fitness=0.8,
            passed_checks=passed,
            failed_checks=("LOW_SUB_UNIVERSE_SHARPE",),
            low_sub_detail=(0.36, 0.25),
        )
        improved = self._record(
            "improved",
            sharpe=1.0,
            fitness=0.7,
            passed_checks=passed,
            failed_checks=("LOW_SUB_UNIVERSE_SHARPE",),
            low_sub_detail=(0.36, 0.30),
        )

        frontier = build_signal_frontiers(
            LearningEvidenceSet(records=(root, improved), settings=()),
            (self._mutation("root", "improved"),),
            ("root",),
        ).records[0]

        self.assertEqual(
            {branch.task_id for branch in frontier.branches},
            {"root", "improved"},
        )

    def test_bounded_frontier_keeps_low_sub_universe_representative(self) -> None:
        passed = (
            "LOW_SHARPE",
            "LOW_FITNESS",
            "CONCENTRATED_WEIGHT",
        )
        points = (
            ("point-0", 1.0, 1.1, 0.25),
            ("point-1", 1.1, 1.0, 0.27),
            ("point-2", 1.2, 0.9, 0.29),
            ("robust", 1.3, 0.8, 0.35),
            ("point-4", 1.4, 0.7, 0.32),
        )
        records = tuple(
            self._record(
                task_id,
                sharpe=sharpe,
                fitness=fitness,
                passed_checks=passed,
                failed_checks=("LOW_SUB_UNIVERSE_SHARPE",),
                low_sub_detail=(0.36, actual),
            )
            for task_id, sharpe, fitness, actual in points
        )

        frontier = build_signal_frontiers(
            LearningEvidenceSet(records=records, settings=()),
            tuple(self._mutation("point-0", task_id) for task_id, *_rest in points[1:]),
            ("point-0",),
        ).records[0]

        self.assertEqual(
            tuple(branch.task_id for branch in frontier.branches),
            ("point-0", "robust", "point-4"),
        )

    def test_bounded_frontier_counts_repeated_representative_once(self) -> None:
        records = (
            self._record(
                "best-metrics",
                sharpe=1.2,
                fitness=1.0,
                passed_checks=("CONCENTRATED_WEIGHT",),
                failed_checks=(
                    "LOW_SHARPE",
                    "LOW_FITNESS",
                    "LOW_SUB_UNIVERSE_SHARPE",
                ),
            ),
            self._record(
                "sharpe-progress",
                sharpe=1.1,
                fitness=0.7,
                passed_checks=("LOW_SHARPE", "CONCENTRATED_WEIGHT"),
                failed_checks=("LOW_FITNESS", "LOW_SUB_UNIVERSE_SHARPE"),
            ),
            self._record(
                "robust",
                sharpe=1.0,
                fitness=0.8,
                passed_checks=(
                    "LOW_SUB_UNIVERSE_SHARPE",
                    "CONCENTRATED_WEIGHT",
                ),
                failed_checks=("LOW_SHARPE", "LOW_FITNESS"),
            ),
            self._record(
                "fitness-progress",
                sharpe=1.0,
                fitness=0.9,
                passed_checks=("LOW_FITNESS", "CONCENTRATED_WEIGHT"),
                failed_checks=("LOW_SHARPE", "LOW_SUB_UNIVERSE_SHARPE"),
            ),
        )

        frontier = build_signal_frontiers(
            LearningEvidenceSet(records=records, settings=()),
            tuple(
                self._mutation("best-metrics", record.task_id)
                for record in records[1:]
            ),
            ("best-metrics",),
        ).records[0]

        self.assertEqual(
            tuple(branch.task_id for branch in frontier.branches),
            ("fitness-progress", "robust", "best-metrics"),
        )

    def test_bounded_frontier_keeps_branch_closest_to_all_targets(self) -> None:
        records = (
            self._record(
                "sharpe",
                sharpe=1.4,
                fitness=0.7,
                passed_checks=("LOW_SHARPE", "CONCENTRATED_WEIGHT"),
                failed_checks=("LOW_FITNESS", "LOW_SUB_UNIVERSE_SHARPE"),
            ),
            self._record(
                "fitness",
                sharpe=1.0,
                fitness=1.2,
                passed_checks=("LOW_FITNESS", "CONCENTRATED_WEIGHT"),
                failed_checks=("LOW_SHARPE", "LOW_SUB_UNIVERSE_SHARPE"),
            ),
            self._record(
                "robust",
                sharpe=1.0,
                fitness=0.8,
                passed_checks=(
                    "LOW_SUB_UNIVERSE_SHARPE",
                    "CONCENTRATED_WEIGHT",
                ),
                failed_checks=("LOW_SHARPE", "LOW_FITNESS"),
            ),
            self._record(
                "closest",
                sharpe=1.2,
                fitness=1.0,
                passed_checks=(
                    "LOW_SHARPE",
                    "LOW_FITNESS",
                    "CONCENTRATED_WEIGHT",
                ),
                failed_checks=("LOW_SUB_UNIVERSE_SHARPE",),
                low_sub_detail=(0.36, 0.3564),
            ),
        )

        frontier = build_signal_frontiers(
            LearningEvidenceSet(records=records, settings=()),
            tuple(self._mutation("sharpe", record.task_id) for record in records[1:]),
            ("sharpe",),
        ).records[0]

        self.assertEqual(
            tuple(branch.task_id for branch in frontier.branches),
            ("fitness", "closest", "sharpe"),
        )

    def test_incomplete_or_boundary_turnover_children_leave_the_frontier(self) -> None:
        root = self._record("root", sharpe=1.0, fitness=0.7)
        children = (
            self._record(
                "incomplete",
                sharpe=1.0,
                fitness=0.8,
                check_set_complete=False,
            ),
            self._record(
                "lower-bound",
                sharpe=1.1,
                fitness=0.9,
                turnover=0.01,
            ),
            self._record(
                "upper-bound",
                sharpe=1.2,
                fitness=1.0,
                turnover=0.70,
            ),
        )

        frontier = build_signal_frontiers(
            LearningEvidenceSet(records=(root, *children), settings=()),
            tuple(self._mutation("root", child.task_id) for child in children),
            ("root",),
        ).records[0]

        self.assertEqual(
            tuple(branch.task_id for branch in frontier.branches),
            ("root",),
        )

    @staticmethod
    def _record(
        task_id: str,
        *,
        sharpe: float,
        fitness: float,
        settings_key: str = "settings",
        finished_at: str = "2026-08-31T00:01:00+00:00",
        passed_checks: tuple[str, ...] = (
            "LOW_SUB_UNIVERSE_SHARPE",
            "CONCENTRATED_WEIGHT",
        ),
        failed_checks: tuple[str, ...] = ("LOW_SHARPE", "LOW_FITNESS"),
        low_sub_detail: tuple[float, float] | None = None,
        turnover: float = 0.12,
        check_set_complete: bool = True,
    ) -> LearningEvidenceRecord:
        return LearningEvidenceRecord(
            task_id=task_id,
            account_scope="group-account",
            formula=f"rank({task_id})",
            formula_fingerprint=f"formula-{task_id}",
            settings_key=settings_key,
            finished_at=finished_at,
            check_details_captured=True,
            checks=(
                (
                    BacktestCheckRecord(
                        name="LOW_SUB_UNIVERSE_SHARPE",
                        status="FAIL",
                        threshold=low_sub_detail[0],
                        actual=low_sub_detail[1],
                        platform_date="2026-08-31",
                    ),
                )
                if low_sub_detail is not None
                else ()
            ),
            outcome="failed" if failed_checks else "passed",
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            failure_categories=(),
            pending_checks=(),
            missing_checks=() if check_set_complete else ("HIGH_TURNOVER",),
            unexpected_checks=(),
            check_set_complete=check_set_complete,
            non_sc_check_set_complete=check_set_complete,
            sharpe=sharpe,
            fitness=fitness,
            turnover=turnover,
            returns=0.08,
            drawdown=0.04,
            margin=0.001,
            book_size=20_000_000,
            pnl=100_000,
        )

    @staticmethod
    def _mutation(parent: str, child: str) -> BacktestMutationRecord:
        return BacktestMutationRecord(
            child_task_id=child,
            parent_task_id=parent,
            action="structural",
            location="formula",
            before=parent,
            after=child,
        )


if __name__ == "__main__":
    unittest.main()
