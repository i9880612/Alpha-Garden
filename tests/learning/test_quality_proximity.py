from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from learning.evidence import LearningEvidenceRecord, LearningEvidenceSet
from learning.optimization_targets import (
    OptimizationTarget,
    ParentOptimizationTargetSet,
    ParentOptimizationTargets,
)
from learning.quality_proximity import (
    LOCAL_POLISHING_STAGE,
    QUALIFIED_EVOLUTION_STAGE,
    STRUCTURAL_EVOLUTION_STAGE,
    build_parent_improvement_stages,
)


class QualityProximityTests(unittest.TestCase):
    def test_platform_gap_at_boundary_enters_local_polishing(self) -> None:
        parent = self._record(
            "parent",
            failed="LOW_SHARPE",
            sharpe=1.1875,
            fitness=1.1,
        )
        stage = self._build(
            (parent,),
            parent,
            targets=(
                self._target(
                    "LOW_SHARPE",
                    details=True,
                    threshold=1.25,
                    actual=1.1875,
                    normalized_gap=0.05,
                    gap_state="available",
                ),
            ),
        )

        self.assertEqual(stage.stage, LOCAL_POLISHING_STAGE)
        self.assertEqual(stage.proximity.source, "platform_check_detail")
        self.assertAlmostEqual(stage.proximity.normalized_gap, 0.05)

    def test_platform_gap_above_boundary_stays_structural(self) -> None:
        parent = self._record(
            "parent",
            failed="LOW_SHARPE",
            sharpe=1.18,
            fitness=1.1,
        )
        stage = self._build(
            (parent,),
            parent,
            targets=(
                self._target(
                    "LOW_SHARPE",
                    details=True,
                    threshold=1.25,
                    actual=1.18,
                    normalized_gap=0.056,
                    gap_state="available",
                ),
            ),
        )

        self.assertEqual(stage.stage, STRUCTURAL_EVOLUTION_STAGE)
        self.assertIsNone(stage.proximity)

    def test_current_platform_values_missing_never_use_historical_fallback(
        self,
    ) -> None:
        parent = self._record(
            "parent",
            failed="LOW_SHARPE",
            sharpe=0.96,
            fitness=1.1,
        )
        cohort = (
            parent,
            *(
                self._record(
                    f"fail-{index}",
                    failed="LOW_SHARPE",
                    sharpe=0.8 + index / 100,
                    fitness=1.1,
                )
                for index in range(4)
            ),
            *(
                self._record(
                    f"pass-{index}",
                    failed=None,
                    sharpe=1.0 + index / 100,
                    fitness=1.1,
                )
                for index in range(5)
            ),
        )

        stage = self._build(
            cohort,
            parent,
            targets=(
                self._target(
                    "LOW_SHARPE",
                    details=True,
                    gap_state="platform_values_missing",
                ),
            ),
        )

        self.assertEqual(stage.stage, STRUCTURAL_EVOLUTION_STAGE)
        self.assertIsNone(stage.proximity)

    def test_historical_cohort_can_supply_a_strict_observed_pass_floor(self) -> None:
        parent = self._record(
            "parent",
            failed="LOW_SHARPE",
            sharpe=0.96,
            fitness=1.1,
        )
        cohort = (
            parent,
            *(
                self._record(
                    f"fail-{index}",
                    failed="LOW_SHARPE",
                    sharpe=0.8 + index / 100,
                    fitness=1.1,
                )
                for index in range(4)
            ),
            *(
                self._record(
                    f"pass-{index}", failed=None, sharpe=1.0 + index / 100, fitness=1.1
                )
                for index in range(5)
            ),
        )

        stage = self._build(
            cohort,
            parent,
            targets=(self._target("LOW_SHARPE"),),
        )

        self.assertEqual(stage.stage, LOCAL_POLISHING_STAGE)
        self.assertEqual(
            stage.proximity.source,
            "cohort_observed_pass_floor",
        )
        self.assertEqual(stage.proximity.reference, 1.0)
        self.assertAlmostEqual(stage.proximity.normalized_gap, 0.04)

    def test_historical_cohort_above_boundary_stays_structural(self) -> None:
        parent = self._record(
            "parent",
            failed="LOW_SHARPE",
            sharpe=0.94,
            fitness=1.1,
        )
        cohort = (
            parent,
            *(
                self._record(
                    f"fail-{index}",
                    failed="LOW_SHARPE",
                    sharpe=0.8 + index / 100,
                    fitness=1.1,
                )
                for index in range(4)
            ),
            *(
                self._record(
                    f"pass-{index}",
                    failed=None,
                    sharpe=1.0 + index / 100,
                    fitness=1.1,
                )
                for index in range(5)
            ),
        )

        stage = self._build(
            cohort,
            parent,
            targets=(self._target("LOW_SHARPE"),),
        )

        self.assertEqual(stage.stage, STRUCTURAL_EVOLUTION_STAGE)
        self.assertIsNone(stage.proximity)

    def test_historical_cohort_never_crosses_account_or_settings(self) -> None:
        parent = self._record(
            "parent",
            failed="LOW_SHARPE",
            sharpe=0.96,
            fitness=1.1,
        )
        for dimension, identity in (
            ("account", {"account_scope": "other-account"}),
            ("settings", {"settings_key": "settings-b"}),
        ):
            foreign_records = tuple(
                self._record(
                    f"{dimension}-fail-{index}",
                    failed="LOW_SHARPE",
                    sharpe=0.8 + index / 100,
                    fitness=1.1,
                    **identity,
                )
                for index in range(5)
            ) + tuple(
                self._record(
                    f"{dimension}-pass-{index}",
                    failed=None,
                    sharpe=1.0 + index / 100,
                    fitness=1.1,
                    **identity,
                )
                for index in range(5)
            )

            with self.subTest(dimension=dimension):
                stage = self._build(
                    (parent, *foreign_records),
                    parent,
                    targets=(self._target("LOW_SHARPE"),),
                )

                self.assertEqual(stage.stage, STRUCTURAL_EVOLUTION_STAGE)
                self.assertIsNone(stage.proximity)

    def test_historical_cohort_with_missing_metric_fails_closed(self) -> None:
        parent = self._record(
            "parent",
            failed="LOW_SHARPE",
            sharpe=0.96,
            fitness=1.1,
        )
        cohort = (
            parent,
            *(
                self._record(
                    f"fail-{index}",
                    failed="LOW_SHARPE",
                    sharpe=0.8 + index / 100,
                    fitness=1.1,
                )
                for index in range(4)
            ),
            *(
                self._record(
                    f"pass-{index}",
                    failed=None,
                    sharpe=1.0 + index / 100,
                    fitness=1.1,
                )
                for index in range(5)
            ),
        )
        cases = (
            ("parent", 0),
            ("failed", 1),
            ("passed", 5),
        )

        for name, missing_index in cases:
            records = list(cohort)
            records[missing_index] = replace(records[missing_index], sharpe=None)
            selected_parent = records[0]
            with self.subTest(name=name):
                stage = self._build(
                    tuple(records),
                    selected_parent,
                    targets=(self._target("LOW_SHARPE"),),
                )

                self.assertEqual(stage.stage, STRUCTURAL_EVOLUTION_STAGE)
                self.assertIsNone(stage.proximity)

    def test_cohort_fails_closed_when_evidence_is_insufficient_or_overlaps(
        self,
    ) -> None:
        parent = self._record(
            "parent",
            failed="LOW_SHARPE",
            sharpe=0.96,
            fitness=1.1,
        )
        insufficient = (
            parent,
            *(
                self._record(
                    f"fail-{index}", failed="LOW_SHARPE", sharpe=0.8, fitness=1.1
                )
                for index in range(4)
            ),
            *(
                self._record(f"pass-{index}", failed=None, sharpe=1.0, fitness=1.1)
                for index in range(4)
            ),
        )
        overlapping = insufficient + (
            self._record("pass-extra", failed=None, sharpe=0.9, fitness=1.1),
        )

        for name, cohort in (
            ("insufficient", insufficient),
            ("overlapping", overlapping),
        ):
            with self.subTest(name=name):
                stage = self._build(
                    cohort,
                    parent,
                    targets=(self._target("LOW_SHARPE"),),
                )
                self.assertEqual(stage.stage, STRUCTURAL_EVOLUTION_STAGE)

    def test_two_failed_targets_never_enter_local_polishing(self) -> None:
        parent = self._record(
            "parent",
            failed="LOW_SHARPE",
            additional_failed=("LOW_FITNESS",),
            sharpe=1.24,
            fitness=0.99,
        )
        stage = self._build(
            (parent,),
            parent,
            targets=(
                self._target(
                    "LOW_SHARPE",
                    details=True,
                    threshold=1.25,
                    actual=1.24,
                    normalized_gap=0.008,
                    gap_state="available",
                ),
                self._target(
                    "LOW_FITNESS",
                    details=True,
                    threshold=1.0,
                    actual=0.99,
                    normalized_gap=0.01,
                    gap_state="available",
                ),
            ),
        )

        self.assertEqual(stage.stage, STRUCTURAL_EVOLUTION_STAGE)

    def test_low_sub_universe_target_always_uses_structural_evolution(self) -> None:
        parent = self._record(
            "parent-low-sub",
            failed="LOW_SUB_UNIVERSE_SHARPE",
            sharpe=1.3,
            fitness=1.1,
        )

        stage = self._build(
            (parent,),
            parent,
            targets=(
                self._target(
                    "LOW_SUB_UNIVERSE_SHARPE",
                    details=True,
                    threshold=0.36,
                    actual=0.357,
                    normalized_gap=(0.36 - 0.357) / 0.36,
                    gap_state="available",
                ),
            ),
        )

        self.assertEqual(stage.stage, STRUCTURAL_EVOLUTION_STAGE)
        self.assertIsNone(stage.proximity)

    def test_fully_passing_parent_enters_qualified_evolution_stage(self) -> None:
        parent = self._record(
            "qualified",
            failed=None,
            sharpe=1.3,
            fitness=1.1,
        )

        stage = build_parent_improvement_stages(
            LearningEvidenceSet(records=(parent,), settings=()),
            ParentOptimizationTargetSet(
                records=(
                    ParentOptimizationTargets(
                        parent_task_id=parent.task_id,
                        account_scope=parent.account_scope,
                        settings_key=parent.settings_key,
                        targets=(),
                    ),
                )
            ),
            (parent.task_id,),
            qualified_parent_task_ids=(parent.task_id,),
        ).records[0]

        self.assertEqual(stage.stage, QUALIFIED_EVOLUTION_STAGE)
        self.assertIsNone(stage.proximity)

    def _build(
        self,
        records: tuple[LearningEvidenceRecord, ...],
        parent: LearningEvidenceRecord,
        *,
        targets: tuple[OptimizationTarget, ...],
    ):
        return build_parent_improvement_stages(
            LearningEvidenceSet(records=records, settings=()),
            ParentOptimizationTargetSet(
                records=(
                    ParentOptimizationTargets(
                        parent_task_id=parent.task_id,
                        account_scope=parent.account_scope,
                        settings_key=parent.settings_key,
                        targets=targets,
                    ),
                )
            ),
            (parent.task_id,),
        ).records[0]

    @staticmethod
    def _target(
        check_name: str,
        *,
        details: bool = False,
        threshold: float | None = None,
        actual: float | None = None,
        normalized_gap: float | None = None,
        gap_state: str = "historical_not_captured",
    ) -> OptimizationTarget:
        return OptimizationTarget(
            check_name=check_name,
            details_captured=details,
            threshold=threshold,
            actual=actual,
            platform_date=None,
            normalized_gap=normalized_gap,
            gap_state=gap_state,
            actions=(),
        )

    @staticmethod
    def _record(
        task_id: str,
        *,
        failed: str | None,
        sharpe: float,
        fitness: float,
        additional_failed: tuple[str, ...] = (),
        account_scope: str = "group-account",
        settings_key: str = "settings-a",
    ) -> LearningEvidenceRecord:
        basic = {
            "LOW_FITNESS",
            "LOW_SHARPE",
            "LOW_SUB_UNIVERSE_SHARPE",
        }
        failed_checks = ({failed} if failed is not None else set()) | set(
            additional_failed
        )
        passed_checks = basic - failed_checks
        return LearningEvidenceRecord(
            task_id=task_id,
            account_scope=account_scope,
            formula=f"rank({task_id.replace('-', '_')})",
            formula_fingerprint=f"fingerprint-{task_id}",
            settings_key=settings_key,
            finished_at=f"2026-09-01T00:00:{len(task_id):02d}+00:00",
            check_details_captured=False,
            checks=(),
            outcome="failed" if failed_checks else "passed",
            passed_checks=tuple(sorted(passed_checks)),
            failed_checks=tuple(sorted(failed_checks)),
            failure_categories=("signal_quality",) if failed_checks else (),
            pending_checks=(),
            missing_checks=(),
            unexpected_checks=(),
            check_set_complete=True,
            non_sc_check_set_complete=True,
            sharpe=sharpe,
            fitness=fitness,
            turnover=0.1,
            returns=0.02,
            drawdown=0.1,
            margin=0.001,
            book_size=None,
            pnl=None,
        )


if __name__ == "__main__":
    unittest.main()
