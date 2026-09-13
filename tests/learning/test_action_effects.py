from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from learning.action_effects import (
    ACTION_EFFECT_CONFLICT,
    ACTION_EFFECT_SAFE_PROGRESS,
    ACTION_EFFECT_UNRESOLVED,
    ACTION_STATE_DEPRIORITIZED,
    ACTION_STATE_INSUFFICIENT,
    ACTION_STATE_MIXED,
    ACTION_STATE_PREFERRED,
    ACTION_STRATEGY_EXPLOITATION,
    DefectActionRequest,
    ExplicitSelfCorrelationEvidence,
    TaskRunEvidence,
    build_defect_action_strategies,
)
from learning.evidence import (
    LearningEvidenceRecord,
    LearningEvidenceSet,
    build_mutation_learning_evidence,
)
from persistence.backtests import BacktestCheckRecord, BacktestMutationRecord
from worldquant.backtests import STANDARD_REGULAR_CHECK_NAMES


_RUN_DIVERSE_ASSIGNMENTS = (
    ("root-0", "run-1"),
    ("root-0", "run-2"),
    ("root-1", "run-1"),
    ("root-1", "run-3"),
    ("root-2", "run-2"),
    ("root-2", "run-3"),
    ("root-3", "run-1"),
    ("root-3", "run-2"),
    ("root-4", "run-2"),
    ("root-4", "run-3"),
)


class DefectActionEffectTests(unittest.TestCase):
    def test_direction_seed_owns_later_repairs_without_learning_from_the_reversal(self):
        negative = self._record(
            "negative", defects=("LOW_FITNESS", "LOW_SHARPE"), fitness=-1.1, sharpe=-1.4
        )
        seed = self._record(
            "positive", defects=("LOW_FITNESS",), fitness=0.8, sharpe=1.4
        )
        child = self._record(
            "repair", defects=("LOW_FITNESS",), fitness=0.9, sharpe=1.4
        )
        evidence = LearningEvidenceSet(records=(negative, seed, child), settings=())
        mutations = (
            BacktestMutationRecord(
                child_task_id="positive",
                parent_task_id="negative",
                action="direction_reversal",
                location="formula",
                before="negative",
                after="positive",
            ),
            BacktestMutationRecord(
                child_task_id="repair",
                parent_task_id="positive",
                action="distribution_stabilization",
                location="formula",
                before="positive",
                after="repair",
            ),
        )
        strategies = build_defect_action_strategies(
            evidence,
            build_mutation_learning_evidence(evidence, mutations),
            (
                DefectActionRequest(
                    parent_task_id="positive",
                    target_check_name="LOW_FITNESS",
                    candidate_actions=("distribution_stabilization",),
                ),
            ),
            seed_root_task_ids=("positive",),
            explicit_self_correlations=(),
            task_runs=(
                TaskRunEvidence(task_id="positive", run_id="direction-run"),
                TaskRunEvidence(task_id="repair", run_id="repair-run"),
            ),
        )
        self.assertEqual(len(strategies.observations), 1)
        observation = strategies.observations[0]
        self.assertEqual(observation.root_task_id, "positive")
        self.assertEqual(observation.child_task_id, "repair")
        self.assertEqual(observation.outcome, ACTION_EFFECT_UNRESOLVED)
        self.assertEqual(strategies.records[0].preferred_actions, ())

    def test_cross_lineage_safe_progress_becomes_preferred(self) -> None:
        strategy = self._strategy(
            outcomes=("safe",) * 10,
            assignments=_RUN_DIVERSE_ASSIGNMENTS,
        )

        field_swap = self._action(strategy, "field_swap")
        self.assertEqual(field_swap.state, ACTION_STATE_PREFERRED)
        self.assertEqual(field_swap.resolved_count, 10)
        self.assertEqual(field_swap.root_lineage_count, 5)
        self.assertEqual(field_swap.run_count, 3)
        self.assertEqual(strategy.mode, ACTION_STRATEGY_EXPLOITATION)
        self.assertEqual(strategy.preferred_actions, ("field_swap",))

    def test_sample_and_lineage_minimums_remain_exploration(self) -> None:
        scenarios = {
            "nine conclusions": _RUN_DIVERSE_ASSIGNMENTS[:9],
            "four roots": tuple(
                (f"root-{index % 4}", "run-1") for index in range(10)
            ),
        }
        for name, assignments in scenarios.items():
            with self.subTest(name=name):
                strategy = self._strategy(
                    outcomes=("safe",) * len(assignments),
                    assignments=assignments,
                )
                self.assertEqual(
                    self._action(strategy, "field_swap").state,
                    ACTION_STATE_INSUFFICIENT,
                )
                self.assertEqual(strategy.preferred_actions, ())
                self.assertEqual(strategy.reason, "insufficient_lineage_evidence")

    def test_run_partition_does_not_change_learning_decisions(self) -> None:
        for outcomes, expected in (
            (("safe",) * 10, ACTION_STATE_PREFERRED),
            (("no", "conflict") * 5, ACTION_STATE_DEPRIORITIZED),
            (("safe", "no") * 5, ACTION_STATE_MIXED),
            (("safe",) * 9 + ("unresolved",), ACTION_STATE_INSUFFICIENT),
        ):
            baseline = self._strategy(
                outcomes=outcomes, assignments=_RUN_DIVERSE_ASSIGNMENTS
            )
            for run_count in (1, 2, 10):
                with self.subTest(expected=expected, runs=run_count):
                    strategy = self._strategy(
                        outcomes=outcomes,
                        assignments=tuple(
                            (root, f"run-{index % run_count}")
                            for index, (root, _) in enumerate(_RUN_DIVERSE_ASSIGNMENTS)
                        ),
                    )
                    action = self._action(strategy, "field_swap")
                    self.assertEqual(action.state, expected)
                    self.assertEqual(strategy.mode, baseline.mode)
                    self.assertEqual(strategy.reason, baseline.reason)
                    self.assertEqual(
                        strategy.preferred_actions, baseline.preferred_actions
                    )
                    self.assertEqual(
                        strategy.deprioritized_actions, baseline.deprioritized_actions
                    )
                    self.assertEqual(
                        action.resolved_count,
                        self._action(baseline, "field_swap").resolved_count,
                    )
                    self.assertEqual(action.root_lineage_count, 5)
                    self.assertEqual(
                        action.run_count, min(run_count, action.resolved_count)
                    )

    def test_single_root_run_evidence_is_not_discarded(
        self,
    ) -> None:
        for outcome, expected in (
            ("safe", ACTION_STATE_PREFERRED),
            ("no", ACTION_STATE_MIXED),
            ("conflict", ACTION_STATE_MIXED),
        ):
            with self.subTest(outcome=outcome):
                strategy = self._strategy(
                    outcomes=("safe",) * 10 + (outcome,),
                    assignments=(*_RUN_DIVERSE_ASSIGNMENTS, ("root-5", "run-4")),
                )
                field_swap = self._action(strategy, "field_swap")
                self.assertEqual(field_swap.state, expected)
                self.assertEqual(field_swap.resolved_count, 11)
                self.assertEqual(field_swap.root_lineage_count, 6)
                self.assertEqual(field_swap.run_count, 4)

    def test_prolific_positive_root_cannot_outweigh_other_roots(self) -> None:
        strategy = self._strategy(
            outcomes=("safe",) * 100 + ("no", "conflict") * 4,
            assignments=(("root-0", "run-1"),) * 100
            + tuple((f"root-{index // 2 + 1}", "run-1") for index in range(8)),
        )
        action = self._action(strategy, "field_swap")
        self.assertEqual(action.safe_progress_count, 100)
        self.assertEqual(action.resolved_count, 108)
        self.assertEqual(action.root_lineage_count, 5)
        self.assertEqual(action.state, ACTION_STATE_MIXED)
        self.assertEqual(strategy.preferred_actions, ())

    def test_each_root_requires_a_strict_positive_majority(self) -> None:
        for root_outcomes, expected in (
            (("safe", "safe", "no"), ACTION_STATE_PREFERRED),
            (("safe", "no", "conflict"), ACTION_STATE_MIXED),
        ):
            with self.subTest(expected=expected):
                strategy = self._strategy(
                    outcomes=root_outcomes * 5,
                    assignments=tuple(
                        (f"root-{index // 3}", "run-1") for index in range(15)
                    ),
                )
                self.assertEqual(
                    self._action(strategy, "field_swap").state, expected
                )

    def test_unresolved_observations_do_not_supply_resolved_roots(self) -> None:
        assignments = tuple(
            (f"root-{index % 4}", "run-1") for index in range(10)
        ) + (("root-4", "run-1"),) * 2
        for outcomes, count, roots in (
            (("safe",) * 10 + ("unresolved",) * 2, 10, 4),
            (("unresolved",) * 12, 0, 0),
        ):
            with self.subTest(resolved=count):
                strategy = self._strategy(outcomes=outcomes, assignments=assignments)
                action = self._action(strategy, "field_swap")
                self.assertEqual(action.resolved_count, count)
                self.assertEqual(action.root_lineage_count, roots)
                self.assertEqual(action.unresolved_count, 12 - count)
                self.assertEqual(action.state, ACTION_STATE_INSUFFICIENT)

    def test_lineage_tie_is_mixed_and_clear_failure_is_deprioritized(
        self,
    ) -> None:
        mixed_outcomes = tuple(
            "no" if run_id == "run-3" else "safe"
            for _root_id, run_id in _RUN_DIVERSE_ASSIGNMENTS
        )
        mixed = self._strategy(
            outcomes=mixed_outcomes,
            assignments=_RUN_DIVERSE_ASSIGNMENTS,
        )
        negative = self._strategy(
            outcomes=("no",) * 10,
            assignments=_RUN_DIVERSE_ASSIGNMENTS,
        )

        self.assertEqual(
            self._action(mixed, "field_swap").state,
            ACTION_STATE_MIXED,
        )
        self.assertEqual(
            self._action(negative, "field_swap").state,
            ACTION_STATE_DEPRIORITIZED,
        )
        self.assertEqual(negative.deprioritized_actions, ("field_swap",))

    def test_formal_self_correlation_resolves_only_the_matching_child(
        self,
    ) -> None:
        built = self._build(
            outcomes=("safe", "conflict", "formal-other-failure", "unresolved"),
            assignments=(
                ("root-0", "run-1"),
                ("root-0", "run-1"),
                ("root-0", "run-1"),
                ("root-0", "run-1"),
            ),
        )

        outcomes = [item.outcome for item in built.observations]
        self.assertEqual(outcomes.count(ACTION_EFFECT_SAFE_PROGRESS), 1)
        self.assertEqual(outcomes.count(ACTION_EFFECT_CONFLICT), 1)
        self.assertEqual(outcomes.count(ACTION_EFFECT_UNRESOLVED), 2)

    def test_missing_low_sub_universe_gap_stays_unresolved(self) -> None:
        built = self._build(
            outcomes=("safe",),
            assignments=(("root-0", "run-1"),),
            defects=("LOW_SUB_UNIVERSE_SHARPE",),
            capture_low_sub_gap=False,
        )

        self.assertEqual(
            built.observations[0].outcome,
            ACTION_EFFECT_UNRESOLVED,
        )

    def test_exact_defect_set_prevents_cross_context_transfer(self) -> None:
        built = self._build(
            outcomes=("safe",) * 10,
            assignments=_RUN_DIVERSE_ASSIGNMENTS,
            defects=("LOW_FITNESS", "LOW_SHARPE"),
            request_defects=("LOW_FITNESS",),
        )

        strategy = built.records[0]
        field_swap = self._action(strategy, "field_swap")
        self.assertEqual(strategy.defect_checks, ("LOW_FITNESS",))
        self.assertEqual(field_swap.resolved_count, 0)
        self.assertEqual(field_swap.state, ACTION_STATE_INSUFFICIENT)

    def test_account_and_full_settings_prevent_cross_context_transfer(self) -> None:
        for name, request_account, request_settings in (
            ("account", "other-account", "settings-a"),
            ("settings", "group-account", "settings-b"),
        ):
            with self.subTest(name=name):
                built = self._build(
                    outcomes=("safe",) * 10,
                    assignments=_RUN_DIVERSE_ASSIGNMENTS,
                    request_account_scope=request_account,
                    request_settings_key=request_settings,
                )
                field_swap = self._action(built.records[0], "field_swap")
                self.assertEqual(field_swap.resolved_count, 0)
                self.assertEqual(field_swap.state, ACTION_STATE_INSUFFICIENT)

    def test_multiple_generations_from_one_seed_remain_one_root_lineage(
        self,
    ) -> None:
        root = self._record(
            "root",
            defects=("LOW_FITNESS",),
            fitness=0.8,
            sharpe=1.1,
        )
        records = [root]
        mutations: list[BacktestMutationRecord] = []
        task_runs: list[TaskRunEvidence] = []
        explicit_sc: list[ExplicitSelfCorrelationEvidence] = []
        parent_id = root.task_id
        for index in range(10):
            child_id = f"generation-{index:02d}"
            records.append(
                self._record(
                    child_id,
                    defects=("LOW_FITNESS",),
                    fitness=0.81 + index * 0.01,
                    sharpe=1.1,
                )
            )
            mutations.append(
                BacktestMutationRecord(
                    child_task_id=child_id,
                    parent_task_id=parent_id,
                    action="field_swap",
                    location="formula.arguments[0]",
                    before=parent_id,
                    after=child_id,
                )
            )
            task_runs.append(
                TaskRunEvidence(task_id=child_id, run_id=f"run-{index % 3}")
            )
            explicit_sc.append(
                ExplicitSelfCorrelationEvidence(
                    task_id=child_id,
                    status="PASS",
                    formal_check_state="passed",
                    observed_at="2026-09-01T00:00:00+00:00",
                )
            )
            parent_id = child_id
        evidence = LearningEvidenceSet(records=tuple(records), settings=())
        built = build_defect_action_strategies(
            evidence,
            build_mutation_learning_evidence(evidence, mutations),
            (
                DefectActionRequest(
                    parent_task_id=root.task_id,
                    target_check_name="LOW_FITNESS",
                    candidate_actions=("field_swap",),
                ),
            ),
            seed_root_task_ids=(root.task_id,),
            task_runs=task_runs,
            explicit_self_correlations=explicit_sc,
        )

        action = built.records[0].actions[0]
        self.assertEqual(action.resolved_count, 10)
        self.assertEqual(action.root_lineage_count, 1)
        self.assertEqual(action.state, ACTION_STATE_INSUFFICIENT)

    def _strategy(self, *, outcomes, assignments):
        return self._build(outcomes=outcomes, assignments=assignments).records[0]

    def _build(
        self,
        *,
        outcomes: tuple[str, ...],
        assignments: tuple[tuple[str, str], ...],
        defects: tuple[str, ...] = ("LOW_FITNESS",),
        request_defects: tuple[str, ...] | None = None,
        request_account_scope: str = "group-account",
        request_settings_key: str = "settings-a",
        capture_low_sub_gap: bool = True,
    ):
        self.assertEqual(len(outcomes), len(assignments))
        records: dict[str, LearningEvidenceRecord] = {}
        mutations: list[BacktestMutationRecord] = []
        task_runs: list[TaskRunEvidence] = []
        explicit_sc: list[ExplicitSelfCorrelationEvidence] = []
        roots = tuple(sorted({root_id for root_id, _run_id in assignments}))
        for index, ((root_id, run_id), outcome) in enumerate(
            zip(assignments, outcomes, strict=True)
        ):
            parent = records.setdefault(
                root_id,
                self._record(
                    root_id,
                    defects=defects,
                    fitness=0.8,
                    sharpe=1.1,
                    capture_low_sub_gap=capture_low_sub_gap,
                    low_sub_actual=0.20,
                ),
            )
            child_id = f"child-{index:02d}"
            if outcome == "no":
                child_fitness = parent.fitness
                child_sharpe = parent.sharpe
            else:
                child_fitness = parent.fitness + 0.1
                child_sharpe = parent.sharpe + 0.1
            records[child_id] = self._record(
                child_id,
                defects=defects,
                fitness=child_fitness,
                sharpe=child_sharpe,
                capture_low_sub_gap=capture_low_sub_gap,
                low_sub_actual=(0.25 if outcome != "no" else 0.20),
            )
            mutations.append(
                BacktestMutationRecord(
                    child_task_id=child_id,
                    parent_task_id=root_id,
                    action="field_swap",
                    location="formula.arguments[0]",
                    before="close",
                    after=f"field_{index}",
                )
            )
            task_runs.append(TaskRunEvidence(task_id=child_id, run_id=run_id))
            if outcome in {"safe", "conflict", "formal-other-failure"}:
                explicit_sc.append(
                    ExplicitSelfCorrelationEvidence(
                        task_id=child_id,
                        status="FAIL" if outcome == "conflict" else "PASS",
                        formal_check_state=(
                            "passed"
                            if outcome == "safe"
                            else "failed"
                        ),
                        observed_at="2026-09-01T00:00:00+00:00",
                    )
                )

        requested_defects = request_defects or defects
        request_parent_id = roots[0]
        if (
            requested_defects != defects
            or request_account_scope != "group-account"
            or request_settings_key != "settings-a"
        ):
            request_parent_id = "request-parent"
            records[request_parent_id] = self._record(
                request_parent_id,
                defects=requested_defects,
                fitness=0.8,
                sharpe=1.1,
                account_scope=request_account_scope,
                settings_key=request_settings_key,
            )
        evidence = LearningEvidenceSet(
            records=tuple(records.values()),
            settings=(),
        )
        mutation_evidence = build_mutation_learning_evidence(evidence, mutations)
        return build_defect_action_strategies(
            evidence,
            mutation_evidence,
            (
                DefectActionRequest(
                    parent_task_id=request_parent_id,
                    target_check_name=requested_defects[0],
                    candidate_actions=("field_swap", "window_mutation"),
                ),
            ),
            seed_root_task_ids=roots,
            task_runs=task_runs,
            explicit_self_correlations=explicit_sc,
        )

    @staticmethod
    def _action(strategy, action: str):
        return next(item for item in strategy.actions if item.action == action)

    @staticmethod
    def _record(
        task_id: str,
        *,
        defects: tuple[str, ...],
        fitness: float,
        sharpe: float,
        capture_low_sub_gap: bool = True,
        low_sub_actual: float = 0.20,
        account_scope: str = "group-account",
        settings_key: str = "settings-a",
    ) -> LearningEvidenceRecord:
        statuses = {name: "PASS" for name in STANDARD_REGULAR_CHECK_NAMES}
        statuses.update({name: "FAIL" for name in defects})
        statuses["SELF_CORRELATION"] = "PENDING"
        checks = tuple(
            BacktestCheckRecord(
                name=name,
                status=status,
                threshold=(
                    0.36
                    if name == "LOW_SUB_UNIVERSE_SHARPE"
                    and capture_low_sub_gap
                    else None
                ),
                actual=(
                    low_sub_actual
                    if name == "LOW_SUB_UNIVERSE_SHARPE"
                    and capture_low_sub_gap
                    else None
                ),
                platform_date="2026-09-01" if capture_low_sub_gap else None,
            )
            for name, status in sorted(statuses.items())
        )
        passed = tuple(sorted(name for name, status in statuses.items() if status == "PASS"))
        failed = tuple(sorted(name for name, status in statuses.items() if status == "FAIL"))
        pending = tuple(
            sorted(name for name, status in statuses.items() if status == "PENDING")
        )
        return LearningEvidenceRecord(
            task_id=task_id,
            account_scope=account_scope,
            formula=task_id,
            formula_fingerprint=f"fingerprint-{task_id}",
            settings_key=settings_key,
            finished_at="2026-08-31T00:00:00+00:00",
            check_details_captured=capture_low_sub_gap,
            checks=checks,
            outcome="failed",
            passed_checks=passed,
            failed_checks=failed,
            failure_categories=("signal_quality",),
            pending_checks=pending,
            missing_checks=(),
            unexpected_checks=(),
            check_set_complete=True,
            non_sc_check_set_complete=True,
            sharpe=sharpe,
            fitness=fitness,
            turnover=0.08,
            returns=0.02,
            drawdown=0.10,
            margin=0.0002,
            book_size=None,
            pnl=None,
        )


if __name__ == "__main__":
    unittest.main()
