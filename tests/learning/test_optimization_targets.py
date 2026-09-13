from __future__ import annotations

import json
import sys
import unittest
from hashlib import sha256
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from learning.evidence import (
    build_learning_evidence,
    build_mutation_learning_evidence,
)
from learning.optimization_targets import build_parent_optimization_targets
from persistence.backtests import (
    BacktestCheckRecord,
    BacktestMutationRecord,
    BacktestResultRecord,
    BacktestSnapshot,
    BacktestTaskRecord,
)
from worldquant.backtests import STANDARD_REGULAR_CHECK_NAMES


class ParentOptimizationTargetTests(unittest.TestCase):
    def test_historical_failure_becomes_status_target_without_guessed_values(
        self,
    ) -> None:
        parent = self._snapshot(
            "parent",
            checks={
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_FITNESS": "FAIL",
                "LOW_SHARPE": "PASS",
                "SELF_CORRELATION": "PENDING",
            },
            details_captured=False,
            details={
                "LOW_FITNESS": (1.0, 0.8, "2026-08-31"),
            },
        )
        evidence = build_learning_evidence((parent,))

        built = build_parent_optimization_targets(
            evidence,
            build_mutation_learning_evidence(evidence, ()),
            (parent.task.task_id,),
        )

        self.assertEqual(len(built.records), 1)
        self.assertEqual(len(built.records[0].targets), 1)
        target = built.records[0].targets[0]
        self.assertEqual(target.check_name, "LOW_FITNESS")
        self.assertFalse(target.details_captured)
        self.assertIsNone(target.threshold)
        self.assertIsNone(target.actual)
        self.assertIsNone(target.platform_date)
        self.assertIsNone(target.normalized_gap)
        self.assertEqual(target.gap_state, "historical_not_captured")
        self.assertEqual(target.actions, ())

    def test_non_failure_statuses_do_not_become_targets(self) -> None:
        for status in ("PASS", "PENDING", "ERROR", "WARNING"):
            with self.subTest(status=status):
                parent = self._snapshot(
                    f"parent-{status.lower()}",
                    checks={
                        "LOW_FITNESS": status,
                        "LOW_SHARPE": "PASS",
                    },
                )
                evidence = build_learning_evidence((parent,))

                targets = (
                    build_parent_optimization_targets(
                        evidence,
                        build_mutation_learning_evidence(evidence, ()),
                        (parent.task.task_id,),
                    )
                    .records[0]
                    .targets
                )

                self.assertEqual(targets, ())

    def test_captured_gap_requires_values_consistent_with_platform_failure(
        self,
    ) -> None:
        parent = self._snapshot(
            "parent",
            checks={
                "LOW_FITNESS": "FAIL",
                "LOW_SHARPE": "FAIL",
                "LOW_SUB_UNIVERSE_SHARPE": "FAIL",
            },
            details_captured=True,
            details={
                "LOW_FITNESS": (1.0, 1.1, None),
                "LOW_SHARPE": (1.25, 1.0, "2026-08-31"),
                "LOW_SUB_UNIVERSE_SHARPE": (0.36, 0.25, "2026-08-31"),
            },
        )
        evidence = build_learning_evidence((parent,))

        targets = (
            build_parent_optimization_targets(
                evidence,
                build_mutation_learning_evidence(evidence, ()),
                (parent.task.task_id,),
            )
            .records[0]
            .targets
        )

        by_name = {target.check_name: target for target in targets}
        self.assertEqual(
            by_name["LOW_FITNESS"].gap_state,
            "status_value_conflict",
        )
        self.assertIsNone(by_name["LOW_FITNESS"].normalized_gap)
        self.assertEqual(by_name["LOW_SHARPE"].gap_state, "available")
        self.assertAlmostEqual(by_name["LOW_SHARPE"].normalized_gap, 0.2)
        self.assertEqual(by_name["LOW_SHARPE"].platform_date, "2026-08-31")
        self.assertEqual(
            by_name["LOW_SUB_UNIVERSE_SHARPE"].gap_state,
            "available",
        )
        self.assertAlmostEqual(
            by_name["LOW_SUB_UNIVERSE_SHARPE"].normalized_gap,
            (0.36 - 0.25) / 0.36,
        )

    def test_action_evidence_uses_only_direct_same_scope_comparisons(self) -> None:
        parent = self._snapshot(
            "parent",
            checks={
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_FITNESS": "FAIL",
                "LOW_SHARPE": "PASS",
                "SELF_CORRELATION": "PENDING",
            },
        )
        safe = self._snapshot(
            "safe",
            checks={
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_FITNESS": "PASS",
                "LOW_SHARPE": "PASS",
                "SELF_CORRELATION": "PASS",
            },
        )
        risky = self._snapshot(
            "risky",
            checks={
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_FITNESS": "PASS",
                "LOW_SHARPE": "FAIL",
                "SELF_CORRELATION": "PENDING",
            },
        )
        unresolved = self._snapshot(
            "unresolved",
            checks={
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_FITNESS": "PENDING",
                "LOW_SHARPE": "PASS",
                "SELF_CORRELATION": "PENDING",
            },
        )
        missing_check = self._snapshot(
            "missing-check",
            checks={
                "LOW_FITNESS": "PASS",
                "LOW_SHARPE": "PASS",
                "SELF_CORRELATION": "PENDING",
            },
            complete_checks=False,
        )
        other_settings = self._snapshot(
            "other-settings",
            settings={"delay": 0},
            checks={"LOW_FITNESS": "PASS", "LOW_SHARPE": "PASS"},
        )
        other_account = self._snapshot(
            "other-account",
            account_scope="other-account",
            checks={"LOW_FITNESS": "PASS", "LOW_SHARPE": "PASS"},
        )
        grandchild = self._snapshot(
            "grandchild",
            checks={"LOW_FITNESS": "PASS", "LOW_SHARPE": "PASS"},
        )
        snapshots = (
            parent,
            safe,
            risky,
            unresolved,
            missing_check,
            other_settings,
            other_account,
            grandchild,
        )
        mutations = (
            self._mutation(parent, safe, "field_swap"),
            self._mutation(parent, risky, "field_swap"),
            self._mutation(parent, unresolved, "window_mutation"),
            self._mutation(parent, missing_check, "operator_replace"),
            self._mutation(parent, other_settings, "operator_replace"),
            self._mutation(parent, other_account, "binary_operator_flip"),
            self._mutation(safe, grandchild, "operator_wrap"),
        )
        evidence = build_learning_evidence(snapshots)

        target = (
            build_parent_optimization_targets(
                evidence,
                build_mutation_learning_evidence(evidence, mutations),
                (parent.task.task_id,),
            )
            .records[0]
            .targets[0]
        )

        actions = {item.action: item for item in target.actions}
        self.assertEqual(
            set(actions),
            {"field_swap", "window_mutation", "operator_replace"},
        )
        self.assertEqual(actions["field_swap"].comparable_count, 2)
        self.assertEqual(actions["field_swap"].repaired_count, 2)
        self.assertEqual(actions["field_swap"].safe_repaired_count, 1)
        self.assertEqual(actions["window_mutation"].unresolved_count, 1)
        self.assertEqual(actions["operator_replace"].repaired_count, 1)
        self.assertEqual(actions["operator_replace"].safe_repaired_count, 0)

    def test_pending_self_correlation_is_not_a_safe_repair(self) -> None:
        parent = self._snapshot(
            "parent-pending-sc",
            checks={
                "LOW_FITNESS": "FAIL",
                "LOW_SHARPE": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "SELF_CORRELATION": "PENDING",
            },
        )
        child = self._snapshot(
            "child-pending-sc",
            checks={
                "LOW_FITNESS": "PASS",
                "LOW_SHARPE": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "SELF_CORRELATION": "PENDING",
            },
        )
        evidence = build_learning_evidence((parent, child))

        action = (
            build_parent_optimization_targets(
                evidence,
                build_mutation_learning_evidence(
                    evidence,
                    (self._mutation(parent, child, "structural"),),
                ),
                (parent.task.task_id,),
            )
            .records[0]
            .targets[0]
            .actions[0]
        )

        self.assertEqual(action.repaired_count, 1)
        self.assertEqual(action.safe_repaired_count, 0)

    def test_missing_self_correlation_is_not_a_safe_repair(self) -> None:
        parent = self._snapshot(
            "parent-missing-sc",
            checks={
                "LOW_FITNESS": "FAIL",
                "LOW_SHARPE": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "SELF_CORRELATION": "PENDING",
            },
        )
        child = self._snapshot(
            "child-missing-sc",
            checks={
                "LOW_FITNESS": "PASS",
                "LOW_SHARPE": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_TURNOVER": "PASS",
                "HIGH_TURNOVER": "PASS",
                "MATCHES_COMPETITION": "PASS",
            },
            complete_checks=False,
        )
        evidence = build_learning_evidence((parent, child))

        action = (
            build_parent_optimization_targets(
                evidence,
                build_mutation_learning_evidence(
                    evidence,
                    (self._mutation(parent, child, "structural"),),
                ),
                (parent.task.task_id,),
            )
            .records[0]
            .targets[0]
            .actions[0]
        )

        self.assertEqual(action.repaired_count, 1)
        self.assertEqual(action.safe_repaired_count, 0)

    def test_shared_incomplete_check_set_is_not_a_safe_repair(self) -> None:
        parent = self._snapshot(
            "parent-shared-missing",
            checks={
                "LOW_FITNESS": "FAIL",
                "LOW_SHARPE": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
            },
            complete_checks=False,
        )
        child = self._snapshot(
            "child-shared-missing",
            checks={
                "LOW_FITNESS": "PASS",
                "LOW_SHARPE": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
            },
            complete_checks=False,
        )
        evidence = build_learning_evidence((parent, child))

        target = (
            build_parent_optimization_targets(
                evidence,
                build_mutation_learning_evidence(
                    evidence,
                    (self._mutation(parent, child, "structural"),),
                ),
                (parent.task.task_id,),
            )
            .records[0]
            .targets[0]
        )

        self.assertEqual(target.actions[0].repaired_count, 1)
        self.assertEqual(target.actions[0].safe_repaired_count, 0)

    @staticmethod
    def _mutation(
        parent: BacktestSnapshot,
        child: BacktestSnapshot,
        action: str,
    ) -> BacktestMutationRecord:
        return BacktestMutationRecord(
            child_task_id=child.task.task_id,
            parent_task_id=parent.task.task_id,
            action=action,
            location="formula.arguments[0]",
            before="close",
            after="open",
        )

    @staticmethod
    def _snapshot(
        name: str,
        *,
        checks: dict[str, str],
        details_captured: bool = True,
        details: dict[str, tuple[float | None, float | None, str | None]] | None = None,
        settings: dict[str, object] | None = None,
        account_scope: str = "group-account",
        complete_checks: bool = True,
    ) -> BacktestSnapshot:
        formula = f"rank({name.replace('-', '_')})"
        settings_json = json.dumps(
            settings or {"delay": 1},
            sort_keys=True,
            separators=(",", ":"),
        )
        identity = sha256(f"{formula}|{settings_json}".encode("utf-8")).hexdigest()
        task_id = f"backtest_{identity}"
        task = BacktestTaskRecord(
            task_id=task_id,
            account_scope=account_scope,
            formula=formula,
            formula_fingerprint=identity,
            settings_json=settings_json,
            request_fingerprint=identity,
            status="completed",
            remote_id=f"simulation_{identity}",
            platform_alpha_id=f"alpha_{identity}",
            created_at="2026-08-31T00:00:00+00:00",
            submission_started_at="2026-08-31T00:01:00+00:00",
            last_observed_at="2026-08-31T00:02:00+00:00",
            retry_not_before=None,
            finished_at="2026-08-31T00:02:00+00:00",
            failure_code=None,
            failure_message=None,
        )
        check_details = details or {}
        recorded_checks = (
            {name: "PASS" for name in STANDARD_REGULAR_CHECK_NAMES}
            if complete_checks
            else {}
        )
        recorded_checks.update(checks)
        return BacktestSnapshot(
            task=task,
            result=BacktestResultRecord(
                task_id=task_id,
                sharpe=1.0,
                fitness=0.8,
                turnover=0.1,
                returns=0.05,
                drawdown=0.1,
                margin=0.001,
                book_size=20_000_000,
                pnl=100_000,
                long_count=None,
                short_count=None,
                check_details_captured=details_captured,
                checks=tuple(
                    BacktestCheckRecord(
                        name=check_name,
                        status=status,
                        threshold=check_details.get(
                            check_name,
                            (None, None, None),
                        )[0],
                        actual=check_details.get(
                            check_name,
                            (None, None, None),
                        )[1],
                        platform_date=check_details.get(
                            check_name,
                            (None, None, None),
                        )[2],
                    )
                    for check_name, status in sorted(recorded_checks.items())
                ),
            ),
            yearly_stats=(),
        )


if __name__ == "__main__":
    unittest.main()
