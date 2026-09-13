from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from learning.evidence import (
    build_learning_evidence,
    build_mutation_learning_evidence,
)
from persistence.backtests import (
    BacktestCheckRecord,
    BacktestMutationRecord,
    BacktestResultRecord,
    BacktestSnapshot,
    BacktestTaskRecord,
    BacktestYearlyStatRecord,
)
from worldquant.backtests import STANDARD_REGULAR_CHECK_NAMES


class LearningEvidenceTests(unittest.TestCase):
    def test_position_counts_and_yearly_facts_do_not_enter_learning_evidence(
        self,
    ) -> None:
        baseline = self._snapshot(
            "rank(close)",
            checks={"LOW_SHARPE": "FAIL"},
        )
        assert baseline.result is not None
        enriched = replace(
            baseline,
            result=replace(
                baseline.result,
                long_count=731,
                short_count=694,
            ),
            yearly_stats=(
                BacktestYearlyStatRecord(
                    task_id=baseline.task.task_id,
                    year=2024,
                    pnl=-25_000,
                    book_size=20_000_000,
                    long_count=705,
                    short_count=680,
                    turnover=0.12,
                    sharpe=-0.2,
                    returns=-0.01,
                    drawdown=0.2,
                    margin=-0.0001,
                    fitness=-0.05,
                    stage="IS",
                ),
            ),
        )

        self.assertEqual(
            build_learning_evidence((baseline,)),
            build_learning_evidence((enriched,)),
        )

    def test_single_failed_result_is_recorded_but_not_comparable(self) -> None:
        evidence = build_learning_evidence(
            [
                self._snapshot(
                    "rank(close)",
                    checks={
                        "LOW_SHARPE": "FAIL",
                        "SELF_CORRELATION": "PENDING",
                    },
                )
            ]
        )

        self.assertEqual(len(evidence.records), 1)
        self.assertEqual(evidence.records[0].outcome, "failed")
        self.assertEqual(evidence.records[0].pending_checks, ("SELF_CORRELATION",))
        self.assertEqual(evidence.records[0].failure_categories, ("signal_quality",))
        self.assertFalse(evidence.settings[0].comparison_available)
        self.assertFalse(evidence.settings[0].outcome_contrast_available)

    def test_non_sc_pending_result_has_no_learning_label(self) -> None:
        evidence = build_learning_evidence(
            [
                self._snapshot(
                    "rank(close)",
                    checks={
                        "LOW_SHARPE": "PENDING",
                        "SELF_CORRELATION": "PENDING",
                    },
                )
            ]
        )

        self.assertIsNone(evidence.records[0].outcome)
        self.assertEqual(evidence.settings[0].labeled_count, 0)
        self.assertEqual(evidence.settings[0].pending_count, 1)
        self.assertFalse(evidence.settings[0].comparison_available)

    def test_incomplete_check_set_without_explicit_failure_has_no_label(self) -> None:
        evidence = build_learning_evidence(
            [
                self._snapshot(
                    "rank(close)",
                    checks={"LOW_SHARPE": "PASS"},
                    complete_checks=False,
                )
            ]
        )

        record = evidence.records[0]
        self.assertIsNone(record.outcome)
        self.assertFalse(record.check_set_complete)
        self.assertFalse(record.non_sc_check_set_complete)
        self.assertIn("LOW_FITNESS", record.missing_checks)

    def test_missing_sc_does_not_hide_complete_base_quality(self) -> None:
        checks = {
            name: "PASS"
            for name in STANDARD_REGULAR_CHECK_NAMES
            if name != "SELF_CORRELATION"
        }
        evidence = build_learning_evidence(
            [
                self._snapshot(
                    "rank(close)",
                    checks=checks,
                    complete_checks=False,
                )
            ]
        )

        record = evidence.records[0]
        self.assertEqual(record.outcome, "passed")
        self.assertFalse(record.check_set_complete)
        self.assertTrue(record.non_sc_check_set_complete)

    def test_sc_pending_does_not_hide_passed_base_quality(self) -> None:
        evidence = build_learning_evidence(
            [
                self._snapshot(
                    "rank(close)",
                    checks={
                        "CONCENTRATED_WEIGHT": "PASS",
                        "HIGH_TURNOVER": "PASS",
                        "LOW_FITNESS": "PASS",
                        "LOW_SHARPE": "PASS",
                        "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                        "LOW_TURNOVER": "PASS",
                        "MATCHES_COMPETITION": "PASS",
                        "SELF_CORRELATION": "PENDING",
                    },
                )
            ]
        )

        self.assertEqual(evidence.records[0].outcome, "passed")
        self.assertEqual(evidence.records[0].pending_checks, ("SELF_CORRELATION",))
        self.assertEqual(evidence.settings[0].labeled_count, 1)
        self.assertEqual(evidence.settings[0].passed_count, 1)
        self.assertEqual(evidence.settings[0].pending_count, 0)

    def test_sc_failure_is_separate_from_base_quality_outcome(self) -> None:
        evidence = build_learning_evidence(
            [
                self._snapshot(
                    "rank(close)",
                    checks={
                        "LOW_FITNESS": "PASS",
                        "LOW_SHARPE": "PASS",
                        "SELF_CORRELATION": "FAIL",
                    },
                )
            ]
        )

        self.assertEqual(evidence.records[0].outcome, "passed")
        self.assertEqual(evidence.records[0].failed_checks, ("SELF_CORRELATION",))
        self.assertEqual(evidence.records[0].failure_categories, ())

    def test_only_equal_settings_are_grouped_for_comparison(self) -> None:
        evidence = build_learning_evidence(
            [
                self._snapshot(
                    "rank(close)",
                    checks={"LOW_SHARPE": "FAIL"},
                ),
                self._snapshot(
                    "rank(open)",
                    checks={"LOW_SHARPE": "PASS"},
                    finished_at="2026-08-29T00:04:00+00:00",
                ),
                self._snapshot(
                    "rank(low)",
                    settings={"delay": 0},
                    checks={"LOW_SHARPE": "PASS"},
                    finished_at="2026-08-29T00:05:00+00:00",
                ),
            ]
        )

        summaries = sorted(evidence.settings, key=lambda item: item.total_count)
        self.assertEqual(tuple(item.total_count for item in summaries), (1, 2))
        self.assertFalse(summaries[0].comparison_available)
        self.assertTrue(summaries[1].comparison_available)
        self.assertTrue(summaries[1].outcome_contrast_available)

    def test_duplicate_task_cannot_be_counted_twice(self) -> None:
        snapshot = self._snapshot("rank(close)", checks={"LOW_SHARPE": "FAIL"})
        with self.assertRaisesRegex(ValueError, "learning_task_duplicated"):
            build_learning_evidence([snapshot, snapshot])

    def test_same_formula_in_different_environment_is_grouped_separately(self) -> None:
        sector = self._snapshot(
            "rank(close)",
            settings={"neutralization": "SECTOR"},
            checks={"LOW_SHARPE": "FAIL"},
        )
        subindustry = self._snapshot(
            "rank(close)",
            settings={"neutralization": "SUBINDUSTRY"},
            checks={"LOW_SHARPE": "PASS"},
            finished_at="2026-08-29T00:04:00+00:00",
        )
        subindustry = replace(
            subindustry,
            task=replace(subindustry.task, task_id="different_task"),
            result=replace(subindustry.result, task_id="different_task"),
        )

        evidence = build_learning_evidence((sector, subindustry))

        self.assertEqual(len(evidence.records), 2)
        self.assertEqual(len(evidence.settings), 2)

    def test_same_settings_from_different_accounts_are_not_mixed(self) -> None:
        group = self._snapshot(
            "rank(close)",
            account_scope="group-account",
            checks={"LOW_SHARPE": "FAIL"},
        )
        other = self._snapshot(
            "rank(open)",
            account_scope="other-account",
            checks={"LOW_SHARPE": "PASS"},
        )

        evidence = build_learning_evidence((group, other))

        self.assertEqual(len(evidence.settings), 2)
        self.assertEqual(
            {summary.account_scope for summary in evidence.settings},
            {"group-account", "other-account"},
        )
        self.assertTrue(
            all(summary.total_count == 1 for summary in evidence.settings)
        )

    def test_direct_parent_child_facts_produce_recomputable_deltas(self) -> None:
        parent = self._snapshot(
            "rank(close)",
            checks={
                "LOW_SHARPE": "FAIL",
                "SELF_CORRELATION": "PENDING",
            },
        )
        child = self._snapshot(
            "rank(open)",
            checks={
                "LOW_SHARPE": "PASS",
                "LOW_FITNESS": "FAIL",
                "SELF_CORRELATION": "PENDING",
            },
            finished_at="2026-08-29T00:04:00+00:00",
        )
        assert child.result is not None
        child = replace(
            child,
            result=replace(
                child.result,
                sharpe=0.2,
                fitness=0.05,
                returns=0.01,
                drawdown=0.2,
                margin=0.0001,
            ),
        )
        evidence = build_learning_evidence((parent, child))

        mutation_evidence = build_mutation_learning_evidence(
            evidence,
            (
                BacktestMutationRecord(
                    child_task_id=child.task.task_id,
                    parent_task_id=parent.task.task_id,
                    action="field_swap",
                    location="formula.arguments[0]",
                    before="close",
                    after="open",
                ),
            ),
        )

        record = mutation_evidence.records[0]
        self.assertTrue(record.comparison_available)
        self.assertEqual(record.repaired_checks, ("LOW_SHARPE",))
        self.assertEqual(record.introduced_failed_checks, ("LOW_FITNESS",))
        self.assertEqual(record.introduced_pending_checks, ())
        self.assertEqual(record.child_missing_checks, ())
        self.assertEqual(record.remaining_failed_checks, ())
        self.assertEqual(
            record.parent_pending_checks,
            ("SELF_CORRELATION",),
        )
        self.assertEqual(record.child_pending_checks, ("SELF_CORRELATION",))
        self.assertAlmostEqual(record.sharpe_delta, 0.28)
        self.assertAlmostEqual(record.fitness_delta, 0.08)
        self.assertAlmostEqual(record.drawdown_delta, -0.1539)

    def test_cross_environment_mutation_does_not_claim_metric_improvement(self) -> None:
        parent = self._snapshot(
            "rank(close)",
            settings={"delay": 1, "neutralization": "SECTOR"},
            checks={"LOW_SHARPE": "FAIL"},
        )
        child = self._snapshot(
            "rank(open)",
            settings={"delay": 1, "neutralization": "SUBINDUSTRY"},
            checks={"LOW_SHARPE": "PASS"},
            finished_at="2026-08-29T00:04:00+00:00",
        )
        evidence = build_learning_evidence((parent, child))

        mutation_evidence = build_mutation_learning_evidence(
            evidence,
            (
                BacktestMutationRecord(
                    child_task_id=child.task.task_id,
                    parent_task_id=parent.task.task_id,
                    action="field_swap",
                    location="formula.arguments[0]",
                    before="close",
                    after="open",
                ),
            ),
        )

        record = mutation_evidence.records[0]
        self.assertFalse(record.comparison_available)
        self.assertNotEqual(record.parent_settings_key, record.child_settings_key)
        self.assertEqual(record.repaired_checks, ())
        self.assertIsNone(record.sharpe_delta)
        self.assertIsNone(record.fitness_delta)

    @staticmethod
    def _snapshot(
        formula: str,
        *,
        checks: dict[str, str],
        settings: dict[str, object] | None = None,
        account_scope: str = "group-account",
        finished_at: str = "2026-08-29T00:03:00+00:00",
        complete_checks: bool = True,
    ) -> BacktestSnapshot:
        formula_identity = sha256(formula.encode("utf-8")).hexdigest()
        task_id = f"backtest_{formula_identity}"
        settings_json = json.dumps(
            settings or {"delay": 1},
            sort_keys=True,
            separators=(",", ":"),
        )
        task = BacktestTaskRecord(
            task_id=task_id,
            account_scope=account_scope,
            formula=formula,
            formula_fingerprint=formula_identity,
            settings_json=settings_json,
            request_fingerprint=sha256(
                f"{formula}|{settings_json}".encode("utf-8")
            ).hexdigest(),
            status="completed",
            remote_id=f"remote_{formula_identity}",
            platform_alpha_id=f"alpha_{formula_identity}",
            created_at="2026-08-29T00:00:00+00:00",
            submission_started_at="2026-08-29T00:01:00+00:00",
            last_observed_at=finished_at,
            retry_not_before=None,
            finished_at=finished_at,
            failure_code=None,
            failure_message=None,
        )
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
                sharpe=-0.08 if "FAIL" in recorded_checks.values() else 1.2,
                fitness=-0.03 if "FAIL" in recorded_checks.values() else 1.0,
                turnover=0.0832,
                returns=-0.0136 if "FAIL" in recorded_checks.values() else 0.1,
                drawdown=0.3539,
                margin=(
                    -0.000328 if "FAIL" in recorded_checks.values() else 0.001
                ),
                book_size=20_000_000,
                pnl=(
                    -674_669 if "FAIL" in recorded_checks.values() else 100_000
                ),
                long_count=None,
                short_count=None,
                check_details_captured=True,
                checks=tuple(
                    BacktestCheckRecord(
                        name=name,
                        status=status,
                        threshold=None,
                        actual=None,
                        platform_date=None,
                    )
                    for name, status in sorted(recorded_checks.items())
                ),
            ),
            yearly_stats=(),
        )


if __name__ == "__main__":
    unittest.main()
