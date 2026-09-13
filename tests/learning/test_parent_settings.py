from __future__ import annotations

import json
import sys
import unittest
from hashlib import sha256
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from learning.evidence import build_learning_evidence
from learning.parent_settings import build_parent_settings_evidence
from persistence.backtests import (
    BacktestCheckRecord,
    BacktestMutationRecord,
    BacktestResultRecord,
    BacktestSnapshot,
    BacktestTaskRecord,
)
from worldquant.backtests import STANDARD_REGULAR_CHECK_NAMES


class ParentSettingsEvidenceTests(unittest.TestCase):
    def test_counts_only_children_with_the_parent_account_and_complete_settings(
        self,
    ) -> None:
        parent = self._snapshot(
            "parent",
            account_scope="group-account",
            settings={"delay": 1, "neutralization": "SECTOR"},
            checks={
                "LOW_SHARPE": "PASS",
                "LOW_FITNESS": "FAIL",
                "SELF_CORRELATION": "PENDING",
            },
        )
        same_settings = self._snapshot(
            "same",
            account_scope="group-account",
            settings={"delay": 1, "neutralization": "SECTOR"},
            checks={
                "LOW_SHARPE": "PASS",
                "LOW_FITNESS": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "SELF_CORRELATION": "PASS",
            },
        )
        other_settings = self._snapshot(
            "settings",
            account_scope="group-account",
            settings={"delay": 1, "neutralization": "SUBINDUSTRY"},
            checks={
                "LOW_SHARPE": "PASS",
                "LOW_FITNESS": "FAIL",
                "SELF_CORRELATION": "FAIL",
            },
        )
        other_account = self._snapshot(
            "account",
            account_scope="other-account",
            settings={"delay": 1, "neutralization": "SECTOR"},
            checks={
                "LOW_SHARPE": "PASS",
                "LOW_FITNESS": "PENDING",
                "SELF_CORRELATION": "PASS",
            },
        )
        children = (same_settings, other_settings, other_account)

        built = build_parent_settings_evidence(
            build_learning_evidence((parent, *children)),
            tuple(self._mutation(parent, child) for child in children),
            (parent.task.task_id,),
        )

        self.assertEqual(len(built.records), 1)
        record = built.records[0]
        self.assertEqual(record.parent_task_id, parent.task.task_id)
        self.assertEqual(record.account_scope, "group-account")
        self.assertEqual(
            record.settings_key,
            sha256(parent.task.settings_json.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(record.child_count, 1)
        self.assertEqual(record.child_quality_passed_count, 1)
        self.assertEqual(record.child_quality_failed_count, 0)
        self.assertEqual(record.child_quality_pending_count, 0)
        self.assertEqual(record.sc_passed_count, 1)
        self.assertEqual(record.sc_pending_count, 1)
        self.assertEqual(record.sc_failed_count, 0)
        self.assertFalse(record.sc_risk_available)
        self.assertIsNone(record.sc_smoothed_risk)

    def test_sc_risk_requires_five_explicit_results_and_uses_smoothing(self) -> None:
        parent = self._snapshot(
            "parent",
            account_scope="group-account",
            settings={"neutralization": "SECTOR"},
            checks={"LOW_SHARPE": "PASS", "SELF_CORRELATION": "FAIL"},
        )
        children = tuple(
            self._snapshot(
                f"child-{index}",
                account_scope="group-account",
                settings={"neutralization": "SECTOR"},
                checks={
                    "LOW_SHARPE": "PASS",
                    "SELF_CORRELATION": "FAIL" if index == 0 else "PASS",
                },
            )
            for index in range(4)
        )

        built = build_parent_settings_evidence(
            build_learning_evidence((parent, *children)),
            tuple(self._mutation(parent, child) for child in children),
            (parent.task.task_id,),
        )

        record = built.records[0]
        self.assertEqual(record.sc_explicit_count, 5)
        self.assertEqual(record.sc_failed_count, 2)
        self.assertEqual(record.sc_passed_count, 3)
        self.assertTrue(record.sc_risk_available)
        self.assertAlmostEqual(record.sc_smoothed_risk, 3 / 7)

    def test_incomplete_non_sc_checks_are_counted_as_pending_quality(self) -> None:
        parent = self._snapshot(
            "parent",
            account_scope="group-account",
            settings={"delay": 1},
            checks={"LOW_FITNESS": "FAIL"},
        )
        child = self._snapshot(
            "child",
            account_scope="group-account",
            settings={"delay": 1},
            checks={
                "LOW_SHARPE": "PASS",
                "LOW_FITNESS": "PASS",
                "CONCENTRATED_WEIGHT": "PASS",
                "LOW_SUB_UNIVERSE_SHARPE": "PASS",
            },
            complete_checks=False,
        )

        record = build_parent_settings_evidence(
            build_learning_evidence((parent, child)),
            (self._mutation(parent, child),),
            (parent.task.task_id,),
        ).records[0]

        self.assertEqual(record.child_quality_passed_count, 0)
        self.assertEqual(record.child_quality_pending_count, 1)

    def test_requested_parent_must_have_a_real_result(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "parent_settings_parent_result_missing",
        ):
            build_parent_settings_evidence(
                build_learning_evidence(()),
                (),
                ("missing-parent",),
            )

    @staticmethod
    def _mutation(
        parent: BacktestSnapshot,
        child: BacktestSnapshot,
    ) -> BacktestMutationRecord:
        return BacktestMutationRecord(
            child_task_id=child.task.task_id,
            parent_task_id=parent.task.task_id,
            action="field_swap",
            location="formula.arguments[0]",
            before="close",
            after="open",
        )

    @staticmethod
    def _snapshot(
        name: str,
        *,
        account_scope: str,
        settings: dict[str, object],
        checks: dict[str, str],
        complete_checks: bool = True,
    ) -> BacktestSnapshot:
        formula = f"rank({name.replace('-', '_')})"
        identity = sha256(formula.encode("utf-8")).hexdigest()
        settings_json = json.dumps(
            settings,
            sort_keys=True,
            separators=(",", ":"),
        )
        task_id = f"backtest_{identity}"
        task = BacktestTaskRecord(
            task_id=task_id,
            account_scope=account_scope,
            formula=formula,
            formula_fingerprint=identity,
            settings_json=settings_json,
            request_fingerprint=sha256(
                f"{account_scope}|{formula}|{settings_json}".encode("utf-8")
            ).hexdigest(),
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
                sharpe=1.2,
                fitness=0.9,
                turnover=0.1,
                returns=0.05,
                drawdown=0.1,
                margin=0.001,
                book_size=20_000_000,
                pnl=100_000,
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
