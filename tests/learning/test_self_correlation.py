import json
import sys
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from learning.self_correlation import (
    SelfCorrelationCheckEvidence,
    select_self_correlation_reference,
)
from persistence.backtests import (
    BacktestTaskRecord,
    BacktestSnapshot,
    BacktestResultRecord,
    BacktestCheckRecord,
    BacktestYearlyStatRecord,
)
from persistence.submissions import PlatformSubmittedAlphaRecord
from worldquant.backtests import BacktestSettings, STANDARD_REGULAR_CHECK_NAMES


class SelfCorrelationEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.fromisoformat("2026-09-07T10:00:00+00:00")
        self.settings = BacktestSettings(
            "EQUITY",
            "USA",
            "TOP3000",
            1,
            4,
            "SECTOR",
            0.08,
            "ON",
            "VERIFY",
            "OFF",
            "FASTEXPR",
            False,
            "OFF",
            "OFF",
        ).as_platform_dict()
        task = BacktestTaskRecord(
            "parent",
            "account",
            "rank(close)",
            "fp",
            json.dumps(self.settings),
            "request",
            "completed",
            "remote",
            "alpha-parent",
            "2026-09-07T08:00:00+00:00",
            "2026-09-07T08:00:01+00:00",
            "2026-09-07T08:10:00+00:00",
            None,
            "2026-09-07T08:10:00+00:00",
            None,
            None,
        )
        checks = tuple(
            BacktestCheckRecord(
                name,
                "PENDING" if name == "SELF_CORRELATION" else "PASS",
                None,
                None,
                None,
            )
            for name in sorted(STANDARD_REGULAR_CHECK_NAMES)
        )
        result = BacktestResultRecord(
            "parent",
            1.5,
            1.2,
            0.1,
            0.05,
            0.1,
            0.01,
            None,
            None,
            None,
            None,
            True,
            checks,
        )
        yearly = BacktestYearlyStatRecord(
            "parent", 2023, None, None, None, None, 0.1, 1.5, 0.05, 0.1, 0.01, 1.2, "IS"
        )
        self.parent = BacktestSnapshot(task, result, (yearly,))
        self.check = SelfCorrelationCheckEvidence(
            "parent",
            "ineligible",
            "2026-09-07T09:00:00+00:00",
            tuple(
                (name, "FAIL" if name == "SELF_CORRELATION" else "PASS")
                for name in sorted(STANDARD_REGULAR_CHECK_NAMES)
            ),
            "reference",
            0.8,
        )
        self.reference = PlatformSubmittedAlphaRecord(
            "account",
            "reference",
            "rank(open)",
            "ACTIVE",
            "2026-09-01T00:00:00+00:00",
            False,
            {"settings": self.settings},
            "2026-09-07T09:00:00+00:00",
        )

    def select(self, *, parent=None, check=None, reference=None):
        return select_self_correlation_reference(
            parent or self.parent,
            check or self.check,
            (reference or self.reference,),
            observed_at=self.now,
        )

    def test_complete_failure_selects_same_setting_peer_despite_old_pending_sc(self):
        selected = self.select()
        self.assertEqual(selected.correlation, 0.8)
        self.assertEqual(
            (selected.parent_task_id, selected.formula), ("parent", "rank(open)")
        )
        reference = replace(
            self.reference,
            raw_payload={
                "settings": {
                    **self.settings,
                    "startDate": "2019-01-01",
                    "endDate": "2023-12-31",
                }
            },
        )
        self.assertEqual(self.select(reference=reference), selected)

    def test_submitted_parent_uses_own_structure_without_inventing_sc_result(self):
        reference = replace(self.reference, platform_alpha_id="alpha-parent", formula="rank(close)")
        for check in (None, replace(self.check, attempt_status="submitted"),
                      replace(self.check, reference_id=None)):
            selected = select_self_correlation_reference(self.parent, check, (reference,), observed_at=self.now)
            self.assertEqual((selected.formula, selected.platform_alpha_id), ("rank(close)", "alpha-parent"))
            self.assertIsNone(selected.correlation)
        # Same formula under another ID is already submitted only in the same settings/account.
        selected = select_self_correlation_reference(self.parent, None,
            (replace(reference, platform_alpha_id="other-id"),), observed_at=self.now)
        self.assertIsNotNone(selected)
        for invalid in (replace(reference, account_scope="other"),
                        replace(reference, status="INACTIVE"),
                        replace(reference, observed_at="2026-09-08T00:00:00+00:00"),
                        replace(reference, raw_payload={"settings": {**self.settings, "decay": 5}}),
                        replace(reference, formula="rank(open)")):
            self.assertIsNone(select_self_correlation_reference(self.parent, None, (invalid,), observed_at=self.now))
        self.assertIsNone(select_self_correlation_reference(self.parent, None, (), observed_at=self.now))

    def test_submitted_parent_keeps_quality_and_existing_conflict_requirements(self):
        own = replace(self.reference, platform_alpha_id="alpha-parent", formula="rank(close)")
        selected = select_self_correlation_reference(self.parent, self.check,
            (own, self.reference), observed_at=self.now)
        self.assertEqual((selected.formula, selected.correlation), ("rank(open)", .8))
        bad = replace(self.parent, result=replace(self.parent.result,
            checks=tuple(replace(c, status="FAIL") if c.name == "LOW_FITNESS" else c for c in self.parent.result.checks)))
        for parent in (bad, replace(self.parent, yearly_stats=()),
                       replace(self.parent, task=replace(self.parent.task, finished_at="2026-09-08T00:00:00+00:00"))):
            self.assertIsNone(select_self_correlation_reference(parent, None, (own,), observed_at=self.now))

    def test_missing_or_invalid_correlation_does_not_create_a_low_band(self):
        for value in (None, True, float("nan"), float("inf"), -1, 1.01, "0.74"):
            with self.subTest(value=value):
                self.assertIsNone(self.select(check=replace(self.check, correlation=value)))

    def test_pending_missing_pass_and_other_failure_do_not_trigger(self):
        for name, status in (
            ("SELF_CORRELATION", "PASS"),
            ("SELF_CORRELATION", "PENDING"),
            ("LOW_FITNESS", "FAIL"),
            ("CONCENTRATED_WEIGHT", "PENDING"),
        ):
            check = replace(
                self.check,
                statuses=tuple(
                    (n, status if n == name else s) for n, s in self.check.statuses
                ),
            )
            self.assertIsNone(self.select(check=check))
        self.assertIsNone(
            self.select(check=replace(self.check, statuses=self.check.statuses[:-1]))
        )
        self.assertIsNone(self.select(check=replace(self.check, reference_id=None)))

    def test_formal_unknown_active_or_submitted_is_not_new_repair_permission(self):
        for status in (
            "check_pending",
            "ready",
            "submitting",
            "submission_unknown",
            "submitted",
            "failed",
        ):
            self.assertIsNone(
                self.select(check=replace(self.check, attempt_status=status))
            )

    def test_foreign_inactive_self_future_and_mismatched_settings_are_excluded(self):
        for reference in (
            replace(self.reference, account_scope="other"),
            replace(self.reference, status="INACTIVE"),
            replace(self.reference, observed_at="2026-09-08T00:00:00+00:00"),
            replace(self.reference, raw_payload={}),
            replace(
                self.reference, raw_payload={"settings": {**self.settings, "decay": 5}}
            ),
        ):
            self.assertIsNone(self.select(reference=reference))
        self.assertIsNone(
            self.select(
                parent=replace(
                    self.parent,
                    task=replace(self.parent.task, platform_alpha_id="reference"),
                )
            )
        )

    def test_quality_failure_and_stale_or_future_check_are_not_repaired(self):
        result = replace(
            self.parent.result,
            checks=tuple(
                replace(c, status="FAIL") if c.name == "LOW_FITNESS" else c
                for c in self.parent.result.checks
            ),
        )
        self.assertIsNone(self.select(parent=replace(self.parent, result=result)))
        for timestamp in ("2026-09-07T07:00:00+00:00", "2026-09-08T07:00:00+00:00"):
            self.assertIsNone(
                self.select(check=replace(self.check, observed_at=timestamp))
            )

    def test_same_lineage_is_not_a_once_only_gate(self):
        child = replace(
            self.parent,
            task=replace(self.parent.task, task_id="child", formula="rank(open/close)"),
            result=replace(self.parent.result, task_id="child"),
        )
        self.assertIsNotNone(self.select())
        self.assertIsNotNone(
            self.select(parent=child, check=replace(self.check, task_id="child"))
        )
