import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.execution.catalog_fixture import initialize_test_generation_catalog
from execution.backtest_batches import (
    AutomatedCandidateBacktest,
    prepare_automated_candidate_backtest_batch,
)
from execution.backtest_reconciliation import (
    expire_unknown_backtest_submissions,
    reconcile_run_backtest_submissions,
)
from execution.backtests import (
    prepare_backtest_task,
    record_submission_accepted,
    record_submission_unknown,
)
from execution.runs import (
    AutomatedRunLimits,
    fail_automated_run,
    prepare_automated_run,
    start_automated_run,
)
from execution.real_backtests import advance_real_backtest, account_in_flight_backtest_count
from generation.candidate import exploration_candidate
from generation.parser import parse_formula
from persistence.backtests import get_backtest_task
from persistence.database import open_database
from persistence.schema import initialize_database_schema
from worldquant.alphas import UserAlphaPage, UserAlphaRecord
from worldquant.backtests import BacktestSettings
from worldquant.backtests import WorldQuantProtocolError


class ReconciliationClient:
    def __init__(self, records: tuple[UserAlphaRecord, ...] = ()) -> None:
        self.records = records
        self.calls: list[tuple[int, str, bool]] = []

    def fetch_user_alpha_page(self, *, limit, offset, hidden, status=None):
        self.calls.append((offset, status, hidden))
        selected = (
            self.records
            if status is None and hidden is False
            else ()
        )
        return UserAlphaPage(
            total_count=len(selected),
            records=selected[offset : offset + limit],
            has_next=offset + limit < len(selected),
        )


class ChangingReconciliationClient(ReconciliationClient):
    def __init__(
        self,
        first_records: tuple[UserAlphaRecord, ...],
        second_records: tuple[UserAlphaRecord, ...],
    ) -> None:
        super().__init__()
        self.first_records = first_records
        self.second_records = second_records
        self.unhidden_scan_count = 0

    def fetch_user_alpha_page(self, *, limit, offset, hidden, status=None):
        self.calls.append((offset, status, hidden))
        if status is not None or hidden:
            records = ()
        else:
            self.unhidden_scan_count += 1
            records = (
                self.first_records
                if self.unhidden_scan_count == 1
                else self.second_records
            )
        return UserAlphaPage(
            total_count=len(records),
            records=records[offset : offset + limit],
            has_next=offset + limit < len(records),
        )


class BacktestReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.database_path = root / "reconciliation.sqlite3"
        self.settings_path = root / "backtest.json"
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
        self.settings_path.write_text(
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
                        "byFieldCategory": {"sample": "SECTOR"},
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
            initialize_test_generation_catalog(connection)

    def test_unknown_deadline_releases_slot_but_consumes_original_identity(self) -> None:
        _run_id, task_id, created_id = self._prepare_failed_run()
        with open_database(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            before = get_backtest_task(connection, task_id)
            self.assertEqual(account_in_flight_backtest_count(connection, "group-account"), 1)
            self.assertEqual(expire_unknown_backtest_submissions(
                connection, account_scope="group-account", observed_at="2026-09-04T08:50:09+00:00",
            ), ())
            self.assertEqual(expire_unknown_backtest_submissions(
                connection, account_scope="other-account", observed_at="2026-09-04T08:50:10+00:00",
            ), ())
            [expired] = expire_unknown_backtest_submissions(
                connection, account_scope="group-account", observed_at="2026-09-04T08:50:10+00:00",
            )
            self.assertEqual(expired.task.failure_code, "submission_outcome_timeout")
            self.assertIn("平台是否接受仍未知", expired.task.failure_message)
            self.assertEqual(expired.task.request_fingerprint, before.task.request_fingerprint)
            self.assertEqual(account_in_flight_backtest_count(connection, "group-account"), 0)
            self.assertIsNone(expired.result)
            self.assertIsNone(expired.yearly_stats)
            self.assertIsNone(expired.task.platform_alpha_id)
            self.assertEqual(get_backtest_task(connection, created_id).task.status, "created")
            same = prepare_backtest_task(
                connection, account_scope="group-account", formula=before.task.formula,
                settings=self.settings.as_platform_dict(), created_at="2026-09-04T08:51:00+00:00",
            )
            self.assertEqual(same, expired)
            self.assertEqual(expire_unknown_backtest_submissions(
                connection, account_scope="group-account", observed_at="2026-09-04T08:51:00+00:00",
            ), ())
        advanced = advance_real_backtest(
            self.database_path, object(), task_id, allow_submission=True,
            observed_at="2026-09-04T08:51:00+00:00",
        )
        self.assertEqual(advanced.action, "already_failed")
        self.assertFalse(advanced.platform_request_performed)

    def test_unknown_expiry_does_not_touch_pending_or_unowned_tasks(self) -> None:
        _, unknown_id, pending_id = self._prepare_failed_run()
        with open_database(self.database_path) as connection:
            pending = record_submission_accepted(
                connection, pending_id, remote_id="simulation-known",
                observed_at="2026-09-04T08:40:11+00:00",
            )
            task = prepare_backtest_task(
                connection, account_scope="group-account", formula="rank(volume)",
                settings=self.settings.as_platform_dict(), created_at="2026-09-04T08:40:00+00:00",
            )
            unowned = record_submission_unknown(
                connection, task.task.task_id, observed_at="2026-09-04T08:40:10+00:00",
            )
        with open_database(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = expire_unknown_backtest_submissions(
                connection, account_scope="group-account", observed_at="2026-09-04T08:50:10+00:00",
            )
            self.assertEqual([item.task.task_id for item in result], [unknown_id])
            self.assertEqual(get_backtest_task(connection, pending_id), pending)
            self.assertEqual(get_backtest_task(connection, unowned.task.task_id), unowned)

    def test_unknown_expiry_waits_for_the_active_request_commit_window(self) -> None:
        _, task_id, _ = self._prepare_failed_run(max_pending_seconds=30)
        with open_database(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self.assertEqual(expire_unknown_backtest_submissions(
                connection, account_scope="group-account",
                observed_at="2026-09-04T08:41:40+00:00",
            ), ())
            [expired] = expire_unknown_backtest_submissions(
                connection, account_scope="group-account",
                observed_at="2026-09-04T08:41:40.000001+00:00",
            )
            self.assertEqual(expired.task.task_id, task_id)
            self.assertEqual(expired.task.failure_code, "submission_outcome_timeout")

    def test_unique_positive_evidence_ends_unknown_without_result(self) -> None:
        run_id, unknown_id, created_id = self._prepare_failed_run()
        client = ReconciliationClient((self._alpha("alpha-1"),))

        result = reconcile_run_backtest_submissions(
            self.database_path,
            client,
            run_id,
            observed_at="2026-09-04T08:42:00+00:00",
        )

        self.assertEqual(result.action, "confirmed")
        self.assertEqual(result.platform_request_count, 4)
        self.assertEqual(result.reconciled_task_count, 1)
        self.assertEqual(result.unresolved_task_count, 0)
        with open_database(self.database_path) as connection:
            unknown = get_backtest_task(connection, unknown_id)
            created = get_backtest_task(connection, created_id)
        assert unknown is not None
        assert created is not None
        self.assertEqual(unknown.task.status, "failed")
        self.assertEqual(unknown.task.platform_alpha_id, "alpha-1")
        self.assertEqual(
            unknown.task.failure_code,
            "remote_accepted_without_simulation_id",
        )
        self.assertIsNone(unknown.result)
        self.assertIsNone(unknown.yearly_stats)
        self.assertEqual(created.task.status, "created")
        self.assertTrue(all(call[1] is None for call in client.calls))

    def test_same_recorded_start_and_finish_uses_bounded_request_window(self) -> None:
        run_id, unknown_id, _ = self._prepare_failed_run(finish_after_seconds=10)

        result = reconcile_run_backtest_submissions(
            self.database_path,
            ReconciliationClient((self._alpha("alpha-1"),)),
            run_id,
            observed_at="2026-09-04T08:42:16+00:00",
        )

        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, unknown_id)
        assert snapshot is not None
        self.assertEqual(result.action, "confirmed")
        self.assertEqual(snapshot.task.status, "failed")
        self.assertEqual(snapshot.task.platform_alpha_id, "alpha-1")

    def test_zero_matches_preserve_unknown_state(self) -> None:
        run_id, unknown_id, _ = self._prepare_failed_run()

        result = reconcile_run_backtest_submissions(
            self.database_path,
            ReconciliationClient(),
            run_id,
            observed_at="2026-09-04T08:42:00+00:00",
        )

        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, unknown_id)
        assert snapshot is not None
        self.assertEqual(result.action, "unresolved")
        self.assertEqual(snapshot.task.status, "submission_unknown")
        self.assertIsNone(snapshot.task.platform_alpha_id)

    def test_alpha_created_after_two_claim_lock_waits_is_reconciled_without_post(self) -> None:
        # Start 08:40:10; BEGIN waits 20s, COMMIT waits 20s, platform takes 25s.
        # Each wait/request is within its limit; creation is 65s after intent.
        run_id, unknown_id, created_id = self._prepare_failed_run(finish_after_seconds=90)
        client = ReconciliationClient((self._alpha(
            "alpha-delayed", created_at="2026-09-04T08:41:15+00:00",
        ),))
        result = reconcile_run_backtest_submissions(
            self.database_path, client, run_id,
            observed_at="2026-09-04T08:43:00+00:00",
        )
        self.assertEqual(result.action, "confirmed")
        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, unknown_id)
            unsent = get_backtest_task(connection, created_id)
        self.assertEqual(snapshot.task.platform_alpha_id, "alpha-delayed")
        self.assertEqual(snapshot.task.failure_code, "remote_accepted_without_simulation_id")
        self.assertIsNone(snapshot.result)
        self.assertEqual(unsent.task.status, "created")
        blocked = advance_real_backtest(
            self.database_path, client, unknown_id,
            observed_at="2026-09-04T08:44:00+00:00", allow_submission=True,
        )
        self.assertEqual(blocked.action, "already_failed")
        self.assertFalse(blocked.platform_request_performed)

    def test_visibility_settling_preserves_unknown_without_remote_request(self) -> None:
        run_id, unknown_id, _ = self._prepare_failed_run()
        client = ReconciliationClient((self._alpha("alpha-1"),))

        result = reconcile_run_backtest_submissions(
            self.database_path,
            client,
            run_id,
            observed_at="2026-09-04T08:41:10+00:00",
        )

        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, unknown_id)
        assert snapshot is not None
        self.assertEqual(result.action, "unresolved")
        self.assertEqual(result.platform_request_count, 0)
        self.assertEqual(client.calls, [])
        self.assertEqual(snapshot.task.status, "submission_unknown")

    def test_multiple_matches_preserve_unknown_state(self) -> None:
        run_id, unknown_id, _ = self._prepare_failed_run()

        result = reconcile_run_backtest_submissions(
            self.database_path,
            ReconciliationClient(
                (self._alpha("alpha-1"), self._alpha("alpha-2"))
            ),
            run_id,
            observed_at="2026-09-04T08:42:00+00:00",
        )

        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, unknown_id)
        assert snapshot is not None
        self.assertEqual(result.action, "unresolved")
        self.assertEqual(snapshot.task.status, "submission_unknown")
        self.assertIsNone(snapshot.task.platform_alpha_id)

    def test_formula_setting_and_time_mismatches_do_not_confirm(self) -> None:
        run_id, unknown_id, _ = self._prepare_failed_run()
        missing_setting = self._alpha("alpha-missing-setting")
        assert missing_setting.settings is not None
        incomplete_settings = dict(missing_setting.settings)
        incomplete_settings.pop("maxTrade")
        alternatives = (
            self._alpha(
                "alpha-late",
                created_at="2026-09-04T08:41:11+00:00",
            ),
            self._alpha("alpha-formula", formula="rank(open)"),
            self._alpha("alpha-settings", universe="TOP1000"),
            UserAlphaRecord(
                platform_alpha_id=missing_setting.platform_alpha_id,
                alpha_type=missing_setting.alpha_type,
                status=missing_setting.status,
                formula=missing_setting.formula,
                settings=incomplete_settings,
                created_at=missing_setting.created_at,
                hidden=missing_setting.hidden,
            ),
            self._alpha(
                "alpha-time",
                created_at="2026-09-04T08:39:59+00:00",
            ),
        )

        result = reconcile_run_backtest_submissions(
            self.database_path,
            ReconciliationClient(alternatives),
            run_id,
            observed_at="2026-09-04T08:42:00+00:00",
        )

        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, unknown_id)
        assert snapshot is not None
        self.assertEqual(result.action, "unresolved")
        self.assertEqual(snapshot.task.status, "submission_unknown")

    def test_incomplete_local_settings_fail_closed(self) -> None:
        run_id, unknown_id, _ = self._prepare_failed_run()
        with open_database(self.database_path) as connection:
            connection.execute(
                "UPDATE backtest_tasks SET settings_json = ? WHERE task_id = ?",
                (json.dumps({"delay": 1}), unknown_id),
            )

        with self.assertRaisesRegex(
            ValueError,
            "backtest_reconciliation_settings_invalid",
        ):
            reconcile_run_backtest_submissions(
                self.database_path,
                ReconciliationClient((self._alpha("alpha-1"),)),
                run_id,
                observed_at="2026-09-04T08:42:00+00:00",
            )

        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, unknown_id)
        assert snapshot is not None
        self.assertEqual(snapshot.task.status, "submission_unknown")
        self.assertIsNone(snapshot.task.platform_alpha_id)

    def test_changed_remote_snapshot_fails_without_changing_unknown(self) -> None:
        run_id, unknown_id, _ = self._prepare_failed_run()
        first = (self._alpha("alpha-1"),)
        second = (self._alpha("alpha-1"), self._alpha("alpha-2"))

        with self.assertRaisesRegex(
            WorldQuantProtocolError,
            "worldquant_user_alphas_snapshot_changed",
        ):
            reconcile_run_backtest_submissions(
                self.database_path,
                ChangingReconciliationClient(first, second),
                run_id,
                observed_at="2026-09-04T08:42:00+00:00",
            )

        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, unknown_id)
        assert snapshot is not None
        self.assertEqual(snapshot.task.status, "submission_unknown")
        self.assertIsNone(snapshot.task.platform_alpha_id)

    def test_complete_pagination_is_checked_until_window_is_covered(self) -> None:
        run_id, unknown_id, _ = self._prepare_failed_run()
        records = tuple(
            self._alpha(
                f"unrelated-{index}",
                formula="rank(open)",
                created_at=(
                    datetime.fromisoformat("2026-09-04T08:41:49+00:00")
                    - timedelta(milliseconds=index)
                ).isoformat(),
            )
            for index in range(100)
        ) + (self._alpha("alpha-target"),)
        client = ReconciliationClient(records)

        result = reconcile_run_backtest_submissions(
            self.database_path,
            client,
            run_id,
            observed_at="2026-09-04T08:42:00+00:00",
        )

        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, unknown_id)
        assert snapshot is not None
        self.assertEqual(result.action, "confirmed")
        self.assertEqual(result.platform_request_count, 6)
        self.assertEqual(snapshot.task.platform_alpha_id, "alpha-target")

    def test_existing_alpha_binding_rejects_reconciliation(self) -> None:
        run_id, unknown_id, _ = self._prepare_failed_run()
        with open_database(self.database_path) as connection:
            conflict = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula="rank(volume)",
                settings=self.settings.as_platform_dict(),
                created_at="2026-09-04T08:39:00+00:00",
            )
            record_submission_unknown(
                connection,
                conflict.task.task_id,
                observed_at="2026-09-04T08:39:01+00:00",
            )
            record_submission_accepted(
                connection,
                conflict.task.task_id,
                remote_id=(
                    "https://api.worldquantbrain.com/simulations/conflict"
                ),
                observed_at="2026-09-04T08:39:02+00:00",
                platform_alpha_id="alpha-conflict",
            )

        with self.assertRaisesRegex(
            ValueError,
            "backtest_reconciliation_alpha_identity_conflict",
        ):
            reconcile_run_backtest_submissions(
                self.database_path,
                ReconciliationClient((self._alpha("alpha-conflict"),)),
                run_id,
                observed_at="2026-09-04T08:42:00+00:00",
            )

        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, unknown_id)
        assert snapshot is not None
        self.assertEqual(snapshot.task.status, "submission_unknown")
        self.assertIsNone(snapshot.task.platform_alpha_id)

    def test_out_of_order_page_fails_without_changing_unknown(self) -> None:
        run_id, unknown_id, _ = self._prepare_failed_run()
        records = (
            self._alpha(
                "older",
                formula="rank(open)",
                created_at="2026-09-04T08:40:20+00:00",
            ),
            self._alpha(
                "newer",
                formula="rank(open)",
                created_at="2026-09-04T08:40:30+00:00",
            ),
        )

        with self.assertRaisesRegex(
            WorldQuantProtocolError,
            "worldquant_user_alphas_order_invalid",
        ):
            reconcile_run_backtest_submissions(
                self.database_path,
                ReconciliationClient(records),
                run_id,
                observed_at="2026-09-04T08:42:00+00:00",
            )

        with open_database(self.database_path) as connection:
            snapshot = get_backtest_task(connection, unknown_id)
        assert snapshot is not None
        self.assertEqual(snapshot.task.status, "submission_unknown")

    def _prepare_failed_run(
        self,
        *,
        finish_after_seconds: int = 60,
        max_pending_seconds: int = 600,
    ) -> tuple[str, str, str]:
        base = datetime.fromisoformat("2026-09-04T08:40:00+00:00")
        run = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=3,
                backtest_count=2,
                max_cycles=1,
                max_backtests=2,
                max_pending_seconds=max_pending_seconds,
                max_consecutive_failures=2,
                max_request_failures=3,
                max_in_flight_backtests=3,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ),
            created_at=base.isoformat(),
        )
        start_automated_run(
            self.database_path,
            run.run_id,
            started_at=(base + timedelta(seconds=1)).isoformat(),
        )
        prepared = prepare_automated_candidate_backtest_batch(
            self.database_path,
            run_id=run.run_id,
            candidates=(
                AutomatedCandidateBacktest(
                    exploration_candidate(parse_formula("rank(close)").expression),
                    self.settings,
                ),
                AutomatedCandidateBacktest(
                    exploration_candidate(parse_formula("rank(open)").expression),
                    self.settings,
                ),
            ),
            created_at=(base + timedelta(seconds=2)).isoformat(),
        )
        with open_database(self.database_path) as connection:
            record_submission_unknown(
                connection,
                prepared[0].task.task_id,
                observed_at=(base + timedelta(seconds=10)).isoformat(),
            )
        fail_automated_run(
            self.database_path,
            run.run_id,
            failed_at=(base + timedelta(seconds=finish_after_seconds)).isoformat(),
            reason="submission_reconciliation_required",
        )
        return (
            run.run_id,
            prepared[0].task.task_id,
            prepared[1].task.task_id,
        )

    def _alpha(
        self,
        alpha_id: str,
        *,
        formula: str = "rank(close)",
        universe: str = "TOP3000",
        created_at: str = "2026-09-04T08:40:30+00:00",
    ) -> UserAlphaRecord:
        settings = self.settings.as_platform_dict()
        settings["universe"] = universe
        settings.update(
            {"startDate": "2020-01-01", "endDate": "2025-01-01"}
        )
        return UserAlphaRecord(
            platform_alpha_id=alpha_id,
            alpha_type="REGULAR",
            status="UNSUBMITTED",
            formula=formula,
            settings=settings,
            created_at=datetime.fromisoformat(created_at),
            hidden=False,
        )


if __name__ == "__main__":
    unittest.main()
