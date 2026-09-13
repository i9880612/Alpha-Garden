from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.execution.catalog_fixture import initialize_test_generation_catalog
from execution.runs import (
    AutomatedRunLimits,
    complete_automated_run_after_candidate_planning_stop,
    fail_automated_run,
    prepare_automated_run,
    record_automated_cycle_settlement,
    resume_automated_run_after_submission_reconciliation,
    start_automated_run,
)
from execution.backtests import (
    prepare_backtest_task,
    record_submission_accepted,
    record_submission_not_accepted,
    record_submission_unknown,
)
from persistence.backtests import get_backtest_task, initialize_backtest_schema
from persistence.database import open_database
from persistence.run_allocations import create_run_allocation, get_run_allocation
from execution.run_config import load_automated_run_limits
from persistence.run_diagnostics import (
    CandidatePlanningDiagnostic,
    CandidatePlanningExclusionCount,
)
from persistence.runs import (
    AutomatedRunBacktestRecord,
    attach_backtest_to_automated_run,
    get_automated_run,
    initialize_run_schema,
)
from persistence.submissions import initialize_submission_schema


class AutomatedRunExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.database_path = root / "runs.sqlite3"
        self.policy_path = root / "backtest.json"
        self.policy_path.write_text(
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
            initialize_backtest_schema(connection)
            initialize_run_schema(connection)
            initialize_submission_schema(connection)
            initialize_test_generation_catalog(connection)

    def test_preparation_requires_current_account_catalog_before_creating_run(self) -> None:
        with open_database(self.database_path) as connection:
            connection.execute("DELETE FROM platform_catalog_syncs")

        with self.assertRaisesRegex(ValueError, "generation_catalog_sync_missing"):
            self._prepare()

        with open_database(self.database_path) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM automated_runs"
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_run_freezes_config_percentages_and_rejects_overwriting_them(self):
        config_path = self.database_path.with_suffix(".json")
        payload = json.loads((Path(__file__).resolve().parents[2] / "config/run.default.json").read_text())
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        original_limits = load_automated_run_limits(config_path, cycles=1)
        prepared = prepare_automated_run(self.database_path, self.policy_path,
            account_scope="group-account", limits=original_limits,
            created_at="2026-08-30T00:00:00+08:00")
        payload.update(explorationPercent=20, selfCorrelationPercent=50,
                       mutationPercent=30, directionValidationPercent=5)
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        updated_limits = load_automated_run_limits(config_path, cycles=1)
        self.assertEqual(updated_limits.self_correlation_percent, 50)
        with open_database(self.database_path) as connection:
            frozen = get_run_allocation(connection, prepared.run_id)
            self.assertEqual((frozen.exploration_percent, frozen.self_correlation_percent,
                              frozen.mutation_percent, frozen.direction_validation_percent),
                             (30, 30, 40, 3))
            create_run_allocation(connection, frozen)
            with self.assertRaisesRegex(ValueError, "allocation_conflict"):
                create_run_allocation(connection, replace(frozen, direction_validation_percent=5))
            self.assertEqual(get_run_allocation(connection, prepared.run_id), frozen)
        start_automated_run(self.database_path, prepared.run_id,
            started_at="2026-08-30T00:00:30+08:00")
        fail_automated_run(self.database_path, prepared.run_id,
            failed_at="2026-08-30T00:01:00+08:00", reason="submission_reconciliation_required")
        updated = prepare_automated_run(self.database_path, self.policy_path,
            account_scope="group-account", limits=updated_limits,
            created_at="2026-08-30T00:00:00+08:00")
        self.assertNotEqual(updated.run_id, prepared.run_id)
        self.assertEqual(updated.minimum_exploration_backtests, 20)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_run_allocation(connection, updated.run_id).self_correlation_percent, 50)
            self.assertEqual(get_run_allocation(connection, prepared.run_id), frozen)

    def test_unqualified_cycles_continue_until_an_explicit_run_limit(self) -> None:
        prepared = self._prepare()
        self.assertEqual(prepared.minimum_exploration_backtests, 6)
        self.assertEqual(prepared.exploration_seed_attempt_multiplier, 4)
        started = start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        first, _, _ = self._record_cycle_settlement(
            prepared.run_id,
            cycle_number=1,
            outcome="not_qualified",
            frontier_advanced=False,
            observed_at="2026-08-30T00:02:00+08:00",
        )
        second, _, _ = self._record_cycle_settlement(
            prepared.run_id,
            cycle_number=2,
            outcome="not_qualified",
            frontier_advanced=False,
            observed_at="2026-08-30T00:03:00+08:00",
        )

        self.assertEqual(started.status, "running")
        self.assertEqual(first.status, "running")
        self.assertEqual(second.status, "running")
        self.assertEqual(second.current_cycle, 2)
        self.assertIsNone(second.stop_reason)

    def test_candidate_planning_stop_is_atomic_and_idempotent(self) -> None:
        prepared = self._prepare()
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        diagnostic = CandidatePlanningDiagnostic(
            cycle_number=1,
            exploration_generation_target_count=3,
            exploration_backtest_target_count=2,
            seed_attempt_limit=12,
            attempted_seed_count=12,
            generated_candidate_count=1,
            selected_candidate_count=None,
            generation_exclusions=(
                CandidatePlanningExclusionCount("test_rejection", 11),
            ),
            selection_rejections=(),
        )

        stopped = complete_automated_run_after_candidate_planning_stop(
            self.database_path,
            prepared.run_id,
            completed_at="2026-08-30T00:02:00+08:00",
            reason="generation_attempt_budget_exhausted",
            diagnostic=diagnostic,
        )
        repeated = complete_automated_run_after_candidate_planning_stop(
            self.database_path,
            prepared.run_id,
            completed_at="2026-08-30T00:03:00+08:00",
            reason="generation_attempt_budget_exhausted",
            diagnostic=diagnostic,
        )

        self.assertEqual(stopped, repeated)
        self.assertEqual(stopped.status, "completed")
        self.assertEqual(stopped.current_cycle, 0)
        self.assertEqual(
            stopped.candidate_planning_stop_diagnostic_json,
            diagnostic.canonical_json(
                stop_reason="generation_attempt_budget_exhausted"
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "automated_run_candidate_planning_stop_conflict",
        ):
            complete_automated_run_after_candidate_planning_stop(
                self.database_path,
                prepared.run_id,
                completed_at="2026-08-30T00:03:00+08:00",
                reason="generation_attempt_budget_exhausted",
                diagnostic=replace(
                    diagnostic,
                    generation_exclusions=(
                        CandidatePlanningExclusionCount(
                            "different_rejection",
                            11,
                        ),
                    ),
                ),
            )

    def test_candidate_planning_stop_rejects_a_cycle_with_tasks(self) -> None:
        prepared = self._prepare()
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        with open_database(self.database_path) as connection:
            task = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula="rank(close)",
                settings=self._backtest_settings(),
                created_at="2026-08-30T00:01:30+08:00",
            ).task
            attach_backtest_to_automated_run(
                connection,
                AutomatedRunBacktestRecord(
                    run_id=prepared.run_id,
                    task_id=task.task_id,
                    cycle_number=1,
                ),
            )
        diagnostic = CandidatePlanningDiagnostic(
            cycle_number=1,
            exploration_generation_target_count=3,
            exploration_backtest_target_count=2,
            seed_attempt_limit=12,
            attempted_seed_count=12,
            generated_candidate_count=1,
            selected_candidate_count=None,
            generation_exclusions=(
                CandidatePlanningExclusionCount("test_rejection", 11),
            ),
            selection_rejections=(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_candidate_planning_cycle_not_empty",
        ):
            complete_automated_run_after_candidate_planning_stop(
                self.database_path,
                prepared.run_id,
                completed_at="2026-08-30T00:02:00+08:00",
                reason="generation_attempt_budget_exhausted",
                diagnostic=diagnostic,
            )
        with open_database(self.database_path) as connection:
            recovered = get_automated_run(connection, prepared.run_id)

        self.assertEqual(recovered.status, "running")
        self.assertIsNone(recovered.candidate_planning_stop_diagnostic_json)

    def test_completed_run_rejects_an_active_linked_backtest(self) -> None:
        prepared = prepare_automated_run(
            self.database_path,
            self.policy_path,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=2,
                backtest_count=1,
                max_cycles=1,
                max_backtests=1,
                max_pending_seconds=60,
                max_consecutive_failures=1,
                max_request_failures=3,
                max_in_flight_backtests=1,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ),
            created_at="2026-08-30T00:00:00+08:00",
        )
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        with open_database(self.database_path) as connection:
            task = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula="rank(close)",
                settings=self._backtest_settings(),
                created_at="2026-08-30T00:01:01+08:00",
            ).task
            record_submission_unknown(
                connection,
                task.task_id,
                observed_at="2026-08-30T00:01:02+08:00",
            )
            record_submission_accepted(
                connection,
                task.task_id,
                remote_id=(
                    "https://api.worldquantbrain.com/simulations/active"
                ),
                observed_at="2026-08-30T00:01:03+08:00",
            )
            attach_backtest_to_automated_run(
                connection,
                AutomatedRunBacktestRecord(
                    run_id=prepared.run_id,
                    task_id=task.task_id,
                    cycle_number=1,
                ),
            )

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_completion_has_active_backtests",
        ):
            with open_database(self.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                record_automated_cycle_settlement(
                    connection,
                    prepared.run_id,
                    cycle_number=1,
                    outcome="not_qualified",
                    frontier_advanced=False,
                    observed_at="2026-08-30T00:02:00+08:00",
                )

        with open_database(self.database_path) as connection:
            recovered = get_automated_run(connection, prepared.run_id)
        self.assertEqual(recovered.status, "running")
        self.assertEqual(recovered.current_cycle, 0)

    def test_non_failure_cycle_resets_failure_counter_and_recovers(self) -> None:
        prepared = self._prepare()
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        self._record_cycle_settlement(
            prepared.run_id,
            cycle_number=1,
            outcome="failed",
            frontier_advanced=False,
            observed_at="2026-08-30T00:02:00+08:00",
        )
        recovered, _, _ = self._record_cycle_settlement(
            prepared.run_id,
            cycle_number=2,
            outcome="frontier_advanced",
            frontier_advanced=True,
            observed_at="2026-08-30T00:03:00+08:00",
        )
        with open_database(self.database_path) as connection:
            persisted = get_automated_run(connection, prepared.run_id)

        self.assertEqual(recovered.consecutive_failures, 0)
        self.assertEqual(persisted, recovered)

    def test_consecutive_failures_close_the_run(self) -> None:
        prepared = self._prepare()
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        self._record_cycle_settlement(
            prepared.run_id,
            cycle_number=1,
            outcome="failed",
            frontier_advanced=False,
            observed_at="2026-08-30T00:02:00+08:00",
        )
        stopped, _, _ = self._record_cycle_settlement(
            prepared.run_id,
            cycle_number=2,
            outcome="failed",
            frontier_advanced=False,
            observed_at="2026-08-30T00:03:00+08:00",
        )

        self.assertEqual(stopped.status, "failed")
        self.assertEqual(stopped.stop_reason, "consecutive_failures_reached")

    def test_max_cycles_closes_a_successful_run(self) -> None:
        prepared = prepare_automated_run(
            self.database_path,
            self.policy_path,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=10,
                backtest_count=2,
                max_cycles=1,
                max_backtests=2,
                max_pending_seconds=3600,
                max_consecutive_failures=1,
                max_request_failures=3,
                max_in_flight_backtests=3,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ),
            created_at="2026-08-30T00:00:00+08:00",
        )
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )

        stopped, _, _ = self._record_cycle_settlement(
            prepared.run_id,
            cycle_number=1,
            outcome="frontier_advanced",
            frontier_advanced=True,
            observed_at="2026-08-30T00:02:00+08:00",
        )

        self.assertEqual(stopped.status, "completed")
        self.assertEqual(stopped.stop_reason, "max_cycles_reached")

    def test_final_failed_cycle_closes_run_as_failed(self) -> None:
        prepared = prepare_automated_run(
            self.database_path,
            self.policy_path,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=10,
                backtest_count=2,
                max_cycles=1,
                max_backtests=2,
                max_pending_seconds=3600,
                max_consecutive_failures=2,
                max_request_failures=3,
                max_in_flight_backtests=3,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ),
            created_at="2026-08-30T00:00:00+08:00",
        )
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )

        stopped, _, _ = self._record_cycle_settlement(
            prepared.run_id,
            cycle_number=1,
            outcome="failed",
            frontier_advanced=False,
            observed_at="2026-08-30T00:02:00+08:00",
        )

        self.assertEqual(stopped.status, "failed")
        self.assertEqual(stopped.stop_reason, "final_cycle_failed")

    def test_unknown_external_backtest_allows_preparation_and_start_without_state_change(self) -> None:
        task_id = self._prepare_unresolved_backtest("submission_unknown")
        with open_database(self.database_path) as connection:
            before = get_backtest_task(connection, task_id)
        run = self._prepare()
        started = start_automated_run(self.database_path, run.run_id, started_at="2026-08-30T00:02:00+08:00")
        self.assertEqual(started.status, "running")
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, task_id), before)

    def test_pending_external_backtest_still_blocks_authorized_preparation(self) -> None:
        self._prepare_unresolved_backtest("pending")
        with self.assertRaisesRegex(
            ValueError,
            "automated_run_backtest_reconciliation_required",
        ):
            self._prepare()

        with open_database(self.database_path) as connection:
            run_count = connection.execute(
                "SELECT COUNT(*) FROM automated_runs"
            ).fetchone()[0]
        self.assertEqual(run_count, 0)

    def test_unresolved_external_backtest_blocks_start_after_preparation(self) -> None:
        prepared = self._prepare()
        self._prepare_unresolved_backtest("pending")

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_backtest_reconciliation_required",
        ):
            start_automated_run(
                self.database_path,
                prepared.run_id,
                started_at="2026-08-30T00:02:00+08:00",
            )

        with open_database(self.database_path) as connection:
            persisted = get_automated_run(connection, prepared.run_id)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.status, "created")

    def test_unresolved_backtest_from_another_account_does_not_block(self) -> None:
        self._prepare_unresolved_backtest("pending")
        with open_database(self.database_path) as connection:
            connection.execute(
                "UPDATE platform_catalog_syncs SET account_scope = 'other-account'"
            )

        prepared = prepare_automated_run(
            self.database_path,
            self.policy_path,
            account_scope="other-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=10,
                backtest_count=2,
                max_cycles=1,
                max_backtests=2,
                max_pending_seconds=3600,
                max_consecutive_failures=1,
                max_request_failures=3,
                max_in_flight_backtests=1,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ),
            created_at="2026-08-30T00:01:00+08:00",
        )

        self.assertEqual(prepared.account_scope, "other-account")

    def test_unresolved_backtest_allows_original_run_to_resume_free_slots(self) -> None:
        prepared = self._prepare()
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        with open_database(self.database_path) as connection:
            task = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula="rank(close)",
                settings=self._backtest_settings(),
                created_at="2026-08-30T00:01:00+08:00",
            )
            attach_backtest_to_automated_run(
                connection,
                AutomatedRunBacktestRecord(
                    run_id=prepared.run_id,
                    task_id=task.task.task_id,
                    cycle_number=1,
                ),
            )
            record_submission_unknown(
                connection,
                task.task.task_id,
                observed_at="2026-08-30T00:02:00+08:00",
            )
        fail_automated_run(
            self.database_path,
            prepared.run_id,
            failed_at="2026-08-30T00:03:00+08:00",
            reason="submission_reconciliation_required",
        )

        resumed = resume_automated_run_after_submission_reconciliation(
            self.database_path,
            prepared.run_id,
        )

        self.assertEqual(resumed.status, "running")
        self.assertIsNone(resumed.finished_at)
        self.assertIsNone(resumed.stop_reason)
        with open_database(self.database_path) as connection:
            unknown = get_backtest_task(connection, task.task.task_id)
        self.assertEqual(unknown.task.status, "submission_unknown")
        self.assertEqual(unknown.task.submission_started_at, "2026-08-30T00:02:00+08:00")

    def test_other_failed_runs_cannot_use_reconciliation_resume(self) -> None:
        prepared = self._prepare()
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        fail_automated_run(
            self.database_path,
            prepared.run_id,
            failed_at="2026-08-30T00:02:00+08:00",
            reason="request_failure_limit_reached",
        )

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_reconciliation_resume_invalid",
        ):
            resume_automated_run_after_submission_reconciliation(
                self.database_path,
                prepared.run_id,
            )

    def test_reconciliation_resume_requires_original_catalog_identity(self) -> None:
        prepared = self._prepare()
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        fail_automated_run(
            self.database_path,
            prepared.run_id,
            failed_at="2026-08-30T00:02:00+08:00",
            reason="submission_reconciliation_required",
        )
        with open_database(self.database_path) as connection:
            connection.execute(
                "UPDATE platform_catalog_syncs SET account_scope = ?",
                ("other-account",),
            )

        with self.assertRaisesRegex(
            ValueError,
            "generation_catalog_account_scope_mismatch",
        ):
            resume_automated_run_after_submission_reconciliation(
                self.database_path,
                prepared.run_id,
            )

        with open_database(self.database_path) as connection:
            record = get_automated_run(connection, prepared.run_id)
        self.assertEqual(record.status, "failed")
        self.assertEqual(
            record.stop_reason,
            "submission_reconciliation_required",
        )

    def test_reconciliation_resume_blocks_unrelated_pending_task(self) -> None:
        prepared = self._prepare()
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        with open_database(self.database_path) as connection:
            own_task = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula="rank(close)",
                settings=self._backtest_settings(),
                created_at="2026-08-30T00:01:30+08:00",
            )
            attach_backtest_to_automated_run(
                connection,
                AutomatedRunBacktestRecord(
                    run_id=prepared.run_id,
                    task_id=own_task.task.task_id,
                    cycle_number=1,
                ),
            )
            record_submission_unknown(
                connection,
                own_task.task.task_id,
                observed_at="2026-08-30T00:02:00+08:00",
            )
        fail_automated_run(
            self.database_path,
            prepared.run_id,
            failed_at="2026-08-30T00:03:00+08:00",
            reason="submission_reconciliation_required",
        )
        with open_database(self.database_path) as connection:
            record_submission_not_accepted(
                connection,
                own_task.task.task_id,
            )
        self._prepare_unresolved_backtest("pending", formula="rank(open)")

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_backtest_reconciliation_required",
        ):
            resume_automated_run_after_submission_reconciliation(
                self.database_path,
                prepared.run_id,
            )

    def _record_cycle_settlement(
        self,
        run_id: str,
        *,
        cycle_number: int,
        outcome: str,
        frontier_advanced: bool,
        observed_at: str,
    ):
        with open_database(self.database_path) as connection:
            return record_automated_cycle_settlement(
                connection,
                run_id,
                cycle_number=cycle_number,
                outcome=outcome,
                frontier_advanced=frontier_advanced,
                observed_at=observed_at,
            )

    def _prepare(self):
        return prepare_automated_run(
            self.database_path,
            self.policy_path,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=500,
                backtest_count=20,
                max_cycles=5,
                max_backtests=100,
                max_pending_seconds=3600,
                max_consecutive_failures=2,
                max_request_failures=3,
                max_in_flight_backtests=3,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ),
            created_at="2026-08-30T00:00:00+08:00",
        )

    def _prepare_unresolved_backtest(
        self,
        status: str,
        *,
        formula: str = "rank(close)",
    ) -> str:
        with open_database(self.database_path) as connection:
            task = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula=formula,
                settings=self._backtest_settings(),
                created_at="2026-08-30T00:00:30+08:00",
            )
            unknown = record_submission_unknown(
                connection,
                task.task.task_id,
                observed_at="2026-08-30T00:00:31+08:00",
            )
            if status == "submission_unknown":
                return unknown.task.task_id
            if status != "pending":
                raise AssertionError(f"unsupported test status: {status}")
            pending = record_submission_accepted(
                connection,
                task.task.task_id,
                remote_id="https://api.worldquantbrain.com/simulations/existing",
                observed_at="2026-08-30T00:00:32+08:00",
            )
            return pending.task.task_id

    @staticmethod
    def _backtest_settings() -> dict[str, object]:
        return {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 4,
            "neutralization": "SECTOR",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "OFF",
            "language": "FASTEXPR",
            "visualization": False,
            "maxTrade": "OFF",
            "maxPosition": "OFF",
        }


if __name__ == "__main__":
    unittest.main()
