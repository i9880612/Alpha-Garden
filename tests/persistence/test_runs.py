from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.backtests import (
    prepare_backtest_task,
    record_submission_accepted,
    record_submission_unknown,
)
from persistence.backtests import initialize_backtest_schema
from persistence.database import open_database
from persistence.run_diagnostics import (
    CandidatePlanningDiagnostic,
    CandidatePlanningExclusionCount,
)
from persistence.runs import (
    AutomatedCycleSettlementRecord,
    AutomatedRunBacktestRecord,
    AutomatedRunRecord,
    attach_backtest_to_automated_run,
    create_automated_cycle_settlement,
    create_automated_run,
    get_automated_run_backtest_by_task,
    get_automated_run,
    initialize_run_schema,
    list_automated_run_backtests,
    list_submission_reconciliation_runs,
    replace_automated_run,
)
from persistence.submissions import initialize_submission_schema


class AutomatedRunPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "runs.sqlite3"
        with open_database(self.database_path) as connection:
            initialize_backtest_schema(connection)
            initialize_run_schema(connection)
            initialize_submission_schema(connection)

    def test_created_plan_round_trips_without_extra_state(self) -> None:
        record = self._record("run-a")
        with open_database(self.database_path) as connection:
            created = create_automated_run(connection, record)
            loaded = get_automated_run(connection, record.run_id)

        self.assertEqual(created, record)
        self.assertEqual(loaded, record)

    def test_optimization_mode_is_frozen_and_cannot_change_on_resume(self):
        record = replace(self._record("run-opt"), optimization_only=True)
        with open_database(self.database_path) as connection:
            create_automated_run(connection, record)
            self.assertEqual(get_automated_run(connection, record.run_id), record)
            with self.assertRaisesRegex(ValueError, "identity_conflict"):
                replace_automated_run(connection, replace(record, optimization_only=False),
                                     expected_status="created")

    def test_unlimited_continuous_plan_round_trips(self) -> None:
        record = replace(
            self._record("run-unlimited"),
            max_cycles=-1,
            max_backtests=0,
        )

        with open_database(self.database_path) as connection:
            created = create_automated_run(connection, record)
            loaded = get_automated_run(connection, record.run_id)

        self.assertEqual(created, record)
        self.assertEqual(loaded, record)

    def test_positive_cycle_plan_requires_a_backtest_limit(self) -> None:
        invalid = replace(self._record("run-a"), max_backtests=0)

        with open_database(self.database_path) as connection:
            with self.assertRaisesRegex(
                ValueError,
                "automated_run_backtest_limit_invalid",
            ):
                create_automated_run(connection, invalid)

    def test_only_one_active_plan_can_exist(self) -> None:
        with open_database(self.database_path) as connection:
            create_automated_run(connection, self._record("run-a"))
            with self.assertRaisesRegex(ValueError, "automated_run_active_exists"):
                create_automated_run(connection, self._record("run-b"))

    def test_external_authorization_and_limits_must_agree(self) -> None:
        invalid = replace(
            self._record("run-a"),
            real_backtests_authorized=False,
        )
        with open_database(self.database_path) as connection:
            with self.assertRaisesRegex(
                ValueError,
                "automated_run_backtest_limit_invalid",
            ):
                create_automated_run(connection, invalid)

    def test_automatic_submission_requires_real_backtest_authorization(self) -> None:
        invalid = replace(
            self._record("run-a"),
            real_backtests_authorized=False,
            max_backtests=0,
            automatic_submissions_enabled=True,
        )
        with open_database(self.database_path) as connection:
            with self.assertRaisesRegex(
                ValueError,
                "automated_run_automatic_submission_without_backtests",
            ):
                create_automated_run(connection, invalid)

    def test_exploration_reserve_cannot_exceed_the_cycle_backtest_count(self) -> None:
        invalid = replace(
            self._record("run-a"),
            minimum_exploration_backtests=21,
        )
        with open_database(self.database_path) as connection:
            with self.assertRaisesRegex(
                ValueError,
                "automated_run_exploration_reserve_invalid",
            ):
                create_automated_run(connection, invalid)

    def test_exploration_attempt_multiplier_is_positive_and_immutable(self) -> None:
        record = self._record("run-a")
        with open_database(self.database_path) as connection:
            with self.assertRaisesRegex(
                ValueError,
                "automated_run_exploration_attempt_multiplier_invalid",
            ):
                create_automated_run(
                    connection,
                    replace(record, exploration_seed_attempt_multiplier=0),
                )
            create_automated_run(connection, record)
            with self.assertRaisesRegex(
                ValueError,
                "automated_run_identity_conflict",
            ):
                replace_automated_run(
                    connection,
                    replace(record, exploration_seed_attempt_multiplier=5),
                    expected_status="created",
                )

    def test_planning_stop_diagnostic_must_match_terminal_run(self) -> None:
        created = self._record("run-a")
        running = replace(
            created,
            status="running",
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
        ).canonical_json(stop_reason="generation_attempt_budget_exhausted")
        completed = replace(
            running,
            status="completed",
            finished_at="2026-08-30T00:02:00+08:00",
            stop_reason="generation_attempt_budget_exhausted",
            candidate_planning_stop_diagnostic_json=diagnostic,
        )

        with open_database(self.database_path) as connection:
            create_automated_run(connection, created)
            replace_automated_run(connection, running, expected_status="created")
            saved = replace_automated_run(
                connection,
                completed,
                expected_status="running",
            )
            with self.assertRaisesRegex(
                ValueError,
                "automated_run_candidate_planning_diagnostic_missing",
            ):
                replace_automated_run(
                    connection,
                    replace(
                        completed,
                        candidate_planning_stop_diagnostic_json=None,
                    ),
                    expected_status="completed",
                )

        self.assertEqual(saved, completed)

    def test_backtest_link_round_trips_as_a_minimal_fact(self) -> None:
        run = self._record("run-a")
        with open_database(self.database_path) as connection:
            create_automated_run(connection, run)
            task = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula="rank(close)",
                settings={"delay": 1},
                created_at="2026-08-30T00:01:00+08:00",
            ).task
            link = AutomatedRunBacktestRecord(
                run_id=run.run_id,
                task_id=task.task_id,
                cycle_number=1,
            )
            attach_backtest_to_automated_run(connection, link)
            loaded = get_automated_run_backtest_by_task(connection, task.task_id)
            listed = list_automated_run_backtests(connection, run.run_id)

        self.assertEqual(loaded, link)
        self.assertEqual(listed, (link,))

    def test_backtest_link_requires_the_run_account(self) -> None:
        run = self._record("run-a")
        with open_database(self.database_path) as connection:
            create_automated_run(connection, run)
            task = prepare_backtest_task(
                connection,
                account_scope="other-account",
                formula="rank(close)",
                settings={"delay": 1},
                created_at="2026-08-30T00:01:00+08:00",
            ).task
            with self.assertRaisesRegex(
                ValueError,
                "automated_run_backtest_account_scope_mismatch",
            ):
                attach_backtest_to_automated_run(
                    connection,
                    AutomatedRunBacktestRecord(
                        run_id=run.run_id,
                        task_id=task.task_id,
                        cycle_number=1,
                    ),
                )

        with open_database(self.database_path) as connection:
            self.assertEqual(
                list_automated_run_backtests(connection, run.run_id),
                (),
            )

    def test_terminal_run_rejects_new_backtest_links(self) -> None:
        with open_database(self.database_path) as connection:
            for status in ("completed", "failed"):
                with self.subTest(status=status):
                    created = self._record(f"run-{status}")
                    running = replace(
                        created,
                        status="running",
                        started_at="2026-08-30T00:01:00+08:00",
                    )
                    terminal = replace(
                        running,
                        status=status,
                        finished_at="2026-08-30T00:02:00+08:00",
                        stop_reason="test_terminal",
                    )
                    create_automated_run(connection, created)
                    replace_automated_run(
                        connection,
                        running,
                        expected_status="created",
                    )
                    replace_automated_run(
                        connection,
                        terminal,
                        expected_status="running",
                    )
                    task = prepare_backtest_task(
                        connection,
                        account_scope="group-account",
                        formula="rank(close)",
                        settings={"delay": 1},
                        created_at="2026-08-30T00:03:00+08:00",
                    ).task

                    with self.assertRaisesRegex(
                        ValueError,
                        "automated_run_backtest_run_status_invalid",
                    ):
                        attach_backtest_to_automated_run(
                            connection,
                            AutomatedRunBacktestRecord(
                                run_id=terminal.run_id,
                                task_id=task.task_id,
                                cycle_number=1,
                            ),
                        )

                    self.assertEqual(
                        list_automated_run_backtests(
                            connection,
                            terminal.run_id,
                        ),
                        (),
                    )

    def test_unknown_submission_requires_reconciliation_terminal_state(self) -> None:
        created = self._record("run-a")
        running = replace(
            created,
            status="running",
            started_at="2026-08-30T00:01:00+08:00",
        )
        with open_database(self.database_path) as connection:
            create_automated_run(connection, created)
            replace_automated_run(
                connection,
                running,
                expected_status="created",
            )
            task = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula="rank(close)",
                settings={"delay": 1},
                created_at="2026-08-30T00:01:00+08:00",
            ).task
            attach_backtest_to_automated_run(
                connection,
                AutomatedRunBacktestRecord(
                    run_id=running.run_id,
                    task_id=task.task_id,
                    cycle_number=1,
                ),
            )
            record_submission_unknown(
                connection,
                task.task_id,
                observed_at="2026-08-30T00:02:00+08:00",
            )
            invalid = replace(
                running,
                status="completed",
                finished_at="2026-08-30T00:03:00+08:00",
                stop_reason="manual_stop",
            )

            with self.assertRaisesRegex(
                ValueError,
                "automated_run_submission_reconciliation_required",
            ):
                replace_automated_run(
                    connection,
                    invalid,
                    expected_status="running",
                )

            requiring_reconciliation = list_submission_reconciliation_runs(
                connection
            )

        self.assertEqual(requiring_reconciliation, (running,))

    def test_terminal_run_rejects_linked_active_tasks_at_write_boundary(
        self,
    ) -> None:
        created = self._record("run-a")
        running = replace(
            created,
            status="running",
            started_at="2026-08-30T00:01:00+08:00",
        )
        with open_database(self.database_path) as connection:
            create_automated_run(connection, created)
            replace_automated_run(connection, running, expected_status="created")
            task = prepare_backtest_task(
                connection,
                account_scope="group-account",
                formula="rank(close)",
                settings={"delay": 1},
                created_at="2026-08-30T00:01:01+08:00",
            ).task
            attach_backtest_to_automated_run(
                connection,
                AutomatedRunBacktestRecord(
                    run_id=running.run_id,
                    task_id=task.task_id,
                    cycle_number=1,
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "automated_run_failure_has_unsubmitted_backtests",
            ):
                replace_automated_run(
                    connection,
                    replace(
                        running,
                        status="failed",
                        finished_at="2026-08-30T00:01:30+08:00",
                        stop_reason="manual_failure",
                    ),
                    expected_status="running",
                )
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

            with self.assertRaisesRegex(
                ValueError,
                "automated_run_completion_has_active_backtests",
            ):
                replace_automated_run(
                    connection,
                    replace(
                        running,
                        status="completed",
                        finished_at="2026-08-30T00:02:00+08:00",
                        stop_reason="manual_stop",
                    ),
                    expected_status="running",
                )
            recovered = get_automated_run(connection, running.run_id)

        self.assertEqual(recovered, running)

    def test_cycle_settlement_outcome_must_match_frontier_fact(self) -> None:
        with open_database(self.database_path) as connection:
            create_automated_run(connection, self._record("run-a"))
            for outcome, frontier_advanced in (
                ("frontier_advanced", False),
                ("not_qualified", True),
                ("failed", True),
            ):
                with self.subTest(outcome=outcome):
                    with self.assertRaisesRegex(
                        ValueError,
                        "automated_cycle_settlement_frontier_conflict",
                    ):
                        create_automated_cycle_settlement(
                            connection,
                            AutomatedCycleSettlementRecord(
                                run_id="run-a",
                                cycle_number=1,
                                outcome=outcome,
                                frontier_advanced=frontier_advanced,
                                settled_at="2026-08-30T00:02:00+08:00",
                            ),
                        )

    def test_cycle_settlement_time_cannot_move_backwards(self) -> None:
        with open_database(self.database_path) as connection:
            create_automated_run(connection, self._record("run-a"))
            create_automated_cycle_settlement(
                connection,
                AutomatedCycleSettlementRecord(
                    run_id="run-a",
                    cycle_number=1,
                    outcome="not_qualified",
                    frontier_advanced=False,
                    settled_at="2026-08-30T00:10:00+08:00",
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "automated_cycle_settlement_time_order_invalid",
            ):
                create_automated_cycle_settlement(
                    connection,
                    AutomatedCycleSettlementRecord(
                        run_id="run-a",
                        cycle_number=2,
                        outcome="not_qualified",
                        frontier_advanced=False,
                        settled_at="2026-08-30T00:05:00+08:00",
                    ),
                )

    @staticmethod
    def _record(run_id: str) -> AutomatedRunRecord:
        settings_policy_json = json.dumps(
            {
                "catalogContext": {
                    "delay": 1,
                    "instrumentType": "EQUITY",
                    "region": "USA",
                    "universe": "TOP3000",
                },
                "decay": 4,
                "language": "FASTEXPR",
                "nanHandling": "OFF",
                "neutralization": {
                    "byFieldCategory": {"sample": "SECTOR"},
                    "default": "SECTOR",
                    "rootGroupNeutralize": "NONE",
                },
                "pasteurization": "ON",
                "truncation": {"default": 0.08, "tailRisk": 0.05},
                "unitHandling": "VERIFY",
                "visualization": False,
                "maxTrade": "OFF",
                "maxPosition": "OFF",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return AutomatedRunRecord(
            run_id=run_id,
            account_scope="group-account",
            settings_policy_json=settings_policy_json,
            settings_policy_key=sha256(
                settings_policy_json.encode("utf-8")
            ).hexdigest(),
            generation_count=500,
            backtest_count=20,
            minimum_exploration_backtests=15,
            exploration_seed_attempt_multiplier=4,
            max_cycles=5,
            max_backtests=100,
            max_pending_seconds=3600,
            max_consecutive_failures=2,
            max_request_failures=3,
            max_in_flight_backtests=3,
            real_backtests_authorized=True,
            automatic_submissions_enabled=False,
            status="created",
            current_cycle=0,
            consecutive_failures=0,
            request_failure_count=0,
            last_request_failure_code=None,
            last_request_status_code=None,
            last_request_failure_at=None,
            last_request_retry_after_seconds=None,
            created_at="2026-08-30T00:00:00+08:00",
            started_at=None,
            retry_not_before=None,
            finished_at=None,
            stop_reason=None,
            candidate_planning_stop_diagnostic_json=None,
        )


if __name__ == "__main__":
    unittest.main()
