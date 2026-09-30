from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tests.execution import test_cycles as cycle_fixture
from execution.backtest_batches import AutomatedCandidateBacktest, prepare_automated_candidate_backtest_batch
from execution.cycles import plan_automated_cycle
from execution.cycle_backtests import advance_automated_cycle_backtests
from execution.driver import advance_automated_run
from execution.launch import launch_automated_run, resume_automated_run
from execution.run_config import load_automated_run_limits
from execution.runs import prepare_automated_run, start_automated_run, record_automated_cycle_settlement
from execution.sc_research import SelfCorrelationResearchPlan, preview_self_correlation_research_plan
from execution.seeds import load_signal_frontiers
from generation.self_correlation import SELF_CORRELATION_RESEARCH_FAMILIES
from persistence.backtests import BacktestMutationRecord, create_backtest_mutation, get_backtest_mutation, get_backtest_task
from persistence.catalog import OperatorRoleRecord, replace_operator_roles
from persistence.database import open_database
from persistence.runs import get_automated_run, replace_automated_run
from persistence.seeds import list_signal_seeds
from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
from persistence.pnl import list_pnl_series, save_pnl_series
from persistence.submission_queue import list_formal_submission_queue
from persistence.submissions import list_platform_submitted_alphas, record_platform_submitted_alphas
from persistence.submitted_sync import complete_submitted_sync, invalidate_submitted_sync
from selection.settings import load_backtest_settings_policy
from worldquant.backtests import BacktestSettings, STANDARD_REGULAR_CHECK_NAMES
from worldquant.pnl import PnlObservation
from worldquant.submissions import FormalCheckObservation
from tests.learning.test_seed_correlation import series
from submission.formal import assess_formal_check_payload


class SelfCorrelationResearchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = cycle_fixture.AutomatedCyclePlanningTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f._replace_catalog(fields=("model", "earnings", "estimate", "close", "open"),
            field_categories={"model": "model", "earnings": "fundamental", "estimate": "analyst"},
            field_datasets={"model": "models", "earnings": "accounts", "estimate": "forecasts"},
            cross_sectional=("rank", "zscore"), time_series=("ts_delta", "ts_av_diff", "ts_mean", "ts_rank"),
            windows=(5,22,66,120), extra_operators=(
                f._operator("days_from_last_change", "Time Series", (("x", "expr"),)),
                f._operator("trade_when", "Logical", (("x", "expr"), ("y", "expr"), ("z", "expr"))),
            ))
        with open_database(f.database_path) as connection:
            replace_operator_roles(connection, (
                OperatorRoleRecord("rank", "cross_sectional_normalization"),
                OperatorRoleRecord("zscore", "cross_sectional_normalization"),
                OperatorRoleRecord("ts_delta", "time_series_change"),
                OperatorRoleRecord("ts_av_diff", "time_series_change"),
                OperatorRoleRecord("ts_mean", "time_series_smoothing"),
                OperatorRoleRecord("ts_rank", "time_series_normalization"),
            ))
        self.parent_id = f._completed_backtest("rank(model)", sharpe=1.8, fitness=1.2,
            sharpe_status="PASS", fitness_status="PASS", time_offset=2)
        with open_database(f.database_path) as connection:
            connection.execute("INSERT INTO backtest_yearly_stats(task_id,stage,year,sharpe) VALUES (?, 'IS', 2023, 1.1)", (self.parent_id,))
        f._record_submitted_alpha("zscore(model)", settings=f.settings, sharpe=2,
            observed_at="2026-08-30T00:02:30+08:00")
        self.payload = {"is": {"checks": [
            {"name": n, "result": "FAIL" if n == "SELF_CORRELATION" else "PASS", "value": .9, "limit": .7}
            for n in STANDARD_REGULAR_CHECK_NAMES], "selfCorrelated": {
                "max": .9, "schema": {"properties": [{"name": "id"}, {"name": "correlation"}]},
                "records": [["submitted-alpha", .9]]}}}
        with open_database(f.database_path) as connection:
            save_submission_check(connection, SubmissionCheckRecord(self.parent_id,
                "2026-08-30T00:03:00+08:00", json.dumps(self.payload), None))

    def preview(self):
        return preview_self_correlation_research_plan(self.fixture.database_path,
            load_backtest_settings_policy(self.fixture.policy_path), parent_task_id=self.parent_id,
            observed_at=datetime.fromisoformat("2026-08-30T00:04:00+08:00"))

    def start(self, key=None):
        config = Path(__file__).resolve().parents[2] / "config" / "run.default.json"
        limits = load_automated_run_limits(config, cycles=3,
            self_correlation_parent_task_id=self.parent_id, self_correlation_plan_key=key)
        run = prepare_automated_run(self.fixture.database_path, self.fixture.policy_path,
            account_scope="group-account", limits=limits, created_at="2026-08-30T00:04:00+08:00")
        start_automated_run(self.fixture.database_path, run.run_id, started_at="2026-08-30T00:04:01+08:00")
        return run

    def test_preview_is_read_only_and_six_legal_unique_trials_round_trip(self):
        with open_database(self.fixture.database_path) as c:
            before = tuple(c.execute("SELECT count(*) FROM " + t).fetchone()[0]
                           for t in ("backtest_tasks", "signal_seeds", "automated_runs"))
        with patch('socket.socket.connect', side_effect=AssertionError("network forbidden")):
            plan = self.preview()
        self.assertEqual(plan.correlation, .9)
        self.assertEqual(plan.required_sharpe, 2.2)
        self.assertEqual(Counter(c.change.action for c in plan.candidates), dict.fromkeys(SELF_CORRELATION_RESEARCH_FAMILIES, 2))
        self.assertEqual(SelfCorrelationResearchPlan.from_json(plan.canonical_json()), plan)
        with open_database(self.fixture.database_path) as c:
            after = tuple(c.execute("SELECT count(*) FROM " + t).fetchone()[0]
                          for t in ("backtest_tasks", "signal_seeds", "automated_runs"))
        self.assertEqual(before, after)

    def test_launch_and_resume_keep_baseline_gate_and_frozen_sc_budget(self):
        f = self.fixture
        environment = f.database_path.parent / "fixture.env"
        environment.write_text("WQB_ACCOUNT_SCOPE=group-account\n"
                               "WQB_BASE_URL=https://api.worldquantbrain.com\n"
                               "WQB_SESSION_TOKEN=synthetic-test-token\n", encoding="utf-8")
        config = Path(__file__).resolve().parents[2] / "config" / "run.default.json"
        limits = load_automated_run_limits(config, cycles=3,
            self_correlation_parent_task_id=self.parent_id)
        clock = lambda: datetime.fromisoformat("2026-08-30T00:04:00+08:00")
        factory = Mock(side_effect=RuntimeError("fixture_stop_before_platform_access"))
        with self.assertRaisesRegex(ValueError, "submitted_baseline_incomplete"):
            launch_automated_run(f.database_path, f.policy_path, environment,
                limits=limits, clock=clock, client_factory=factory)
        factory.assert_not_called()
        with open_database(f.database_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM automated_runs").fetchone()[0], 0)
            self.assertEqual(list_signal_seeds(connection), ())
            save_pnl_series(connection, replace(series("submitted-alpha", [i % 3 - 1 for i in range(300)],
                account="group-account"), observed_at="2026-08-30T00:03:00+08:00"))
            complete_submitted_sync(connection, account_scope="group-account",
                alpha_ids=("submitted-alpha",), completed_at="2026-08-30T00:03:00+08:00")
        created = []
        with self.assertRaisesRegex(RuntimeError, "fixture_stop_before_platform_access"):
            launch_automated_run(f.database_path, f.policy_path, environment,
                limits=limits, clock=clock, client_factory=factory, run_created=created.append)
        factory.assert_called_once()
        frozen = created[0]
        self.assertEqual((frozen.backtest_count, frozen.max_cycles, frozen.max_backtests), (2, 3, 6))
        self.assertFalse(frozen.automatic_submissions_enabled)
        self.assertEqual(SelfCorrelationResearchPlan.from_json(frozen.self_correlation_plan_json).remaining_attempts, 20)
        factory.reset_mock()
        with self.assertRaisesRegex(ValueError, "existing_run_active_resume_required"):
            launch_automated_run(f.database_path, f.policy_path, environment,
                limits=limits, clock=clock, client_factory=factory)
        factory.assert_not_called()
        with open_database(f.database_path) as connection:
            self.assertEqual(get_automated_run(connection, frozen.run_id), frozen)
            invalidate_submitted_sync(connection, account_scope="group-account")
        with self.assertRaisesRegex(ValueError, "submitted_baseline_incomplete"):
            resume_automated_run(f.database_path, environment, frozen.run_id, client_factory=factory)
        factory.assert_not_called()
        with open_database(f.database_path) as connection:
            self.assertEqual(get_automated_run(connection, frozen.run_id), frozen)
            complete_submitted_sync(connection, account_scope="group-account",
                alpha_ids=("submitted-alpha",), completed_at="2026-08-30T00:04:00+08:00")
        with self.assertRaisesRegex(RuntimeError, "fixture_stop_before_platform_access"):
            resume_automated_run(f.database_path, environment, frozen.run_id, client_factory=factory)
        factory.assert_called_once()
        with open_database(f.database_path) as connection:
            self.assertEqual(get_automated_run(connection, frozen.run_id), frozen)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM automated_runs").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM backtest_mutations").fetchone()[0], 0)

    def test_frozen_six_trials_keep_original_parent_after_frontier_advances_and_resume(self):
        preview = self.preview()
        run = self.start(preview.fingerprint)
        self.assertFalse(run.automatic_submissions_enabled)
        self.assertEqual((run.backtest_count, run.max_cycles, run.max_backtests), (2, 3, 6))
        counts = Counter()
        for cycle in range(1,4):
            minute = 10 * cycle
            plan = plan_automated_cycle(self.fixture.database_path, run_id=run.run_id,
                created_at=f"2026-08-30T00:{minute:02d}:00+08:00")
            restored = plan_automated_cycle(self.fixture.database_path, run_id=run.run_id,
                created_at=f"2026-08-30T00:{minute:02d}:01+08:00")
            self.assertEqual(plan.backtests, restored.backtests)
            self.assertEqual(plan.exploration_backtest_count, 0)
            self.assertEqual({s.task.formula for s in plan.backtests},
                             {c.formula for c in preview.candidates[(cycle-1)*2:cycle*2]})
            with open_database(self.fixture.database_path) as connection:
                for s in plan.backtests:
                    m = get_backtest_mutation(connection, s.task.task_id)
                    self.assertEqual(m.parent_task_id, self.parent_id)
                    counts[m.action] += 1
            self.fixture._complete_plan(plan, observed_at=f"2026-08-30T00:{minute+1:02d}:00+08:00",
                qualified_task_id=plan.backtests[0].task.task_id if cycle == 1 else None, qualified_sharpe=3.0)
            with open_database(self.fixture.database_path) as c:
                record_automated_cycle_settlement(c, run.run_id, cycle_number=cycle,
                    outcome="not_qualified", frontier_advanced=False, observed_at=f"2026-08-30T00:{minute+2:02d}:00+08:00")
                self.assertEqual(get_automated_run(c, run.run_id).self_correlation_plan_json, preview.canonical_json())
            if cycle == 1:
                with self.assertRaisesRegex(ValueError, "initial_trials_already_started"):
                    self.preview()
        self.assertEqual(counts, dict.fromkeys(SELF_CORRELATION_RESEARCH_FAMILIES, 2))
        with open_database(self.fixture.database_path) as c:
            self.assertEqual([s.root_task_id for s in list_signal_seeds(c)], [self.parent_id])
            attempts = c.execute("SELECT count(*) FROM backtest_mutations m JOIN backtest_tasks t ON t.task_id=m.child_task_id WHERE m.parent_task_id=? AND t.submission_started_at IS NOT NULL", (self.parent_id,)).fetchone()[0]
            self.assertEqual(20-attempts, 14)
            self.assertNotIn(self.parent_id, load_signal_frontiers(c).active_branch_task_ids)

    def test_unrelated_unsupported_submitted_expression_does_not_hide_known_blocker(self):
        self.fixture._record_submitted_alpha("x=rank(estimate);x", alpha_id="unsupported", settings=self.fixture.settings,
            sharpe=2, observed_at="2026-08-30T00:02:30+08:00")
        self.assertEqual(self.preview().blockers, ("submitted-alpha",))

    def test_preview_keeps_a_reported_decommissioned_blocker_in_the_research_target(self):
        with open_database(self.fixture.database_path) as connection:
            reference = list_platform_submitted_alphas(connection, account_scope="group-account")[0]
            record_platform_submitted_alphas(connection, (replace(reference,
                status="DECOMMISSIONED", raw_payload={**reference.raw_payload, "status": "DECOMMISSIONED",
                    "settings": {**reference.raw_payload["settings"], "simulationMode": "FULL"}}),))
        plan = self.preview()
        self.assertEqual(plan.blockers, ("submitted-alpha",))
        self.assertEqual(len(plan.candidates), 6)
        self.assertTrue(all(c.change.conflict_reference_alpha_id == "submitted-alpha" for c in plan.candidates))

    def test_current_sc_target_uses_its_own_budget_when_a_better_quality_descendant_exists(self):
        # Admit the existing root without starting any of its six research trials.
        from execution.seeds import synchronize_signal_seeds
        with open_database(self.fixture.database_path) as connection:
            synchronize_signal_seeds(connection, observed_at="2026-08-30T00:03:00+08:00",
                candidate_task_ids=(self.parent_id,))
        better = self.fixture._completed_backtest("rank(earnings)", sharpe=2.8, fitness=1.8,
            sharpe_status="PASS", fitness_status="PASS", time_offset=3)
        with open_database(self.fixture.database_path) as connection:
            create_backtest_mutation(connection, BacktestMutationRecord(better, self.parent_id,
                "internal_field_replacement", "formula.arguments[0]", "model", "earnings"))
            self.assertNotIn(self.parent_id, load_signal_frontiers(connection).active_branch_task_ids)
        plan = self.preview()
        self.assertEqual((plan.parent_task_id, plan.remaining_attempts), (self.parent_id, 19))
        self.assertEqual(len(plan.candidates), 6)

    def test_latest_pending_or_other_failure_rejects_plan_without_new_run(self):
        with open_database(self.fixture.database_path) as c:
            save_submission_check(c, SubmissionCheckRecord(self.parent_id,
                "2026-08-30T00:03:30+08:00", None, "request_error", attempt_count=2))
        with self.assertRaisesRegex(ValueError, "not_only_sc_failed"):
            self.preview()
        with open_database(self.fixture.database_path) as c:
            self.assertEqual(c.execute("SELECT count(*) FROM automated_runs").fetchone()[0], 0)

    def test_reviewed_plan_change_rolls_back_seed_and_run_creation(self):
        with self.assertRaisesRegex(ValueError, "reviewed_plan_changed"):
            self.start("0" * 64)
        with open_database(self.fixture.database_path) as c:
            self.assertEqual(list_signal_seeds(c), ())
            self.assertEqual(c.execute("SELECT count(*) FROM automated_runs").fetchone()[0], 0)

    def test_run_identity_and_frozen_tasks_cannot_be_replaced(self):
        run = self.start()
        plan = SelfCorrelationResearchPlan.from_json(run.self_correlation_plan_json)
        with open_database(self.fixture.database_path) as c:
            current = get_automated_run(c, run.run_id)
            with self.assertRaisesRegex(ValueError, "identity_conflict"):
                replace_automated_run(c, replace(current, self_correlation_plan_json=plan.canonical_json().replace('"remaining_attempts":20','"remaining_attempts":19')), expected_status="running")
        settings = BacktestSettings.from_platform_dict(json.loads(plan.settings_json))
        with self.assertRaisesRegex(ValueError, "frozen_candidates_mismatch"):
            prepare_automated_candidate_backtest_batch(self.fixture.database_path, run_id=run.run_id,
                candidates=tuple(AutomatedCandidateBacktest(c, settings) for c in plan.candidates[2:4]),
                cycle_number=1, created_at="2026-08-30T00:05:00+08:00")

    def test_focus_planning_does_not_request_unrelated_historical_evidence(self):
        run = self.start()
        with patch('execution.driver.capture_next_seed_series', side_effect=AssertionError("unrelated PnL")), \
             patch('execution.driver.capture_next_recovery_series', side_effect=AssertionError("unrelated recovery")), \
             patch('execution.driver.advance_stopped_run_check', side_effect=AssertionError("unrelated checks")):
            result = advance_automated_run(self.fixture.database_path, object(), run.run_id,
                observed_at="2026-08-30T00:05:00+08:00")
        self.assertFalse(result.platform_request_performed)

    def test_batch_records_sc_and_curves_even_when_quality_fails_without_backfill_or_submission(self):
        run = self.start()
        plan = plan_automated_cycle(self.fixture.database_path, run_id=run.run_id,
            created_at="2026-08-30T00:05:00+08:00")
        self.fixture._complete_plan(plan, observed_at="2026-08-30T00:06:00+08:00")
        payload = {"is": {"checks": [
            {"name": name, "result": "FAIL" if name in {"LOW_SHARPE", "LOW_FITNESS"} else "PASS"}
            for name in STANDARD_REGULAR_CHECK_NAMES]}}
        client = Mock()
        client.fetch_formal_submission_check.return_value = FormalCheckObservation(payload, None)
        client.fetch_pnl.side_effect = lambda *, platform_alpha_id: PnlObservation(
            series(platform_alpha_id, [i % 3 - 1 for i in range(300)]).points)
        for _ in range(12):
            result = advance_automated_cycle_backtests(self.fixture.database_path, client, run.run_id,
                observed_at="2026-08-30T00:07:00+08:00")
            if result.action == "cycle_terminal":
                break
        self.assertEqual(result.action, "cycle_terminal")
        with open_database(self.fixture.database_path) as connection:
            from persistence.submission_checks import get_submission_check
            ids = {get_backtest_task(connection, s.task.task_id).task.platform_alpha_id for s in plan.backtests}
            for candidate in plan.backtests:
                assessment = assess_formal_check_payload(json.loads(get_submission_check(
                    connection, candidate.task.task_id).payload_json))
                self.assertEqual(dict(assessment.statuses)["SELF_CORRELATION"], "PASS")
                self.assertEqual(assessment.state, "failed")
            self.assertEqual({s.platform_alpha_id for s in list_pnl_series(connection)},
                             ids | {"submitted-alpha"})
            self.assertEqual(list_formal_submission_queue(connection), ())
        self.assertEqual({c.kwargs["platform_alpha_id"] for c in client.fetch_formal_submission_check.call_args_list}, ids)
        self.assertEqual({c.kwargs["platform_alpha_id"] for c in client.fetch_pnl.call_args_list}, ids | {"submitted-alpha"})
        client.submit_backtest.assert_not_called()
        client.submit_formal_alpha.assert_not_called()


if __name__ == "__main__":
    unittest.main()
