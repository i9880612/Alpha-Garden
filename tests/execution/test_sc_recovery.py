from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from execution.backtest_batches import AutomatedCandidateBacktest, prepare_automated_candidate_backtest_batch
from execution.backtests import record_submission_accepted
from execution.cycles import plan_automated_cycle
from execution.cycle_backtests import settle_failed_automated_cycle_if_terminal
from execution.launch import launch_automated_run
from execution.run_config import load_automated_run_limits
from execution.runner import run_automated_run
from execution.run_recovery import advance_stopped_run_check
from execution.runs import fail_automated_run, prepare_automated_run, start_automated_run
from execution.submission_checks import check_completed_backtest, advance_deferred_submission_check
from generation.candidate import exploration_candidate
from generation.parser import parse_formula
from persistence.backtests import get_backtest_task
from persistence.database import open_database
from persistence.pnl import save_pnl_series
from persistence.runs import get_automated_run, get_automated_cycle_settlement
from persistence.submission_checks import get_submission_check
from persistence.submission_queue import list_formal_submission_queue
from persistence.submitted_sync import complete_submitted_sync
from tests.execution import test_sc_research as sc_fixture
from tests.execution.test_launch import FakeTime, LaunchClient
from tests.learning.test_seed_correlation import series
from worldquant.backtests import BacktestSettings, BacktestPollObservation, STANDARD_REGULAR_CHECK_NAMES
from worldquant.client import WorldQuantRequestError
from worldquant.submissions import FormalCheckObservation


@pytest.fixture
def research():
    fixture = sc_fixture.SelfCorrelationResearchTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def low_quality_check(sc="PASS"):
    return {"is": {"checks": [{"name": name, "result": sc if name == "SELF_CORRELATION"
        else "FAIL" if name in {"LOW_SHARPE", "LOW_FITNESS"} else "PASS"}
        for name in STANDARD_REGULAR_CHECK_NAMES]}}


def completed_research(research):
    run = research.start()
    plan = plan_automated_cycle(research.fixture.database_path, run_id=run.run_id,
                               created_at="2026-08-30T00:05:00+08:00")
    research.fixture._complete_plan(plan, observed_at="2026-08-30T00:06:00+08:00")
    return run, plan


def test_real_sc_launch_drains_old_pending_slots_under_their_original_deadline(research):
    f = research.fixture
    config = Path(__file__).resolve().parents[2] / "config/run.default.json"
    old = prepare_automated_run(f.database_path, f.policy_path, account_scope="group-account",
        limits=replace(load_automated_run_limits(config, cycles=1), max_pending_seconds=60,
                       generation_count=6, backtest_count=2, max_backtests=2, max_in_flight_backtests=2),
        created_at="2026-08-30T00:03:00+08:00")
    old_id = old.run_id
    start_automated_run(f.database_path, old_id, started_at="2026-08-30T00:03:01+08:00")
    tasks = prepare_automated_candidate_backtest_batch(f.database_path, run_id=old_id,
        candidates=tuple(AutomatedCandidateBacktest(exploration_candidate(parse_formula(formula).expression),
            BacktestSettings.from_platform_dict(f.settings)) for formula in ("rank(close)", "rank(open)")),
        created_at="2026-08-30T00:04:00+08:00")
    with open_database(f.database_path) as connection:
        for index, task in enumerate(tasks):
            record_submission_accepted(connection, task.task.task_id, remote_id=f"https://api.worldquantbrain.com/simulations/old-{index}",
                                       observed_at="2026-08-30T00:04:00+08:00")
        save_pnl_series(connection, replace(series("submitted-alpha", [i % 3 - 1 for i in range(300)], account="group-account"),
                                           observed_at="2026-08-30T00:03:00+08:00"))
        complete_submitted_sync(connection, account_scope="group-account", alpha_ids=("submitted-alpha",),
                                completed_at="2026-08-30T00:04:00+08:00")
    fail_automated_run(f.database_path, old_id, failed_at="2026-08-30T00:04:01+08:00",
                      reason="platform_request_not_retryable:test", request_failure_code="test", request_status_code=404)
    environment = f.database_path.parent / "fixture.env"
    environment.write_text("WQB_ACCOUNT_SCOPE=group-account\nWQB_BASE_URL=https://api.worldquantbrain.com\n"
                           "WQB_SESSION_TOKEN=synthetic-test-token\n", encoding="utf-8")
    limits = replace(load_automated_run_limits(Path(__file__).resolve().parents[2] / "config/run.default.json",
        cycles=3, self_correlation_parent_task_id=research.parent_id), max_in_flight_backtests=2)
    time = FakeTime("2026-08-30T00:04:10+08:00")
    client = LaunchClient()
    client.poll_backtest = Mock(return_value=BacktestPollObservation("pending", None, "PENDING", None))
    first_submission = client.submit_backtest
    def submit(**arguments):
        if client.submit_backtest.call_count == 2:
            raise RuntimeError("new_sc_slot_available")
        return first_submission(**arguments)
    client.submit_backtest = Mock(side_effect=submit)
    def wait(seconds):
        time.wait(seconds)
        assert sum(time.waits) < 120, "SC run is stuck behind old pending tasks"
    with pytest.raises(RuntimeError, match="new_sc_slot_available"):
        launch_automated_run(f.database_path, f.policy_path, environment, limits=limits,
                            clock=time.now, waiter=wait, client_factory=lambda _: client)
    assert client.poll_backtest.call_count >= 2
    assert client.submit_backtest.call_count == 2
    with open_database(f.database_path) as connection:
        assert get_automated_run(connection, old_id).status == "failed"
        assert all(get_backtest_task(connection, task.task.task_id).task.failure_code == "platform_pending_timeout" for task in tasks)


def test_failed_sc_runner_checks_low_quality_children_before_settlement(research):
    run, plan = completed_research(research)
    database = research.fixture.database_path
    fail_automated_run(database, run.run_id, failed_at="2026-08-30T00:06:01+08:00",
                      reason="platform_request_not_retryable:test", request_failure_code="test", request_status_code=404)
    client = SimpleNamespace(authenticated=True, fetch_formal_submission_check=Mock(
        return_value=FormalCheckObservation(low_quality_check(), None)))
    time = FakeTime("2026-08-30T00:07:00+08:00")
    result = run_automated_run(database, client, run.run_id, clock=time.now, waiter=time.wait)
    assert result.run.status == "failed"
    assert client.fetch_formal_submission_check.call_count == 2
    with open_database(database) as connection:
        assert get_automated_cycle_settlement(connection, run.run_id, 1) is not None
        assert all(get_submission_check(connection, item.task.task_id).attempt_count == 1 for item in plan.backtests)
        assert list_formal_submission_queue(connection) == ()


def test_explicit_failed_sc_resume_can_finish_saved_deferred_checks_after_settlement(research):
    run, plan = completed_research(research)
    database = research.fixture.database_path
    pending = SimpleNamespace(fetch_formal_submission_check=Mock(side_effect=WorldQuantRequestError(
        "temporary", status_code=503, retryable=True, outcome_unknown=False)))
    time = FakeTime("2026-08-30T00:07:00+08:00")
    for _ in range(3):
        assert check_completed_backtest(database, pending, plan.backtests[0].task.task_id, observed_at=time.now().isoformat())
        time.wait(120)
    client = SimpleNamespace(authenticated=True, fetch_formal_submission_check=Mock(
        return_value=FormalCheckObservation(low_quality_check(), None)))
    assert check_completed_backtest(database, client, plan.backtests[1].task.task_id, observed_at=time.now().isoformat())
    client.fetch_formal_submission_check.reset_mock()
    fail_automated_run(database, run.run_id, failed_at=time.now().isoformat(),
                      reason="platform_request_not_retryable:test", request_failure_code="test", request_status_code=404)
    settlement = settle_failed_automated_cycle_if_terminal(database, run.run_id, observed_at=time.now().isoformat())
    assert settlement is not None
    with open_database(database) as connection:
        saved_settlement = get_automated_cycle_settlement(connection, run.run_id, 1)
    # Generic old-result draining does not wait on already settled deferred work.
    assert advance_stopped_run_check(database, client, account_scope="group-account", observed_at=time.now().isoformat()) is None
    result = run_automated_run(database, client, run.run_id, clock=time.now, waiter=time.wait)
    assert result.run.status == "failed"
    client.fetch_formal_submission_check.assert_called_once()
    with open_database(database) as connection:
        assert get_submission_check(connection, plan.backtests[0].task.task_id).attempt_count == 4
        assert get_automated_cycle_settlement(connection, run.run_id, 1) == saved_settlement


@pytest.mark.parametrize("response", ["503", "quality_failed_sc_pending"])
def test_sc_deferred_checks_keep_owner_scope_and_stop_after_six_attempts(research, response):
    run, plan = completed_research(research)
    database, task_id = research.fixture.database_path, plan.backtests[0].task.task_id
    fetch = Mock(side_effect=WorldQuantRequestError("temporary", status_code=503, retryable=True, outcome_unknown=False))
    if response != "503":
        fetch = Mock(return_value=FormalCheckObservation(low_quality_check("PENDING"), None))
    client = SimpleNamespace(authenticated=True, fetch_formal_submission_check=fetch)
    time = FakeTime("2026-08-30T00:07:00+08:00")
    for _ in range(3):
        assert check_completed_backtest(database, client, task_id, observed_at=time.now().isoformat())
        time.wait(120)
    with open_database(database) as connection:
        saved = get_submission_check(connection, task_id)
    time.value = time.value.fromisoformat(saved.retry_not_before)
    assert not advance_deferred_submission_check(database, client, account_scope="group-account",
        run_id="another-run", observed_at=time.now().isoformat())
    assert not advance_deferred_submission_check(database, client, account_scope="another-account",
        run_id=run.run_id, observed_at=time.now().isoformat())
    # The actual SC runner must perform one due read before advancing its work.
    with patch("execution.runner.advance_automated_run", side_effect=RuntimeError("after_deferred_read")):
        with pytest.raises(RuntimeError, match="after_deferred_read"):
            run_automated_run(database, client, run.run_id, clock=time.now, waiter=time.wait)
    assert fetch.call_count == 4
    for _ in range(2):
        time.wait(600)
        assert advance_deferred_submission_check(database, client, account_scope="group-account",
            run_id=run.run_id, observed_at=time.now().isoformat())
    time.wait(600)
    assert not advance_deferred_submission_check(database, client, account_scope="group-account",
        run_id=run.run_id, observed_at=time.now().isoformat())
    assert fetch.call_count == 6
    with open_database(database) as connection:
        assert get_backtest_task(connection, task_id).task.status == "completed"
        assert get_submission_check(connection, task_id).retry_not_before is None
        assert list_formal_submission_queue(connection) == ()
