from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from execution.backtests import record_submission_accepted
from execution.cycles import plan_automated_cycle
from execution.runner import run_automated_run
from persistence.backtests import get_backtest_task
from persistence.catalog import list_operator_catalog
from persistence.database import open_database
from persistence.runs import list_automated_run_backtests, get_automated_run
from tests.execution.test_sc_recovery import research
from tests.execution.test_launch import FakeTime
from worldquant.backtests import BacktestPollObservation
from worldquant.client import WorldQuantRequestError


def live_client(research, unavailable):
    with open_database(research.fixture.database_path) as connection:
        operators = tuple(SimpleNamespace(name=r.operator_name, scope=r.scope)
                          for r in list_operator_catalog(connection) if r.operator_name not in unavailable)
    return SimpleNamespace(authenticated=True, fetch_operators=Mock(return_value=operators),
        submit_backtest=Mock(side_effect=AssertionError("no new backtest allowed")),
        poll_backtest=Mock(return_value=BacktestPollObservation("failed", None, "ERROR", None,
                                                               "synthetic_platform_rejection")))


@pytest.mark.parametrize("missing", ["rank", "trade_when"])
def test_live_preflight_checks_parent_and_all_six_before_first_post(research, missing):
    run = research.start()
    database = research.fixture.database_path
    client = live_client(research, {missing})
    clock = FakeTime("2026-08-30T00:05:00+08:00")
    result = run_automated_run(database, client, run.run_id, clock=clock.now, waiter=clock.wait)
    assert result.run.status == "failed"
    assert result.run.stop_reason == "sc_research_operator_unavailable:" + missing
    client.submit_backtest.assert_not_called()
    client.fetch_operators.assert_called_once()
    with open_database(database) as connection:
        assert list_automated_run_backtests(connection, run.run_id) == ()
        assert get_automated_run(connection, run.run_id).self_correlation_plan_json == run.self_correlation_plan_json


def test_unavailable_operator_on_resume_drains_sent_work_without_spending_more(research):
    run = research.start()
    database = research.fixture.database_path
    plan = plan_automated_cycle(database, run_id=run.run_id, created_at="2026-08-30T00:05:00+08:00")
    sent, unsent = plan.backtests
    with open_database(database) as connection:
        record_submission_accepted(connection, sent.task.task_id,
            remote_id="https://api.worldquantbrain.com/simulations/synthetic",
            observed_at="2026-08-30T00:05:01+08:00")
    client = live_client(research, {"trade_when"})
    clock = FakeTime("2026-08-30T00:05:02+08:00")
    result = run_automated_run(database, client, run.run_id, clock=clock.now, waiter=clock.wait)
    assert result.run.status == "failed"
    client.submit_backtest.assert_not_called()
    client.poll_backtest.assert_called_once()
    with open_database(database) as connection:
        assert get_backtest_task(connection, sent.task.task_id).task.failure_code == "platform_error"
        cancelled = get_backtest_task(connection, unsent.task.task_id).task
        assert cancelled.submission_started_at is None
        assert cancelled.failure_code == "automated_run_stopped_before_submission"
        assert len(list_automated_run_backtests(connection, run.run_id)) == 2
        assert get_automated_run(connection, run.run_id).self_correlation_plan_json == run.self_correlation_plan_json


def test_operator_request_failures_are_bounded_and_not_recorded_as_missing_operators(research):
    run = research.start()
    client = live_client(research, set())
    client.fetch_operators.side_effect = WorldQuantRequestError("temporary", status_code=503,
        retryable=True, outcome_unknown=False)
    clock = FakeTime("2026-08-30T00:05:00+08:00")
    result = run_automated_run(research.fixture.database_path, client, run.run_id,
                             clock=clock.now, waiter=clock.wait)
    assert result.run.stop_reason == "request_failure_limit_reached"
    assert client.fetch_operators.call_count == run.max_request_failures
    client.submit_backtest.assert_not_called()


def test_local_preview_rejects_parent_removed_from_regular_catalog_without_network(research):
    database = research.fixture.database_path
    with open_database(database) as connection:
        connection.execute("UPDATE platform_operators SET scope_json='[\"COMBO\"]' WHERE operator_name='rank'")
        before = connection.execute("SELECT COUNT(*) FROM backtest_tasks").fetchone()[0]
    with pytest.raises(ValueError, match="sc_research_operator_unavailable:rank"):
        research.preview()
    with open_database(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM backtest_tasks").fetchone()[0] == before
