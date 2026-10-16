import io
import math
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from execution.backtests import record_submission_accepted
from execution.real_backtests import advance_real_backtest
from execution.driver import advance_automated_run
from execution.runner import run_automated_run
from execution.runs import fail_automated_run
from alpha_garden.console import run_progress_console
from execution.progress import live_progress
from persistence.database import open_database
from persistence.backtests import get_backtest_task
from persistence.pnl import list_pnl_series
from persistence.runs import get_automated_run, list_automated_run_backtests
from tests.execution import test_real_backtests as fixtures
from tests.execution import test_driver as driver_fixtures
from tests.execution import test_runner as runner_fixtures
from tests.learning.test_seed_correlation import series
from worldquant.backtests import (
    BacktestPollObservation, BacktestYearlyStatsObservation,
    WorldQuantDetailMetricUnavailable, WorldQuantProtocolError, WorldQuantZeroCapitalResult,
    parse_backtest_detail,
)
from worldquant.client import WorldQuantRequestError
from worldquant.pnl import PnlObservation


@pytest.fixture
def task():
    case = fixtures.RealBacktestExecutionTests()
    case.setUp()
    prepared = case._prepare()
    with open_database(case.database_path) as connection:
        record_submission_accepted(connection, prepared.task.task_id,
            remote_id="https://api.worldquantbrain.com/simulations/one",
            observed_at="2026-08-29T00:02:00+00:00")
    client = Mock()
    client.poll_backtest.return_value = BacktestPollObservation("completed", "alpha-1", "COMPLETE", None)
    client.fetch_backtest_detail.return_value = parse_backtest_detail(case._detail_payload(),
        expected_alpha_id="alpha-1", expected_formula=case.formula, expected_settings=case.settings.as_platform_dict())
    client.fetch_backtest_yearly_stats.return_value = BacktestYearlyStatsObservation("ready", (), None)
    client.fetch_pnl.return_value = PnlObservation(series("alpha-1", [math.sin(i) for i in range(300)]).points)
    try:
        yield case, prepared.task.task_id, client
    finally:
        case.doCleanups()


def advance(task, second):
    case, task_id, client = task
    return advance_real_backtest(case.database_path, client, task_id,
        observed_at=f"2026-08-29T00:{3 + second // 60:02d}:{second % 60:02d}+00:00")


@pytest.mark.parametrize("quality_passed", [True, False])
def test_score_then_pnl_before_yearly_stats_without_waiting_for_batch(task, quality_passed):
    case, _, client = task
    if not quality_passed:
        payload = case._detail_payload()
        payload["is"]["checks"] = [{"name": "LOW_SHARPE", "result": "FAIL"}]
        client.fetch_backtest_detail.return_value = parse_backtest_detail(payload,
            expected_alpha_id="alpha-1", expected_formula=case.formula, expected_settings=case.settings.as_platform_dict())
    assert advance(task, 0).action == "detail_ready"
    assert advance(task, 1).action == "detail_captured"
    client.fetch_pnl.assert_not_called()
    assert advance(task, 2).action == "pnl_captured"
    client.fetch_backtest_yearly_stats.assert_not_called()
    with open_database(case.database_path) as connection:
        assert list_pnl_series(connection)[0].points == client.fetch_pnl.return_value.points
    assert advance(task, 3).action == "completed"
    assert advance(task, 4).action == "already_completed"
    client.fetch_pnl.assert_called_once_with(platform_alpha_id="alpha-1")
    assert [call[0] for call in client.method_calls] == [
        "poll_backtest", "fetch_backtest_detail", "fetch_pnl", "fetch_backtest_yearly_stats"]


@pytest.mark.parametrize("state", ["pending", "failed", "error"])
def test_no_pnl_without_returned_formula(task, state):
    case, _, client = task
    if state == "error":
        client.poll_backtest.side_effect = WorldQuantRequestError("poll_failed", status_code=503,
            retryable=True, outcome_unknown=False)
        with pytest.raises(WorldQuantRequestError):
            advance(task, 0)
    else:
        client.poll_backtest.return_value = BacktestPollObservation(state, None, state.upper(), None)
        assert advance(task, 0).action == state
    client.fetch_pnl.assert_not_called()
    client.fetch_backtest_detail.assert_not_called()
    with open_database(case.database_path) as connection:
        assert list_pnl_series(connection) == ()


@pytest.mark.parametrize("error", [
    WorldQuantRequestError("detail_failed", status_code=503, retryable=True, outcome_unknown=False),
    WorldQuantProtocolError("worldquant_detail_formula_mismatch"),
    WorldQuantZeroCapitalResult(),
    WorldQuantDetailMetricUnavailable("sharpe"),
])
def test_no_pnl_on_invalid_or_failed_score(task, error):
    case, _, client = task
    advance(task, 0)
    client.fetch_backtest_detail.side_effect = error
    if isinstance(error, WorldQuantRequestError):
        with pytest.raises(WorldQuantRequestError):
            advance(task, 1)
    else:
        assert advance(task, 1).action == "failed"
    client.fetch_pnl.assert_not_called()
    with open_database(case.database_path) as connection:
        assert list_pnl_series(connection) == ()


def test_partial_checks_without_saved_score_do_not_collect_pnl(task):
    case, _, client = task
    advance(task, 0)
    payload = case._detail_payload()
    payload["is"]["checks"] = [{"name": "LOW_SHARPE", "result": "PASS"}]
    client.fetch_backtest_detail.return_value = parse_backtest_detail(payload,
        expected_alpha_id="alpha-1", expected_formula=case.formula, expected_settings=case.settings.as_platform_dict())
    assert advance(task, 1).action == "checks_pending"
    client.fetch_pnl.assert_not_called()


def test_pnl_failure_preserves_score_and_cooldown_while_yearly_stats_continue(task):
    case, _, client = task
    advance(task, 0)
    advance(task, 1)
    client.fetch_pnl.side_effect = WorldQuantRequestError("pnl_failed", status_code=503,
        retryable=True, outcome_unknown=False)
    pending = advance(task, 2)
    assert pending.action == "pnl_pending"
    assert pending.snapshot.result.sharpe == 1.3
    with open_database(case.database_path) as connection:
        saved = list_pnl_series(connection)[0]
        assert saved.points is None
        assert saved.retry_not_before == "2026-08-29T00:13:02+00:00"
    assert advance(task, 3).action == "completed"
    client.fetch_pnl.assert_called_once()


@pytest.mark.parametrize("status", [401, 403, 429])
def test_pnl_account_errors_propagate_without_erasing_score(task, status):
    case, _, client = task
    advance(task, 0)
    advance(task, 1)
    client.fetch_pnl.side_effect = WorldQuantRequestError("pnl_account_error", status_code=status,
        retryable=status == 429, outcome_unknown=False)
    with pytest.raises(WorldQuantRequestError):
        advance(task, 2)
    client.fetch_backtest_yearly_stats.assert_not_called()
    with open_database(case.database_path) as connection:
        assert list_pnl_series(connection) == ()


def test_pnl_account_limit_is_not_rewritten_as_simulation_timeout(task):
    case, task_id, client = task
    advance(task, 0)
    advance(task, 1)
    client.fetch_pnl.side_effect = WorldQuantRequestError("worldquant_pnl_http_error", status_code=429,
        retryable=True, outcome_unknown=False, retry_after_seconds=120)
    with pytest.raises(WorldQuantRequestError) as error:
        advance_real_backtest(case.database_path, client, task_id,
            observed_at="2026-08-29T01:00:00+00:00", max_pending_seconds=60)
    assert error.value.status_code == 429
    with open_database(case.database_path) as connection:
        saved = get_backtest_task(connection, task_id)
        assert saved.task.status == "pending"
        assert saved.task.failure_code is None
        assert saved.result.sharpe == 1.3


@pytest.mark.parametrize("retry_after,expected_delay", [(120, 120), (None, 60)])
def test_pnl_rate_limit_pauses_all_peers_and_survives_restart(retry_after, expected_delay):
    case = driver_fixtures.AutomatedRunDriverTests()
    case.setUp()
    try:
        run_id = case._prepare_run(backtest_count=3, max_backtests=3)
        client = driver_fixtures.DriverClient(*(case._accepted(f"peer-{i}") for i in range(3)))
        case._advance(client, minute=1)
        case._prepare_backtests(run_id, ("rank(close)", "rank(open)", "rank(volume)"),
            created_at="2026-08-30T00:01:30+08:00")
        now = datetime.fromisoformat("2026-08-30T00:02:00+08:00")
        with patch.object(client, "fetch_pnl", side_effect=WorldQuantRequestError(
                "worldquant_pnl_http_error", status_code=429, retryable=True, outcome_unknown=False,
                retry_after_seconds=retry_after)) as pnl:
            for _ in range(20):
                result = advance_automated_run(case.database_path, client, run_id, observed_at=now.isoformat())
                if result.action == "request_retry_scheduled":
                    break
                now += timedelta(seconds=1)
            else:
                pytest.fail("PnL rate limit did not reach the run boundary")
            pnl.assert_called_once()
        with open_database(case.database_path) as connection:
            run = get_automated_run(connection, run_id)
            tasks = [get_backtest_task(connection, link.task_id)
                     for link in list_automated_run_backtests(connection, run_id)]
        assert run.request_failure_count == 1
        assert run.retry_not_before == (now + timedelta(seconds=expected_delay)).isoformat()
        assert all(item.task.status == "pending" and item.task.retry_not_before is None for item in tasks)
        assert any(item.result is not None for item in tasks)

        # New process/client, same database: no submit, polling, check or PnL
        # request is allowed before the persisted account cooldown expires.
        resumed_client = driver_fixtures.DriverClient()
        clock = runner_fixtures.FakeTime((now + timedelta(seconds=1)).isoformat())
        def stop_during_cooldown(seconds):
            assert seconds == expected_delay - 1
            assert resumed_client.calls == []
            raise KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            run_automated_run(case.database_path, resumed_client, run_id,
                clock=clock.now, waiter=stop_during_cooldown)
        resumed = advance_automated_run(case.database_path, resumed_client, run_id,
            observed_at=(now + timedelta(seconds=expected_delay)).isoformat())
        assert resumed.platform_request_performed
        assert resumed.run.request_failure_count == 0
        assert resumed.run.retry_not_before is None
        assert "submit" not in resumed_client.calls
        with open_database(case.database_path) as connection:
            assert len(list_automated_run_backtests(connection, run_id)) == 3
    finally:
        case.doCleanups()


@pytest.mark.parametrize("poll_interval,retry_after,expected_delay", [(1, None, 2), (5, None, 5), (1, 10, 10)])
@pytest.mark.parametrize("recover_old_task", [False, True])
def test_pnl_pacing_applies_before_peer_requests_and_does_not_extend_deadlines(
        poll_interval, retry_after, expected_delay, recover_old_task):
    case = runner_fixtures.AutomatedRunnerTests()
    case.setUp()
    try:
        run_id = case._prepare_running_task(max_pending_seconds=5,
            formulas=("rank(close)", "rank(open)"))
        client = driver_fixtures.DriverClient()
        with open_database(case.database_path) as connection:
            task_ids = [link.task_id for link in list_automated_run_backtests(connection, run_id)]
            for i, task_id in enumerate(task_ids):
                record_submission_accepted(connection, task_id,
                    remote_id=f"https://api.worldquantbrain.com/simulations/peer-{i}",
                    observed_at="2026-08-30T00:01:31+08:00")
        # First candidate is scored, while its peer still occupies a slot.
        advance_real_backtest(case.database_path, client, task_ids[0],
            observed_at="2026-08-30T00:01:32+08:00")
        advance_real_backtest(case.database_path, client, task_ids[0],
            observed_at="2026-08-30T00:01:33+08:00")
        owner_id = run_id
        if recover_old_task:
            fail_automated_run(case.database_path, run_id,
                failed_at="2026-08-30T00:01:34+08:00", reason="test_interruption")
            run_id = case._prepare_running_task(max_pending_seconds=5, formulas=("rank(volume)",))
            client.submissions.append(case._accepted("current"))
        clock = runner_fixtures.FakeTime("2026-08-30T00:01:35+08:00")
        requests = []
        with ExitStack() as stack:
            for name in ("submit_backtest", "poll_backtest", "fetch_backtest_detail", "fetch_pnl",
                         "fetch_backtest_yearly_stats", "fetch_formal_submission_check"):
                original = getattr(client, name)
                def request(*args, _name=name, _original=original, **kwargs):
                    requests.append((_name, clock.now()))
                    if _name == "fetch_pnl" and retry_after is not None:
                        return PnlObservation(None, retry_after)
                    return _original(*args, **kwargs)
                stack.enter_context(patch.object(client, name, side_effect=request))
            completed = run_automated_run(case.database_path, client, run_id,
                poll_interval_seconds=poll_interval, clock=clock.now, waiter=clock.wait)
        assert completed.run.status == "completed"
        pnl_requests = [i for i, (name, _) in enumerate(requests) if name == "fetch_pnl"]
        assert pnl_requests
        for i in pnl_requests:
            assert i + 1 < len(requests)
            assert (requests[i + 1][1] - requests[i][1]).total_seconds() >= expected_delay
        with open_database(case.database_path) as connection:
            assert get_automated_run(connection, owner_id).max_pending_seconds == 5
            for task_id in task_ids:
                assert get_backtest_task(connection, task_id).task.submission_started_at == "2026-08-30T00:01:31+08:00"
    finally:
        case.doCleanups()


@pytest.mark.parametrize("tty", [False, True])
def test_pnl_logs_are_separate_persistent_lines_even_with_live_progress(task, tty):
    class Output(io.StringIO):
        def isatty(self):
            return tty

    _, _, client = task
    advance(task, 0)
    advance(task, 1)
    output = Output()
    observation = client.fetch_pnl.return_value

    def fetch_pnl(**kwargs):
        # The start line must already be visible while the HTTP read is pending.
        assert output.getvalue().endswith("[PnL采集] alpha-1 开始获取每日曲线\n")
        return observation

    client.fetch_pnl.side_effect = fetch_pnl
    with redirect_stdout(output), run_progress_console():
        live_progress("等待其他公式")
        assert advance(task, 2).action == "pnl_captured"
        live_progress("继续等待其他公式")
        advance(task, 3)
    lines = [line[line.index("[PnL采集]"):] for line in output.getvalue().splitlines()
             if "[PnL采集]" in line]
    assert lines == ["[PnL采集] alpha-1 开始获取每日曲线",
                     f"[PnL采集] alpha-1 每日曲线已保存（{len(observation.points)} 个点）"]
