import math
from unittest.mock import Mock

import pytest

from execution.backtests import (
    apply_backtest_detail, prepare_backtest_task, record_backtest_yearly_stats, record_submission_accepted,
)
from execution.seed_evidence import capture_next_seed_series
from execution.seeds import load_signal_frontiers
from persistence.database import open_database
from persistence.pnl import list_pnl_series, save_pnl_series
from persistence.seeds import list_signal_seeds
from persistence.submissions import PlatformSubmittedAlphaRecord, record_platform_submitted_alphas
from tests.execution import test_seeds as seed_tests
from tests.learning.test_seed_correlation import series
from worldquant.client import WorldQuantRequestError
from worldquant.backtests import (
    BacktestCheck, BacktestDetail, BacktestYearlyStat, STANDARD_REGULAR_CHECK_NAMES, WorldQuantProtocolError,
)
from worldquant.pnl import PnlObservation


@pytest.fixture
def pending_seed():
    case = seed_tests.SignalSeedExecutionTests()
    case.setUp()
    with open_database(case.database_path) as connection:
        snapshot = case._completed(connection, "rank(close)", "1", 1.2, 0.8)
        payload = {"id": "reference", "status": "ACTIVE", "hidden": False,
                   "dateSubmitted": "2026-09-01T00:00:00+00:00", "regular": {"code": "rank(reference)"}}
        record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
            "group-account", "reference", "rank(reference)", "ACTIVE", payload["dateSubmitted"],
            False, payload, "2026-09-09T00:00:00+00:00"),))
    yield case, snapshot
    case.doCleanups()


def advance(case, snapshot, client, observed="2026-09-09T00:00:00+00:00"):
    return capture_next_seed_series(case.database_path, client, account_scope="group-account",
                                   observed_at=observed, candidate_task_ids=(snapshot.task.task_id,))


def test_pnl_collection_admits_only_after_all_pairs_and_reuses_captured_data(pending_seed, caplog):
    case, snapshot = pending_seed
    caplog.set_level("INFO", logger="execution.progress")
    client = Mock()
    observations = iter([
        PnlObservation(series("reference", [math.sin(i) for i in range(300)]).points),
        PnlObservation(series("child", [math.cos(i) for i in range(300)]).points),
    ])
    def fetch_pnl(*, platform_alpha_id):
        record = caplog.records[-1]
        assert f"正在获取时序数据 {platform_alpha_id}" in record.getMessage()
        assert not getattr(record, "transient", False)  # Must reach the web log handler before the request finishes.
        return next(observations)
    client.fetch_pnl.side_effect = fetch_pnl
    assert advance(case, snapshot, client) == 0
    assert "获取完成（已获取 1/2）" in caplog.messages[-1]
    with open_database(case.database_path) as connection:
        assert list_signal_seeds(connection) == ()
    assert advance(case, snapshot, client) == 0
    assert "获取完成（已获取 2/2）" in caplog.messages[-1]
    with open_database(case.database_path) as connection:
        assert tuple(s.root_task_id for s in list_signal_seeds(connection)) == (snapshot.task.task_id,)
        assert load_signal_frontiers(connection).active_branch_task_ids == (snapshot.task.task_id,)
    count = len(caplog.records)
    assert advance(case, snapshot, client) is None
    assert len(caplog.records) == count
    assert client.fetch_pnl.call_count == 2
    client.submit_backtest.assert_not_called()
    client.submit_formal_alpha.assert_not_called()


def test_pending_pnl_does_not_spin_or_admit_and_can_resume_after_cooldown(pending_seed, caplog):
    case, snapshot = pending_seed
    client = Mock()
    client.fetch_pnl.return_value = PnlObservation(None, 30.0)
    assert advance(case, snapshot, client) == 30
    assert "暂未就绪，证据待定，600 秒后可重试" in caplog.messages[-1]
    assert "获取完成" not in caplog.text
    assert advance(case, snapshot, client, "2026-09-09T00:00:30+00:00") == 30
    assert advance(case, snapshot, client, "2026-09-09T00:01:00+00:00") is None
    with open_database(case.database_path) as connection:
        assert not list_signal_seeds(connection)
        assert all(p.points is None for p in list_pnl_series(connection))
    assert advance(case, snapshot, client, "2026-09-09T00:10:01+00:00") == 30


def test_failed_request_does_not_fail_seed_or_research_batch(pending_seed, caplog):
    case, snapshot = pending_seed
    client = Mock()
    client.fetch_pnl.side_effect = WorldQuantRequestError("pnl_network_error", retryable=True,
                                                       outcome_unknown=False, retry_after_seconds=45)
    assert advance(case, snapshot, client) == 45
    assert "读取失败（pnl_network_error），证据待定，600 秒后可重试" in caplog.messages[-1]
    with open_database(case.database_path) as connection:
        assert not list_signal_seeds(connection)
        record = list_pnl_series(connection)[0]
        assert record.points is None and record.retry_not_before is not None


def checked_result(connection, identity, *, failed_check=None):
    task = prepare_backtest_task(connection, account_scope="group-account", formula=f"ts_mean(close,{identity})",
                                 settings={"delay": 1}, created_at=f"2026-09-01T00:0{identity}:00+00:00")
    record_submission_accepted(connection, task.task.task_id, remote_id=f"simulation_{identity}",
                               observed_at=f"2026-09-01T00:0{identity}:10+00:00")
    finished = f"2026-09-01T00:0{identity}:20+00:00"
    checks = tuple(BacktestCheck(name, "PENDING" if name == "SELF_CORRELATION" else
                                 "FAIL" if name == failed_check else "PASS", None, None, None)
                   for name in sorted(STANDARD_REGULAR_CHECK_NAMES))
    apply_backtest_detail(connection, task.task.task_id, BacktestDetail(
        platform_alpha_id=f"alpha_{identity}", sharpe=1.5, fitness=1.2, turnover=0.12, returns=0.08,
        drawdown=0.04, margin=0.001, book_size=20_000_000, pnl=100_000, checks=checks, grade="GOOD",
    ), observed_at=finished)
    return record_backtest_yearly_stats(connection, task.task.task_id, (
        BacktestYearlyStat(2023, 100_000, 20_000_000, 0.12, 1.5, 0.08, 0.04, 0.001, 1.2, 100, 100, "IS"),
    ), observed_at=finished)


def test_unchecked_result_passing_non_sc_checks_collects_local_sc_evidence(pending_seed):
    case, _ = pending_seed
    with open_database(case.database_path) as connection:
        eligible = checked_result(connection, "7")
        checked_result(connection, "8", failed_check="LOW_FITNESS")
    client = Mock()
    client.fetch_pnl.side_effect = lambda *, platform_alpha_id: PnlObservation(
        series(platform_alpha_id, [math.sin(i) for i in range(300)]).points)
    results = [capture_next_seed_series(case.database_path, client, account_scope="group-account",
                                        observed_at="2026-09-09T00:00:00+00:00", candidate_task_ids=())
               for _ in range(3)]
    assert results == [0.0, 0.0, None]
    assert [call.kwargs["platform_alpha_id"] for call in client.fetch_pnl.call_args_list] == [
        "reference", eligible.task.platform_alpha_id]
    client.fetch_formal_submission_check.assert_not_called()
    client.submit_formal_alpha.assert_not_called()


@pytest.mark.parametrize("admit_seeds", [True, False])
def test_backfill_resumes_after_cooldown_without_blocking_other_candidates(pending_seed, admit_seeds):
    case, _ = pending_seed
    with open_database(case.database_path) as connection:
        first, second = (checked_result(connection, identity) for identity in ("7", "8"))
    attempts = {}

    def fetch_pnl(*, platform_alpha_id):
        attempts[platform_alpha_id] = attempts.get(platform_alpha_id, 0) + 1
        if platform_alpha_id == first.task.platform_alpha_id and attempts[platform_alpha_id] == 1:
            raise WorldQuantRequestError("pnl_network_error", retryable=True, outcome_unknown=False)
        increments = [math.sin(i) if platform_alpha_id == "reference" else math.cos(i) for i in range(300)]
        return PnlObservation(series(platform_alpha_id, increments).points)

    client = Mock()
    client.fetch_pnl.side_effect = fetch_pnl
    def capture(observed="2026-09-09T00:00:00+00:00"):
        return capture_next_seed_series(case.database_path, client, account_scope="group-account",
            observed_at=observed, candidate_task_ids=(first.task.task_id, second.task.task_id), admit_seeds=admit_seeds)

    assert [capture() for _ in range(4)] == [0.0, 0.0, 0.0, None]
    with open_database(case.database_path) as connection:
        assert {s.root_task_id for s in list_signal_seeds(connection)} == (
            {second.task.task_id} if admit_seeds else set())
        failed = next(p for p in list_pnl_series(connection) if p.platform_alpha_id == first.task.platform_alpha_id)
        assert failed.points is None and failed.retry_not_before is not None
    assert capture("2026-09-09T00:10:01+00:00") == 0.0
    assert capture("2026-09-09T00:10:02+00:00") is None
    with open_database(case.database_path) as connection:
        assert {s.root_task_id for s in list_signal_seeds(connection)} == (
            {first.task.task_id, second.task.task_id} if admit_seeds else set())
    assert attempts == {"reference": 1, first.task.platform_alpha_id: 2, second.task.platform_alpha_id: 1}
    client.submit_backtest.assert_not_called()
    client.submit_formal_alpha.assert_not_called()


@pytest.mark.parametrize("add_blocking_reference", [False, True])
def test_final_seed_pass_uses_current_references_for_cached_candidate(pending_seed, add_blocking_reference):
    case, _ = pending_seed
    with open_database(case.database_path) as connection:
        cached, missing = (checked_result(connection, identity) for identity in ("7", "8"))
        save_pnl_series(connection, series(cached.task.platform_alpha_id,
                        [math.cos(i) for i in range(300)], account="group-account"))
    client = Mock()
    client.fetch_pnl.side_effect = lambda *, platform_alpha_id: PnlObservation(series(platform_alpha_id,
        [math.cos(i) if platform_alpha_id == "blocking" else math.sin(i) for i in range(300)]).points)
    def capture():
        return capture_next_seed_series(case.database_path, client, account_scope="group-account",
            observed_at="2026-09-09T00:00:00+00:00", candidate_task_ids=(cached.task.task_id, missing.task.task_id))
    assert capture() == 0.0  # First reference is now available; cached candidate has not been admitted.
    if add_blocking_reference:
        with open_database(case.database_path) as connection:
            record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
                "group-account", "blocking", "rank(blocking)", "ACTIVE", "2026-09-08T00:00:00+00:00", False,
                {"id": "blocking", "status": "ACTIVE", "hidden": False,
                 "dateSubmitted": "2026-09-08T00:00:00+00:00", "regular": {"code": "rank(blocking)"},
                 "is": {"sharpe": 1.5}}, "2026-09-09T00:00:00+00:00"),))
        assert capture() == 0.0
    assert [capture() for _ in range(2)] == [0.0, None]
    with open_database(case.database_path) as connection:
        assert {s.root_task_id for s in list_signal_seeds(connection)} == (
            set() if add_blocking_reference else {cached.task.task_id})
    assert {call.kwargs["platform_alpha_id"] for call in client.fetch_pnl.call_args_list} == {
        "reference", missing.task.platform_alpha_id} | ({"blocking"} if add_blocking_reference else set())


def test_protocol_error_is_deferred_but_unrelated_program_error_is_not_swallowed(pending_seed):
    case, snapshot = pending_seed
    client = Mock()
    client.fetch_pnl.side_effect = WorldQuantProtocolError("worldquant_pnl_response_invalid")
    assert advance(case, snapshot, client) == 0
    with open_database(case.database_path) as connection:
        assert not list_signal_seeds(connection)
        assert list_pnl_series(connection)[0].points is None
    client.fetch_pnl.side_effect = ValueError("program_invariant_broken")
    with pytest.raises(ValueError, match="program_invariant_broken"):
        advance(case, snapshot, client, "2026-09-09T00:00:01+00:00")
