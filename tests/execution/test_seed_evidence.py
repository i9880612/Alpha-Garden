import math
from unittest.mock import Mock

import pytest

from execution.seed_evidence import capture_next_seed_series
from execution.seeds import load_signal_frontiers
from persistence.database import open_database
from persistence.pnl import list_pnl_series
from persistence.seeds import list_signal_seeds
from persistence.submissions import PlatformSubmittedAlphaRecord, record_platform_submitted_alphas
from tests.execution import test_seeds as seed_tests
from tests.learning.test_seed_correlation import series
from worldquant.client import WorldQuantRequestError
from worldquant.backtests import WorldQuantProtocolError
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


def test_pnl_collection_admits_only_after_all_pairs_and_reuses_captured_data(pending_seed):
    case, snapshot = pending_seed
    client = Mock()
    client.fetch_pnl.side_effect = [
        PnlObservation(series("reference", [math.sin(i) for i in range(300)]).points),
        PnlObservation(series("child", [math.cos(i) for i in range(300)]).points),
    ]
    assert advance(case, snapshot, client) == 0
    with open_database(case.database_path) as connection:
        assert list_signal_seeds(connection) == ()
    assert advance(case, snapshot, client) == 0
    with open_database(case.database_path) as connection:
        assert tuple(s.root_task_id for s in list_signal_seeds(connection)) == (snapshot.task.task_id,)
        assert load_signal_frontiers(connection).active_branch_task_ids == (snapshot.task.task_id,)
    assert advance(case, snapshot, client) is None
    assert client.fetch_pnl.call_count == 2
    client.submit_backtest.assert_not_called()
    client.submit_formal_alpha.assert_not_called()


def test_pending_pnl_does_not_spin_or_admit_and_can_resume_after_cooldown(pending_seed):
    case, snapshot = pending_seed
    client = Mock()
    client.fetch_pnl.return_value = PnlObservation(None, 30.0)
    assert advance(case, snapshot, client) == 30
    assert advance(case, snapshot, client, "2026-09-09T00:00:30+00:00") == 30
    assert advance(case, snapshot, client, "2026-09-09T00:01:00+00:00") is None
    with open_database(case.database_path) as connection:
        assert not list_signal_seeds(connection)
        assert all(p.points is None for p in list_pnl_series(connection))
    assert advance(case, snapshot, client, "2026-09-09T00:10:01+00:00") == 30


def test_failed_request_does_not_fail_seed_or_research_batch(pending_seed):
    case, snapshot = pending_seed
    client = Mock()
    client.fetch_pnl.side_effect = WorldQuantRequestError("pnl_network_error", retryable=True,
                                                       outcome_unknown=False, retry_after_seconds=45)
    assert advance(case, snapshot, client) == 45
    with open_database(case.database_path) as connection:
        assert not list_signal_seeds(connection)
        record = list_pnl_series(connection)[0]
        assert record.points is None and record.retry_not_before is not None


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
