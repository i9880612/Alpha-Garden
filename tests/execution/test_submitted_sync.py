from copy import deepcopy
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from execution.submitted_sync import SubmittedSyncPolicy, sync_submitted_alphas
from persistence.database import open_database
from persistence.pnl import list_pnl_series
from persistence.schema import initialize_database_schema
from persistence.submissions import list_platform_submitted_alphas
from persistence.submitted_sync import require_submitted_baseline
from worldquant.alphas import parse_user_alpha_page
from worldquant.client import WorldQuantRequestError, WorldQuantClient, WorldQuantCredentials, WorldQuantResponse
from worldquant.pnl import PnlObservation


STAMP = "2026-09-01T00:00:00+00:00"


@pytest.mark.parametrize("statuses,success", [
    ([201, 401, 503, 201, 200, 200, 200, 200], True),
    ([201, 401, 503, 503, 503], False),
    ([201, 401, 403], False),
])
def test_refresh_failure_reauthenticates_inside_the_bounded_read_retry_budget(setup, statuses, success):
    database, _, run = setup
    replies = list(statuses)
    requests = []
    def execute(request, timeout):
        requests.append((request.get_method(), request.full_url.split("?")[0]))
        status = replies.pop(0)
        body = {"count": 0, "next": None, "previous": None, "results": []}
        return WorldQuantResponse(status, {}, json.dumps(body).encode())
    client = WorldQuantClient(base_url="https://api.worldquantbrain.com",
        credentials=WorldQuantCredentials(bearer_token="synthetic-test-token"), request_executor=execute)
    if success:
        assert run(client).completed
        assert [method for method, _ in requests] == ["POST", "GET", "POST", "POST", "GET", "GET", "GET", "GET"]
    else:
        with pytest.raises(WorldQuantRequestError) as error:
            run(client)
        assert error.value.status_code == statuses[-1]
        with pytest.raises(ValueError, match="baseline_incomplete"):
            require_complete(database)
    assert not replies
    assert len(requests) == len(statuses)


def test_identical_resync_keeps_evidence_time_but_updates_completed_scan_time(setup):
    database, clock, run = setup
    client = ReadOnlyClient([payload()])
    run(client)
    clock[0] += timedelta(days=1)
    run(client)
    with open_database(database) as connection:
        assert list_platform_submitted_alphas(connection, account_scope="fixture-account")[0].observed_at == STAMP
        assert connection.execute("SELECT completed_at FROM platform_submitted_alpha_syncs").fetchone()[0] == clock[0].isoformat()
    # Changed economic evidence is current evidence, never backdated to the first scan.
    client.values[0]["is"]["sharpe"] = 2.0
    clock[0] += timedelta(days=1)
    run(client)
    with open_database(database) as connection:
        record, = list_platform_submitted_alphas(connection, account_scope="fixture-account")
        assert record.observed_at == clock[0].isoformat()
        assert record.raw_payload["is"]["sharpe"] == 2.0


def payload(alpha="a", *, hidden=False, status="ACTIVE"):
    return {"id": alpha, "type": "REGULAR", "status": status, "hidden": hidden,
            "dateCreated": STAMP, "dateSubmitted": STAMP,
            "regular": {"code": f"rank({alpha})"}, "settings": {"delay": 1},
            "is": {"sharpe": 1.5}, "futureField": {"nested": [1, True]}}


class ReadOnlyClient:
    def __init__(self, values):
        self.values = values
        self.authenticated = False
        self.authenticate = Mock(side_effect=lambda: setattr(self, "authenticated", True))
        self.fetch_user_alpha_page = Mock(side_effect=self.page)
        self.fetch_alpha_detail = Mock(side_effect=lambda *, platform_alpha_id:
            SimpleNamespace(payload=deepcopy(next(p for p in self.values if p["id"] == platform_alpha_id))))
        start = datetime.fromisoformat(STAMP)
        self.points = tuple(((start + timedelta(days=i)).date().isoformat(), float(i * i)) for i in range(260))
        self.fetch_pnl = Mock(return_value=PnlObservation(self.points))

    def page(self, *, limit, offset, hidden, exclude_status, order):
        assert exclude_status == "UNSUBMITTED" and order == "-dateSubmitted"
        values = [deepcopy(p) for p in self.values if p["hidden"] == hidden]
        return parse_user_alpha_page({"count": len(values), "previous": None,
            "next": "next-page" if offset + limit < len(values) else None,
            "results": values[offset:offset + limit]})


@pytest.fixture
def setup(tmp_path):
    database = tmp_path / "research.sqlite3"
    with open_database(database) as connection:
        initialize_database_schema(connection)
    environment = tmp_path / "fixture.env"
    environment.write_text("WQB_ACCOUNT_SCOPE=fixture-account\nWQB_BASE_URL=https://api.worldquantbrain.com\n"
                           "WQB_SESSION_TOKEN=synthetic-test-token\n", encoding="utf-8")
    current = [datetime.fromisoformat(STAMP)]
    def run(client, **kwargs):
        return sync_submitted_alphas(database, environment, client_factory=lambda settings: client,
            clock=lambda: current[0], waiter=lambda seconds: None,
            policy=SubmittedSyncPolicy(page_size=1, request_interval_seconds=0), **kwargs)
    return database, current, run


def require_complete(database, account="fixture-account"):
    with open_database(database) as connection:
        require_submitted_baseline(connection, account_scope=account)


def test_all_pages_hidden_decommissioned_payload_pnl_and_no_seed_import(setup):
    database, _, run = setup
    client = ReadOnlyClient([payload("a"), payload("b", status="DECOMMISSIONED"), payload("c", hidden=True)])
    result = run(client)
    assert (result.submitted_count, result.hidden_count, result.pnl_captured_count, result.completed) == (3, 1, 3, True)
    assert client.fetch_user_alpha_page.call_count == 6  # Both routes, both complete scans.
    with open_database(database) as connection:
        records = list_platform_submitted_alphas(connection, account_scope="fixture-account")
        assert [record.raw_payload for record in records] == client.values
        assert len(list_pnl_series(connection, account_scope="fixture-account")) == 3
        assert connection.execute("SELECT COUNT(*) FROM signal_seeds").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM backtest_tasks").fetchone()[0] == 0
    require_complete(database)
    with pytest.raises(ValueError, match="baseline_incomplete"):
        require_complete(database, "another-account")
    assert run(client).completed
    assert client.fetch_pnl.call_count == 3  # Rescans facts; never overwrites captured curves.


def test_verified_empty_account_is_distinct_from_unsynced_empty_database(setup):
    database, _, run = setup
    with pytest.raises(ValueError, match="baseline_incomplete"):
        require_complete(database)
    client = ReadOnlyClient([])
    assert run(client).completed
    assert client.fetch_user_alpha_page.call_count == 4
    client.fetch_alpha_detail.assert_not_called()
    client.fetch_pnl.assert_not_called()
    require_complete(database)


def test_pnl_pending_is_resumable_and_does_not_block_other_reads_or_pass_gate(setup):
    database, current, run = setup
    client = ReadOnlyClient([payload("a"), payload("b")])
    client.fetch_pnl.side_effect = [PnlObservation(None, 30), PnlObservation(client.points)]
    first = run(client)
    assert (first.completed, first.pnl_captured_count, first.pnl_pending_count) == (False, 1, 1)
    with pytest.raises(ValueError, match="baseline_incomplete"):
        require_complete(database)
    assert not run(client).completed
    assert client.fetch_pnl.call_count == 2
    current[0] += timedelta(seconds=601)
    client.fetch_pnl.side_effect = None
    assert run(client).completed
    assert client.fetch_pnl.call_count == 3
    require_complete(database)


@pytest.mark.parametrize("failure", ["duplicate", "count", "hidden", "unsubmitted", "missing-date", "type", "detail"])
def test_incomplete_or_inconsistent_scan_never_sets_complete_marker(setup, failure):
    database, _, run = setup
    values = [payload("a"), payload("b")]
    client = ReadOnlyClient(values)
    if failure == "duplicate":
        values[1]["id"] = "a"
    elif failure == "count":
        original = client.page
        def changed(**kwargs):
            page = original(**kwargs)
            return SimpleNamespace(total_count=3 if kwargs["offset"] else page.total_count,
                                   has_next=page.has_next, records=page.records)
        client.fetch_user_alpha_page.side_effect = changed
    elif failure == "hidden":
        original = client.page
        def wrong(**kwargs):
            return original(**{**kwargs, "hidden": False})
        client.fetch_user_alpha_page.side_effect = wrong
    elif failure == "unsubmitted":
        values[0]["status"] = "UNSUBMITTED"
    elif failure == "missing-date":
        values[0]["dateSubmitted"] = None
    elif failure == "type":
        values[0]["type"] = "SUPER"
    else:
        broken = payload("a")
        broken["regular"]["code"] = "rank(changed)"
        client.fetch_alpha_detail.return_value = SimpleNamespace(payload=broken)
        client.fetch_alpha_detail.side_effect = None
    with pytest.raises(ValueError):
        run(client)
    with pytest.raises(ValueError, match="baseline_incomplete"):
        require_complete(database)


def test_changed_final_snapshot_keeps_captured_facts_but_not_completion(setup):
    database, _, run = setup
    client = ReadOnlyClient([payload()])
    original = client.page
    def page(**kwargs):
        if client.fetch_user_alpha_page.call_count == 3:
            client.values.append(payload("new"))
        return original(**kwargs)
    client.fetch_user_alpha_page.side_effect = page
    with pytest.raises(ValueError, match="snapshot_changed"):
        run(client)
    with open_database(database) as connection:
        assert len(list_pnl_series(connection, account_scope="fixture-account")) == 1
    with pytest.raises(ValueError, match="baseline_incomplete"):
        require_complete(database)


def test_retry_exhaustion_invalidates_previous_completion(setup):
    database, _, run = setup
    assert run(ReadOnlyClient([])).completed
    client = ReadOnlyClient([])
    client.fetch_user_alpha_page.side_effect = WorldQuantRequestError(
        "fixture-network", retryable=True, outcome_unknown=False)
    with pytest.raises(WorldQuantRequestError, match="fixture-network"):
        run(client)
    assert client.fetch_user_alpha_page.call_count == 3
    with pytest.raises(ValueError, match="baseline_incomplete"):
        require_complete(database)


def test_explicit_sync_adds_only_metadata_table_to_previous_schema(setup):
    database, _, run = setup
    with open_database(database) as connection:
        connection.execute("DROP TABLE platform_submitted_alpha_syncs")
        connection.execute("INSERT INTO generation_windows VALUES (22, 'month')")
    assert run(ReadOnlyClient([])).completed
    with open_database(database) as connection:
        initialize_database_schema(connection)
        assert tuple(connection.execute("SELECT * FROM generation_windows").fetchone()) == (22, "month")
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()


def test_missing_captured_reference_blocks_baseline_after_sync(setup):
    database, _, run = setup
    assert run(ReadOnlyClient([payload()])).completed
    with open_database(database) as connection:
        connection.execute("DELETE FROM platform_pnl_series")
    with pytest.raises(ValueError, match="pnl_incomplete"):
        require_complete(database)


@pytest.mark.parametrize("retry_after, attempted_reads", [(120, 2), (1, 4)])
def test_exhausted_account_throttle_stops_all_further_reads_preserving_captures(setup, retry_after, attempted_reads):
    database, _, run = setup
    client = ReadOnlyClient([payload("a"), payload("b"), payload("c")])
    throttle = WorldQuantRequestError("worldquant_pnl_http_error", retryable=True,
        outcome_unknown=False, status_code=429, retry_after_seconds=retry_after)
    client.fetch_pnl.side_effect = [PnlObservation(client.points), throttle, throttle, throttle]
    with pytest.raises(WorldQuantRequestError) as raised:
        run(client)
    assert raised.value is throttle
    assert client.fetch_pnl.call_count == attempted_reads
    assert {call.kwargs["platform_alpha_id"] for call in client.fetch_pnl.call_args_list} == {"a", "b"}
    assert client.fetch_user_alpha_page.call_count == 4  # No second scan after throttling.
    with open_database(database) as connection:
        assert len(list_platform_submitted_alphas(connection, account_scope="fixture-account")) == 3
        assert [row.platform_alpha_id for row in list_pnl_series(connection, account_scope="fixture-account")] == ["a"]
    with pytest.raises(ValueError, match="baseline_incomplete"):
        require_complete(database)
