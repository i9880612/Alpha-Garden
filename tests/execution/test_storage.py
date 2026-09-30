import hashlib
import sqlite3
from datetime import datetime, timezone

import pytest

from execution.backtests import (
    apply_backtest_detail, prepare_backtest_task, record_backtest_yearly_stats, record_submission_accepted,
)
from execution.storage import inspect_storage, preview_pnl_prune
from persistence.database import open_database, read_database
from persistence.pnl import PnlSeriesRecord, save_pnl_series
from persistence.schema import initialize_database_schema
from persistence.submissions import PlatformSubmittedAlphaRecord, record_platform_submitted_alphas
from worldquant.backtests import BacktestCheck, BacktestDetail, BacktestYearlyStat, STANDARD_REGULAR_CHECK_NAMES


OLD = "2026-01-01T00:00:00+00:00"
NOW = datetime(2026, 9, 30, tzinfo=timezone.utc)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "storage.sqlite3"
    with open_database(path) as connection:
        initialize_database_schema(connection)
    return path


def curve(connection, alpha, *, account="account", observed=OLD, pending=False):
    save_pnl_series(connection, PnlSeriesRecord(
        account, alpha, observed, None if pending else (("2020-01-01", 1.0), ("2020-01-02", 2.0)),
        "2026-10-01T00:00:00+00:00" if pending else None,
    ))


def completed(connection, alpha, *, account="account", failed_check="LOW_FITNESS"):
    task = prepare_backtest_task(connection, account_scope=account, formula=f"rank({alpha})",
                                 settings={"delay": 1}, created_at=OLD)
    record_submission_accepted(connection, task.task.task_id, remote_id=alpha, observed_at=OLD)
    apply_backtest_detail(connection, task.task.task_id, BacktestDetail(
        platform_alpha_id=alpha, sharpe=1.5, fitness=0.8, turnover=0.12, returns=0.08,
        drawdown=0.04, margin=0.001, book_size=20_000_000, pnl=100_000,
        checks=tuple(BacktestCheck(name, "FAIL" if name == failed_check else "PASS", None, None, None)
                     for name in sorted(STANDARD_REGULAR_CHECK_NAMES)), grade="GOOD",
    ), observed_at=OLD)
    record_backtest_yearly_stats(connection, task.task.task_id, (
        BacktestYearlyStat(2023, 100_000, 20_000_000, 0.12, 1.5, 0.08, 0.04, 0.001, 0.8, 100, 100, "IS"),
    ), observed_at=OLD)
    curve(connection, alpha, account=account)
    return task.task.task_id


def submitted(connection, alpha, *, account="account"):
    payload = {"id": alpha, "status": "DECOMMISSIONED", "hidden": True,
               "dateSubmitted": OLD, "regular": {"code": f"rank({alpha})"}}
    record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
        account, alpha, f"rank({alpha})", "DECOMMISSIONED", OLD, True, payload, OLD,
    ),))


def test_preview_protects_unknown_sc_only_recent_and_shared_account_evidence(database):
    with open_database(database) as connection:
        completed(connection, "old_failure")
        completed(connection, "sc_only", failed_check="SELF_CORRELATION")
        completed(connection, "qualified", failed_check=None)
        task = completed(connection, "recent")
        connection.execute("UPDATE backtest_tasks SET last_observed_at=? WHERE task_id=?", (NOW.isoformat(), task))
        curve(connection, "unlinked")
        curve(connection, "pending", pending=True)
        completed(connection, "shared")
        completed(connection, "shared", account="another-account")
        submitted(connection, "shared")
        # The same Alpha has both a failed and a passing result in the same account.
        task = completed(connection, "duplicate_identity", failed_check=None)
        connection.execute("UPDATE backtest_tasks SET platform_alpha_id='old_failure' WHERE task_id=?", (task,))
    before = hashlib.sha256(database.read_bytes()).digest()
    report = preview_pnl_prune(database, observed_at=NOW)
    assert report.candidate_count == 1  # Only the other account's shared ID.
    reasons = dict(report.retained_counts)
    assert reasons == {"no_definite_rejection": 3, "retention_window": 1,
                       "unlinked_evidence": 2, "pending_evidence": 1, "submitted_reference": 1}
    assert report.candidate_payload_bytes > 0
    assert hashlib.sha256(database.read_bytes()).digest() == before


def test_preview_protects_all_lineage_members_and_bound_recovery_reference(database):
    with open_database(database) as connection:
        root = completed(connection, "root")
        child = completed(connection, "child")
        completed(connection, "bound_reference")
        connection.execute("INSERT INTO signal_seeds VALUES (?, ?)", (root, OLD))
        connection.execute("INSERT INTO backtest_mutations VALUES (?, ?, ?, ?, ?, ?)",
                           (child, root, "field", "root", "root", "child"))
        connection.execute("INSERT INTO backtest_mutation_references VALUES (?, ?, ?)",
                           (child, "bound_reference", "rank(bound_reference)"))
        checked = completed(connection, "checked")
        connection.execute("INSERT INTO submission_checks VALUES (?, ?, ?, ?, ?, ?)",
                           (checked, OLD, None, "request_error", 1, None))
    preview = preview_pnl_prune(database, observed_at=NOW)
    assert preview.candidate_count == 0
    assert dict(preview.retained_counts) == {"research_or_submission_dependency": 3, "recovery_reference": 1}


def test_capacity_counts_bytes_and_preview_age_is_inclusive(database):
    with open_database(database) as connection:
        completed(connection, "failure")
        curve(connection, "pending", pending=True)
        connection.execute("UPDATE platform_pnl_series SET observed_at=? WHERE platform_alpha_id='failure'",
                           ("2026-07-02T00:00:00+00:00",))  # Exactly 90 days ago.
    before = database.read_bytes()
    capacity = inspect_storage(database)
    assert capacity.database_file_bytes == len(before)
    assert capacity.wal_file_bytes == 0
    assert capacity.measurements.page_count * capacity.measurements.page_size == len(before)
    assert (capacity.measurements.pnl_count, capacity.measurements.captured_count) == (2, 1)
    assert capacity.measurements.payload_bytes > 0
    assert preview_pnl_prune(database, observed_at=NOW).candidate_count == 0
    assert preview_pnl_prune(database, observed_at=NOW, retention_days=89).candidate_count == 1
    with read_database(database) as connection:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM platform_pnl_series")
    assert database.read_bytes() == before


def test_preview_keeps_run_queue_archive_and_uncertain_submission_dependencies():
    from tests.execution.test_runs import AutomatedRunExecutionTests
    from persistence.submission_queue import initialize_submission_queue_schema
    case = AutomatedRunExecutionTests()
    case.setUp()
    try:
        run = case._prepare()
        with open_database(case.database_path) as connection:
            initialize_submission_queue_schema(connection)
            ids = {name: completed(connection, name, account="group-account")
                   for name in ("run_member", "queued", "family_root", "archived", "uncertain", "unused")}
            connection.execute("INSERT INTO automated_run_backtests VALUES (?, ?, 1)", (run.run_id, ids["run_member"]))
            connection.execute("INSERT INTO submission_queue VALUES (?, ?, ?, 1, ?, ?, ?)",
                               (ids["queued"], "group-account", run.run_id, ids["family_root"], "rank(queued)", OLD))
            connection.execute("INSERT INTO qualified_alpha_archive (task_id, archived_at) VALUES (?, ?)",
                               (ids["archived"], OLD))
            connection.execute("""INSERT INTO formal_submission_attempts (
                task_id, run_id, cycle_number, family_root_task_id, submission_mode,
                status, check_attempt_count, created_at, updated_at
            ) VALUES (?, ?, 1, ?, 'manual', 'submission_unknown', 0, ?, ?)""",
                (ids["uncertain"], run.run_id, ids["family_root"], OLD, OLD))
        preview = preview_pnl_prune(case.database_path, observed_at=NOW)
        assert preview.candidate_count == 1
        assert dict(preview.retained_counts) == {"research_or_submission_dependency": 5}
    finally:
        case.doCleanups()


@pytest.mark.parametrize("operation", [inspect_storage, preview_pnl_prune])
def test_missing_database_is_not_created(tmp_path, operation):
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        operation(path)
    assert not path.exists()


def test_preview_rejects_short_retention_and_naive_time(database):
    with pytest.raises(ValueError, match="at_least_30"):
        preview_pnl_prune(database, retention_days=29, observed_at=NOW)
    with pytest.raises(ValueError, match="timezone_required"):
        preview_pnl_prune(database, observed_at=NOW.replace(tzinfo=None))
