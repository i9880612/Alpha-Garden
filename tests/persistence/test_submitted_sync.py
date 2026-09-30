from dataclasses import replace

import pytest

from persistence.database import open_database
from persistence.pnl import PnlSeriesRecord, save_pnl_series
from persistence.schema import add_submitted_sync_storage, add_submission_check_storage, initialize_database_schema
from persistence.submissions import PlatformSubmittedAlphaRecord, record_platform_submitted_alphas
from persistence.submitted_sync import complete_submitted_sync, invalidate_submitted_sync, require_submitted_baseline


def test_metadata_migration_is_explicit_transactional_and_rejects_unrelated_changes(tmp_path):
    with open_database(tmp_path / "fixture.sqlite3") as connection:
        initialize_database_schema(connection)
        connection.execute("DROP TABLE platform_submitted_alpha_syncs")
        with pytest.raises(ValueError, match="requires_transaction"):
            add_submitted_sync_storage(connection)
        connection.execute("BEGIN IMMEDIATE")
        add_submitted_sync_storage(connection)
        connection.rollback()
        assert connection.execute("SELECT 1 FROM sqlite_master WHERE name='platform_submitted_alpha_syncs'").fetchone() is None
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="schema_mismatch"):
            add_submitted_sync_storage(connection)
        connection.rollback()


def test_earlier_migration_can_finish_before_new_sync_metadata_migration(tmp_path):
    with open_database(tmp_path / "fixture.sqlite3") as connection:
        initialize_database_schema(connection)
        connection.execute("DROP TABLE platform_submitted_alpha_syncs")
        connection.execute("DROP TABLE submission_checks")
        connection.execute("BEGIN IMMEDIATE")
        add_submission_check_storage(connection)
        add_submitted_sync_storage(connection)
        initialize_database_schema(connection)


def test_same_alpha_id_and_sync_markers_stay_account_isolated(tmp_path):
    stamp = "2026-09-01T00:00:00+00:00"
    raw = {"id": "shared-id", "status": "ACTIVE", "dateSubmitted": stamp,
           "hidden": False, "regular": {"code": "rank(close)"}}
    record = PlatformSubmittedAlphaRecord("first", "shared-id", "rank(close)", "ACTIVE", stamp, False, raw, stamp)
    with open_database(tmp_path / "fixture.sqlite3") as connection:
        initialize_database_schema(connection)
        for account in ("first", "second"):
            record_platform_submitted_alphas(connection, (replace(record, account_scope=account),))
            save_pnl_series(connection, PnlSeriesRecord(account, "shared-id", stamp, (("2020-01-01", 0.0),)))
            complete_submitted_sync(connection, account_scope=account, alpha_ids=("shared-id",), completed_at=stamp)
        invalidate_submitted_sync(connection, account_scope="first")
        with pytest.raises(ValueError, match="baseline_incomplete"):
            require_submitted_baseline(connection, account_scope="first")
        require_submitted_baseline(connection, account_scope="second")
        assert connection.execute("SELECT COUNT(*) FROM platform_submitted_alphas").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM platform_pnl_series").fetchone()[0] == 2


def test_completion_requires_exact_record_coverage_and_every_pnl_capture(tmp_path):
    stamp = "2026-09-01T00:00:00+00:00"
    raw = {"id": "alpha", "status": "ACTIVE", "dateSubmitted": stamp,
           "hidden": False, "regular": {"code": "rank(close)"}}
    with open_database(tmp_path / "fixture.sqlite3") as connection:
        initialize_database_schema(connection)
        record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
            "first", "alpha", "rank(close)", "ACTIVE", stamp, False, raw, stamp),))
        with pytest.raises(ValueError, match="record_set_mismatch"):
            complete_submitted_sync(connection, account_scope="first", alpha_ids=(), completed_at=stamp)
        with pytest.raises(ValueError, match="pnl_incomplete"):
            complete_submitted_sync(connection, account_scope="first", alpha_ids=("alpha",), completed_at=stamp)
