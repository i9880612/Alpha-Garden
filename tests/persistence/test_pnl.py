import sqlite3

import pytest

from persistence.pnl import PnlSeriesRecord, list_pnl_series, save_pnl_series
from persistence.schema import add_pnl_storage, initialize_database_schema


def test_explicit_additive_migration_preserves_facts_and_is_repeatable():
    with sqlite3.connect(":memory:") as connection:
        initialize_database_schema(connection)
        connection.execute("DROP TABLE platform_pnl_series")
        connection.execute("INSERT INTO generation_windows VALUES (22, 'month')")
        add_pnl_storage(connection)
        add_pnl_storage(connection)
        assert connection.execute(
            "SELECT value, horizon FROM generation_windows"
        ).fetchall() == [(22, "month")]
        record = PnlSeriesRecord(
            "account", "alpha", "2026-09-07T00:00:00+00:00", (("2020-01-01", 1.0),)
        )
        save_pnl_series(connection, record)
        save_pnl_series(connection, record)
        assert list_pnl_series(connection) == (record,)
        with pytest.raises(ValueError, match="capture_conflict"):
            save_pnl_series(
                connection,
                PnlSeriesRecord(
                    "account", "alpha", record.observed_at, (("2020-01-01", 2.0),)
                ),
            )
        assert list_pnl_series(connection) == (record,)


def test_migration_refuses_unrelated_schema():
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER)")
        with pytest.raises(ValueError, match="schema_mismatch"):
            add_pnl_storage(connection)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall() == [("unrelated",)]


def test_read_only_series_selection_filters_account_and_requested_alphas():
    with sqlite3.connect(":memory:") as connection:
        initialize_database_schema(connection)
        records = tuple(PnlSeriesRecord(account, alpha, "2026-09-07T00:00:00+00:00",
                                        (("2020-01-01", float(i)),))
                        for i, (account, alpha) in enumerate(
                            (("one", "shared"), ("two", "shared"), ("one", "extra"))))
        for record in records:
            save_pnl_series(connection, record)
        connection.commit()
        connection.execute("PRAGMA query_only=ON")
        assert list_pnl_series(connection, account_scope="one",
                               platform_alpha_ids=frozenset({"shared", "missing"})) == (records[0],)
        assert set(list_pnl_series(connection, account_scope="one")) == {records[0], records[2]}
        assert set(list_pnl_series(connection, platform_alpha_ids=frozenset({"shared"}))) == set(records[:2])
        assert list_pnl_series(connection, platform_alpha_ids=frozenset()) == ()
        assert list_pnl_series(connection, account_scope="missing") == ()
