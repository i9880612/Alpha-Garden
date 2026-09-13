import sqlite3

import pytest

from persistence.schema import add_run_allocation_storage, initialize_database_schema


def test_explicit_migration_preserves_old_facts_without_inventing_run_policies():
    with sqlite3.connect(":memory:") as connection:
        initialize_database_schema(connection)
        connection.execute("DROP TABLE automated_run_allocations")
        connection.execute("INSERT INTO generation_windows VALUES (22, 'month')")
        with pytest.raises(ValueError, match="database_schema_incompatible"):
            initialize_database_schema(connection)
        add_run_allocation_storage(connection)
        add_run_allocation_storage(connection)
        assert connection.execute("SELECT * FROM generation_windows").fetchall() == [(22, "month")]
        assert connection.execute("SELECT * FROM automated_run_allocations").fetchall() == []
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migration_rolls_back_with_its_callers_transaction():
    with sqlite3.connect(":memory:") as connection:
        initialize_database_schema(connection)
        connection.execute("DROP TABLE automated_run_allocations")
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        add_run_allocation_storage(connection)
        connection.rollback()
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='automated_run_allocations'").fetchall() == []


def test_migration_rejects_unrelated_schema_without_writing():
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER)")
        with pytest.raises(ValueError, match="schema_mismatch"):
            add_run_allocation_storage(connection)
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("unrelated",)]
