from __future__ import annotations

import sqlite3
import re
from contextlib import closing

from persistence.backtests import initialize_backtest_schema
from persistence.pnl import initialize_pnl_schema
from persistence.qualified_archive import initialize_qualified_archive_schema
from persistence.catalog import initialize_catalog_schema
from persistence.runs import initialize_run_schema
from persistence.run_allocations import initialize_run_allocation_schema
from persistence.submission_queue import initialize_submission_queue_schema
from persistence.submissions import initialize_submission_schema
from persistence.submission_checks import initialize_submission_check_schema


PROJECT_TABLE_NAMES = frozenset(
    {
        "automated_cycle_settlements",
        "automated_run_backtests",
        "automated_runs",
        "automated_run_allocations",
        "backtest_checks",
        "backtest_mutations",
        "backtest_results",
        "backtest_tasks",
        "backtest_yearly_stats",
        "backtest_yearly_stats_captures",
        "generation_operator_outputs",
        "generation_operator_roles",
        "generation_windows",
        "formal_submission_attempts",
        "platform_catalog_syncs",
        "platform_fields",
        "platform_operators",
        "platform_submitted_alphas",
        "platform_pnl_series",
        "signal_seeds",
        "qualified_alpha_archive",
        "submission_queue",
        "submission_checks",
    }
)


def initialize_database_schema(connection: sqlite3.Connection) -> None:
    expected_schema = _reference_schema_objects()
    if _schema_objects(connection):
        _require_current_schema(connection, expected_schema)
        return
    _create_current_schema(connection)
    _require_current_schema(connection, expected_schema)


def add_pnl_storage(connection: sqlite3.Connection) -> None:
    """Explicit additive migration; reject unrelated or partially altered schemas.

    The caller owns the transaction. Existing tables and data are never rewritten.
    """
    expected = _reference_schema_objects()
    actual = _schema_objects(connection)
    if actual == expected:
        return
    previous = {
        key: value for key, value in expected.items() if key[2] != "platform_pnl_series"
    }
    if actual != previous:
        raise ValueError("pnl_storage_migration_schema_mismatch")
    initialize_pnl_schema(connection)
    _require_current_schema(connection, expected)


def add_submission_research_storage(connection: sqlite3.Connection) -> None:
    """Add grade evidence and retirement storage, including the observed grade-only schema."""
    expected = _reference_schema_objects()
    actual = _schema_objects(connection)
    if actual == expected:
        return
    table_key = ("table", "backtest_results", "backtest_results")
    archive_key = ("table", "qualified_alpha_archive", "qualified_alpha_archive")
    grade_missing = actual.get(table_key) != expected[table_key]
    archive_missing = archive_key not in actual
    archive_fields = {"grade": ("TEXT", "backtest_results"),
                      "platform_alpha_id": ("TEXT", "backtest_tasks"),
                      "sharpe": ("REAL", "backtest_results"),
                      "fitness": ("REAL", "backtest_results"),
                      "turnover": ("REAL", "backtest_results")}
    present_columns = [row[1] for row in connection.execute("PRAGMA table_info(qualified_alpha_archive)")]
    expected_columns = ["task_id", "archived_at", *archive_fields]
    if not archive_missing and present_columns != expected_columns[:len(present_columns)]:
        raise ValueError("submission_research_migration_schema_mismatch")
    missing_archive_fields = {name: definition for name, definition in archive_fields.items()
                              if name not in present_columns}
    previous = dict(expected)
    if grade_missing:
        previous[table_key] = expected[table_key].replace(", grade TEXT", "")
    if archive_missing:
        del previous[archive_key]
    else:
        for name, (kind, _) in missing_archive_fields.items():
            previous[archive_key] = previous[archive_key].replace(f", {name} {kind}", "")
    if actual != previous:
        raise ValueError("submission_research_migration_schema_mismatch")
    if not connection.in_transaction:
        raise ValueError("submission_research_migration_requires_transaction")
    if grade_missing:
        connection.execute("ALTER TABLE backtest_results ADD COLUMN grade TEXT")
    if archive_missing:
        initialize_qualified_archive_schema(connection)
    else:
        for name, (kind, source) in missing_archive_fields.items():
            connection.execute(f"ALTER TABLE qualified_alpha_archive ADD COLUMN {name} {kind}")
            connection.execute(
                f"UPDATE qualified_alpha_archive SET {name} = "
                f"(SELECT {name} FROM {source} r WHERE r.task_id = qualified_alpha_archive.task_id)"
            )
    _require_current_schema(connection, expected)


def add_submission_source_storage(connection: sqlite3.Connection) -> None:
    """Explicit additive migration; historical attempts retain the queue policy."""
    expected = _reference_schema_objects()
    actual = _schema_objects(connection)
    if actual == expected:
        return
    key = ("table", "formal_submission_attempts", "formal_submission_attempts")
    column = ("source TEXT NOT NULL DEFAULT 'queue' CHECK ( "
              "source = 'queue' OR (source = 'qualified_archive' AND submission_mode = 'manual'))")
    previous = dict(expected)
    previous[key] = expected[key].replace(", " + column, "")
    if actual != previous:
        raise ValueError("submission_source_migration_schema_mismatch")
    if not connection.in_transaction:
        raise ValueError("submission_source_migration_requires_transaction")
    connection.execute("ALTER TABLE formal_submission_attempts ADD COLUMN " + column)
    _require_current_schema(connection, expected)


def add_optimization_run_storage(connection: sqlite3.Connection) -> None:
    """Freeze the explicit optimization mode; historical runs remain ordinary runs."""
    expected = _reference_schema_objects()
    actual = _schema_objects(connection)
    if actual == expected:
        return
    key = ("table", "automated_runs", "automated_runs")
    column = "optimization_only INTEGER NOT NULL DEFAULT 0 CHECK (optimization_only IN (0, 1))"
    previous = dict(expected)
    previous[key] = expected[key].replace(", " + column, "")
    if actual != previous:
        raise ValueError("optimization_run_migration_schema_mismatch")
    if not connection.in_transaction:
        raise ValueError("optimization_run_migration_requires_transaction")
    connection.execute("ALTER TABLE automated_runs ADD COLUMN " + column)
    _require_current_schema(connection, expected)


def add_run_allocation_storage(connection: sqlite3.Connection) -> None:
    """Explicit additive migration; caller owns the transaction, no history rewrite."""
    expected = _reference_schema_objects()
    actual = _schema_objects(connection)
    if actual == expected:
        return
    previous = {
        key: value for key, value in expected.items()
        if key[2] != "automated_run_allocations"
    }
    if actual != previous:
        raise ValueError("run_allocation_migration_schema_mismatch")
    initialize_run_allocation_schema(connection)
    _require_current_schema(connection, expected)


def add_submission_check_storage(connection: sqlite3.Connection) -> None:
    """Explicit additive migration; no existing rows or tables are changed."""
    expected = _reference_schema_objects()
    actual = _schema_objects(connection)
    if actual == expected:
        return
    previous = {key: value for key, value in expected.items() if key[2] != "submission_checks"}
    if actual != previous:
        raise ValueError("submission_check_migration_schema_mismatch")
    initialize_submission_check_schema(connection)
    _require_current_schema(connection, expected)


def migrate_cancelled_backtest_reservations(connection: sqlite3.Connection) -> None:
    """Preserve every task and reference while releasing only unsent cancellations.

    Caller disables foreign keys before BEGIN IMMEDIATE and restores them after
    commit/rollback. Renaming the original table would redirect child references.
    """
    expected = _reference_schema_objects()
    actual = _schema_objects(connection)
    if actual == expected:
        return
    table_key = ("table", "backtest_tasks", "backtest_tasks")
    previous = {key: value for key, value in expected.items()
                if key[1] not in {"idx_backtest_request_reservation", "idx_backtest_formula_reservation"}}
    previous[table_key] = expected[table_key].replace(
        "request_fingerprint TEXT NOT NULL,", "request_fingerprint TEXT NOT NULL UNIQUE,"
    ).replace("UNIQUE (account_scope, remote_id),",
              "UNIQUE (account_scope, formula_fingerprint, settings_json), UNIQUE (account_scope, remote_id),")
    if actual != previous:
        raise ValueError("backtest_reservation_migration_schema_mismatch")
    if not connection.in_transaction or connection.execute("PRAGMA foreign_keys").fetchone()[0]:
        raise ValueError("backtest_reservation_migration_requires_transaction_without_foreign_keys")
    connection.execute(expected[table_key].replace("backtest_tasks (", "backtest_tasks_replacement (", 1))
    connection.execute("INSERT INTO backtest_tasks_replacement SELECT * FROM backtest_tasks")
    connection.execute("DROP TABLE backtest_tasks")
    connection.execute("ALTER TABLE backtest_tasks_replacement RENAME TO backtest_tasks")
    for (kind, name, table), sql in expected.items():
        if kind == "index" and table == "backtest_tasks":
            connection.execute(sql)
    # SQLite may quote the renamed table; compare schema after normalizing that
    # harmless representation in the common reader.
    _require_current_schema(connection, expected)
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise ValueError("backtest_reservation_migration_foreign_key_violation")


def _create_current_schema(connection: sqlite3.Connection) -> None:
    initialize_catalog_schema(connection)
    initialize_backtest_schema(connection)
    initialize_run_schema(connection)
    initialize_submission_schema(connection)
    initialize_submission_queue_schema(connection)


def _reference_schema_objects() -> dict[tuple[str, str, str], str]:
    with closing(sqlite3.connect(":memory:")) as reference:
        reference.execute("PRAGMA foreign_keys = ON")
        _create_current_schema(reference)
        table_names = _database_table_names(reference)
        if table_names != PROJECT_TABLE_NAMES:
            raise AssertionError("project_table_inventory_invalid")
        return _schema_objects(reference)


def _require_current_schema(
    connection: sqlite3.Connection,
    expected: dict[tuple[str, str, str], str],
) -> None:
    actual = _schema_objects(connection)
    if actual == expected:
        return
    differing = sorted(
        {
            key[1]
            for key in actual.keys() | expected.keys()
            if actual.get(key) != expected.get(key)
        }
    )
    raise ValueError("database_schema_incompatible:" + ",".join(differing))


def _database_table_names(connection: sqlite3.Connection) -> frozenset[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        """
    ).fetchall()
    return frozenset(row[0] for row in rows)


def _schema_objects(
    connection: sqlite3.Connection,
) -> dict[tuple[str, str, str], str]:
    rows = connection.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_master
        WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    def normalized(sql: str) -> str:
        sql = " ".join(sql.replace('"backtest_tasks"', 'backtest_tasks').split())
        # ALTER TABLE can move whitespace before commas and closing parentheses.
        # Normalize SQL spacing without changing quoted CHECK/default values.
        parts = re.split(r"('(?:''|[^'])*')", sql)
        return "".join(re.sub(r"\s+([,)])", r"\1", part) if i % 2 == 0 else part
                       for i, part in enumerate(parts))
    return {(row[0], row[1], row[2]): normalized(row[3]) for row in rows}
