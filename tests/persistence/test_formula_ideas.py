from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persistence.database import open_database
from persistence.formula_ideas import (
    FormulaIdeaRecord,
    create_run_idea_policy,
    get_formula_idea,
    get_run_idea_percent,
    initialize_run_idea_policy_schema,
    list_formula_ideas,
    record_formula_idea,
)
from persistence.schema import add_formula_idea_storage, initialize_database_schema


def _record(task_id="task-a", **values):
    fields = {"task_id": task_id, "template_id": "revision", "family": "analyst",
              "field_ids": ("est_eps",), "hypothesis": "分析师上调预期的股票会涨"}
    fields.update(values)
    return FormulaIdeaRecord(**fields)


def _insert_task(connection, task_id):
    connection.execute(
        "INSERT INTO backtest_tasks (task_id, account_scope, formula, formula_fingerprint, "
        "settings_json, request_fingerprint, status, created_at) "
        "VALUES (?, 'acct', ?, ?, '{}', ?, 'created', '2026-10-19T00:00:00+00:00')",
        (task_id, f"rank({task_id})", task_id, task_id),
    )


@pytest.fixture()
def connection(tmp_path):
    with open_database(tmp_path / "ideas.sqlite3") as connection:
        initialize_database_schema(connection)
        yield connection


def test_idea_record_is_written_once_and_read_back(connection):
    _insert_task(connection, "task-a")
    record = _record(field_ids=("anl4_ebit_value", "est_ebit"))
    record_formula_idea(connection, record)
    record_formula_idea(connection, record)
    assert get_formula_idea(connection, "task-a") == record
    assert list_formula_ideas(connection) == (record,)
    assert get_formula_idea(connection, "missing") is None


def test_idea_record_cannot_be_rewritten(connection):
    _insert_task(connection, "task-a")
    record_formula_idea(connection, _record())
    with pytest.raises(ValueError, match="formula_idea_conflict"):
        record_formula_idea(connection, _record(template_id="forward_value"))


def test_idea_record_requires_an_existing_task(connection):
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        record_formula_idea(connection, _record("no-such-task"))


@pytest.mark.parametrize("values, code", [
    ({"template_id": " "}, "formula_idea_template_invalid"),
    ({"field_ids": ()}, "formula_idea_fields_invalid"),
    ({"field_ids": ["est_eps"]}, "formula_idea_fields_invalid"),
    ({"hypothesis": ""}, "formula_idea_hypothesis_invalid"),
])
def test_idea_record_rejects_incomplete_values(values, code):
    with pytest.raises(ValueError, match=code):
        _record(**values)


def test_run_policy_is_frozen_and_absent_means_no_templates():
    import sqlite3
    connection = sqlite3.connect(":memory:")  # policy table alone; run rows are not needed here
    initialize_run_idea_policy_schema(connection)
    assert get_run_idea_percent(connection, "run-a") is None
    create_run_idea_policy(connection, run_id="run-a", idea_percent=50)
    create_run_idea_policy(connection, run_id="run-a", idea_percent=50)
    assert get_run_idea_percent(connection, "run-a") == 50
    with pytest.raises(ValueError, match="automated_run_idea_policy_conflict"):
        create_run_idea_policy(connection, run_id="run-a", idea_percent=90)
    for bad in (-1, 101, True, 1.5):
        with pytest.raises(ValueError, match="automated_run_idea_percent_invalid"):
            create_run_idea_policy(connection, run_id="run-b", idea_percent=bad)


def test_idea_storage_upgrade_is_explicit_atomic_and_keeps_history(tmp_path):
    with open_database(tmp_path / "upgrade.sqlite3") as connection:
        initialize_database_schema(connection)
        _insert_task(connection, "old-task")
        connection.execute("DROP TABLE formula_ideas")
        connection.execute("DROP TABLE automated_run_idea_policies")
        connection.commit()
        before = tuple(connection.iterdump())
        with pytest.raises(ValueError, match="requires_transaction"):
            add_formula_idea_storage(connection)
        connection.execute("BEGIN IMMEDIATE")
        add_formula_idea_storage(connection)
        connection.rollback()
        assert tuple(connection.iterdump()) == before
        connection.execute("BEGIN IMMEDIATE")
        add_formula_idea_storage(connection)
        add_formula_idea_storage(connection)
        connection.commit()
        initialize_database_schema(connection)
        assert get_formula_idea(connection, "old-task") is None
        assert connection.execute("SELECT COUNT(*) FROM backtest_tasks").fetchone()[0] == 1


def test_idea_storage_upgrade_rejects_unrelated_schema(tmp_path):
    with open_database(tmp_path / "upgrade.sqlite3") as connection:
        initialize_database_schema(connection)
        connection.execute("DROP TABLE formula_ideas")
        connection.execute("DROP TABLE automated_run_idea_policies")
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="formula_idea_migration_schema_mismatch"):
            add_formula_idea_storage(connection)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name='formula_ideas'").fetchone() is None
