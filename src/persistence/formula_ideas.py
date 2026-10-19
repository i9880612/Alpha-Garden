"""Which idea template produced a task, and each run's frozen template share.

Both facts are written once, in the same transaction as the task or run they describe,
and are never rewritten. Tasks without an idea record came from another generator.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FormulaIdeaRecord:
    task_id: str
    template_id: str
    family: str
    field_ids: tuple[str, ...]
    hypothesis: str

    def __post_init__(self) -> None:
        for value, code in (
            (self.task_id, "formula_idea_task_id_invalid"),
            (self.template_id, "formula_idea_template_invalid"),
            (self.family, "formula_idea_family_invalid"),
            (self.hypothesis, "formula_idea_hypothesis_invalid"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(code)
        if (not isinstance(self.field_ids, tuple) or not self.field_ids
                or any(not isinstance(item, str) or not item.strip() for item in self.field_ids)):
            raise ValueError("formula_idea_fields_invalid")


def initialize_formula_idea_schema(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS formula_ideas (
        task_id TEXT PRIMARY KEY REFERENCES backtest_tasks(task_id),
        template_id TEXT NOT NULL CHECK (length(trim(template_id)) > 0),
        family TEXT NOT NULL CHECK (length(trim(family)) > 0),
        field_ids_json TEXT NOT NULL CHECK (length(field_ids_json) > 2),
        hypothesis TEXT NOT NULL CHECK (length(trim(hypothesis)) > 0)
    )""")


def initialize_run_idea_policy_schema(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS automated_run_idea_policies (
        run_id TEXT PRIMARY KEY NOT NULL REFERENCES automated_runs(run_id),
        idea_percent INTEGER NOT NULL CHECK (
            typeof(idea_percent) = 'integer' AND idea_percent BETWEEN 0 AND 100
        )
    )""")


def record_formula_idea(connection: sqlite3.Connection, record: FormulaIdeaRecord) -> None:
    existing = get_formula_idea(connection, record.task_id)
    if existing is not None:
        if existing != record:
            raise ValueError("formula_idea_conflict")
        return
    connection.execute(
        "INSERT INTO formula_ideas VALUES (?, ?, ?, ?, ?)",
        (record.task_id, record.template_id, record.family,
         json.dumps(list(record.field_ids), ensure_ascii=False), record.hypothesis),
    )


def get_formula_idea(connection: sqlite3.Connection, task_id: str) -> FormulaIdeaRecord | None:
    row = connection.execute(
        "SELECT task_id, template_id, family, field_ids_json, hypothesis "
        "FROM formula_ideas WHERE task_id = ?", (task_id,),
    ).fetchone()
    return _record(row) if row is not None else None


def list_formula_ideas(connection: sqlite3.Connection) -> tuple[FormulaIdeaRecord, ...]:
    return tuple(_record(row) for row in connection.execute(
        "SELECT task_id, template_id, family, field_ids_json, hypothesis "
        "FROM formula_ideas ORDER BY task_id"
    ))


def create_run_idea_policy(connection: sqlite3.Connection, *, run_id: str, idea_percent: int) -> None:
    if isinstance(idea_percent, bool) or not isinstance(idea_percent, int) \
            or not 0 <= idea_percent <= 100:
        raise ValueError("automated_run_idea_percent_invalid")
    existing = get_run_idea_percent(connection, run_id)
    if existing is not None:
        if existing != idea_percent:
            raise ValueError("automated_run_idea_policy_conflict")
        return
    connection.execute("INSERT INTO automated_run_idea_policies VALUES (?, ?)",
                       (run_id, idea_percent))


def get_run_idea_percent(connection: sqlite3.Connection, run_id: str) -> int | None:
    row = connection.execute(
        "SELECT idea_percent FROM automated_run_idea_policies WHERE run_id = ?", (run_id,),
    ).fetchone()
    return int(row[0]) if row is not None else None


def _record(row) -> FormulaIdeaRecord:
    return FormulaIdeaRecord(
        task_id=row[0], template_id=row[1], family=row[2],
        field_ids=tuple(json.loads(row[3])), hypothesis=row[4],
    )
