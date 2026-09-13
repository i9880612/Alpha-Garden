from __future__ import annotations

import sqlite3
from dataclasses import astuple, dataclass


@dataclass(frozen=True, slots=True)
class RunAllocationRecord:
    run_id: str
    exploration_percent: int
    self_correlation_percent: int
    mutation_percent: int
    direction_validation_percent: int


def initialize_run_allocation_schema(connection: sqlite3.Connection) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS automated_run_allocations (
            run_id TEXT PRIMARY KEY NOT NULL REFERENCES automated_runs(run_id),
            exploration_percent INTEGER NOT NULL CHECK (
                typeof(exploration_percent) = 'integer'
                AND exploration_percent BETWEEN 1 AND 100
            ),
            self_correlation_percent INTEGER NOT NULL CHECK (
                typeof(self_correlation_percent) = 'integer'
                AND self_correlation_percent BETWEEN 0 AND 100
            ),
            mutation_percent INTEGER NOT NULL CHECK (
                typeof(mutation_percent) = 'integer'
                AND mutation_percent BETWEEN 0 AND 100
            ),
            direction_validation_percent INTEGER NOT NULL CHECK (
                typeof(direction_validation_percent) = 'integer'
                AND direction_validation_percent BETWEEN 0 AND mutation_percent
            ),
            CHECK (exploration_percent + self_correlation_percent + mutation_percent = 100)
        )
    """)


def get_run_allocation(
    connection: sqlite3.Connection, run_id: str,
) -> RunAllocationRecord | None:
    row = connection.execute("""
        SELECT run_id, exploration_percent, self_correlation_percent,
               mutation_percent, direction_validation_percent
        FROM automated_run_allocations WHERE run_id = ?
    """, (run_id,)).fetchone()
    return RunAllocationRecord(*row) if row is not None else None


def create_run_allocation(
    connection: sqlite3.Connection, record: RunAllocationRecord,
) -> None:
    existing = get_run_allocation(connection, record.run_id)
    if existing is not None:
        if existing != record:
            raise ValueError("automated_run_allocation_conflict")
        return
    connection.execute("""
        INSERT INTO automated_run_allocations (
            run_id, exploration_percent, self_correlation_percent,
            mutation_percent, direction_validation_percent
        ) VALUES (?, ?, ?, ?, ?)
    """, astuple(record))
