from __future__ import annotations

import sqlite3

from execution.qualified_candidates import load_qualified_evolution
from persistence.backtests import (
    BACKTEST_TERMINAL_STATUSES,
    get_backtest_task,
    list_backtest_mutations,
    list_backtest_tasks,
)
from persistence.qualified_archive import (
    QualifiedAlphaArchiveRecord,
    archive_qualified_alpha,
    delete_qualified_alpha_archive_item,
    list_qualified_alpha_archive,
)
from persistence.submissions import normalize_submitted_formula


def synchronize_qualified_alpha_archive(
    connection: sqlite3.Connection, *, observed_at: str,
) -> tuple[QualifiedAlphaArchiveRecord, ...]:
    """Retain each retired checked formula after its started children settle.

    Membership records retirement, not submission approval or a second budget.
    An unknown historical grade remains unknown in its original result.
    """
    tasks = {task.task_id: task for task in list_backtest_tasks(connection)}
    archived = {record.task_id for record in list_qualified_alpha_archive(connection)}
    decisions = load_qualified_evolution(connection)
    candidates = tuple(record.task_id for record in decisions
                       if record.retired and record.task_id not in archived)
    eligible = {record.task_id for record in decisions}
    for task_id in archived:
        if task_id not in eligible:
            delete_qualified_alpha_archive_item(connection, task_id)
    if not candidates:
        return ()
    unsettled_parents = {
        mutation.parent_task_id for mutation in list_backtest_mutations(connection)
        if tasks[mutation.child_task_id].submission_started_at is not None
        and tasks[mutation.child_task_id].status not in BACKTEST_TERMINAL_STATUSES
    }
    created = []
    for task_id in candidates:
        if task_id in unsettled_parents or task_id not in eligible:
            continue
        created.append(archive_qualified_alpha(
            connection, task_id=task_id, archived_at=observed_at,
        ))
    return tuple(created)


def consume_qualified_alpha_archive(connection: sqlite3.Connection, *, task_id: str) -> None:
    """Remove this account's matching archive entries when a submission is claimed/confirmed."""
    source = get_backtest_task(connection, task_id)
    if source is None:
        raise ValueError("qualified_archive_source_missing")
    formula = normalize_submitted_formula(source.task.formula)
    for record in list_qualified_alpha_archive(connection):
        snapshot = get_backtest_task(connection, record.task_id)
        assert snapshot is not None
        if snapshot.task.account_scope != source.task.account_scope:
            continue
        if (snapshot.task.platform_alpha_id == source.task.platform_alpha_id
                or normalize_submitted_formula(snapshot.task.formula) == formula):
            delete_qualified_alpha_archive_item(connection, record.task_id)
