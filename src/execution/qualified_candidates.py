from __future__ import annotations

import json
import sqlite3

from learning.qualified_evolution import QualifiedEvolutionRecord, assess_qualified_evolution
from learning.frontiers import PARENT_ATTEMPT_BUDGET, count_parent_attempts
from persistence.backtests import (
    BACKTEST_TERMINAL_STATUSES, BacktestSnapshot, get_backtest_task, list_backtest_tasks, list_backtest_mutations,
    list_completed_backtests,
)
from persistence.submission_checks import get_submission_check
from persistence.qualified_archive import list_qualified_alpha_archive
from persistence.submissions import (
    list_formal_submission_attempts, list_platform_submitted_alphas, normalize_submitted_formula,
)
from submission.formal import (
    BELOW_TARGET_GRADES, assess_formal_check_payload, local_formal_submission_eligible, qualified_archive_eligible,
)


def load_unconsumed_qualified_candidates(
    connection: sqlite3.Connection, task_ids: set[str],
) -> tuple[BacktestSnapshot, ...]:
    """Read fully checked candidates without duplicating submission or archive state."""
    if not task_ids:
        return ()
    tasks = {task.task_id: task for task in list_backtest_tasks(connection)}
    consumed = {
        (tasks[attempt.task_id].account_scope,
         normalize_submitted_formula(tasks[attempt.task_id].formula))
        for attempt in list_formal_submission_attempts(connection)
    }
    submitted_ids = set()
    for account in {task.account_scope for task in tasks.values()}:
        for record in list_platform_submitted_alphas(connection, account_scope=account):
            consumed.add((account, record.normalized_formula))
            submitted_ids.add((account, record.platform_alpha_id))
    candidates = []
    for task_id in sorted(task_ids):
        snapshot = get_backtest_task(connection, task_id)
        if snapshot is None:
            raise ValueError("qualified_candidate_source_missing")
        task = snapshot.task
        if ((task.account_scope, normalize_submitted_formula(task.formula)) in consumed
                or (task.account_scope, task.platform_alpha_id) in submitted_ids):
            continue
        check = get_submission_check(connection, task.task_id)
        payload = json.loads(check.payload_json) if check and check.payload_json else None
        if qualified_archive_eligible(snapshot, payload):
            candidates.append(snapshot)
    return tuple(candidates)


def load_exhausted_qualified_candidates(
    connection: sqlite3.Connection, *, account_scope: str,
) -> tuple[BacktestSnapshot, ...]:
    """Only archived, fully checked formulas whose own sent-child budget is exhausted."""
    archived = {record.task_id for record in list_qualified_alpha_archive(connection)}
    if not archived:
        return ()
    tasks = {task.task_id: task for task in list_backtest_tasks(connection)}
    mutations = list_backtest_mutations(connection)
    counts = count_parent_attempts(mutations, frozenset(
        task.task_id for task in tasks.values() if task.submission_started_at is not None
    ))
    unsettled_parents = {
        mutation.parent_task_id for mutation in mutations
        if tasks[mutation.child_task_id].submission_started_at is not None
        and tasks[mutation.child_task_id].status not in BACKTEST_TERMINAL_STATUSES
    }
    eligible = {task_id for task_id in archived
                if tasks[task_id].account_scope == account_scope
                and counts[task_id] >= PARENT_ATTEMPT_BUDGET
                and task_id not in unsettled_parents}
    return tuple(snapshot for snapshot in load_unconsumed_qualified_candidates(connection, eligible)
                 if snapshot.result is not None and snapshot.result.grade in BELOW_TARGET_GRADES)


def load_qualified_evolution(
    connection: sqlite3.Connection, *, excluded_task_ids: frozenset[str] = frozenset(),
) -> tuple[QualifiedEvolutionRecord, ...]:
    """Read individual checked candidates and retirement from the original facts."""
    checked = []
    for snapshot in list_completed_backtests(connection):
        if snapshot.task.task_id in excluded_task_ids or not local_formal_submission_eligible(snapshot):
            continue
        check = get_submission_check(connection, snapshot.task.task_id)
        payload = json.loads(check.payload_json) if check and check.payload_json else None
        if assess_formal_check_payload(payload).state == "passed":
            checked.append(snapshot)
    eligible = {s.task.task_id for s in load_unconsumed_qualified_candidates(
        connection, {s.task.task_id for s in checked},
    )}
    decisions = assess_qualified_evolution(
        tuple(checked), list_backtest_mutations(connection),
        frozenset(task.task_id for task in list_backtest_tasks(connection)
                  if task.submission_started_at is not None and task.task_id not in excluded_task_ids),
    )
    return tuple(record for record in decisions if record.task_id in eligible)
