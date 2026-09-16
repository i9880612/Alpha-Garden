from __future__ import annotations

import json
import sqlite3

from learning.qualified_evolution import QualifiedEvolutionRecord, assess_qualified_evolution
from learning.pnl import PnlCorrelations
from persistence.backtests import (
    BacktestMutationRecord, BacktestSnapshot, get_backtest_task,
    list_active_backtest_tasks, list_backtest_mutations, list_completed_backtests, list_started_backtest_task_ids,
)
from persistence.submission_checks import list_submission_checks
from persistence.submission_queue import list_formal_submission_queue
from persistence.pnl import list_pnl_series
from persistence.qualified_archive import list_qualified_alpha_archive
from persistence.submissions import (
    list_formal_submission_formulas, list_platform_submitted_alphas, normalize_submitted_formula,
)
from submission.formal import (
    BELOW_TARGET_GRADES, assess_formal_check_payload, local_formal_submission_eligible, qualified_archive_eligible,
)
from submission.opportunities import select_submission_opportunities


def load_unconsumed_qualified_candidates(
    connection: sqlite3.Connection, task_ids: set[str],
) -> tuple[BacktestSnapshot, ...]:
    """Read fully checked candidates without duplicating submission or archive state."""
    if not task_ids:
        return ()
    snapshots = tuple(get_backtest_task(connection, task_id) for task_id in sorted(task_ids))
    if any(snapshot is None for snapshot in snapshots):
        raise ValueError("qualified_candidate_source_missing")
    return _unconsumed_candidates(connection, snapshots)


def _unconsumed_candidates(
    connection: sqlite3.Connection,
    snapshots: tuple[BacktestSnapshot, ...],
) -> tuple[BacktestSnapshot, ...]:
    if not snapshots:
        return ()
    consumed = {
        (account, normalize_submitted_formula(formula))
        for account, formula in list_formal_submission_formulas(connection)
    }
    submitted_ids = set()
    for account in {snapshot.task.account_scope for snapshot in snapshots}:
        for record in list_platform_submitted_alphas(connection, account_scope=account):
            consumed.add((account, record.normalized_formula))
            submitted_ids.add((account, record.platform_alpha_id))
    candidates = []
    checks = {record.task_id: record for record in list_submission_checks(connection)}
    for snapshot in sorted(snapshots, key=lambda item: item.task.task_id):
        task = snapshot.task
        if ((task.account_scope, normalize_submitted_formula(task.formula)) in consumed
                or (task.account_scope, task.platform_alpha_id) in submitted_ids):
            continue
        check = checks.get(task.task_id)
        payload = json.loads(check.payload_json) if check and check.payload_json else None
        if qualified_archive_eligible(snapshot, payload):
            candidates.append(snapshot)
    return tuple(candidates)


def load_submittable_qualified_candidates(
    connection: sqlite3.Connection, *, account_scope: str,
) -> tuple[BacktestSnapshot, ...]:
    """The archive page and manual submission use exactly the same current steps."""
    archived = {record.task_id for record in list_qualified_alpha_archive(connection)}
    ids = set(load_submission_opportunity_ids(connection, account_scope=account_scope)) & archived
    return tuple(s for s in list_completed_backtests(connection, task_ids=frozenset(ids))
                 if s.result.grade in BELOW_TARGET_GRADES)


def load_submission_opportunity_ids(
    connection: sqlite3.Connection, *, account_scope: str,
    include_optimization: bool = False, reserved_task_id: str | None = None,
) -> tuple[str, ...]:
    """Plan globally before applying source, grade, page or explicit-selection filters.

    Archive rows record completed research. Eligibility is derived from current
    facts, never written by readers. A claimed item is reintroduced only for the
    final pre-submit guard; an uncertain submission is never retried here.
    """
    passed_ids = {record.task_id for record in list_submission_checks(connection)
                  if assess_formal_check_payload(json.loads(record.payload_json)
                     if record.payload_json else None).state == "passed"}
    if reserved_task_id:
        passed_ids.add(reserved_task_id)
    checked = tuple(s for s in list_completed_backtests(connection, task_ids=frozenset(passed_ids))
                    if s.task.account_scope == account_scope and local_formal_submission_eligible(s)
                    and s.result.grade in BELOW_TARGET_GRADES | {"SPECTACULAR"})
    if not checked:
        return ()
    references = list_platform_submitted_alphas(connection, account_scope=account_scope)
    consumed = {normalize_submitted_formula(formula) for account, formula
                in list_formal_submission_formulas(connection) if account == account_scope}
    submitted_ids = {ref.platform_alpha_id for ref in references}
    consumed.update(ref.normalized_formula for ref in references)
    series = list_pnl_series(connection, platform_alpha_ids=frozenset(
        [s.task.platform_alpha_id for s in checked] + [r.platform_alpha_id for r in references]),
        account_scope=account_scope)
    correlations = PnlCorrelations(series)
    mutations = list_backtest_mutations(connection)
    decisions = {r.task_id: r for r in assess_qualified_evolution(
        checked, mutations, list_started_backtest_task_ids(connection), series, correlations=correlations)}
    available = tuple(s for s in checked if s.task.task_id == reserved_task_id or (
        s.task.platform_alpha_id not in submitted_ids
        and normalize_submitted_formula(s.task.formula) not in consumed))
    unsettled_tasks = {task.task_id for task in list_active_backtest_tasks(connection)
                       if task.submission_started_at is not None}
    unsettled_parents = {
        mutation.parent_task_id for mutation in mutations
        if mutation.child_task_id in unsettled_tasks
    }
    archived = {r.task_id for r in list_qualified_alpha_archive(connection)}
    queued = {r.task_id for r in list_formal_submission_queue(connection, account_scope=account_scope)}
    candidates = tuple(s for s in available if s.task.task_id not in unsettled_parents and (
        s.task.task_id == reserved_task_id or s.task.task_id in queued
        or decisions[s.task.task_id].retired and s.task.task_id in archived
        or include_optimization and not decisions[s.task.task_id].retired))
    protected = tuple(s for s in available if s.task.task_id != reserved_task_id
                      and not decisions[s.task.task_id].retired and s.task.task_id not in queued)
    return select_submission_opportunities(candidates, protected, references, series, correlations=correlations)


def load_qualified_evolution(
    connection: sqlite3.Connection, *, excluded_task_ids: frozenset[str] = frozenset(),
    mutations: tuple[BacktestMutationRecord, ...] | None = None,
) -> tuple[QualifiedEvolutionRecord, ...]:
    """Read individual checked candidates and retirement from the original facts."""
    passed_ids = set()
    for check in list_submission_checks(connection):
        if check.task_id in excluded_task_ids:
            continue
        payload = json.loads(check.payload_json) if check.payload_json else None
        if assess_formal_check_payload(payload).state == "passed":
            passed_ids.add(check.task_id)
    checked = tuple(snapshot for snapshot in list_completed_backtests(connection, task_ids=frozenset(passed_ids))
                    if local_formal_submission_eligible(snapshot))
    eligible = {s.task.task_id for s in _unconsumed_candidates(connection, checked)}
    decisions = assess_qualified_evolution(
        checked, list_backtest_mutations(connection) if mutations is None else mutations,
        list_started_backtest_task_ids(connection) - excluded_task_ids,
        list_pnl_series(connection, platform_alpha_ids=frozenset(s.task.platform_alpha_id for s in checked)),
    )
    return tuple(record for record in decisions if record.task_id in eligible)
