from __future__ import annotations

import sqlite3
from execution.qualified_candidates import load_qualified_evolution
from generation.direction import DIRECTION_REVERSAL, reverse_direction_candidate
from generation.parser import parse_formula
from learning.direction import negative_direction_is_testable
from learning.evidence import build_learning_evidence
from learning.frontiers import (
    PARENT_ATTEMPT_BUDGET,
    SignalFrontier,
    SignalFrontierBranch,
    SignalFrontierSet,
    build_signal_frontiers,
)
from learning.seeds import assess_signal_seed
from learning.seed_correlation import assess_seed_correlation
from learning.recovery import recovery_task_ids
from execution.recovery import load_recovery_comparisons
from persistence.pnl import list_pnl_series
from persistence.qualified_archive import list_qualified_alpha_archive
from persistence.submissions import list_platform_submitted_alphas
from persistence.backtests import (
    BacktestMutationRecord,
    BacktestSnapshot,
    list_backtest_mutations,
    list_started_backtest_task_ids,
    list_completed_backtests,
)
from persistence.seeds import (
    SignalSeedRecord,
    create_signal_seed,
    get_signal_seed,
    list_signal_seeds,
)
from submission.formal import BELOW_TARGET_GRADES


def synchronize_signal_seeds(
    connection: sqlite3.Connection,
    *,
    candidate_task_ids: tuple[str, ...] | None = None,
) -> tuple[SignalSeedRecord, ...]:
    snapshots = list_completed_backtests(connection)
    candidates = _candidate_snapshots(snapshots, candidate_task_ids)
    mutation_children = {
        mutation.child_task_id: mutation
        for mutation in list_backtest_mutations(connection)
    }
    snapshots_by_id = {snapshot.task.task_id: snapshot for snapshot in snapshots}
    created: list[SignalSeedRecord] = []
    references = _seed_references(connection, snapshots)
    series = list_pnl_series(connection)
    for snapshot in candidates:
        mutation = mutation_children.get(snapshot.task.task_id)
        if mutation is not None:
            parent = snapshots_by_id.get(mutation.parent_task_id)
            if (
                mutation.action != DIRECTION_REVERSAL
                or parent is None
                or parent.task.task_id in mutation_children
                or not negative_direction_is_testable(parent)
                or parent.task.account_scope != snapshot.task.account_scope
                or parent.task.settings_json != snapshot.task.settings_json
            ):
                continue
            expected = reverse_direction_candidate(
                parse_formula(parent.task.formula).expression,
                parent_task_id=parent.task.task_id,
            )
            assert expected.change is not None
            if (
                snapshot.task.formula != expected.formula
                or mutation.location != expected.change.location
                or mutation.before != expected.change.before
                or mutation.after != expected.change.after
            ):
                continue
        if not assess_signal_seed(snapshot).eligible:
            continue
        if get_signal_seed(connection, snapshot.task.task_id) is not None:
            continue
        if assess_seed_correlation(snapshot, references, series).state not in {"passed", "submitted"}:
            continue
        assert snapshot.task.finished_at is not None
        created.append(
            create_signal_seed(
                connection,
                SignalSeedRecord(
                    root_task_id=snapshot.task.task_id,
                    promoted_at=snapshot.task.finished_at,
                ),
            )
        )
    return tuple(created)


def load_signal_frontiers(
    connection: sqlite3.Connection,
    *,
    excluded_task_ids: frozenset[str] = frozenset(),
    optimization_only: bool = False,
) -> SignalFrontierSet:
    """Research safe descendants of existing roots independently of new-seed admission."""
    if not isinstance(excluded_task_ids, frozenset) or any(
        not isinstance(task_id, str) or not task_id.strip()
        for task_id in excluded_task_ids
    ):
        raise ValueError("signal_frontier_excluded_tasks_invalid")
    mutations = list_backtest_mutations(connection)
    roots = tuple(seed.root_task_id for seed in list_signal_seeds(connection))
    decisions = load_qualified_evolution(
        connection, excluded_task_ids=excluded_task_ids, mutations=mutations,
    ) if optimization_only else ()
    required = None
    if optimization_only:
        # Evolution already considered all checked descendants, including submitted
        # replacements. Frontier membership only needs each active candidate's
        # complete ancestor chain and the roots, not unrelated completed backtests.
        parents = {mutation.child_task_id: mutation.parent_task_id for mutation in mutations}
        required = set(roots)
        pending = [decision.task_id for decision in decisions if not decision.retired]
        while pending:
            task_id = pending.pop()
            if task_id in required:
                continue
            required.add(task_id)
            if task_id in parents:
                pending.append(parents[task_id])
    completed = tuple(
        snapshot
        for snapshot in list_completed_backtests(
            connection, task_ids=frozenset(required) if required is not None else None,
        )
        if snapshot.task.task_id not in excluded_task_ids
    )
    evidence = build_learning_evidence(completed)
    frontiers = build_signal_frontiers(
        evidence,
        mutations,
        roots,
        _attempted_mutation_child_task_ids(
            connection,
            mutations,
            excluded_task_ids=excluded_task_ids,
        ),
        # Optimization uses checked lineage membership, which is independent of
        # recovery branches. Avoid rebuilding historical PnL correlations here.
        recovery_task_ids=frozenset() if optimization_only else recovery_task_ids(
            load_recovery_comparisons(connection, snapshots=completed),
            list_pnl_series(connection),
        ),
    )
    if not optimization_only:
        return frontiers
    root_by_task = {task_id: frontier.root_task_id for frontier in frontiers.records
                    for task_id in frontier.qualified_task_ids}
    facts = {record.task_id: record for record in evidence.records}
    grades = {snapshot.task.task_id: snapshot.result.grade for snapshot in completed}
    archived = {record.task_id for record in list_qualified_alpha_archive(connection)}
    selected = []
    for decision in decisions:
        task_id = decision.task_id
        if (decision.retired or task_id in archived or task_id not in root_by_task
                or grades[task_id] not in BELOW_TARGET_GRADES):
            continue
        fact = facts[task_id]
        branch = SignalFrontierBranch(
            root_task_id=root_by_task[task_id], task_id=task_id, account_scope=fact.account_scope,
            settings_key=fact.settings_key, finished_at=fact.finished_at,
            sharpe=fact.sharpe, fitness=fact.fitness, remaining_attempts=decision.remaining_attempts,
        )
        selected.append(SignalFrontier(
            root_task_id=branch.root_task_id, branches=(branch,), qualified_task_ids=(task_id,),
            qualified_parent_attempt_count=PARENT_ATTEMPT_BUDGET - decision.remaining_attempts,
            qualified_parent_task_id=task_id,
        ))
    return SignalFrontierSet(records=tuple(selected))


def signal_seed_evidence_task_ids(connection: sqlite3.Connection) -> tuple[str, ...]:
    """Potential roots/parents for evidence collection, never a generation pool."""
    frontiers = load_signal_frontiers(connection)
    return tuple(dict.fromkeys(
        task for frontier in frontiers.records
        for task in (frontier.root_task_id, *(b.task_id for b in frontier.branches))
    ))


def _seed_references(connection, snapshots):
    return tuple(ref for account in sorted({s.task.account_scope for s in snapshots})
                 for ref in list_platform_submitted_alphas(connection, account_scope=account))


def batch_advances_signal_frontier(
    connection: sqlite3.Connection,
    candidate_task_ids: tuple[str, ...],
) -> bool:
    candidate_ids = _candidate_ids(candidate_task_ids)
    seeds = list_signal_seeds(connection)
    candidate_roots = {
        seed.root_task_id for seed in seeds if seed.root_task_id in candidate_ids
    }
    if candidate_roots:
        return True

    before = load_signal_frontiers(
        connection,
        excluded_task_ids=frozenset(candidate_ids),
    )
    after = load_signal_frontiers(connection)
    before_by_root = {record.root_task_id: record for record in before.records}
    for current in after.records:
        previous = before_by_root.get(current.root_task_id)
        if previous is None:
            continue
        if not previous.qualified and current.qualified:
            return True
        previous_tasks = {branch.task_id for branch in previous.branches}
        current_tasks = {branch.task_id for branch in current.branches}
        if current_tasks != previous_tasks and current_tasks.intersection(candidate_ids):
            return True
    return False


def _attempted_mutation_child_task_ids(
    connection: sqlite3.Connection,
    mutations: tuple[BacktestMutationRecord, ...],
    *,
    excluded_task_ids: frozenset[str],
) -> tuple[str, ...]:
    started = list_started_backtest_task_ids(connection) - excluded_task_ids
    return tuple(
        sorted(
            mutation.child_task_id
            for mutation in mutations
            if mutation.child_task_id in started
        )
    )


def _candidate_snapshots(
    snapshots: tuple[BacktestSnapshot, ...],
    candidate_task_ids: tuple[str, ...] | None,
) -> tuple[BacktestSnapshot, ...]:
    if candidate_task_ids is None:
        return snapshots
    requested = _candidate_ids(candidate_task_ids)
    return tuple(
        snapshot for snapshot in snapshots if snapshot.task.task_id in requested
    )


def _candidate_ids(candidate_task_ids: tuple[str, ...]) -> set[str]:
    if not isinstance(candidate_task_ids, tuple) or any(
        not isinstance(task_id, str) or not task_id.strip()
        for task_id in candidate_task_ids
    ):
        raise ValueError("signal_seed_candidate_tasks_invalid")
    if len(set(candidate_task_ids)) != len(candidate_task_ids):
        raise ValueError("signal_seed_candidate_task_duplicated")
    return set(candidate_task_ids)
