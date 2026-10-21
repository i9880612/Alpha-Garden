from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from execution.progress import LOGGER_NAME
from execution.self_correlation import load_self_correlation_references
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
from learning.pnl import PnlCorrelations
from learning.seed_correlation import (
    assess_seed_correlation,
    assess_seed_redundancy,
    blocking_seeds,
    clearly_stronger,
    submitted_seed,
)
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
    delete_signal_seed,
    SignalSeedRecord,
    create_signal_seed,
    list_signal_seeds,
)
from submission.formal import BELOW_TARGET_GRADES


def synchronize_signal_seeds(
    connection: sqlite3.Connection,
    *,
    candidate_task_ids: tuple[str, ...] | None = None,
    observed_at: str | None = None,
) -> tuple[SignalSeedRecord, ...]:
    requested = None if candidate_task_ids is None else _candidate_ids(candidate_task_ids)
    if requested is not None and not requested:
        return ()
    mutation_children = {
        mutation.child_task_id: mutation
        for mutation in list_backtest_mutations(connection)
    }
    seed_ids = {seed.root_task_id for seed in list_signal_seeds(connection)}
    required = None if requested is None else requested | {
        mutation.parent_task_id for child, mutation in mutation_children.items() if child in requested
    } | seed_ids
    snapshots = list_completed_backtests(connection, task_ids=frozenset(required) if required is not None else None)
    # Strongest first: within a group of near-copies the best one becomes the seed and
    # blocks the rest (seed-to-seed overlap has no Sharpe exception).
    candidates = tuple(sorted(
        (snapshot for snapshot in _candidate_snapshots(snapshots, candidate_task_ids)
         if snapshot.task.task_id not in seed_ids and assess_signal_seed(snapshot).eligible),
        key=lambda snapshot: (-abs(snapshot.result.sharpe), snapshot.task.task_id),
    ))
    if not candidates:
        return ()
    comparison_ids = seed_ids | set(load_signal_frontiers(connection).active_branch_task_ids) if seed_ids else set()
    loaded_ids = {snapshot.task.task_id for snapshot in snapshots}
    if comparison_ids - loaded_ids:
        snapshots += list_completed_backtests(connection, task_ids=frozenset(comparison_ids - loaded_ids))
    snapshots_by_id = {snapshot.task.task_id: snapshot for snapshot in snapshots}
    created: list[SignalSeedRecord] = []
    references = _seed_references(connection, snapshots)
    seed_snapshots = tuple(snapshots_by_id[task_id] for task_id in sorted(comparison_ids) if task_id in snapshots_by_id)
    series = list_pnl_series(connection, platform_alpha_ids=frozenset(
        [ref.platform_alpha_id for ref in references]
        + [s.task.platform_alpha_id for s in candidates if s.task.platform_alpha_id is not None]
        + [s.task.platform_alpha_id for s in seed_snapshots if s.task.platform_alpha_id is not None]
    ))
    correlations = PnlCorrelations(series)
    cutoff = datetime.fromisoformat(observed_at) if observed_at is not None else datetime.now().astimezone()
    repair_roots = {
        reference.parent_task_id
        for account in {snapshot.task.account_scope for snapshot in candidates}
        for reference in load_self_correlation_references(
            connection, parents=tuple(candidates), submitted_alphas=references,
            account_scope=account, observed_at=cutoff,
        )
        if reference.correlation is not None and reference.required_sharpe is not None
    }
    # Newly admitted seeds block later candidates in the same batch, so batch
    # admission matches sequential admission.
    admitted_snapshots = list(seed_snapshots)
    parented = {mutation.parent_task_id for mutation in mutation_children.values()}
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
        if (assess_seed_correlation(snapshot, references, series, correlations=correlations).state not in {"passed", "submitted"}
                and snapshot.task.task_id not in repair_roots):
            continue
        redundancy = assess_seed_redundancy(snapshot, tuple(admitted_snapshots), correlations=correlations)
        if redundancy == "failed":
            # A clearly stronger version of an idea whose seed has not been researched
            # yet replaces it, so each idea is represented by its best formula instead of
            # whichever version finished first. Researched seeds keep their lineage.
            blockers = blocking_seeds(snapshot, tuple(admitted_snapshots), correlations=correlations)
            replaceable = bool(blockers) and all(
                blocker.task.task_id in (seed_ids | {record.root_task_id for record in created})
                and blocker.task.task_id not in parented
                and blocker.task.task_id not in repair_roots
                and not submitted_seed(blocker, references)
                and clearly_stronger(snapshot, blocker)
                for blocker in blockers
            )
            remaining = tuple(item for item in admitted_snapshots if item not in blockers)
            if not replaceable or assess_seed_redundancy(
                    snapshot, remaining, correlations=correlations) != "passed":
                continue
            for blocker in blockers:
                delete_signal_seed(connection, blocker.task.task_id)
                created = [record for record in created if record.root_task_id != blocker.task.task_id]
                logging.getLogger(LOGGER_NAME).info(
                    "[种子] %s 由更强的同类公式 %s 接替（原种子尚未研究，回测结果保留）",
                    blocker.task.platform_alpha_id, snapshot.task.platform_alpha_id)
            admitted_snapshots = list(remaining)
        elif redundancy != "passed":
            continue
        assert snapshot.task.finished_at is not None
        created.append(
            create_signal_seed(
                connection,
                SignalSeedRecord(
                    root_task_id=snapshot.task.task_id,
                    promoted_at=cutoff.isoformat() if snapshot.task.task_id in repair_roots else snapshot.task.finished_at,
                ),
            )
        )
        admitted_snapshots.append(snapshot)
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
    # Ordinary research needs only the three curves of each actual recovery
    # comparison. Unrelated historical curves cannot affect these branches.
    comparisons = () if optimization_only else load_recovery_comparisons(connection, snapshots=completed)
    recovery_alphas: dict[str, set[str]] = {}
    for comparison in comparisons:
        recovery_alphas.setdefault(comparison.account_scope, set()).update((
            comparison.parent_alpha_id, comparison.child_alpha_id, comparison.reference_alpha_id,
        ))
    recovery_series = tuple(record for account, alphas in recovery_alphas.items()
                            for record in list_pnl_series(connection, account_scope=account,
                                                          platform_alpha_ids=frozenset(alphas)))
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
        recovery_task_ids=recovery_task_ids(comparisons, recovery_series),
    )
    if not optimization_only:
        return frontiers
    root_by_task = {task_id: frontier.root_task_id for frontier in frontiers.records
                    for task_id in frontier.qualified_task_ids}
    facts = {record.task_id: record for record in evidence.records}
    grades = {snapshot.task.task_id: snapshot.result.grade for snapshot in completed}
    archived = {record.task_id for record in list_qualified_alpha_archive(connection)}
    optimizable = [decision for decision in decisions if not decision.retired and decision.task_id not in archived
                   and grades[decision.task_id] in BELOW_TARGET_GRADES]
    root_by_task.update(_independent_quality_roots(
        connection, [decision.task_id for decision in optimizable if decision.task_id not in root_by_task],
        admitted=[task_id for task_id in (decision.task_id for decision in optimizable) if task_id in root_by_task],
        snapshots={snapshot.task.task_id: snapshot for snapshot in completed},
        mutations=mutations, seed_roots=frozenset(roots),
    ))
    selected = []
    for decision in optimizable:
        task_id = decision.task_id
        if task_id not in root_by_task:
            continue
        fact = facts[task_id]
        branch = SignalFrontierBranch(
            root_task_id=root_by_task[task_id], task_id=task_id, account_scope=fact.account_scope,
            settings_key=fact.settings_key, finished_at=fact.finished_at,
            sharpe=fact.sharpe, fitness=fact.fitness, remaining_attempts=decision.remaining_attempts,
        )
        selected.append(SignalFrontier(
            root_task_id=branch.root_task_id, branches=(branch,), qualified_task_ids=(task_id,),
            qualified_parent_attempt_count=decision.parent_attempt_count,
            qualified_parent_task_id=task_id,
        ))
    return SignalFrontierSet(records=tuple(selected))


def _independent_quality_roots(
    connection: sqlite3.Connection,
    candidate_task_ids: list[str],
    *,
    admitted: list[str],
    snapshots: dict[str, BacktestSnapshot],
    mutations: tuple[BacktestMutationRecord, ...],
    seed_roots: frozenset[str],
) -> dict[str, str]:
    """Give a qualified alpha outside every seed lineage its own quality lineage.

    Seed admission keeps near-copies of an idea from splitting one mutation budget,
    so a passing combo that resembles an unqualified seed never becomes a root.
    Quality optimization is a different job: it may improve the grade of any
    qualified alpha, as long as that alpha is not a near-copy of one already being
    optimized (another lineage) and is not blocked by a submitted alpha. Strongest
    first, so the best version of an idea is the one optimized. The root is the
    same one the quality budget charges, so attempts stay counted per lineage.
    """
    parents = {mutation.child_task_id: mutation.parent_task_id for mutation in mutations}

    def lineage_root(task_id: str) -> str:
        visited = set()
        while task_id not in seed_roots and task_id in parents and task_id not in visited:
            visited.add(task_id)
            task_id = parents[task_id]
        return task_id

    candidates = []
    for task_id in candidate_task_ids:
        root = lineage_root(task_id)
        # Inside a seed lineage the frontier already decided; never second-guess it here.
        if root not in seed_roots:
            candidates.append((snapshots[task_id], root))
    if not candidates:
        return {}
    candidates.sort(key=lambda item: (-abs(item[0].result.sharpe), item[0].task.finished_at, item[0].task.task_id))
    kept = [(snapshots[task_id], None) for task_id in admitted]
    accounts = {snapshot.task.account_scope for snapshot, _ in candidates}
    references = tuple(ref for account in sorted(accounts)
                       for ref in list_platform_submitted_alphas(connection, account_scope=account))
    series = tuple(record for account in sorted(accounts) for record in list_pnl_series(
        connection, account_scope=account, platform_alpha_ids=frozenset(
            [s.task.platform_alpha_id for s, _ in (*candidates, *kept) if s.task.account_scope == account]
            + [ref.platform_alpha_id for ref in references if ref.account_scope == account])))
    correlations = PnlCorrelations(series)
    selected: dict[str, str] = {}
    for snapshot, root in candidates:
        if assess_seed_correlation(snapshot, references, series, correlations=correlations).state != "passed":
            continue
        others = tuple(other for other, other_root in kept if other_root != root)
        if assess_seed_redundancy(snapshot, others, correlations=correlations) != "passed":
            continue
        kept.append((snapshot, root))
        selected[snapshot.task.task_id] = root
    return selected


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
