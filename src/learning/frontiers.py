from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from learning.evidence import LearningEvidenceRecord, LearningEvidenceSet
from learning.optimization_targets import normalized_failure_gap
from learning.seeds import (
    SIGNAL_TARGET_CHECKS,
    signal_branch_is_safe,
)
from persistence.backtests import BacktestMutationRecord


MAX_SIGNAL_FRONTIER_BRANCHES = 3
PARENT_ATTEMPT_BUDGET = 20


@dataclass(frozen=True, slots=True)
class SignalFrontierBranch:
    root_task_id: str
    task_id: str
    account_scope: str
    settings_key: str
    finished_at: str
    sharpe: float
    fitness: float
    remaining_attempts: int


@dataclass(frozen=True, slots=True)
class SignalFrontier:
    root_task_id: str
    branches: tuple[SignalFrontierBranch, ...]
    qualified_task_ids: tuple[str, ...]
    qualified_parent_attempt_count: int
    qualified_parent_task_id: str | None

    @property
    def qualified(self) -> bool:
        return bool(self.qualified_task_ids)

    @property
    def qualified_evolution_active(self) -> bool:
        return any(branch.task_id in self.qualified_task_ids for branch in self.branches)

    @property
    def qualified_parent_remaining_attempts(self) -> int | None:
        if not self.qualified:
            return None
        return max(
            0,
            PARENT_ATTEMPT_BUDGET
            - self.qualified_parent_attempt_count,
        )


@dataclass(frozen=True, slots=True)
class SignalFrontierSet:
    records: tuple[SignalFrontier, ...]

    @property
    def active_branch_task_ids(self) -> tuple[str, ...]:
        return tuple(
            branch.task_id
            for record in self.records
            for branch in record.branches
        )


def build_signal_frontiers(
    evidence: LearningEvidenceSet,
    mutations: Iterable[BacktestMutationRecord],
    root_task_ids: Iterable[str],
    attempted_mutation_child_task_ids: Iterable[str] = (),
    *,
    recovery_task_ids: frozenset[str] = frozenset(),
) -> SignalFrontierSet:
    if not isinstance(evidence, LearningEvidenceSet):
        raise ValueError("signal_frontier_evidence_invalid")
    roots = tuple(root_task_ids)
    if any(not isinstance(value, str) or not value.strip() for value in roots):
        raise ValueError("signal_frontier_root_id_invalid")
    if len(set(roots)) != len(roots):
        raise ValueError("signal_frontier_root_duplicated")

    attempted_child_ids = tuple(attempted_mutation_child_task_ids)
    if any(
        not isinstance(task_id, str) or not task_id.strip()
        for task_id in attempted_child_ids
    ):
        raise ValueError("signal_frontier_attempted_child_id_invalid")
    if len(set(attempted_child_ids)) != len(attempted_child_ids):
        raise ValueError("signal_frontier_attempted_child_id_duplicated")

    records_by_task = {record.task_id: record for record in evidence.records}
    if len(records_by_task) != len(evidence.records):
        raise ValueError("signal_frontier_evidence_duplicated")
    children_by_parent: defaultdict[str, list[str]] = defaultdict(list)
    child_task_ids: set[str] = set()
    for mutation in mutations:
        if not isinstance(mutation, BacktestMutationRecord):
            raise ValueError("signal_frontier_mutation_invalid")
        if mutation.child_task_id in child_task_ids:
            raise ValueError("signal_frontier_child_duplicated")
        child_task_ids.add(mutation.child_task_id)
        children_by_parent[mutation.parent_task_id].append(mutation.child_task_id)
    if not set(attempted_child_ids) <= child_task_ids:
        raise ValueError("signal_frontier_attempted_child_missing")
    attempt_count_by_parent = count_parent_attempts(mutations, frozenset(attempted_child_ids))
    for children in children_by_parent.values():
        children.sort()

    built: list[SignalFrontier] = []
    for root_task_id in roots:
        root = records_by_task.get(root_task_id)
        if root is None:
            raise ValueError("signal_frontier_root_result_missing")
        if not signal_branch_is_safe(root):
            raise ValueError("signal_frontier_root_not_safe")
        lineage = _comparable_lineage(
            root,
            records_by_task,
            children_by_parent,
            frozenset(roots),
        )
        qualified_records = tuple(
            record
            for record in lineage
            if SIGNAL_TARGET_CHECKS <= set(record.passed_checks)
        )
        qualified_task_ids = tuple(
            record.task_id
            for record in sorted(
                qualified_records,
                key=lambda item: (item.finished_at, item.task_id),
            )
        )
        qualified_parent_attempt_count = 0
        if qualified_records:
            qualified_parent = min(
                qualified_records,
                key=lambda item: (-item.sharpe, item.finished_at, item.task_id),
            )
            qualified_parent_attempt_count = attempt_count_by_parent[
                qualified_parent.task_id
            ]
            active_records = (
                (qualified_parent,)
                if qualified_parent_attempt_count < PARENT_ATTEMPT_BUDGET
                else ()
            )
        else:
            candidates = tuple(
                record
                for record in lineage
                if not SIGNAL_TARGET_CHECKS <= set(record.passed_checks)
            )
            # Retired facts still dominate worse/equivalent descendants: expiry
            # must not revive an older point or grant a duplicate fresh budget.
            active_records = _bounded_frontier(
                tuple(
                    record
                    for record in _nondominated(candidates)
                    if attempt_count_by_parent[record.task_id] < PARENT_ATTEMPT_BUDGET
                )
            )
        # A qualified ancestor must not suppress a measured decorrelated branch.
        # Keep the existing lineage/root; reserve at most two of the three slots
        # for its quality frontier. Dominance is evaluated before budget expiry.
        if qualified_records:
            recovery_candidates = tuple(
                record
                for record in lineage
                if record.task_id in recovery_task_ids
                and record.task_id != qualified_parent.task_id
            )
            recovery_frontier = _bounded_frontier(_nondominated(recovery_candidates))
            active_recovery = tuple(
                record
                for record in recovery_frontier
                if attempt_count_by_parent[record.task_id] < PARENT_ATTEMPT_BUDGET
            )
            active_records = (
                *active_records,
                *active_recovery[: MAX_SIGNAL_FRONTIER_BRANCHES - 1],
            )
        branches = tuple(
            _frontier_branch(
                root_task_id, record, attempt_count_by_parent[record.task_id]
            )
            for record in active_records
        )
        built.append(
            SignalFrontier(
                root_task_id=root_task_id,
                branches=branches,
                qualified_task_ids=qualified_task_ids,
                qualified_parent_attempt_count=qualified_parent_attempt_count,
                qualified_parent_task_id=qualified_parent.task_id if qualified_records else None,
            )
        )
    return SignalFrontierSet(records=tuple(built))


def count_parent_attempts(
    mutations: Iterable[BacktestMutationRecord], attempted_child_task_ids: frozenset[str],
) -> defaultdict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for mutation in mutations:
        if mutation.child_task_id in attempted_child_task_ids:
            counts[mutation.parent_task_id] += 1
    return counts


def _frontier_branch(
    root_task_id: str,
    record: LearningEvidenceRecord,
    attempted_count: int,
) -> SignalFrontierBranch:
    return SignalFrontierBranch(
        root_task_id=root_task_id,
        task_id=record.task_id,
        account_scope=record.account_scope,
        settings_key=record.settings_key,
        finished_at=record.finished_at,
        sharpe=record.sharpe,
        fitness=record.fitness,
        remaining_attempts=max(0, PARENT_ATTEMPT_BUDGET - attempted_count),
    )


def _comparable_lineage(
    root: LearningEvidenceRecord,
    records_by_task: dict[str, LearningEvidenceRecord],
    children_by_parent: dict[str, list[str]],
    roots: frozenset[str],
) -> tuple[LearningEvidenceRecord, ...]:
    accepted: list[LearningEvidenceRecord] = []

    def visit(task_id: str, path: frozenset[str]) -> None:
        if task_id in path:
            raise ValueError("signal_frontier_cycle_detected")
        # A retained descendant root owns its subtree exactly once. Mutation
        # links and attempt counts remain intact when a rejected root is removed.
        if task_id != root.task_id and task_id in roots:
            return
        record = records_by_task.get(task_id)
        if record is None:
            return
        if (
            record.account_scope != root.account_scope
            or record.settings_key != root.settings_key
            or not signal_branch_is_safe(record)
        ):
            return
        accepted.append(record)
        next_path = path | {task_id}
        for child_task_id in children_by_parent.get(task_id, ()):
            visit(child_task_id, next_path)

    visit(root.task_id, frozenset())
    return tuple(accepted)


def _nondominated(
    candidates: tuple[LearningEvidenceRecord, ...],
) -> tuple[LearningEvidenceRecord, ...]:
    unique_by_point: dict[tuple[object, ...], LearningEvidenceRecord] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (item.finished_at, item.task_id),
    ):
        unique_by_point.setdefault(_frontier_point(candidate), candidate)
    unique = tuple(unique_by_point.values())
    return tuple(
        candidate
        for candidate in unique
        if not any(
            _dominates(other, candidate)
            for other in unique
            if other.task_id != candidate.task_id
        )
    )


def _dominates(
    left: LearningEvidenceRecord,
    right: LearningEvidenceRecord,
) -> bool:
    left_passed = set(left.passed_checks) & SIGNAL_TARGET_CHECKS
    right_passed = set(right.passed_checks) & SIGNAL_TARGET_CHECKS
    if not right_passed <= left_passed:
        return False
    gap_comparison = _low_sub_universe_gap_comparison(left, right)
    if gap_comparison is None or gap_comparison > 0:
        return False
    return (
        left.sharpe >= right.sharpe
        and left.fitness >= right.fitness
        and (
            left.sharpe > right.sharpe
            or left.fitness > right.fitness
            or left_passed > right_passed
            or gap_comparison < 0
        )
    )


def _bounded_frontier(
    candidates: tuple[LearningEvidenceRecord, ...],
) -> tuple[LearningEvidenceRecord, ...]:
    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: (item.sharpe, -item.fitness, item.task_id),
        )
    )
    if len(ordered) <= MAX_SIGNAL_FRONTIER_BRANCHES:
        return ordered
    fitness_representative = min(
        ordered,
        key=lambda item: (
            -item.fitness,
            -item.sharpe,
            item.finished_at,
            item.task_id,
        ),
    )
    sharpe_representative = min(
        ordered,
        key=lambda item: (
            -item.sharpe,
            -item.fitness,
            item.finished_at,
            item.task_id,
        ),
    )
    completion_representative = min(ordered, key=_completion_rank)
    chosen_ids: set[str] = set()
    for representative in (
        fitness_representative,
        sharpe_representative,
        completion_representative,
    ):
        if representative.task_id in chosen_ids:
            continue
        chosen_ids.add(representative.task_id)
    if len(chosen_ids) < MAX_SIGNAL_FRONTIER_BRANCHES:
        middle_index = (len(ordered) - 1) // 2
        for candidate in (ordered[middle_index], *ordered):
            if candidate.task_id in chosen_ids:
                continue
            chosen_ids.add(candidate.task_id)
            if len(chosen_ids) == MAX_SIGNAL_FRONTIER_BRANCHES:
                break
    return tuple(
        candidate for candidate in ordered if candidate.task_id in chosen_ids
    )


def _frontier_point(record: LearningEvidenceRecord) -> tuple[object, ...]:
    passed_targets = tuple(sorted(set(record.passed_checks) & SIGNAL_TARGET_CHECKS))
    gap_state, gap = _low_sub_universe_gap(record)
    return record.sharpe, record.fitness, passed_targets, gap_state, gap


def _low_sub_universe_gap_comparison(
    left: LearningEvidenceRecord,
    right: LearningEvidenceRecord,
) -> int | None:
    left_state, left_gap = _low_sub_universe_gap(left)
    right_state, right_gap = _low_sub_universe_gap(right)
    if left_state == "passed" or right_state == "passed":
        return 0
    if left_state == "available" and right_state == "available":
        assert left_gap is not None and right_gap is not None
        return -1 if left_gap < right_gap else 1 if left_gap > right_gap else 0
    if left_state == right_state:
        return 0
    return None


def _low_sub_universe_gap(
    record: LearningEvidenceRecord,
) -> tuple[str, float | None]:
    check_name = "LOW_SUB_UNIVERSE_SHARPE"
    if check_name in record.passed_checks:
        return "passed", 0.0
    check = next((item for item in record.checks if item.name == check_name), None)
    if check is None or check.status.strip().upper() != "FAIL":
        return "unavailable", None
    gap, state = normalized_failure_gap(
        details_captured=record.check_details_captured,
        threshold=check.threshold,
        actual=check.actual,
    )
    return state, gap


def _completion_rank(record: LearningEvidenceRecord) -> tuple[object, ...]:
    passed_targets = set(record.passed_checks) & SIGNAL_TARGET_CHECKS
    gap_state, gap = _low_sub_universe_gap(record)
    if gap_state == "passed":
        robustness = (0, 0.0)
    elif gap_state == "available" and gap is not None:
        robustness = (1, gap)
    else:
        robustness = (2, 0.0)
    return (
        -len(passed_targets),
        *robustness,
        -record.sharpe,
        -record.fitness,
        record.finished_at,
        record.task_id,
    )
