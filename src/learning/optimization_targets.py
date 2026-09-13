from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from learning.evidence import (
    LearningEvidenceRecord,
    LearningEvidenceSet,
    MutationLearningEvidenceRecord,
    MutationLearningEvidenceSet,
)
from learning.seeds import SIGNAL_TARGET_CHECKS, signal_branch_is_safe


@dataclass(frozen=True, slots=True)
class OptimizationActionEvidence:
    action: str
    comparable_count: int
    repaired_count: int
    safe_repaired_count: int
    remaining_failed_count: int
    unresolved_count: int


@dataclass(frozen=True, slots=True)
class OptimizationTarget:
    check_name: str
    details_captured: bool
    threshold: float | None
    actual: float | None
    platform_date: str | None
    normalized_gap: float | None
    gap_state: str
    actions: tuple[OptimizationActionEvidence, ...]


@dataclass(frozen=True, slots=True)
class ParentOptimizationTargets:
    parent_task_id: str
    account_scope: str
    settings_key: str
    targets: tuple[OptimizationTarget, ...]


@dataclass(frozen=True, slots=True)
class ParentOptimizationTargetSet:
    records: tuple[ParentOptimizationTargets, ...]


@dataclass(slots=True)
class _ActionCounts:
    comparable_count: int = 0
    repaired_count: int = 0
    safe_repaired_count: int = 0
    remaining_failed_count: int = 0
    unresolved_count: int = 0


def build_parent_optimization_targets(
    evidence: LearningEvidenceSet,
    mutation_evidence: MutationLearningEvidenceSet,
    parent_task_ids: Iterable[str],
) -> ParentOptimizationTargetSet:
    if not isinstance(evidence, LearningEvidenceSet):
        raise ValueError("optimization_learning_evidence_invalid")
    if not isinstance(mutation_evidence, MutationLearningEvidenceSet):
        raise ValueError("optimization_mutation_evidence_invalid")
    parent_ids = tuple(parent_task_ids)
    if any(not isinstance(value, str) or not value.strip() for value in parent_ids):
        raise ValueError("optimization_parent_id_invalid")
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("optimization_parent_duplicated")

    records_by_task = {record.task_id: record for record in evidence.records}
    mutations_by_parent: dict[str, list[MutationLearningEvidenceRecord]] = {}
    for record in mutation_evidence.records:
        if not record.comparison_available:
            continue
        mutations_by_parent.setdefault(record.parent_task_id, []).append(record)

    built: list[ParentOptimizationTargets] = []
    for parent_task_id in sorted(parent_ids):
        parent = records_by_task.get(parent_task_id)
        if parent is None:
            raise ValueError("optimization_parent_result_missing")
        checks_by_name = {check.name: check for check in parent.checks}
        target_names = tuple(sorted(set(parent.failed_checks) & SIGNAL_TARGET_CHECKS))
        targets: list[OptimizationTarget] = []
        for check_name in target_names:
            check = checks_by_name.get(check_name)
            if check is None or check.status.strip().upper() != "FAIL":
                raise ValueError("optimization_parent_check_mismatch")
            threshold = check.threshold if parent.check_details_captured else None
            actual = check.actual if parent.check_details_captured else None
            platform_date = (
                check.platform_date if parent.check_details_captured else None
            )
            normalized_gap, gap_state = normalized_failure_gap(
                details_captured=parent.check_details_captured,
                threshold=threshold,
                actual=actual,
            )
            targets.append(
                OptimizationTarget(
                    check_name=check_name,
                    details_captured=parent.check_details_captured,
                    threshold=threshold,
                    actual=actual,
                    platform_date=platform_date,
                    normalized_gap=normalized_gap,
                    gap_state=gap_state,
                    actions=_action_evidence(
                        check_name,
                        mutations_by_parent.get(parent_task_id, ()),
                        records_by_task,
                    ),
                )
            )
        built.append(
            ParentOptimizationTargets(
                parent_task_id=parent.task_id,
                account_scope=parent.account_scope,
                settings_key=parent.settings_key,
                targets=tuple(targets),
            )
        )
    return ParentOptimizationTargetSet(records=tuple(built))


def normalized_failure_gap(
    *,
    details_captured: bool,
    threshold: float | None,
    actual: float | None,
) -> tuple[float | None, str]:
    if not details_captured:
        return None, "historical_not_captured"
    if threshold is None or actual is None:
        return None, "platform_values_missing"
    if threshold == 0:
        return None, "threshold_zero"
    gap = threshold - actual
    if gap <= 0:
        return None, "status_value_conflict"
    return gap / abs(threshold), "available"


def _action_evidence(
    check_name: str,
    records: Iterable[MutationLearningEvidenceRecord],
    records_by_task: dict[str, LearningEvidenceRecord],
) -> tuple[OptimizationActionEvidence, ...]:
    counts_by_action: dict[str, _ActionCounts] = {}
    for record in records:
        counts = counts_by_action.setdefault(record.action, _ActionCounts())
        counts.comparable_count += 1
        if check_name in record.repaired_checks:
            counts.repaired_count += 1
            child = records_by_task.get(record.child_task_id)
            if (
                child is not None
                and signal_branch_is_safe(child)
                and "SELF_CORRELATION" in child.passed_checks
                and not record.introduced_failed_checks
                and not record.introduced_pending_checks
                and not record.child_missing_checks
                and not record.child_unexpected_checks
                and record.child_non_sc_check_set_complete
            ):
                counts.safe_repaired_count += 1
        elif check_name in record.remaining_failed_checks:
            counts.remaining_failed_count += 1
        else:
            counts.unresolved_count += 1
    return tuple(
        OptimizationActionEvidence(
            action=action,
            comparable_count=counts.comparable_count,
            repaired_count=counts.repaired_count,
            safe_repaired_count=counts.safe_repaired_count,
            remaining_failed_count=counts.remaining_failed_count,
            unresolved_count=counts.unresolved_count,
        )
        for action, counts in sorted(counts_by_action.items())
    )
