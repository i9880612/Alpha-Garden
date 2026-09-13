from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from learning.evidence import LearningEvidenceRecord, LearningEvidenceSet
from persistence.backtests import BacktestMutationRecord


_SELF_CORRELATION = "SELF_CORRELATION"
_MINIMUM_EXPLICIT_SC_RESULTS = 5


@dataclass(frozen=True, slots=True)
class ParentSettingsEvidence:
    parent_task_id: str
    account_scope: str
    settings_key: str
    child_count: int
    child_quality_passed_count: int
    child_quality_failed_count: int
    child_quality_pending_count: int
    sc_passed_count: int
    sc_failed_count: int
    sc_pending_count: int
    sc_missing_count: int
    sc_explicit_count: int
    sc_risk_available: bool
    sc_smoothed_risk: float | None


@dataclass(frozen=True, slots=True)
class ParentSettingsEvidenceSet:
    records: tuple[ParentSettingsEvidence, ...]


@dataclass(slots=True)
class _EvidenceCounts:
    child_count: int = 0
    child_quality_passed_count: int = 0
    child_quality_failed_count: int = 0
    child_quality_pending_count: int = 0
    sc_passed_count: int = 0
    sc_failed_count: int = 0
    sc_pending_count: int = 0
    sc_missing_count: int = 0


def build_parent_settings_evidence(
    evidence: LearningEvidenceSet,
    mutations: Iterable[BacktestMutationRecord],
    parent_task_ids: Iterable[str],
) -> ParentSettingsEvidenceSet:
    if not isinstance(evidence, LearningEvidenceSet):
        raise ValueError("parent_settings_learning_evidence_invalid")
    parent_ids = tuple(parent_task_ids)
    if any(not isinstance(value, str) or not value.strip() for value in parent_ids):
        raise ValueError("parent_settings_parent_id_invalid")
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("parent_settings_parent_duplicated")

    records_by_task = {record.task_id: record for record in evidence.records}
    parents: dict[str, LearningEvidenceRecord] = {}
    counts_by_parent: dict[str, _EvidenceCounts] = {}
    for parent_task_id in parent_ids:
        parent = records_by_task.get(parent_task_id)
        if parent is None:
            raise ValueError("parent_settings_parent_result_missing")
        parents[parent_task_id] = parent
        counts = _EvidenceCounts()
        _count_sc(counts, parent)
        counts_by_parent[parent_task_id] = counts

    child_ids: set[str] = set()
    for mutation in mutations:
        if mutation.child_task_id in child_ids:
            raise ValueError("parent_settings_mutation_child_duplicated")
        child_ids.add(mutation.child_task_id)
        parent = parents.get(mutation.parent_task_id)
        child = records_by_task.get(mutation.child_task_id)
        if (
            parent is None
            or child is None
            or child.account_scope != parent.account_scope
            or child.settings_key != parent.settings_key
        ):
            continue
        counts = counts_by_parent[mutation.parent_task_id]
        counts.child_count += 1
        quality = _child_quality(child)
        if quality == "passed":
            counts.child_quality_passed_count += 1
        elif quality == "failed":
            counts.child_quality_failed_count += 1
        else:
            counts.child_quality_pending_count += 1
        _count_sc(counts, child)

    built: list[ParentSettingsEvidence] = []
    for parent_task_id in sorted(parents):
        parent = parents[parent_task_id]
        counts = counts_by_parent[parent_task_id]
        explicit_count = counts.sc_passed_count + counts.sc_failed_count
        risk_available = explicit_count >= _MINIMUM_EXPLICIT_SC_RESULTS
        built.append(
            ParentSettingsEvidence(
                parent_task_id=parent_task_id,
                account_scope=parent.account_scope,
                settings_key=parent.settings_key,
                child_count=counts.child_count,
                child_quality_passed_count=counts.child_quality_passed_count,
                child_quality_failed_count=counts.child_quality_failed_count,
                child_quality_pending_count=counts.child_quality_pending_count,
                sc_passed_count=counts.sc_passed_count,
                sc_failed_count=counts.sc_failed_count,
                sc_pending_count=counts.sc_pending_count,
                sc_missing_count=counts.sc_missing_count,
                sc_explicit_count=explicit_count,
                sc_risk_available=risk_available,
                sc_smoothed_risk=(
                    (counts.sc_failed_count + 1) / (explicit_count + 2)
                    if risk_available
                    else None
                ),
            )
        )
    return ParentSettingsEvidenceSet(records=tuple(built))


def _child_quality(record: LearningEvidenceRecord) -> str:
    return record.outcome if record.outcome is not None else "pending"


def _count_sc(
    counts: _EvidenceCounts,
    record: LearningEvidenceRecord,
) -> None:
    if _SELF_CORRELATION in record.passed_checks:
        counts.sc_passed_count += 1
    elif _SELF_CORRELATION in record.failed_checks:
        counts.sc_failed_count += 1
    elif _SELF_CORRELATION in record.pending_checks:
        counts.sc_pending_count += 1
    else:
        counts.sc_missing_count += 1
