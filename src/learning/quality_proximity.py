from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from learning.evidence import LearningEvidenceRecord, LearningEvidenceSet
from learning.optimization_targets import (
    OptimizationTarget,
    ParentOptimizationTargetSet,
)
from learning.seeds import SIGNAL_TARGET_CHECKS


STRUCTURAL_EVOLUTION_STAGE = "structural_evolution"
LOCAL_POLISHING_STAGE = "local_polishing"
QUALIFIED_EVOLUTION_STAGE = "qualified_evolution"
IMPROVEMENT_STAGES = (
    STRUCTURAL_EVOLUTION_STAGE,
    LOCAL_POLISHING_STAGE,
    QUALIFIED_EVOLUTION_STAGE,
)

MAX_LOCAL_POLISHING_NORMALIZED_GAP = 0.05
MIN_COHORT_OUTCOMES_PER_STATUS = 5

_METRIC_BY_CHECK = {
    "LOW_FITNESS": "fitness",
    "LOW_SHARPE": "sharpe",
}
_LOCAL_POLISHING_TARGET_CHECKS = frozenset(_METRIC_BY_CHECK)
_COHORT_GAP_SOURCE = "cohort_observed_pass_floor"
_PLATFORM_GAP_SOURCE = "platform_check_detail"


@dataclass(frozen=True, slots=True)
class QualityProximity:
    check_name: str
    source: str
    actual: float
    reference: float
    normalized_gap: float


@dataclass(frozen=True, slots=True)
class ParentImprovementStage:
    parent_task_id: str
    stage: str
    proximity: QualityProximity | None


@dataclass(frozen=True, slots=True)
class ParentImprovementStageSet:
    records: tuple[ParentImprovementStage, ...]


def build_parent_improvement_stages(
    evidence: LearningEvidenceSet,
    optimization_targets: ParentOptimizationTargetSet,
    parent_task_ids: Iterable[str],
    *,
    qualified_parent_task_ids: Iterable[str] = (),
) -> ParentImprovementStageSet:
    if not isinstance(evidence, LearningEvidenceSet):
        raise ValueError("quality_proximity_learning_evidence_invalid")
    if not isinstance(optimization_targets, ParentOptimizationTargetSet):
        raise ValueError("quality_proximity_targets_invalid")
    parent_ids = tuple(parent_task_ids)
    if any(not isinstance(value, str) or not value.strip() for value in parent_ids):
        raise ValueError("quality_proximity_parent_id_invalid")
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("quality_proximity_parent_duplicated")
    qualified_parent_ids = tuple(qualified_parent_task_ids)
    if any(
        not isinstance(value, str) or not value.strip()
        for value in qualified_parent_ids
    ):
        raise ValueError("quality_proximity_qualified_parent_id_invalid")
    if len(set(qualified_parent_ids)) != len(qualified_parent_ids):
        raise ValueError("quality_proximity_qualified_parent_duplicated")
    if not set(qualified_parent_ids) <= set(parent_ids):
        raise ValueError("quality_proximity_qualified_parent_unknown")
    qualified_parent_id_set = set(qualified_parent_ids)

    evidence_by_task = {record.task_id: record for record in evidence.records}
    targets_by_parent = {
        record.parent_task_id: record for record in optimization_targets.records
    }
    if len(targets_by_parent) != len(optimization_targets.records):
        raise ValueError("quality_proximity_target_parent_duplicated")

    cohorts: dict[tuple[str, str], list[LearningEvidenceRecord]] = {}
    for record in evidence.records:
        cohorts.setdefault((record.account_scope, record.settings_key), []).append(
            record
        )

    stages: list[ParentImprovementStage] = []
    for parent_task_id in sorted(parent_ids):
        parent = evidence_by_task.get(parent_task_id)
        targets = targets_by_parent.get(parent_task_id)
        if parent is None:
            raise ValueError("quality_proximity_parent_result_missing")
        if targets is None:
            raise ValueError("quality_proximity_parent_target_missing")
        if (
            targets.account_scope != parent.account_scope
            or targets.settings_key != parent.settings_key
        ):
            raise ValueError("quality_proximity_parent_identity_mismatch")
        if parent_task_id in qualified_parent_id_set:
            if (
                targets.targets
                or not SIGNAL_TARGET_CHECKS <= set(parent.passed_checks)
            ):
                raise ValueError("quality_proximity_qualified_parent_invalid")
            stages.append(
                ParentImprovementStage(
                    parent_task_id=parent_task_id,
                    stage=QUALIFIED_EVOLUTION_STAGE,
                    proximity=None,
                )
            )
            continue
        if not targets.targets:
            raise ValueError("quality_proximity_parent_target_missing")

        proximity = _local_polishing_proximity(
            parent,
            targets.targets,
            cohort=tuple(cohorts[(parent.account_scope, parent.settings_key)]),
        )
        stages.append(
            ParentImprovementStage(
                parent_task_id=parent_task_id,
                stage=(
                    LOCAL_POLISHING_STAGE
                    if proximity is not None
                    else STRUCTURAL_EVOLUTION_STAGE
                ),
                proximity=proximity,
            )
        )
    return ParentImprovementStageSet(records=tuple(stages))


def _local_polishing_proximity(
    parent: LearningEvidenceRecord,
    targets: tuple[OptimizationTarget, ...],
    *,
    cohort: tuple[LearningEvidenceRecord, ...],
) -> QualityProximity | None:
    if len(targets) != 1:
        return None
    target = targets[0]
    other_checks = SIGNAL_TARGET_CHECKS - {target.check_name}
    if (
        target.check_name not in _LOCAL_POLISHING_TARGET_CHECKS
        or target.check_name not in parent.failed_checks
        or not other_checks <= set(parent.passed_checks)
    ):
        return None

    direct = _platform_proximity(target)
    if direct is not None:
        return direct if _within_local_polishing_gap(direct.normalized_gap) else None
    if target.gap_state != "historical_not_captured":
        return None
    inferred = _cohort_proximity(parent, target.check_name, cohort)
    if inferred is None or not _within_local_polishing_gap(inferred.normalized_gap):
        return None
    return inferred


def _platform_proximity(target: OptimizationTarget) -> QualityProximity | None:
    if (
        target.gap_state != "available"
        or not target.details_captured
        or target.actual is None
        or target.threshold is None
        or target.normalized_gap is None
        or not all(
            math.isfinite(value)
            for value in (
                target.actual,
                target.threshold,
                target.normalized_gap,
            )
        )
        or target.threshold <= 0
        or target.normalized_gap <= 0
    ):
        return None
    return QualityProximity(
        check_name=target.check_name,
        source=_PLATFORM_GAP_SOURCE,
        actual=target.actual,
        reference=target.threshold,
        normalized_gap=target.normalized_gap,
    )


def _cohort_proximity(
    parent: LearningEvidenceRecord,
    check_name: str,
    cohort: tuple[LearningEvidenceRecord, ...],
) -> QualityProximity | None:
    metric_name = _METRIC_BY_CHECK.get(check_name)
    if metric_name is None:
        return None
    passed = tuple(
        getattr(record, metric_name)
        for record in cohort
        if check_name in record.passed_checks
    )
    failed = tuple(
        getattr(record, metric_name)
        for record in cohort
        if check_name in record.failed_checks
    )
    if (
        len(passed) < MIN_COHORT_OUTCOMES_PER_STATUS
        or len(failed) < MIN_COHORT_OUTCOMES_PER_STATUS
        or any(not _finite_metric(value) for value in (*passed, *failed))
    ):
        return None
    pass_floor = min(passed)
    if pass_floor <= 0 or max(failed) >= pass_floor:
        return None
    actual = getattr(parent, metric_name)
    normalized_gap = (pass_floor - actual) / abs(pass_floor)
    if not math.isfinite(normalized_gap) or normalized_gap <= 0:
        return None
    return QualityProximity(
        check_name=check_name,
        source=_COHORT_GAP_SOURCE,
        actual=actual,
        reference=pass_floor,
        normalized_gap=normalized_gap,
    )


def _within_local_polishing_gap(value: float) -> bool:
    return value <= MAX_LOCAL_POLISHING_NORMALIZED_GAP + 1e-12


def _finite_metric(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )
