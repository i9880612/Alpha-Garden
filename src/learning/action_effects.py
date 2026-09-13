from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from learning.evidence import (
    LearningEvidenceRecord,
    LearningEvidenceSet,
    MutationLearningEvidenceRecord,
    MutationLearningEvidenceSet,
)
from learning.optimization_targets import normalized_failure_gap
from learning.seeds import SIGNAL_TARGET_CHECKS, signal_branch_is_safe


ACTION_EFFECT_SAFE_PROGRESS = "safe_progress"
ACTION_EFFECT_NO_PROGRESS = "no_progress"
ACTION_EFFECT_CONFLICT = "conflict"
ACTION_EFFECT_UNRESOLVED = "unresolved"

ACTION_STATE_INSUFFICIENT = "insufficient"
ACTION_STATE_PREFERRED = "preferred"
ACTION_STATE_DEPRIORITIZED = "deprioritized"
ACTION_STATE_MIXED = "mixed"

ACTION_STRATEGY_EXPLORATION = "exploration"
ACTION_STRATEGY_EXPLOITATION = "exploitation"

MIN_RESOLVED_ACTION_EFFECTS = 10
MIN_ACTION_ROOT_LINEAGES = 5

_SELF_CORRELATION = "SELF_CORRELATION"
_METRIC_DELTA_BY_TARGET = {
    "LOW_FITNESS": "fitness_delta",
    "LOW_SHARPE": "sharpe_delta",
}


@dataclass(frozen=True, slots=True)
class TaskRunEvidence:
    task_id: str
    run_id: str


@dataclass(frozen=True, slots=True)
class ExplicitSelfCorrelationEvidence:
    task_id: str
    status: str
    formal_check_state: str
    observed_at: str


@dataclass(frozen=True, slots=True)
class DefectActionRequest:
    parent_task_id: str
    target_check_name: str
    candidate_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActionEffectObservation:
    child_task_id: str
    parent_task_id: str
    root_task_id: str
    run_id: str
    account_scope: str
    settings_key: str
    defect_checks: tuple[str, ...]
    target_check_name: str
    action: str
    outcome: str


@dataclass(frozen=True, slots=True)
class DefectActionEvidence:
    action: str
    state: str
    resolved_count: int
    safe_progress_count: int
    no_progress_count: int
    conflict_count: int
    unresolved_count: int
    root_lineage_count: int
    run_count: int


@dataclass(frozen=True, slots=True)
class DefectActionStrategy:
    parent_task_id: str
    account_scope: str
    settings_key: str
    defect_checks: tuple[str, ...]
    target_check_name: str
    mode: str
    reason: str
    preferred_actions: tuple[str, ...]
    deprioritized_actions: tuple[str, ...]
    actions: tuple[DefectActionEvidence, ...]


@dataclass(frozen=True, slots=True)
class DefectActionStrategySet:
    records: tuple[DefectActionStrategy, ...]
    observations: tuple[ActionEffectObservation, ...]


def build_defect_action_strategies(
    evidence: LearningEvidenceSet,
    mutation_evidence: MutationLearningEvidenceSet,
    requests: Iterable[DefectActionRequest],
    *,
    seed_root_task_ids: Iterable[str],
    task_runs: Iterable[TaskRunEvidence],
    explicit_self_correlations: Iterable[ExplicitSelfCorrelationEvidence],
) -> DefectActionStrategySet:
    if not isinstance(evidence, LearningEvidenceSet):
        raise ValueError("action_effect_learning_evidence_invalid")
    if not isinstance(mutation_evidence, MutationLearningEvidenceSet):
        raise ValueError("action_effect_mutation_evidence_invalid")

    requested = tuple(requests)
    roots = _validated_roots(seed_root_task_ids)
    runs_by_task = _task_runs_by_task(task_runs)
    explicit_sc_by_task = _self_correlations_by_task(explicit_self_correlations)
    _validate_requests(requested)

    records_by_task = {record.task_id: record for record in evidence.records}
    if len(records_by_task) != len(evidence.records):
        raise ValueError("action_effect_learning_task_duplicated")
    parent_by_child = {
        record.child_task_id: record.parent_task_id
        for record in mutation_evidence.records
    }
    if len(parent_by_child) != len(mutation_evidence.records):
        raise ValueError("action_effect_mutation_child_duplicated")

    observations = _build_observations(
        records_by_task,
        mutation_evidence.records,
        roots=roots,
        parent_by_child=parent_by_child,
        runs_by_task=runs_by_task,
        explicit_sc_by_task=explicit_sc_by_task,
    )
    observations_by_group: defaultdict[
        tuple[str, str, tuple[str, ...], str, str],
        list[ActionEffectObservation],
    ] = defaultdict(list)
    for observation in observations:
        observations_by_group[
            (
                observation.account_scope,
                observation.settings_key,
                observation.defect_checks,
                observation.target_check_name,
                observation.action,
            )
        ].append(observation)

    strategies: list[DefectActionStrategy] = []
    for request in sorted(
        requested,
        key=lambda item: (item.parent_task_id, item.target_check_name),
    ):
        parent = records_by_task.get(request.parent_task_id)
        if parent is None:
            raise ValueError("action_effect_parent_result_missing")
        defect_checks = _eligible_defect_checks(parent)
        if defect_checks is None or request.target_check_name not in defect_checks:
            raise ValueError("action_effect_parent_target_invalid")

        actions = tuple(
            _summarize_action(
                action,
                observations_by_group.get(
                    (
                        parent.account_scope,
                        parent.settings_key,
                        defect_checks,
                        request.target_check_name,
                        action,
                    ),
                    (),
                ),
            )
            for action in sorted(request.candidate_actions)
        )
        preferred = tuple(
            action.action
            for action in actions
            if action.state == ACTION_STATE_PREFERRED
        )
        deprioritized = tuple(
            action.action
            for action in actions
            if action.state == ACTION_STATE_DEPRIORITIZED
        )
        mode = (
            ACTION_STRATEGY_EXPLOITATION
            if preferred
            else ACTION_STRATEGY_EXPLORATION
        )
        strategies.append(
            DefectActionStrategy(
                parent_task_id=parent.task_id,
                account_scope=parent.account_scope,
                settings_key=parent.settings_key,
                defect_checks=defect_checks,
                target_check_name=request.target_check_name,
                mode=mode,
                reason=_strategy_reason(actions, preferred),
                preferred_actions=preferred,
                deprioritized_actions=deprioritized,
                actions=actions,
            )
        )

    return DefectActionStrategySet(
        records=tuple(strategies),
        observations=observations,
    )


def _build_observations(
    records_by_task: dict[str, LearningEvidenceRecord],
    mutation_records: tuple[MutationLearningEvidenceRecord, ...],
    *,
    roots: frozenset[str],
    parent_by_child: dict[str, str],
    runs_by_task: dict[str, str],
    explicit_sc_by_task: dict[str, ExplicitSelfCorrelationEvidence],
) -> tuple[ActionEffectObservation, ...]:
    built: list[ActionEffectObservation] = []
    for mutation in mutation_records:
        if not mutation.comparison_available:
            continue
        parent = records_by_task.get(mutation.parent_task_id)
        child = records_by_task.get(mutation.child_task_id)
        if parent is None or child is None:
            raise ValueError("action_effect_mutation_result_missing")
        root_task_id = _lineage_root(
            parent.task_id,
            parent_by_child=parent_by_child,
            roots=roots,
        )
        run_id = runs_by_task.get(child.task_id)
        defect_checks = _eligible_defect_checks(parent)
        if root_task_id is None or run_id is None or defect_checks is None:
            continue
        for target_check_name in defect_checks:
            built.append(
                ActionEffectObservation(
                    child_task_id=child.task_id,
                    parent_task_id=parent.task_id,
                    root_task_id=root_task_id,
                    run_id=run_id,
                    account_scope=parent.account_scope,
                    settings_key=parent.settings_key,
                    defect_checks=defect_checks,
                    target_check_name=target_check_name,
                    action=mutation.action,
                    outcome=_classify_observation(
                        parent,
                        child,
                        mutation,
                        target_check_name=target_check_name,
                        explicit_sc=explicit_sc_by_task.get(child.task_id),
                    ),
                )
            )
    return tuple(
        sorted(
            built,
            key=lambda item: (
                item.account_scope,
                item.settings_key,
                item.defect_checks,
                item.target_check_name,
                item.action,
                item.run_id,
                item.root_task_id,
                item.parent_task_id,
                item.child_task_id,
            ),
        )
    )


def _eligible_defect_checks(
    parent: LearningEvidenceRecord,
) -> tuple[str, ...] | None:
    passed = set(parent.passed_checks)
    failed = set(parent.failed_checks)
    if (
        not parent.non_sc_check_set_complete
        or not SIGNAL_TARGET_CHECKS <= (passed | failed)
        or not signal_branch_is_safe(parent)
    ):
        return None
    defects = tuple(sorted(failed & SIGNAL_TARGET_CHECKS))
    return defects or None


def _classify_observation(
    parent: LearningEvidenceRecord,
    child: LearningEvidenceRecord,
    mutation: MutationLearningEvidenceRecord,
    *,
    target_check_name: str,
    explicit_sc: ExplicitSelfCorrelationEvidence | None,
) -> str:
    progress = _target_progress(
        parent,
        child,
        mutation,
        target_check_name,
    )
    if progress == ACTION_EFFECT_UNRESOLVED:
        return ACTION_EFFECT_UNRESOLVED
    if progress == ACTION_EFFECT_CONFLICT:
        return ACTION_EFFECT_CONFLICT
    if progress == ACTION_EFFECT_NO_PROGRESS:
        return ACTION_EFFECT_NO_PROGRESS

    if not _child_has_no_safety_regression(child, mutation):
        return ACTION_EFFECT_CONFLICT
    sc_status = _effective_sc_status(child, explicit_sc)
    if sc_status == "failed":
        return ACTION_EFFECT_CONFLICT
    if sc_status != "passed":
        return ACTION_EFFECT_UNRESOLVED
    return ACTION_EFFECT_SAFE_PROGRESS


def _target_progress(
    parent: LearningEvidenceRecord,
    child: LearningEvidenceRecord,
    mutation: MutationLearningEvidenceRecord,
    target_check_name: str,
) -> str:
    if (
        target_check_name in child.pending_checks
        or target_check_name in mutation.child_missing_checks
        or target_check_name
        not in (set(child.passed_checks) | set(child.failed_checks))
    ):
        return ACTION_EFFECT_UNRESOLVED

    if target_check_name in child.passed_checks:
        metric_delta_name = _METRIC_DELTA_BY_TARGET.get(target_check_name)
        if metric_delta_name is not None and not _positive_finite(
            getattr(mutation, metric_delta_name)
        ):
            return ACTION_EFFECT_CONFLICT
        return ACTION_EFFECT_SAFE_PROGRESS

    if target_check_name not in child.failed_checks:
        return ACTION_EFFECT_UNRESOLVED
    metric_delta_name = _METRIC_DELTA_BY_TARGET.get(target_check_name)
    if metric_delta_name is not None:
        delta = getattr(mutation, metric_delta_name)
        if not _finite_number(delta):
            return ACTION_EFFECT_UNRESOLVED
        return (
            ACTION_EFFECT_SAFE_PROGRESS
            if delta > 1e-12
            else ACTION_EFFECT_NO_PROGRESS
        )

    gap_comparison = _sub_universe_gap_improved(parent, child)
    if gap_comparison is None:
        return ACTION_EFFECT_UNRESOLVED
    return (
        ACTION_EFFECT_SAFE_PROGRESS
        if gap_comparison
        else ACTION_EFFECT_NO_PROGRESS
    )


def _child_has_no_safety_regression(
    child: LearningEvidenceRecord,
    mutation: MutationLearningEvidenceRecord,
) -> bool:
    return bool(
        signal_branch_is_safe(child)
        and not (set(mutation.introduced_failed_checks) - {_SELF_CORRELATION})
        and not (set(mutation.introduced_pending_checks) - {_SELF_CORRELATION})
        and not (set(mutation.child_missing_checks) - {_SELF_CORRELATION})
        and not mutation.child_unexpected_checks
        and mutation.child_non_sc_check_set_complete
    )


def _effective_sc_status(
    child: LearningEvidenceRecord,
    explicit_sc: ExplicitSelfCorrelationEvidence | None,
) -> str:
    if _SELF_CORRELATION in child.passed_checks:
        backtest_status = "passed"
    elif _SELF_CORRELATION in child.failed_checks:
        backtest_status = "failed"
    elif _SELF_CORRELATION in child.pending_checks:
        backtest_status = "pending"
    else:
        backtest_status = "missing"
    if explicit_sc is None:
        return backtest_status
    if explicit_sc.status == "PASS" and explicit_sc.formal_check_state != "passed":
        return "unresolved"
    formal_status = {"PASS": "passed", "FAIL": "failed"}[explicit_sc.status]
    if backtest_status in {"passed", "failed"} and backtest_status != formal_status:
        return "conflict"
    return formal_status


def _sub_universe_gap_improved(
    parent: LearningEvidenceRecord,
    child: LearningEvidenceRecord,
) -> bool | None:
    parent_gap, parent_state = _failure_gap(parent, "LOW_SUB_UNIVERSE_SHARPE")
    child_gap, child_state = _failure_gap(child, "LOW_SUB_UNIVERSE_SHARPE")
    if parent_state != "available" or child_state != "available":
        return None
    assert parent_gap is not None and child_gap is not None
    return child_gap + 1e-12 < parent_gap


def _failure_gap(
    record: LearningEvidenceRecord,
    check_name: str,
) -> tuple[float | None, str]:
    check = next((item for item in record.checks if item.name == check_name), None)
    if check is None or check.status.strip().upper() != "FAIL":
        return None, "unavailable"
    return normalized_failure_gap(
        details_captured=record.check_details_captured,
        threshold=check.threshold,
        actual=check.actual,
    )


def _summarize_action(
    action: str,
    observations: Iterable[ActionEffectObservation],
) -> DefectActionEvidence:
    items = tuple(observations)
    safe_count = sum(item.outcome == ACTION_EFFECT_SAFE_PROGRESS for item in items)
    no_progress_count = sum(
        item.outcome == ACTION_EFFECT_NO_PROGRESS for item in items
    )
    conflict_count = sum(item.outcome == ACTION_EFFECT_CONFLICT for item in items)
    unresolved_count = sum(
        item.outcome == ACTION_EFFECT_UNRESOLVED for item in items
    )
    resolved = tuple(
        item for item in items if item.outcome != ACTION_EFFECT_UNRESOLVED
    )
    lineages: defaultdict[str, list[ActionEffectObservation]] = defaultdict(list)
    for item in resolved:
        lineages[item.root_task_id].append(item)
    return DefectActionEvidence(
        action=action,
        state=_action_state(resolved, lineages=lineages),
        resolved_count=len(resolved),
        safe_progress_count=safe_count,
        no_progress_count=no_progress_count,
        conflict_count=conflict_count,
        unresolved_count=unresolved_count,
        root_lineage_count=len(lineages),
        run_count=len({item.run_id for item in resolved}),
    )


def _action_state(
    resolved: tuple[ActionEffectObservation, ...],
    *,
    lineages: dict[str, list[ActionEffectObservation]],
) -> str:
    if (
        len(resolved) < MIN_RESOLVED_ACTION_EFFECTS
        or len(lineages) < MIN_ACTION_ROOT_LINEAGES
    ):
        return ACTION_STATE_INSUFFICIENT

    if all(
        sum(item.outcome == ACTION_EFFECT_SAFE_PROGRESS for item in lineage_items)
        > sum(
            item.outcome in {ACTION_EFFECT_NO_PROGRESS, ACTION_EFFECT_CONFLICT}
            for item in lineage_items
        )
        for lineage_items in lineages.values()
    ):
        return ACTION_STATE_PREFERRED
    if all(
        not any(
            item.outcome == ACTION_EFFECT_SAFE_PROGRESS for item in lineage_items
        )
        and any(
            item.outcome in {ACTION_EFFECT_NO_PROGRESS, ACTION_EFFECT_CONFLICT}
            for item in lineage_items
        )
        for lineage_items in lineages.values()
    ):
        return ACTION_STATE_DEPRIORITIZED
    return ACTION_STATE_MIXED


def _strategy_reason(
    actions: tuple[DefectActionEvidence, ...],
    preferred_actions: tuple[str, ...],
) -> str:
    if preferred_actions:
        return "preferred_action_supported"
    if all(action.state == ACTION_STATE_INSUFFICIENT for action in actions):
        return "insufficient_lineage_evidence"
    if any(action.state == ACTION_STATE_MIXED for action in actions):
        return "mixed_action_evidence"
    return "no_preferred_action"


def _lineage_root(
    task_id: str,
    *,
    parent_by_child: dict[str, str],
    roots: frozenset[str],
) -> str | None:
    current = task_id
    visited: set[str] = set()
    seed_root: str | None = None
    while True:
        if current in visited:
            raise ValueError("action_effect_mutation_lineage_cycle")
        visited.add(current)
        if seed_root is None and current in roots:
            seed_root = current
        parent = parent_by_child.get(current)
        if parent is None:
            return seed_root
        current = parent


def _validated_roots(values: Iterable[str]) -> frozenset[str]:
    roots = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in roots):
        raise ValueError("action_effect_seed_root_invalid")
    if len(set(roots)) != len(roots):
        raise ValueError("action_effect_seed_root_duplicated")
    return frozenset(roots)


def _task_runs_by_task(values: Iterable[TaskRunEvidence]) -> dict[str, str]:
    built: dict[str, str] = {}
    for value in values:
        if (
            not isinstance(value, TaskRunEvidence)
            or not isinstance(value.task_id, str)
            or not value.task_id.strip()
            or not isinstance(value.run_id, str)
            or not value.run_id.strip()
        ):
            raise ValueError("action_effect_task_run_invalid")
        if value.task_id in built:
            raise ValueError("action_effect_task_run_duplicated")
        built[value.task_id] = value.run_id
    return built


def _self_correlations_by_task(
    values: Iterable[ExplicitSelfCorrelationEvidence],
) -> dict[str, ExplicitSelfCorrelationEvidence]:
    built: dict[str, ExplicitSelfCorrelationEvidence] = {}
    for value in values:
        if (
            not isinstance(value, ExplicitSelfCorrelationEvidence)
            or not isinstance(value.task_id, str)
            or not value.task_id.strip()
            or value.status not in {"PASS", "FAIL"}
            or value.formal_check_state not in {"passed", "failed", "pending"}
            or (value.status == "FAIL" and value.formal_check_state != "failed")
        ):
            raise ValueError("action_effect_self_correlation_invalid")
        _timestamp(value.observed_at, "action_effect_self_correlation_invalid")
        if value.task_id in built:
            raise ValueError("action_effect_self_correlation_duplicated")
        built[value.task_id] = value
    return built


def _validate_requests(requests: tuple[DefectActionRequest, ...]) -> None:
    identities: set[tuple[str, str]] = set()
    for request in requests:
        if (
            not isinstance(request, DefectActionRequest)
            or not isinstance(request.parent_task_id, str)
            or not request.parent_task_id.strip()
            or request.target_check_name not in SIGNAL_TARGET_CHECKS
            or not isinstance(request.candidate_actions, tuple)
            or not request.candidate_actions
            or any(
                not isinstance(action, str) or not action.strip()
                for action in request.candidate_actions
            )
            or len(set(request.candidate_actions)) != len(request.candidate_actions)
        ):
            raise ValueError("action_effect_request_invalid")
        identity = (request.parent_task_id, request.target_check_name)
        if identity in identities:
            raise ValueError("action_effect_request_duplicated")
        identities.add(identity)


def _finite_number(value: object) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _positive_finite(value: object) -> bool:
    return _finite_number(value) and value > 1e-12


def _timestamp(value: object, error: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(error) from exc
    if parsed.utcoffset() is None:
        raise ValueError(error)
    return parsed
