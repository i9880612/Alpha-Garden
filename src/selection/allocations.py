from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

from generation.polishing import POLISHING_FAMILIES
from generation.internal_edits import INTERNAL_EDIT_FAMILIES
from generation.self_correlation import (
    SELF_CORRELATION_REPAIR_FAMILIES,
    SELF_CORRELATION_LIGHT_FAMILIES,
    SELF_CORRELATION_INTERNAL_FAMILIES,
)
from generation.transformations import (
    DISTRIBUTION_STABILIZATION,
    GROUP_RELATIVE_REFRAME,
    STRUCTURAL_TRANSFORMATION_FAMILIES,
    TEMPORAL_PERSISTENCE_REFRAME,
)
from learning.action_effects import (
    ACTION_STRATEGY_EXPLOITATION,
    DefectActionStrategy,
    DefectActionStrategySet,
)
from learning.optimization_targets import (
    OptimizationTarget,
    ParentOptimizationTargetSet,
    ParentOptimizationTargets,
)
from learning.frontiers import PARENT_ATTEMPT_BUDGET
from learning.parent_settings import (
    ParentSettingsEvidence,
    ParentSettingsEvidenceSet,
)
from learning.quality_proximity import (
    IMPROVEMENT_STAGES,
    LOCAL_POLISHING_STAGE,
    QUALIFIED_EVOLUTION_STAGE,
    STRUCTURAL_EVOLUTION_STAGE,
)


LOW_SUB_UNIVERSE_SHARPE_STRUCTURAL_FAMILIES = (
    GROUP_RELATIVE_REFRAME,
    DISTRIBUTION_STABILIZATION,
    TEMPORAL_PERSISTENCE_REFRAME,
)
MAX_QUALIFIED_EVOLUTION_BACKTESTS_PER_LINEAGE = 2
QUALIFIED_EVOLUTION_FAMILIES = (
    *INTERNAL_EDIT_FAMILIES,
    *POLISHING_FAMILIES,
    *STRUCTURAL_TRANSFORMATION_FAMILIES,
    *SELF_CORRELATION_REPAIR_FAMILIES,
)
IMPROVEMENT_FAMILIES_BY_STAGE = {
    STRUCTURAL_EVOLUTION_STAGE: (*STRUCTURAL_TRANSFORMATION_FAMILIES, *INTERNAL_EDIT_FAMILIES),
    LOCAL_POLISHING_STAGE: (*POLISHING_FAMILIES, *INTERNAL_EDIT_FAMILIES),
    QUALIFIED_EVOLUTION_STAGE: QUALIFIED_EVOLUTION_FAMILIES,
}


@dataclass(frozen=True, slots=True)
class SignalImprovementFamilyCapacity:
    family: str
    available_leaf_count: int
    blocked_leaf_count: int = 0
    attempted_leaf_count: int = 0


@dataclass(frozen=True, slots=True)
class SignalImprovementSource:
    root_task_id: str
    branch_task_id: str
    stage: str
    family_capacities: tuple[SignalImprovementFamilyCapacity, ...]
    parent_remaining_attempts: int
    is_submitted: bool
    self_correlation: float | None = None


@dataclass(frozen=True, slots=True)
class SignalImprovementAllocation:
    root_task_id: str
    parent_task_id: str
    stage: str
    sc_risk_state: str
    target: OptimizationTarget | None
    candidate_family: str


@dataclass(frozen=True, slots=True)
class BacktestSourceAllocation:
    requested_count: int
    minimum_exploration_count: int
    exploration_count: int
    structural_evolution_count: int
    local_polishing_count: int
    signal_improvements: tuple[SignalImprovementAllocation, ...]
    reason: str
    direction_validation_count: int = 0

    @property
    def improvement_count(self) -> int:
        return len(self.signal_improvements)

    @property
    def qualified_evolution_count(self) -> int:
        return sum(
            item.stage == QUALIFIED_EVOLUTION_STAGE
            for item in self.signal_improvements
        )


def allocate_backtest_sources(
    *,
    requested_count: int,
    minimum_exploration_count: int,
    improvement_sources: tuple[SignalImprovementSource, ...],
    settings_evidence: ParentSettingsEvidenceSet,
    optimization_targets: ParentOptimizationTargetSet,
    action_strategies: DefectActionStrategySet,
    account_scope: str,
    rotation_key: str,
    self_correlation_percent: int,
    direction_validation_percent: int,
    direction_validation_count: int = 0,
    optimization_only: bool = False,
) -> BacktestSourceAllocation:
    _positive_integer(requested_count, "source_allocation_requested_count_invalid")
    for percent in (self_correlation_percent, direction_validation_percent):
        if isinstance(percent, bool) or not isinstance(percent, int) or not 0 <= percent <= 100:
            raise ValueError("source_allocation_percent_invalid")
    if optimization_only:
        if minimum_exploration_count != 0 or self_correlation_percent != 0 or direction_validation_percent != 0:
            raise ValueError("source_allocation_optimization_scope_invalid")
    else:
        _positive_integer(minimum_exploration_count, "source_allocation_exploration_count_invalid")
    if minimum_exploration_count > requested_count:
        raise ValueError("source_allocation_exploration_count_invalid")
    for value, error in (
        (account_scope, "source_allocation_account_scope_missing"),
        (rotation_key, "source_allocation_rotation_key_missing"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(error)
    _validate_sources(improvement_sources)
    if not isinstance(settings_evidence, ParentSettingsEvidenceSet):
        raise ValueError("source_allocation_evidence_invalid")
    if not isinstance(optimization_targets, ParentOptimizationTargetSet):
        raise ValueError("source_allocation_optimization_targets_invalid")
    if not isinstance(action_strategies, DefectActionStrategySet):
        raise ValueError("source_allocation_action_strategies_invalid")
    if (
        isinstance(direction_validation_count, bool)
        or not isinstance(direction_validation_count, int)
        or not (
            0 <= direction_validation_count <= direction_validation_limit(
                requested_count, minimum_exploration_count, direction_validation_percent
            )
        )
    ):
        raise ValueError("source_allocation_direction_count_invalid")
    improvement_capacity = (
        requested_count - minimum_exploration_count - direction_validation_count
    )
    if improvement_capacity == 0:
        return _exploration_only(
            requested_count,
            minimum_exploration_count,
            reason=(
                "direction_validation_uses_remaining_capacity"
                if direction_validation_count
                else "exploration_reserve_uses_full_batch"
            ),
            direction_validation_count=direction_validation_count,
            optimization_only=optimization_only,
        )
    if not improvement_sources:
        return _exploration_only(
            requested_count,
            minimum_exploration_count,
            reason="signal_seed_missing",
            direction_validation_count=direction_validation_count,
            optimization_only=optimization_only,
        )

    source_by_branch = {source.branch_task_id: source for source in improvement_sources}
    eligible_branch_ids = set(source_by_branch)
    settings_by_branch: dict[str, ParentSettingsEvidence] = {}
    for record in settings_evidence.records:
        if (
            record.parent_task_id not in eligible_branch_ids
            or record.account_scope != account_scope
        ):
            continue
        if record.parent_task_id in settings_by_branch:
            raise ValueError("source_allocation_parent_settings_duplicated")
        settings_by_branch[record.parent_task_id] = record
    if not settings_by_branch:
        return _exploration_only(
            requested_count,
            minimum_exploration_count,
            reason="signal_settings_evidence_missing",
            direction_validation_count=direction_validation_count,
            optimization_only=optimization_only,
        )

    targets_by_branch: dict[str, ParentOptimizationTargets] = {}
    for record in optimization_targets.records:
        if (
            record.parent_task_id not in eligible_branch_ids
            or record.account_scope != account_scope
        ):
            continue
        if record.parent_task_id in targets_by_branch:
            raise ValueError("source_allocation_optimization_parent_duplicated")
        targets_by_branch[record.parent_task_id] = record

    strategies_by_target = _action_strategies_by_target(
        action_strategies,
        account_scope=account_scope,
    )

    targets_by_root: dict[
        str,
        dict[str, list[OptimizationTarget | None]],
    ] = {}
    for branch_task_id, target_record in targets_by_branch.items():
        settings_record = settings_by_branch.get(branch_task_id)
        if settings_record is None:
            continue
        if settings_record.settings_key != target_record.settings_key:
            raise ValueError("source_allocation_parent_evidence_mismatch")
        source = source_by_branch[branch_task_id]
        if source.stage == QUALIFIED_EVOLUTION_STAGE:
            if target_record.targets:
                raise ValueError("source_allocation_qualified_parent_target_invalid")
            allocation_targets: list[OptimizationTarget | None] = [None]
        else:
            allocation_targets = list(target_record.targets)
            expected_defects = tuple(
                sorted(target.check_name for target in target_record.targets)
            )
            for target in target_record.targets:
                strategy = strategies_by_target.get(
                    (branch_task_id, target.check_name)
                )
                if strategy is not None and (
                    strategy.settings_key != target_record.settings_key
                    or strategy.defect_checks != expected_defects
                ):
                    raise ValueError("source_allocation_action_strategy_mismatch")
        targets_by_root.setdefault(source.root_task_id, {})[
            branch_task_id
        ] = allocation_targets
    ranked_by_root: dict[
        str,
        tuple[tuple[str, tuple[OptimizationTarget | None, ...]], ...],
    ] = {}
    for root_task_id, targets_by_branch_id in targets_by_root.items():
        ranked_branches: list[
            tuple[str, tuple[OptimizationTarget | None, ...]]
        ] = []
        for branch_task_id, targets in targets_by_branch_id.items():
            targets.sort(
                key=lambda target: (
                    len(
                        _candidate_families_for_target(
                            target,
                            source_by_branch[branch_task_id],
                        )
                    ),
                    _target_priority(target),
                    _rotation_identity(
                        rotation_key,
                        root_task_id,
                        branch_task_id,
                        _target_identity(target),
                    ),
                )
            )
            if targets:
                ranked_branches.append((branch_task_id, tuple(targets)))
        ranked_branches.sort(
            key=lambda item: (
                _high_sc_risk(settings_by_branch[item[0]]),
                _target_priority(item[1][0]),
                _rotation_identity(
                    rotation_key,
                    root_task_id,
                    item[0],
                ),
            )
        )
        if ranked_branches:
            ranked_by_root[root_task_id] = tuple(ranked_branches)
    if not ranked_by_root:
        return _exploration_only(
            requested_count,
            minimum_exploration_count,
            reason="signal_optimization_target_missing",
            direction_validation_count=direction_validation_count,
            optimization_only=optimization_only,
        )

    ranked_branch_ids = {
        branch_task_id
        for ranked in ranked_by_root.values()
        for branch_task_id, _targets in ranked
    }
    selected = _round_robin_roots(
        ranked_by_root,
        source_by_branch=source_by_branch,
        strategies_by_target=strategies_by_target,
        capacity=improvement_capacity,
        sc_capacity=min((requested_count * self_correlation_percent + 99) // 100, improvement_capacity),
        rotation_key=rotation_key,
    )
    allocations = tuple(
        SignalImprovementAllocation(
            root_task_id=root_task_id,
            parent_task_id=branch_task_id,
            stage=source_by_branch[branch_task_id].stage,
            sc_risk_state=_sc_risk_state(settings_by_branch[branch_task_id]),
            target=target,
            candidate_family=family,
        )
        for root_task_id, branch_task_id, target, family in selected
    )
    local_polishing_count = sum(
        item.stage == LOCAL_POLISHING_STAGE for item in allocations
    )
    structural_evolution_count = sum(
        item.stage == STRUCTURAL_EVOLUTION_STAGE for item in allocations
    )
    return BacktestSourceAllocation(
        requested_count=len(allocations) if optimization_only else requested_count,
        minimum_exploration_count=minimum_exploration_count,
        exploration_count=(
            0 if optimization_only else requested_count - len(allocations) - direction_validation_count
        ),
        direction_validation_count=direction_validation_count,
        structural_evolution_count=structural_evolution_count,
        local_polishing_count=local_polishing_count,
        signal_improvements=allocations,
        reason=_allocation_reason(
            selected_count=len(allocations),
            improvement_capacity=improvement_capacity,
            target_request_blocked=_target_request_blocked(
                source_by_branch,
                {
                    branch_task_id: targets
                    for ranked in ranked_by_root.values()
                    for branch_task_id, targets in ranked
                    if branch_task_id in ranked_branch_ids
                },
            ),
        ),
    )


def _round_robin_roots(
    ranked_by_root: dict[
        str,
        tuple[tuple[str, tuple[OptimizationTarget | None, ...]], ...],
    ],
    *,
    source_by_branch: dict[str, SignalImprovementSource],
    strategies_by_target: dict[tuple[str, str], DefectActionStrategy],
    capacity: int,
    sc_capacity: int,
    rotation_key: str,
) -> tuple[tuple[str, str, OptimizationTarget | None, str], ...]:
    roots = tuple(
        sorted(
            ranked_by_root,
            key=lambda root_task_id: (
                _root_used_parent_attempts(
                    ranked_by_root[root_task_id],
                    source_by_branch,
                ),
                _rotation_identity(
                    rotation_key,
                    root_task_id,
                ),
            ),
        )
    )
    selected: list[tuple[str, str, OptimizationTarget | None, str]] = []
    qualified_evolution_counts: dict[str, int] = {
        root_task_id: 0 for root_task_id in roots
    }
    branch_positions = {root_task_id: 0 for root_task_id in roots}
    target_positions = {
        (root_task_id, branch_task_id): 0
        for root_task_id, ranked_branches in ranked_by_root.items()
        for branch_task_id, _targets in ranked_branches
    }
    remaining_by_family = {
        (source.branch_task_id, family.family): (
            family.available_leaf_count
            if family.family not in SELF_CORRELATION_REPAIR_FAMILIES
            or family.family in allowed_self_correlation_families(source.self_correlation)
            else 0
        )
        for source in source_by_branch.values()
        for family in source.family_capacities
    }
    remaining_by_parent = {
        source.branch_task_id: source.parent_remaining_attempts
        for source in source_by_branch.values()
    }
    attempted_by_family = {
        (source.branch_task_id, family.family): family.attempted_leaf_count
        for source in source_by_branch.values()
        for family in source.family_capacities
    }
    exploited_contexts: set[
        tuple[str, str, str, tuple[str, ...], str]
    ] = set()
    sc_selected = 0
    while len(selected) < capacity:
        # Spend existing improvement slots on eligible SC repairs first, rotating
        # across roots. This is the research priority, not a learned quality score.
        prefer_sc_repair = sc_selected < sc_capacity and any(
            source_by_branch[branch_task_id].stage == QUALIFIED_EVOLUTION_STAGE
            and remaining_by_parent[branch_task_id] > 0
            and qualified_evolution_counts[root_task_id]
            < MAX_QUALIFIED_EVOLUTION_BACKTESTS_PER_LINEAGE
            and any(remaining_by_family.get((branch_task_id, family), 0) > 0
                    for family in SELF_CORRELATION_REPAIR_FAMILIES)
            for root_task_id, ranked_branches in ranked_by_root.items()
            for branch_task_id, _targets in ranked_branches
        )
        # Ordinary slots prefer usable unsubmitted parents across all lineages.
        # Re-evaluate after each round so exhausted/capped parents cannot block fallback.
        prefer_unsubmitted = not prefer_sc_repair and any(
            not source_by_branch[branch_task_id].is_submitted
            and remaining_by_parent[branch_task_id] > 0
            and (
                source_by_branch[branch_task_id].stage != QUALIFIED_EVOLUTION_STAGE
                or qualified_evolution_counts[root_task_id]
                < MAX_QUALIFIED_EVOLUTION_BACKTESTS_PER_LINEAGE
            )
            and any(
                family not in SELF_CORRELATION_REPAIR_FAMILIES
                and remaining_by_family.get((branch_task_id, family), 0) > 0
                for target in targets
                for family in _candidate_families_for_target(
                    target, source_by_branch[branch_task_id],
                )
            )
            for root_task_id, ranked_branches in ranked_by_root.items()
            for branch_task_id, targets in ranked_branches
        )
        progressed = False
        for root_task_id in roots:
            ranked_branches = ranked_by_root[root_task_id]
            start = branch_positions[root_task_id]
            for offset in range(len(ranked_branches)):
                index = (start + offset) % len(ranked_branches)
                branch_task_id, targets = ranked_branches[index]
                source = source_by_branch[branch_task_id]
                if prefer_unsubmitted and source.is_submitted:
                    continue
                if prefer_sc_repair and (
                    source.stage != QUALIFIED_EVOLUTION_STAGE
                    or not any(remaining_by_family.get((branch_task_id, family), 0) > 0
                               for family in SELF_CORRELATION_REPAIR_FAMILIES)
                ):
                    continue
                if remaining_by_parent[branch_task_id] == 0:
                    continue
                if (
                    source.stage == QUALIFIED_EVOLUTION_STAGE
                    and qualified_evolution_counts[root_task_id]
                    >= MAX_QUALIFIED_EVOLUTION_BACKTESTS_PER_LINEAGE
                ):
                    continue
                target_position = target_positions[(root_task_id, branch_task_id)]
                selected_target: OptimizationTarget | None = None
                selected_family: str | None = None
                selected_target_position = target_position
                selected_exploitation_context: (
                    tuple[str, str, str, tuple[str, ...], str] | None
                ) = None
                for target_offset in range(len(targets)):
                    current_position = target_position + target_offset
                    target = targets[current_position % len(targets)]
                    strategy = (
                        strategies_by_target.get(
                            (branch_task_id, target.check_name)
                        )
                        if target is not None
                        else None
                    )
                    exploitation_context = (
                        _action_strategy_context(root_task_id, strategy)
                        if strategy is not None
                        else None
                    )
                    use_preference = bool(
                        strategy is not None
                        and strategy.mode == ACTION_STRATEGY_EXPLOITATION
                        and exploitation_context not in exploited_contexts
                    )
                    family_order = _candidate_family_order(
                        target,
                        (
                            allowed_self_correlation_families(source.self_correlation)
                            if prefer_sc_repair
                            else tuple(f for f in _candidate_families_for_target(target, source)
                                       if f not in SELF_CORRELATION_REPAIR_FAMILIES)
                        ),
                        attempted_by_family={
                            item.family: attempted_by_family[
                                (branch_task_id, item.family)
                            ]
                            for item in source.family_capacities
                        },
                        rotation_key=rotation_key,
                        parent_task_id=branch_task_id,
                        slot=len(selected),
                        action_strategy=strategy,
                        use_preference=use_preference,
                    )
                    family = next(
                        (
                            item
                            for item in family_order
                            if remaining_by_family.get(
                                (branch_task_id, item),
                                0,
                            )
                            > 0
                        ),
                        None,
                    )
                    if family is None:
                        continue
                    selected_target = target
                    selected_family = family
                    selected_target_position = current_position
                    if (
                        use_preference
                        and strategy is not None
                        and family in strategy.preferred_actions
                    ):
                        selected_exploitation_context = exploitation_context
                    break
                if selected_family is None:
                    continue
                selected.append(
                    (
                        root_task_id,
                        branch_task_id,
                        selected_target,
                        selected_family,
                    )
                )
                if selected_family in SELF_CORRELATION_REPAIR_FAMILIES:
                    sc_selected += 1
                remaining_by_family[(branch_task_id, selected_family)] -= 1
                remaining_by_parent[branch_task_id] -= 1
                attempted_by_family[(branch_task_id, selected_family)] += 1
                if selected_exploitation_context is not None:
                    exploited_contexts.add(selected_exploitation_context)
                if source.stage == QUALIFIED_EVOLUTION_STAGE:
                    qualified_evolution_counts[root_task_id] += 1
                target_positions[(root_task_id, branch_task_id)] = (
                    selected_target_position + 1
                )
                branch_positions[root_task_id] = (index + 1) % len(ranked_branches)
                progressed = True
                break
            if len(selected) == capacity or (prefer_sc_repair and sc_selected >= sc_capacity):
                break
        if not progressed:
            break
    return tuple(selected)


def _allocation_reason(
    *,
    selected_count: int,
    improvement_capacity: int,
    target_request_blocked: bool,
) -> str:
    if selected_count == improvement_capacity:
        return "signal_improvement_attempts"
    if selected_count:
        return "signal_improvement_capacity_limited"
    if target_request_blocked:
        return "signal_improvement_requests_pending"
    return "signal_improvement_neighborhood_exhausted"


def _target_request_blocked(
    source_by_branch: dict[str, SignalImprovementSource],
    targets_by_branch: dict[str, tuple[OptimizationTarget | None, ...]],
) -> bool:
    for branch_task_id, targets in targets_by_branch.items():
        source = source_by_branch[branch_task_id]
        blocked_by_family = {
            item.family: item.blocked_leaf_count for item in source.family_capacities
        }
        for target in targets:
            if any(
                blocked_by_family.get(family, 0) > 0
                for family in _candidate_families_for_target(target, source)
            ):
                return True
    return False


def _candidate_families_for_target(
    target: OptimizationTarget | None,
    source: SignalImprovementSource,
) -> tuple[str, ...]:
    source_families = tuple(item.family for item in source.family_capacities
                            if item.family not in SELF_CORRELATION_REPAIR_FAMILIES
                            or item.family in allowed_self_correlation_families(source.self_correlation))
    if target is None or target.check_name != "LOW_SUB_UNIVERSE_SHARPE":
        return source_families
    allowed = set(
        (*LOW_SUB_UNIVERSE_SHARPE_STRUCTURAL_FAMILIES, *INTERNAL_EDIT_FAMILIES)
    )
    return tuple(family for family in source_families if family in allowed)


def direction_validation_limit(
    requested_count: int, minimum_exploration_count: int, direction_validation_percent: int
) -> int:
    """Initial spend limit, not an estimate of direction quality or a run limit."""
    # Direction checks are part of ordinary mutation capacity, not extra slots.
    return min(
        (requested_count * direction_validation_percent + 99) // 100,
        max(0, requested_count - minimum_exploration_count),
    )


def _exploration_only(
    requested_count: int,
    minimum_exploration_count: int,
    *,
    reason: str,
    direction_validation_count: int = 0,
    optimization_only: bool = False,
) -> BacktestSourceAllocation:
    return BacktestSourceAllocation(
        requested_count=0 if optimization_only else requested_count,
        minimum_exploration_count=minimum_exploration_count,
        exploration_count=0 if optimization_only else requested_count - direction_validation_count,
        direction_validation_count=direction_validation_count,
        structural_evolution_count=0,
        local_polishing_count=0,
        signal_improvements=(),
        reason=reason,
    )


def _validate_sources(sources: tuple[SignalImprovementSource, ...]) -> None:
    if not isinstance(sources, tuple):
        raise ValueError("source_allocation_improvement_sources_invalid")
    for source in sources:
        if (
            not isinstance(source, SignalImprovementSource)
            or not isinstance(source.root_task_id, str)
            or not source.root_task_id.strip()
            or not isinstance(source.branch_task_id, str)
            or not source.branch_task_id.strip()
            or source.stage not in IMPROVEMENT_STAGES
            or not isinstance(source.is_submitted, bool)
            or not isinstance(source.family_capacities, tuple)
            or not source.family_capacities
        ):
            raise ValueError("source_allocation_improvement_sources_invalid")
        if any(
            not isinstance(item, SignalImprovementFamilyCapacity)
            or not isinstance(item.family, str)
            or not item.family.strip()
            or isinstance(item.available_leaf_count, bool)
            or not isinstance(item.available_leaf_count, int)
            or item.available_leaf_count < 0
            or isinstance(item.blocked_leaf_count, bool)
            or not isinstance(item.blocked_leaf_count, int)
            or item.blocked_leaf_count < 0
            or isinstance(item.attempted_leaf_count, bool)
            or not isinstance(item.attempted_leaf_count, int)
            or item.attempted_leaf_count < 0
            for item in source.family_capacities
        ):
            raise ValueError("source_allocation_improvement_sources_invalid")
        remaining_attempts = source.parent_remaining_attempts
        if source.self_correlation is not None and (
            isinstance(source.self_correlation, bool)
            or not isinstance(source.self_correlation, (int, float))
            or not math.isfinite(source.self_correlation)
            or not 0 < source.self_correlation <= 1
        ):
            raise ValueError("source_allocation_self_correlation_invalid")
        if (
            isinstance(remaining_attempts, bool)
            or not isinstance(remaining_attempts, int)
            or remaining_attempts <= 0
            or remaining_attempts > PARENT_ATTEMPT_BUDGET
        ):
            raise ValueError("source_allocation_improvement_sources_invalid")
        families = tuple(item.family for item in source.family_capacities)
        if (
            len(set(families)) != len(families)
            or not set(families) <= set(IMPROVEMENT_FAMILIES_BY_STAGE[source.stage])
        ):
            raise ValueError("source_allocation_improvement_sources_invalid")
    branch_ids = tuple(source.branch_task_id for source in sources)
    if len(set(branch_ids)) != len(branch_ids):
        raise ValueError("source_allocation_signal_branch_duplicated")


def allowed_self_correlation_families(correlation: float | None) -> tuple[str, ...]:
    """Research allocation boundaries shared with candidate generation."""
    if (isinstance(correlation, bool) or not isinstance(correlation, (int, float))
            or not math.isfinite(correlation) or not 0 < correlation <= 1):
        return ()
    if correlation < 0.75:
        return SELF_CORRELATION_LIGHT_FAMILIES
    if correlation < 0.85:
        return SELF_CORRELATION_INTERNAL_FAMILIES
    return ()


def _high_sc_risk(record: ParentSettingsEvidence) -> bool:
    return bool(
        record.sc_risk_available
        and record.sc_smoothed_risk is not None
        and record.sc_smoothed_risk > 0.5
    )


def _sc_risk_state(record: ParentSettingsEvidence) -> str:
    if not record.sc_risk_available:
        return "not_established"
    return "high" if _high_sc_risk(record) else "acceptable"


def _target_priority(target: OptimizationTarget | None) -> tuple[int, float]:
    if target is None:
        return 2, 0.0
    if target.normalized_gap is not None:
        return 0, -target.normalized_gap
    return 1, 0.0


def _target_identity(target: OptimizationTarget | None) -> str:
    return target.check_name if target is not None else QUALIFIED_EVOLUTION_STAGE


def _candidate_family_order(
    target: OptimizationTarget | None,
    transformations: tuple[str, ...],
    *,
    attempted_by_family: dict[str, int],
    rotation_key: str,
    parent_task_id: str,
    slot: int,
    action_strategy: DefectActionStrategy | None,
    use_preference: bool,
) -> tuple[str, ...]:
    comparison_counts = (
        _action_comparison_counts(target) if target is not None else {}
    )
    preferred = (
        set(action_strategy.preferred_actions)
        if action_strategy is not None and use_preference
        else set()
    )
    deprioritized = (
        set(action_strategy.deprioritized_actions)
        if action_strategy is not None
        else set()
    )
    return tuple(
        sorted(
            transformations,
            key=lambda action: (
                (
                    0
                    if action in preferred
                    else 2 if action in deprioritized else 1
                ),
                comparison_counts.get(action, 0),
                attempted_by_family.get(action, 0),
                _rotation_identity(
                    rotation_key,
                    parent_task_id,
                    _target_identity(target),
                    str(slot),
                    action,
                ),
            ),
        )
    )


def _action_strategies_by_target(
    action_strategies: DefectActionStrategySet,
    *,
    account_scope: str,
) -> dict[tuple[str, str], DefectActionStrategy]:
    built: dict[tuple[str, str], DefectActionStrategy] = {}
    for strategy in action_strategies.records:
        if not isinstance(strategy, DefectActionStrategy):
            raise ValueError("source_allocation_action_strategies_invalid")
        if strategy.account_scope != account_scope:
            continue
        identity = (strategy.parent_task_id, strategy.target_check_name)
        if identity in built:
            raise ValueError("source_allocation_action_strategy_duplicated")
        action_names = tuple(item.action for item in strategy.actions)
        if (
            not strategy.parent_task_id.strip()
            or not strategy.settings_key.strip()
            or not strategy.defect_checks
            or strategy.target_check_name not in strategy.defect_checks
            or len(set(action_names)) != len(action_names)
            or not set(strategy.preferred_actions) <= set(action_names)
            or not set(strategy.deprioritized_actions) <= set(action_names)
            or set(strategy.preferred_actions) & set(strategy.deprioritized_actions)
        ):
            raise ValueError("source_allocation_action_strategies_invalid")
        built[identity] = strategy
    return built


def _action_strategy_context(
    root_task_id: str,
    strategy: DefectActionStrategy,
) -> tuple[str, str, str, tuple[str, ...], str]:
    return (
        root_task_id,
        strategy.account_scope,
        strategy.settings_key,
        strategy.defect_checks,
        strategy.target_check_name,
    )


def _root_used_parent_attempts(
    ranked_branches: tuple[
        tuple[str, tuple[OptimizationTarget | None, ...]],
        ...,
    ],
    source_by_branch: dict[str, SignalImprovementSource],
) -> int:
    counts: list[int] = []
    for branch_task_id, targets in ranked_branches:
        source = source_by_branch[branch_task_id]
        usable_families = {
            family
            for target in targets
            for family in _candidate_families_for_target(target, source)
        }
        if any(
            item.available_leaf_count > 0 and item.family in usable_families
            for item in source.family_capacities
        ):
            # All stages use the same authoritative lifetime parent budget.
            # A fresh but unusable sibling must not promote an older branch.
            counts.append(PARENT_ATTEMPT_BUDGET - source.parent_remaining_attempts)
    return min(counts, default=PARENT_ATTEMPT_BUDGET)


def _action_comparison_counts(target: OptimizationTarget) -> dict[str, int]:
    counts: dict[str, int] = {}
    for evidence in target.actions:
        if evidence.action in counts:
            raise ValueError("source_allocation_target_action_duplicated")
        if (
            isinstance(evidence.comparable_count, bool)
            or not isinstance(evidence.comparable_count, int)
            or evidence.comparable_count < 0
        ):
            raise ValueError("source_allocation_target_action_evidence_invalid")
        counts[evidence.action] = evidence.comparable_count
    return counts


def _rotation_identity(rotation_key: str, *parts: str) -> str:
    payload = "|".join((rotation_key, *parts))
    return sha256(payload.encode("utf-8")).hexdigest()


def _positive_integer(value: object, error: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(error)
