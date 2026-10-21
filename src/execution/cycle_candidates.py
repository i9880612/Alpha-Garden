from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from itertools import chain
from hashlib import sha256

from execution.generation import ExclusionCount, generate_exploration_batch
from execution.progress import phase
from generation.candidate import (
    CandidateChange,
    FormulaCandidate,
    exploration_candidate,
    mutation_candidate,
)
from generation.catalog import GenerationCatalog
from generation.ideas import (
    STRATEGY_LABEL_PREFIX,
    IdeaVariant,
    IdeaVariantPool,
    plan_idea_batch,
)
from generation.parser import FormulaSyntaxError, parse_formula
from generation.combo_enhancement import ComboEnhancementLeaf
from generation.decay_ladder import iter_decay_ladder_leaves
from generation.polishing import POLISHING_FAMILIES, iter_window_mutation_leaves
from generation.internal_edits import (
    iter_internal_edit_candidates,
    INTERNAL_EDIT_FAMILIES,
)
from generation.self_correlation import (
    SELF_CORRELATION_REPAIR_FAMILIES,
    SELF_CORRELATION_INTERNAL_FAMILIES,
    SELF_CORRELATION_RESEARCH_FAMILIES,
    iter_self_correlation_leaves,
    shared_field_replacements,
    self_correlation_research_fields,
)
from generation.transformations import (
    COMPLEMENTARY_SIGNAL_REFRAME,
    TEMPORAL_PERSISTENCE_REFRAME,
    TemporalReframeSpec,
    iter_transformation_leaves,
)
from learning.quality_proximity import (
    LOCAL_POLISHING_STAGE,
    ParentImprovementStageSet,
    QUALIFIED_EVOLUTION_STAGE,
)
from learning.diagnosis import (
    HIGH_TURNOVER,
    SHORT_SMOOTHING_WINDOW,
    ParentDiagnosis,
    diagnose_parent,
)
from learning.self_correlation import SelfCorrelationReference
from persistence.backtests import (
    backtest_was_cancelled_before_submission,
    BACKTEST_ACTIVE_STATUSES,
    BACKTEST_TERMINAL_STATUSES,
    BacktestSnapshot,
    BacktestTaskRecord,
    BacktestMutationRecord,
)
from persistence.run_diagnostics import (
    CandidatePlanningDiagnostic,
    CandidatePlanningExclusionCount,
)
from selection.allocations import BacktestSourceAllocation, allowed_self_correlation_families
from learning.quality import QUALITY_IMPROVEMENT
from selection.candidates import select_candidates, explore_internal_fields
from selection.formulas import (
    formula_polishing_candidate,
    formula_selection_candidate,
)
from selection.settings import BacktestSettingsPolicy, choose_backtest_settings
from worldquant.backtests import BacktestSettings


class OptimizationCandidatesExhausted(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SupportedFormulaIdentities:
    formulas: tuple[str, ...]
    formula_fingerprints: frozenset[str]


def supported_formula_identities(formulas: tuple[str, ...]) -> SupportedFormulaIdentities:
    supported_formulas = []
    fingerprints = set()
    for formula in formulas:
        try:
            candidate = exploration_candidate(parse_formula(formula).expression)
        except (FormulaSyntaxError, ValueError):
            continue
        supported_formulas.append(candidate.formula)
        fingerprints.add(candidate.fingerprint)
    return SupportedFormulaIdentities(tuple(supported_formulas), frozenset(fingerprints))


class CycleCandidatePlanningStopped(ValueError):
    stop_reason: str

    def __init__(
        self,
        message: str,
        *,
        diagnostic: CandidatePlanningDiagnostic,
    ) -> None:
        diagnostic.canonical_json(stop_reason=self.stop_reason)
        super().__init__(message)
        self.diagnostic = diagnostic


class ExplorationAttemptBudgetExhausted(CycleCandidatePlanningStopped):
    stop_reason = "generation_attempt_budget_exhausted"


class CandidateSelectionShortfall(CycleCandidatePlanningStopped):
    stop_reason = "selection_rejection_shortfall"


@dataclass(frozen=True, slots=True)
class CycleCandidate:
    candidate: FormulaCandidate
    settings: BacktestSettings
    quality_root_task_id: str | None = None
    idea: IdeaVariant | None = None


@dataclass(frozen=True, slots=True)
class CycleCandidatePlan:
    candidates: tuple[CycleCandidate, ...]
    exploration_backtest_count: int
    structural_evolution_backtest_count: int
    local_polishing_backtest_count: int
    direction_validation_backtest_count: int = 0

    @property
    def qualified_evolution_backtest_count(self) -> int:
        return len(self.candidates) - (
            self.exploration_backtest_count
            + self.structural_evolution_backtest_count
            + self.local_polishing_backtest_count
            + self.direction_validation_backtest_count
        )


@dataclass(frozen=True, slots=True)
class ImprovementPoolCandidate:
    candidate: FormulaCandidate
    leaf_id: str
    family: str
    # 0 = suits the parent's diagnosis better than its family siblings.
    preference: int = 1


@dataclass(frozen=True, slots=True)
class ImprovementFamilyUsage:
    family: str
    blocked_request_count: int
    consumed_request_count: int
    attempted_request_count: int
    unsubmitted_request_count: int = 0


@dataclass(frozen=True, slots=True)
class ImprovementCandidatePool:
    parent_task_id: str
    stage: str
    settings: BacktestSettings
    candidates: tuple[ImprovementPoolCandidate, ...]
    family_usage: tuple[ImprovementFamilyUsage, ...]
    diagnosis: ParentDiagnosis | None = None

    @property
    def prescribed_families(self) -> tuple[str, ...]:
        return self.diagnosis.families if self.diagnosis is not None else ()

    @property
    def blocked_request_count(self) -> int:
        return sum(item.blocked_request_count for item in self.family_usage)

    @property
    def consumed_request_count(self) -> int:
        return sum(item.consumed_request_count for item in self.family_usage)

    @property
    def attempted_request_count(self) -> int:
        return sum(item.attempted_request_count for item in self.family_usage)


def build_improvement_candidate_pools(
    catalog: GenerationCatalog,
    policy: BacktestSettingsPolicy,
    *,
    eligible_parent_task_ids: tuple[str, ...],
    improvement_stages: ParentImprovementStageSet,
    completed_by_task_id: dict[str, BacktestSnapshot],
    field_candidates: tuple[str, ...],
    group_candidates: tuple[str, ...],
    reserved_tasks: tuple[BacktestTaskRecord, ...],
    mutations: tuple[BacktestMutationRecord, ...] = (),
    excluded_formula_fingerprints: frozenset[str] = frozenset(),
    self_correlation_references: tuple[SelfCorrelationReference, ...] = (),
    internal_field_candidates: tuple[str, ...] | None = None,
    rotation_key: str = "",
    field_relatives: Callable[[str], tuple[str, ...]] | None = None,
    prescriptions: Mapping[str, tuple[str, ...]] | None = None,
    enhancement_leaves: Mapping[str, tuple[ComboEnhancementLeaf, ...]] | None = None,
) -> tuple[ImprovementCandidatePool, ...]:
    _validate_pool_request(
        parent_task_ids=eligible_parent_task_ids,
        improvement_stages=improvement_stages,
        completed_by_task_id=completed_by_task_id,
        reserved_tasks=reserved_tasks,
    )
    _validate_formula_fingerprints(excluded_formula_fingerprints)
    reserved_by_identity = {
        _request_identity(
            account_scope=task.account_scope,
            formula_fingerprint=task.formula_fingerprint,
            settings_json=task.settings_json,
        ): task
        for task in reserved_tasks if not backtest_was_cancelled_before_submission(task)
    }
    claimed_identities: set[tuple[str, str, str]] = set()
    stages_by_parent = {
        record.parent_task_id: record for record in improvement_stages.records
    }
    references_by_parent = {
        item.parent_task_id: item for item in self_correlation_references
    }
    if len(references_by_parent) != len(self_correlation_references):
        raise ValueError("self_correlation_parent_reference_duplicate")
    tasks_by_id = {task.task_id: task for task in reserved_tasks}
    pools: list[ImprovementCandidatePool] = []
    for parent_task_id in sorted(eligible_parent_task_ids):
        stage = stages_by_parent[parent_task_id].stage
        parent = completed_by_task_id.get(parent_task_id)
        if parent is None or parent.result is None:
            raise ValueError("automated_cycle_parent_result_missing")
        settings = _snapshot_settings(parent)
        if not policy.allows(settings):
            pools.append(
                ImprovementCandidatePool(
                    parent_task_id=parent_task_id,
                    stage=stage,
                    settings=settings,
                    candidates=(),
                    family_usage=(),
                )
            )
            continue
        try:
            parsed = parse_formula(parent.task.formula)
        except FormulaSyntaxError as exc:
            raise ValueError("automated_cycle_parent_formula_invalid") from exc
        diagnosis = diagnose_parent(
            sharpe=parent.result.sharpe, fitness=parent.result.fitness,
            turnover=parent.result.turnover,
            failed_checks=frozenset(check.name for check in parent.result.checks
                                    if check.status.strip().upper() == "FAIL"),
            prescriptions=prescriptions,
        )

        available: list[ImprovementPoolCandidate] = []
        blocked_by_family: Counter[str] = Counter()
        consumed_by_family: Counter[str] = Counter()
        attempted_by_family: Counter[str] = Counter()
        if stage == LOCAL_POLISHING_STAGE:
            leaves = iter_window_mutation_leaves(parsed.expression, catalog)
        elif stage == QUALIFIED_EVOLUTION_STAGE:
            leaves = chain(
                iter_window_mutation_leaves(parsed.expression, catalog),
                iter_transformation_leaves(
                    parsed.expression,
                    catalog,
                    field_candidates=field_candidates,
                    group_candidates=group_candidates,
                ),
            )
        else:
            leaves = iter_transformation_leaves(
                parsed.expression,
                catalog,
                field_candidates=field_candidates,
                group_candidates=group_candidates,
            )
        if stage == QUALIFIED_EVOLUTION_STAGE and enhancement_leaves:
            # Proven independent companions come first, best expected gain first.
            leaves = chain(enhancement_leaves.get(parent_task_id, ()), leaves)
        if diagnosis is not None and diagnosis.problem == HIGH_TURNOVER:
            # Only a parent that trades too much climbs the decay ladder.
            leaves = chain(iter_decay_ladder_leaves(parsed.expression, catalog), leaves)
        sc_reference = references_by_parent.get(parent_task_id)
        sc_families = allowed_self_correlation_families(
            sc_reference.correlation if sc_reference is not None else None,
        )
        if stage == QUALIFIED_EVOLUTION_STAGE and sc_families:
            assert sc_reference is not None
            try:
                reference = parse_formula(sc_reference.formula).expression
            except FormulaSyntaxError:
                reference = None
            if reference is not None:
                leaves = chain(
                    iter_self_correlation_leaves(
                        parsed.expression, reference, catalog,
                        families=SELF_CORRELATION_RESEARCH_FAMILIES,
                        neutralization=settings.neutralization,
                        field_candidates=explore_internal_fields(
                            self_correlation_research_fields(catalog,
                                field_candidates if internal_field_candidates is None else internal_field_candidates),
                            rotation_key=f"{parent_task_id}|sc-research",
                        ),
                    ),
                    iter_self_correlation_leaves(
                        parsed.expression, reference, catalog,
                        families=tuple(f for f in sc_families if f in SELF_CORRELATION_REPAIR_FAMILIES
                                       and f not in SELF_CORRELATION_RESEARCH_FAMILIES),
                        neutralization=settings.neutralization,
                        cross_dataset=sc_reference.correlation >= 0.85,
                        field_candidates=explore_internal_fields(
                            shared_field_replacements(
                                parsed.expression, reference, catalog,
                                field_candidates if internal_field_candidates is None
                                else internal_field_candidates,
                                cross_dataset=sc_reference.correlation >= 0.85,
                            ),
                            rotation_key=f"{rotation_key}|{parent_task_id}|sc",
                        ),
                    ),
                    leaves,
                )
        leaves = chain(
            leaves,
            iter_internal_edit_candidates(
                parsed.expression,
                catalog,
                parent_task_id=parent_task_id,
                field_candidates=explore_internal_fields(
                    field_candidates
                    if internal_field_candidates is None
                    else internal_field_candidates,
                    rotation_key=f"{rotation_key}|{parent_task_id}",
                ),
                field_relatives=field_relatives,
            ),
        )
        for leaf in leaves:
            if (sc_reference is not None and sc_reference.correlation is not None
                    and not isinstance(leaf, FormulaCandidate) and leaf.family == COMPLEMENTARY_SIGNAL_REFRAME):
                # Mixing an untested raw field is not evidence for combining two
                # independently qualified, low-return-correlation signals.
                continue
            candidate = (
                leaf
                if isinstance(leaf, FormulaCandidate)
                else mutation_candidate(
                    leaf.expression,
                    parent_task_id=parent_task_id,
                    parent_formula_fingerprint=parsed.fingerprint,
                    change=CandidateChange(
                        action=leaf.family,
                        location=leaf.change.location,
                        before=leaf.change.before,
                        after=leaf.change.after,
                        conflict_reference_alpha_id=(
                            sc_reference.platform_alpha_id
                            if leaf.family in SELF_CORRELATION_REPAIR_FAMILIES and sc_reference is not None else None
                        ),
                        conflict_reference_formula=(
                            sc_reference.formula
                            if leaf.family in SELF_CORRELATION_REPAIR_FAMILIES and sc_reference is not None
                            and sc_reference.platform_alpha_id is not None else None
                        ),
                    ),
                )
            )
            assert candidate.change is not None
            if sc_reference is not None and sc_reference.correlation is not None and sc_reference.platform_alpha_id:
                candidate = replace(candidate, change=replace(candidate.change,
                    conflict_reference_alpha_id=sc_reference.platform_alpha_id,
                    conflict_reference_formula=sc_reference.formula))
            family = candidate.change.action
            leaf_id = (
                f"{family}:{candidate.fingerprint}"
                if isinstance(leaf, FormulaCandidate)
                else leaf.leaf_id
            )
            assessed = (
                formula_polishing_candidate(
                    candidate.expression,
                    parsed.expression,
                    catalog,
                    settings,
                    parent_passed_checks=_passed_checks(parent),
                )
                if family in POLISHING_FAMILIES
                else formula_selection_candidate(
                    candidate.expression,
                    catalog,
                    settings,
                )
            )
            if assessed.rejection_reasons:
                continue
            if candidate.fingerprint in excluded_formula_fingerprints:
                consumed_by_family[family] += 1
                continue
            identity = _request_identity(
                account_scope=parent.task.account_scope,
                formula_fingerprint=candidate.fingerprint,
                settings_json=parent.task.settings_json,
            )
            reserved_task = reserved_by_identity.get(identity)
            if (
                reserved_task is not None
                and reserved_task.status in BACKTEST_ACTIVE_STATUSES
            ):
                blocked_by_family[family] += 1
                if reserved_task.submission_started_at is not None:
                    attempted_by_family[family] += 1
                continue
            if (
                reserved_task is not None
                and reserved_task.status in BACKTEST_TERMINAL_STATUSES
            ):
                consumed_by_family[family] += 1
                if reserved_task.submission_started_at is not None:
                    attempted_by_family[family] += 1
                continue
            if reserved_task is not None:
                raise ValueError("automated_cycle_reserved_task_status_invalid")
            if identity in claimed_identities:
                continue
            claimed_identities.add(identity)
            available.append(
                ImprovementPoolCandidate(
                    candidate=candidate,
                    leaf_id=leaf_id,
                    family=family,
                    preference=_leaf_preference(leaf, diagnosis),
                )
            )
        # Trial history comes from saved task/mutation facts, not re-enumeration
        # of today's catalog. A missing field or a new run cannot reset it.
        unsubmitted_by_family: Counter[str] = Counter()
        for family in SELF_CORRELATION_RESEARCH_FAMILIES:
            attempted_by_family.pop(family, None)
        for mutation in mutations:
            task = tasks_by_id.get(mutation.child_task_id)
            if (mutation.parent_task_id != parent_task_id
                    or mutation.action not in SELF_CORRELATION_RESEARCH_FAMILIES
                    or task is None or task.account_scope != parent.task.account_scope
                    or task.settings_json != parent.task.settings_json
                    or backtest_was_cancelled_before_submission(task)):
                continue
            if task.submission_started_at is not None:
                attempted_by_family[mutation.action] += 1
            elif task.status in BACKTEST_ACTIVE_STATUSES:
                unsubmitted_by_family[mutation.action] += 1
        pools.append(
            ImprovementCandidatePool(
                parent_task_id=parent_task_id,
                stage=stage,
                settings=settings,
                diagnosis=diagnosis,
                candidates=tuple(
                    sorted(
                        available,
                        key=lambda item: (
                            item.preference,
                            sha256(
                                f"{rotation_key}|{parent_task_id}|{item.leaf_id}".encode()
                            ).digest()
                            if item.family in (*INTERNAL_EDIT_FAMILIES, *SELF_CORRELATION_INTERNAL_FAMILIES)
                            else b""
                        ),
                    )
                ),
                family_usage=tuple(
                    ImprovementFamilyUsage(
                        family=family,
                        blocked_request_count=blocked_by_family[family],
                        consumed_request_count=consumed_by_family[family],
                        attempted_request_count=attempted_by_family[family],
                        unsubmitted_request_count=unsubmitted_by_family[family],
                    )
                    for family in sorted(
                        set(blocked_by_family)
                        | set(consumed_by_family)
                        | set(attempted_by_family)
                        | set(unsubmitted_by_family)
                    )
                ),
            )
        )
    return tuple(pools)


def build_cycle_candidates(
    catalog: GenerationCatalog,
    policy: BacktestSettingsPolicy,
    *,
    cycle_number: int,
    requested_generation_count: int,
    requested_backtest_count: int,
    exploration_seed_attempt_multiplier: int,
    source_allocation: BacktestSourceAllocation,
    improvement_pools: tuple[ImprovementCandidatePool, ...],
    exploration_field_candidates: tuple[str, ...],
    group_candidates: tuple[str, ...],
    existing_formulas: tuple[str, ...],
    seed_start: int,
    direction_candidates: tuple[CycleCandidate, ...] = (),
    idea_pool: IdeaVariantPool | None = None,
    idea_percent: int = 0,
    combo_variants: tuple[IdeaVariant, ...] = (),
) -> CycleCandidatePlan:
    if (
        isinstance(idea_percent, bool) or not isinstance(idea_percent, int)
        or not 0 <= idea_percent <= 100
    ):
        raise ValueError("automated_cycle_idea_percent_invalid")
    if (
        isinstance(cycle_number, bool)
        or not isinstance(cycle_number, int)
        or cycle_number <= 0
    ):
        raise ValueError("automated_cycle_number_invalid")
    if (
        isinstance(exploration_seed_attempt_multiplier, bool)
        or not isinstance(exploration_seed_attempt_multiplier, int)
        or exploration_seed_attempt_multiplier <= 0
    ):
        raise ValueError("automated_cycle_exploration_attempt_multiplier_invalid")
    if (
        source_allocation.requested_count != requested_backtest_count
        or source_allocation.exploration_count
        + source_allocation.improvement_count
        + source_allocation.direction_validation_count
        != requested_backtest_count
        or len(source_allocation.signal_improvements)
        != source_allocation.improvement_count
        or len(direction_candidates) != source_allocation.direction_validation_count
    ):
        raise ValueError("automated_cycle_source_allocation_invalid")
    pools_by_parent = {pool.parent_task_id: pool for pool in improvement_pools}
    if len(pools_by_parent) != len(improvement_pools):
        raise ValueError("automated_cycle_improvement_pool_duplicated")

    used_leaf_ids: dict[str, set[str]] = {}
    improvement_candidates: list[CycleCandidate] = list(direction_candidates)
    for allocation in source_allocation.signal_improvements:
        pool = pools_by_parent.get(allocation.parent_task_id)
        if pool is None:
            raise ValueError("automated_cycle_improvement_pool_missing")
        if pool.stage != allocation.stage:
            raise ValueError("automated_cycle_improvement_stage_mismatch")
        used = used_leaf_ids.setdefault(allocation.parent_task_id, set())
        selected = next(
            (
                item
                for item in pool.candidates
                if item.family == allocation.candidate_family
                and item.leaf_id not in used
            ),
            None,
        )
        if selected is None:
            raise ValueError("automated_cycle_improvement_pool_inconsistent")
        used.add(selected.leaf_id)
        improvement_candidates.append(
            CycleCandidate(
                candidate=selected.candidate,
                settings=pool.settings,
                quality_root_task_id=(allocation.root_task_id if
                    (allocation.target is not None and allocation.target.name == QUALITY_IMPROVEMENT)
                    or (allocation.target is None and pool.stage == QUALIFIED_EVOLUTION_STAGE) else None),
            )
        )

    exploration_backtest_count = source_allocation.exploration_count
    initial_candidates = tuple(improvement_candidates)
    random_backtest_count = exploration_backtest_count
    idea_target = (random_backtest_count * idea_percent + 50) // 100 if idea_pool is not None else 0
    if idea_target:
        ideas = _idea_candidates(
            catalog, policy, idea_pool, target_count=idea_target, seed=seed_start,
            existing_formulas=existing_formulas + tuple(item.candidate.formula for item in initial_candidates),
            combo_variants=combo_variants,
        )
        missing = idea_target - len(ideas)
        combo_count = sum(1 for item in ideas if item.idea is not None and item.idea.family == "combo")
        phase(cycle_number, 1, f"想法模板选用 {len(ideas)}/{idea_target} 条（其中组合 {combo_count} 条）"
              + (f"；模板库接近饱和，缺口 {missing} 条由随机探索补齐" if missing else ""))
        initial_candidates += ideas
        random_backtest_count -= len(ideas)
    if random_backtest_count == 0:
        phase(cycle_number, 1, f"候选已选定，共 {len(initial_candidates)} 条")
        phase(cycle_number, 2, "公式筛选与回测计划冻结中...")
        return CycleCandidatePlan(
            candidates=initial_candidates, exploration_backtest_count=exploration_backtest_count,
            structural_evolution_backtest_count=source_allocation.structural_evolution_count,
            local_polishing_backtest_count=source_allocation.local_polishing_count,
            direction_validation_backtest_count=len(direction_candidates),
        )
    exploration_generation_count = requested_generation_count - len(initial_candidates)
    if exploration_generation_count < random_backtest_count:
        raise ValueError("automated_cycle_generation_capacity_invalid")
    seed_count = exploration_generation_count * exploration_seed_attempt_multiplier
    exploration_batch = generate_exploration_batch(
        catalog,
        target_count=exploration_generation_count,
        seeds=tuple(range(seed_start, seed_start + seed_count)),
        field_candidates=exploration_field_candidates,
        group_candidates=group_candidates,
        existing_formulas=(
            existing_formulas
            + tuple(item.candidate.formula for item in initial_candidates)
        ),
    )
    if exploration_batch.shortfall is not None:
        raise ExplorationAttemptBudgetExhausted(
            "automated_cycle_generation_shortfall:"
            f"{exploration_batch.shortfall.missing_count}",
            diagnostic=CandidatePlanningDiagnostic(
                cycle_number=cycle_number,
                exploration_generation_target_count=(exploration_generation_count),
                exploration_backtest_target_count=random_backtest_count,
                seed_attempt_limit=seed_count,
                attempted_seed_count=exploration_batch.attempted_seed_count,
                generated_candidate_count=len(exploration_batch.candidates),
                selected_candidate_count=None,
                generation_exclusions=_planning_exclusion_counts(
                    exploration_batch.exclusions
                ),
                selection_rejections=(),
            ),
        )

    phase(cycle_number, 1, f"公式生成完成，共 {len(exploration_batch.candidates) + len(initial_candidates)} 条候选")
    phase(cycle_number, 2, "公式筛选与回测计划冻结中...")
    exploration_by_id: dict[str, CycleCandidate] = {}
    selection_inputs = []
    for item in exploration_batch.candidates:
        settings = choose_backtest_settings(
            item.candidate.expression,
            catalog,
            policy,
        )
        selection_candidate = formula_selection_candidate(
            item.candidate.expression,
            catalog,
            settings,
        )
        exploration_by_id[selection_candidate.candidate_id] = CycleCandidate(
            candidate=item.candidate,
            settings=settings,
        )
        selection_inputs.append(selection_candidate)
    selection = select_candidates(
        tuple(selection_inputs),
        requested_count=random_backtest_count,
    )
    if selection.missing_count:
        rejection_counts = Counter(
            reason for rejected in selection.rejected for reason in rejected.reasons
        )
        raise CandidateSelectionShortfall(
            f"automated_cycle_selection_shortfall:{selection.missing_count}",
            diagnostic=CandidatePlanningDiagnostic(
                cycle_number=cycle_number,
                exploration_generation_target_count=(exploration_generation_count),
                exploration_backtest_target_count=random_backtest_count,
                seed_attempt_limit=seed_count,
                attempted_seed_count=exploration_batch.attempted_seed_count,
                generated_candidate_count=len(exploration_batch.candidates),
                selected_candidate_count=len(selection.selected),
                generation_exclusions=_planning_exclusion_counts(
                    exploration_batch.exclusions
                ),
                selection_rejections=tuple(
                    CandidatePlanningExclusionCount(code=code, count=count)
                    for code, count in sorted(rejection_counts.items())
                ),
            ),
        )
    exploration_candidates = tuple(
        exploration_by_id[item.candidate_id] for item in selection.selected
    )
    selected_candidates = initial_candidates + exploration_candidates
    if len(selected_candidates) != requested_backtest_count:
        raise ValueError("automated_cycle_source_allocation_incomplete")
    return CycleCandidatePlan(
        candidates=selected_candidates,
        exploration_backtest_count=exploration_backtest_count,
        structural_evolution_backtest_count=(
            source_allocation.structural_evolution_count
        ),
        local_polishing_backtest_count=source_allocation.local_polishing_count,
        direction_validation_backtest_count=len(direction_candidates),
    )


# Share of the template slots offered to combos of already-tested signals; singles fill the rest.
COMBO_SHARE_PERCENT = 30


def _idea_candidates(
    catalog: GenerationCatalog,
    policy: BacktestSettingsPolicy,
    pool: IdeaVariantPool,
    *,
    target_count: int,
    seed: int,
    existing_formulas: tuple[str, ...],
    combo_variants: tuple[IdeaVariant, ...] = (),
) -> tuple[CycleCandidate, ...]:
    """Draw untested combos first, then template variants; rejected ones are skipped, never retried."""
    excluded = set(supported_formula_identities(existing_formulas).formula_fingerprints)
    chosen: list[CycleCandidate] = []

    def take(variant: IdeaVariant, candidate: FormulaCandidate) -> None:
        excluded.add(variant.fingerprint)
        settings = choose_backtest_settings(candidate.expression, catalog, policy)
        selection = formula_selection_candidate(candidate.expression, catalog, settings)
        if not selection.rejection_reasons:
            chosen.append(CycleCandidate(candidate, settings, idea=variant))

    combo_target = target_count * COMBO_SHARE_PERCENT // 100
    for variant in combo_variants:
        if len(chosen) >= combo_target:
            break
        if variant.fingerprint not in excluded:
            take(variant, exploration_candidate(
                variant.expression, strategy_label=STRATEGY_LABEL_PREFIX + variant.template_id))
    attempt = 0
    while len(chosen) < target_count:
        batch = plan_idea_batch(
            pool, target_count=target_count - len(chosen), seed=seed + attempt,
            excluded_fingerprints=frozenset(excluded),
        )
        if not batch.candidates:
            break
        for item in batch.candidates:
            take(item.variant, item.candidate)
        attempt += 1
    return tuple(chosen)


def _planning_exclusion_counts(
    exclusions: tuple[ExclusionCount, ...],
) -> tuple[CandidatePlanningExclusionCount, ...]:
    return tuple(
        CandidatePlanningExclusionCount(code=item.code, count=item.count)
        for item in exclusions
    )


def _validate_formula_fingerprints(values: frozenset[str]) -> None:
    if not isinstance(values, frozenset) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError("excluded_formula_fingerprints_invalid")


def _validate_pool_request(
    *,
    parent_task_ids: tuple[str, ...],
    improvement_stages: ParentImprovementStageSet,
    completed_by_task_id: dict[str, BacktestSnapshot],
    reserved_tasks: tuple[BacktestTaskRecord, ...],
) -> None:
    if (
        not isinstance(parent_task_ids, tuple)
        or any(
            not isinstance(task_id, str) or not task_id.strip()
            for task_id in parent_task_ids
        )
        or len(set(parent_task_ids)) != len(parent_task_ids)
    ):
        raise ValueError("automated_cycle_parent_task_ids_invalid")
    if not isinstance(improvement_stages, ParentImprovementStageSet):
        raise ValueError("automated_cycle_improvement_stages_invalid")
    stage_parent_ids = tuple(
        record.parent_task_id for record in improvement_stages.records
    )
    if len(set(stage_parent_ids)) != len(stage_parent_ids) or set(
        stage_parent_ids
    ) != set(parent_task_ids):
        raise ValueError("automated_cycle_improvement_stages_invalid")
    if not isinstance(completed_by_task_id, dict) or any(
        not isinstance(task_id, str)
        or not isinstance(snapshot, BacktestSnapshot)
        or snapshot.task.task_id != task_id
        for task_id, snapshot in completed_by_task_id.items()
    ):
        raise ValueError("automated_cycle_parent_results_invalid")
    if not isinstance(reserved_tasks, tuple) or any(
        not isinstance(task, BacktestTaskRecord) for task in reserved_tasks
    ):
        raise ValueError("automated_cycle_reserved_tasks_invalid")
    identities = tuple(
        _request_identity(
            account_scope=task.account_scope,
            formula_fingerprint=task.formula_fingerprint,
            settings_json=task.settings_json,
        )
        for task in reserved_tasks if not backtest_was_cancelled_before_submission(task)
    )
    if len(set(identities)) != len(identities):
        raise ValueError("automated_cycle_reserved_task_identity_duplicated")


def _request_identity(
    *,
    account_scope: str,
    formula_fingerprint: str,
    settings_json: str,
) -> tuple[str, str, str]:
    return account_scope, formula_fingerprint, settings_json


def _snapshot_settings(snapshot: BacktestSnapshot) -> BacktestSettings:
    try:
        payload = json.loads(snapshot.task.settings_json)
    except json.JSONDecodeError as exc:
        raise ValueError("automated_cycle_parent_settings_invalid") from exc
    try:
        return BacktestSettings.from_platform_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("automated_cycle_parent_settings_invalid") from exc


def _leaf_preference(leaf: object, diagnosis: ParentDiagnosis | None) -> int:
    """For a high-turnover parent, light smoothing comes before heavy smoothing."""
    if (diagnosis is not None and diagnosis.problem == HIGH_TURNOVER
            and getattr(leaf, "family", None) == TEMPORAL_PERSISTENCE_REFRAME
            and isinstance(getattr(leaf, "spec", None), TemporalReframeSpec)
            and leaf.spec.window <= SHORT_SMOOTHING_WINDOW):
        return 0
    return 1


def _passed_checks(snapshot: BacktestSnapshot) -> tuple[str, ...]:
    if snapshot.result is None:
        raise ValueError("automated_cycle_parent_result_missing")
    return tuple(
        sorted(
            check.name
            for check in snapshot.result.checks
            if check.status.strip().upper() == "PASS"
        )
    )
