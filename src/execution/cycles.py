from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from execution.backtest_batches import (
    AutomatedCandidateBacktest,
    prepare_automated_candidate_backtest_batch,
)
from execution.backtests import backtest_request_fingerprint
from execution.catalog import load_generation_catalog
from execution.direction import build_direction_candidates
from execution.cycle_candidates import (
    OptimizationCandidatesExhausted,
    build_cycle_candidates,
    build_improvement_candidate_pools,
)
from execution.seeds import load_signal_frontiers
from execution.self_correlation import load_self_correlation_references
from execution.cycle_schedule import scheduled_cycle_number
from execution.runs import remaining_automated_run_backtests
from execution.progress import phase
from generation.candidate import exploration_candidate
from generation.parser import FormulaSyntaxError, parse_formula
from learning.action_effects import (
    DefectActionRequest,
    ExplicitSelfCorrelationEvidence,
    TaskRunEvidence,
    build_defect_action_strategies,
)
from learning.evidence import (
    build_learning_evidence,
    build_mutation_learning_evidence,
)
from learning.optimization_targets import (
    ParentOptimizationTargetSet,
    build_parent_optimization_targets,
)
from learning.parent_settings import (
    ParentSettingsEvidenceSet,
    build_parent_settings_evidence,
)
from learning.quality_proximity import (
    QUALIFIED_EVOLUTION_STAGE,
    STRUCTURAL_EVOLUTION_STAGE,
    build_parent_improvement_stages,
)
from learning.seed_correlation import submitted_seed
from persistence.backtests import (
    backtest_was_cancelled_before_submission,
    BacktestSnapshot,
    get_backtest_task,
    list_backtest_mutations,
    list_backtest_tasks,
    list_completed_backtests,
)
from persistence.catalog import FieldCatalogContext
from persistence.database import open_database
from persistence.run_allocations import get_run_allocation
from persistence.runs import (
    AutomatedRunBacktestRecord,
    AutomatedRunRecord,
    get_automated_run,
    list_all_automated_run_backtests,
    list_automated_run_backtests,
)
from persistence.seeds import list_signal_seeds
from persistence.submissions import (
    list_formal_submission_attempts,
    list_platform_submitted_alphas,
)
from selection.allocations import (
    BacktestSourceAllocation,
    IMPROVEMENT_FAMILIES_BY_STAGE,
    SignalImprovementFamilyCapacity,
    SignalImprovementSource,
    allocate_backtest_sources,
    direction_validation_limit,
)
from selection.settings import BacktestSettingsPolicy
from submission.formal import assess_formal_check_payload
from worldquant.backtests import BacktestSettings


@dataclass(frozen=True, slots=True)
class AutomatedCyclePlan:
    run_id: str
    cycle_number: int
    planned_source_allocation: BacktestSourceAllocation | None
    catalog_fingerprint: str | None
    requested_generation_count: int
    requested_backtest_count: int
    exploration_backtest_count: int | None
    structural_evolution_backtest_count: int | None
    local_polishing_backtest_count: int | None
    backtests: tuple[BacktestSnapshot, ...]
    recovered: bool

    @property
    def direction_validation_backtest_count(self) -> int | None:
        return (
            self.planned_source_allocation.direction_validation_count
            if self.planned_source_allocation is not None
            else None
        )

    @property
    def qualified_evolution_backtest_count(self) -> int | None:
        return (
            self.planned_source_allocation.qualified_evolution_count
            if self.planned_source_allocation is not None
            else None
        )


def plan_automated_cycle(
    database_path: str | Path,
    *,
    run_id: str,
    created_at: str,
    cycle_number: int | None = None,
) -> AutomatedCyclePlan:
    with open_database(database_path) as connection:
        run = _required_running_run(connection, run_id)
        if not run.real_backtests_authorized:
            raise ValueError("automated_run_backtests_not_authorized")
        if cycle_number is None:
            cycle_number = scheduled_cycle_number(connection, run)
        links = list_automated_run_backtests(connection, run.run_id)
        prior_links = tuple(link for link in links if link.cycle_number < cycle_number)
        current_links = tuple(
            link for link in links if link.cycle_number == cycle_number
        )
        remaining_backtests = remaining_automated_run_backtests(
            run,
            len(prior_links),
        )
        requested_backtests = (
            run.backtest_count
            if remaining_backtests is None
            else min(run.backtest_count, remaining_backtests)
        )
        if requested_backtests <= 0:
            raise ValueError("automated_run_backtest_limit_reached")
        if ((not run.optimization_only and len(current_links) not in {0, requested_backtests})
                or len(current_links) > requested_backtests):
            raise ValueError("automated_cycle_partial_tasks_detected")
        if current_links:
            # The persisted task set IS the frozen plan. Re-running selection here
            # would let late results or later submissions rewrite that plan.
            recovered = _required_snapshots(connection, current_links)
            _action_evidence_cutoff(created_at, recovered)
            _validate_frozen_cycle(run, recovered)
            return AutomatedCyclePlan(
                run_id=run.run_id,
                cycle_number=cycle_number,
                planned_source_allocation=None,
                catalog_fingerprint=None,
                requested_generation_count=run.generation_count,
                requested_backtest_count=requested_backtests,
                exploration_backtest_count=None,
                structural_evolution_backtest_count=None,
                local_polishing_backtest_count=None,
                backtests=tuple(
                    sorted(
                        recovered,
                        key=lambda item: (
                            item.task.formula_fingerprint,
                            item.task.settings_json,
                        ),
                    )
                ),
                recovered=True,
            )

        allocation_policy = get_run_allocation(connection, run.run_id)
        if allocation_policy is None:
            raise ValueError("automated_run_allocation_missing:start_a_new_run")
        phase(cycle_number, 1, "评级提升专项：读取完整检查通过的活动父代..." if run.optimization_only
              else "公式生成中，读取研究证据并准备探索与改善候选...")
        settings_policy = BacktestSettingsPolicy.from_config_dict(
            json.loads(run.settings_policy_json)
        )
        catalog = load_generation_catalog(
            connection,
            FieldCatalogContext(
                instrument_type=settings_policy.instrument_type,
                region=settings_policy.region,
                universe=settings_policy.universe,
                delay=settings_policy.delay,
            ),
            account_scope=run.account_scope,
        )
        evidence_cutoff = _timestamp(created_at, "action_effect_cutoff_invalid")
        reserved_tasks = tuple(task for task in list_backtest_tasks(connection)
                               if not backtest_was_cancelled_before_submission(task))
        submitted_alphas = list_platform_submitted_alphas(
            connection,
            account_scope=run.account_scope,
        )
        submitted_formula_identities = _supported_formula_identities(
            tuple(record.formula for record in submitted_alphas),
        )
        completed = list_completed_backtests(connection)
        completed_by_task_id = {
            snapshot.task.task_id: snapshot for snapshot in completed
        }
        evidence = build_learning_evidence(completed)
        frontiers = load_signal_frontiers(connection, optimization_only=run.optimization_only)
        parent_task_ids = tuple(
            branch.task_id
            for frontier in frontiers.records
            for branch in frontier.branches
        )
        qualified_parent_task_ids = tuple(
            branch.task_id
            for frontier in frontiers.records
            for branch in frontier.branches
            if branch.task_id in frontier.qualified_task_ids
        )
        exploration_fields = tuple(
            field.field_id
            for field in catalog.fields
            if field.field_type in {"MATRIX", "VECTOR"}
            and field.coverage is not None
            and field.coverage > 0
        )
        groups = tuple(
            field.field_id for field in catalog.fields if field.field_type == "GROUP"
        )
        mutation_fields = tuple(
            field.field_id
            for field in catalog.fields
            if field.field_type == "MATRIX"
            and field.coverage is not None
            and field.coverage > 0
        )
        planning_mutations = list_backtest_mutations(connection)
        effective_exploration_reserve = (
            requested_backtests * allocation_policy.exploration_percent + 99
        ) // 100 if not run.optimization_only else 0
        direction_candidates = () if run.optimization_only else build_direction_candidates(
            catalog,
            settings_policy,
            completed=completed,
            mutations=planning_mutations,
            reserved_tasks=reserved_tasks,
            excluded_formula_fingerprints=(
                submitted_formula_identities.formula_fingerprints
            ),
            account_scope=run.account_scope,
            evidence_cutoff=evidence_cutoff,
            max_candidates=direction_validation_limit(
                requested_backtests, effective_exploration_reserve,
                allocation_policy.direction_validation_percent,
            ),
        )
        parent_evidence = build_parent_settings_evidence(
            evidence,
            planning_mutations,
            parent_task_ids,
        )
        mutation_evidence = build_mutation_learning_evidence(
            evidence,
            planning_mutations,
        )
        optimization_targets = build_parent_optimization_targets(
            evidence,
            mutation_evidence,
            parent_task_ids,
        )
        eligible_parent_task_ids = _eligible_improvement_parent_task_ids(
            parent_task_ids,
            account_scope=run.account_scope,
            settings_evidence=parent_evidence,
            optimization_targets=optimization_targets,
            qualified_parent_task_ids=qualified_parent_task_ids,
        )
        improvement_stages = build_parent_improvement_stages(
            evidence,
            optimization_targets,
            eligible_parent_task_ids,
            qualified_parent_task_ids=tuple(
                parent_task_id
                for parent_task_id in qualified_parent_task_ids
                if parent_task_id in eligible_parent_task_ids
            ),
        )
        sc_references = () if run.optimization_only else load_self_correlation_references(
            connection,
            parents=tuple(completed_by_task_id[task_id] for task_id in eligible_parent_task_ids),
            submitted_alphas=submitted_alphas,
            account_scope=run.account_scope,
            observed_at=evidence_cutoff,
        )
        sc_by_parent = {item.parent_task_id: item.correlation for item in sc_references}
        improvement_pools = build_improvement_candidate_pools(
            catalog,
            settings_policy,
            eligible_parent_task_ids=eligible_parent_task_ids,
            improvement_stages=improvement_stages,
            completed_by_task_id=completed_by_task_id,
            field_candidates=mutation_fields,
            internal_field_candidates=exploration_fields,
            rotation_key=f"{run.run_id}|{cycle_number}",
            group_candidates=groups,
            reserved_tasks=reserved_tasks,
            self_correlation_references=sc_references,
            excluded_formula_fingerprints=(
                submitted_formula_identities.formula_fingerprints
                | frozenset(item.candidate.fingerprint for item in direction_candidates)
            ),
        )
        pools_by_parent = {pool.parent_task_id: pool for pool in improvement_pools}
        signal_source_items: list[SignalImprovementSource] = []
        for frontier in frontiers.records:
            for branch in frontier.branches:
                pool = pools_by_parent.get(branch.task_id)
                stage = pool.stage if pool is not None else STRUCTURAL_EVOLUTION_STAGE
                families = IMPROVEMENT_FAMILIES_BY_STAGE[stage]
                available_by_family = (
                    Counter(candidate.family for candidate in pool.candidates)
                    if pool is not None
                    else Counter()
                )
                usage_by_family = (
                    {item.family: item for item in pool.family_usage}
                    if pool is not None
                    else {}
                )
                signal_source_items.append(
                    SignalImprovementSource(
                        root_task_id=frontier.root_task_id,
                        branch_task_id=branch.task_id,
                        stage=stage,
                        parent_remaining_attempts=branch.remaining_attempts,
                        is_submitted=submitted_seed(
                            completed_by_task_id[branch.task_id], submitted_alphas,
                        ),
                        self_correlation=sc_by_parent.get(branch.task_id),
                        family_capacities=tuple(
                            SignalImprovementFamilyCapacity(
                                family=family,
                                available_leaf_count=available_by_family[family],
                                blocked_leaf_count=(
                                    usage_by_family[family].blocked_request_count
                                    if family in usage_by_family
                                    else 0
                                ),
                                attempted_leaf_count=(
                                    usage_by_family[family].attempted_request_count
                                    if family in usage_by_family
                                    else 0
                                ),
                            )
                            for family in families
                        ),
                    )
                )
        improvement_sources = tuple(signal_source_items)
        action_completed = tuple(
            snapshot
            for snapshot in completed
            if _timestamp(
                snapshot.task.finished_at,
                "action_effect_finished_at_invalid",
            )
            <= evidence_cutoff
        )
        action_evidence = build_learning_evidence(action_completed)
        action_mutation_evidence = build_mutation_learning_evidence(
            action_evidence,
            planning_mutations,
        )
        targets_by_parent = {
            record.parent_task_id: record for record in optimization_targets.records
        }
        action_strategies = build_defect_action_strategies(
            action_evidence,
            action_mutation_evidence,
            tuple(
                DefectActionRequest(
                    parent_task_id=source.branch_task_id,
                    target_check_name=target.check_name,
                    candidate_actions=tuple(
                        capacity.family for capacity in source.family_capacities
                    ),
                )
                for source in improvement_sources
                if source.stage != QUALIFIED_EVOLUTION_STAGE
                for target in targets_by_parent[source.branch_task_id].targets
            ),
            seed_root_task_ids=tuple(
                seed.root_task_id
                for seed in list_signal_seeds(connection)
                if _timestamp(
                    seed.promoted_at,
                    "action_effect_seed_timestamp_invalid",
                )
                <= evidence_cutoff
            ),
            task_runs=tuple(
                TaskRunEvidence(task_id=link.task_id, run_id=link.run_id)
                for link in list_all_automated_run_backtests(connection)
            ),
            explicit_self_correlations=_explicit_self_correlation_evidence(
                connection,
                account_scope=run.account_scope,
                evidence_cutoff=evidence_cutoff,
            ),
        )
        rotation_key = f"{run.run_id}:{cycle_number}"
        source_allocation = allocate_backtest_sources(
            requested_count=requested_backtests,
            minimum_exploration_count=effective_exploration_reserve,
            improvement_sources=improvement_sources,
            settings_evidence=parent_evidence,
            optimization_targets=optimization_targets,
            action_strategies=action_strategies,
            account_scope=run.account_scope,
            rotation_key=rotation_key,
            direction_validation_count=len(direction_candidates),
            self_correlation_percent=0 if run.optimization_only else allocation_policy.self_correlation_percent,
            direction_validation_percent=0 if run.optimization_only else allocation_policy.direction_validation_percent,
            optimization_only=run.optimization_only,
        )
        if run.optimization_only and source_allocation.requested_count == 0:
            raise OptimizationCandidatesExhausted("optimization_candidates_exhausted")
        existing_formulas = (
            tuple(task.formula for task in reserved_tasks)
            + submitted_formula_identities.formulas
        )
        candidate_plan = build_cycle_candidates(
            catalog,
            settings_policy,
            cycle_number=cycle_number,
            requested_generation_count=run.generation_count,
            requested_backtest_count=source_allocation.requested_count,
            exploration_seed_attempt_multiplier=(
                run.exploration_seed_attempt_multiplier
            ),
            source_allocation=source_allocation,
            improvement_pools=improvement_pools,
            exploration_field_candidates=exploration_fields,
            group_candidates=groups,
            existing_formulas=existing_formulas,
            seed_start=_cycle_seed_start(run.run_id, cycle_number),
            direction_candidates=direction_candidates,
        )
        selected_candidates = candidate_plan.candidates
    prepared = prepare_automated_candidate_backtest_batch(
        database_path,
        run_id=run.run_id,
        candidates=tuple(
            AutomatedCandidateBacktest(
                candidate=item.candidate,
                settings=item.settings,
            )
            for item in selected_candidates
        ),
        created_at=created_at,
        cycle_number=cycle_number,
    )

    phase(cycle_number, 2, f"筛选完成，已冻结本轮 {len(prepared)} 条回测任务")
    return AutomatedCyclePlan(
        run_id=run.run_id,
        cycle_number=cycle_number,
        planned_source_allocation=source_allocation,
        catalog_fingerprint=catalog.fingerprint,
        requested_generation_count=run.generation_count,
        requested_backtest_count=requested_backtests,
        exploration_backtest_count=candidate_plan.exploration_backtest_count,
        structural_evolution_backtest_count=(
            candidate_plan.structural_evolution_backtest_count
        ),
        local_polishing_backtest_count=candidate_plan.local_polishing_backtest_count,
        backtests=tuple(
            sorted(
                prepared,
                key=lambda item: (
                    item.task.formula_fingerprint,
                    item.task.settings_json,
                ),
            )
        ),
        recovered=False,
    )


def _required_snapshots(
    connection: sqlite3.Connection,
    links: tuple[AutomatedRunBacktestRecord, ...],
) -> tuple[BacktestSnapshot, ...]:
    snapshots: list[BacktestSnapshot] = []
    for link in links:
        snapshot = get_backtest_task(connection, link.task_id)
        if snapshot is None:
            raise ValueError("automated_run_backtest_task_missing")
        snapshots.append(snapshot)
    return tuple(snapshots)


def _action_evidence_cutoff(
    created_at: str,
    current_snapshots: tuple[BacktestSnapshot, ...],
) -> datetime:
    requested = _timestamp(created_at, "action_effect_cutoff_invalid")
    if not current_snapshots:
        return requested
    existing_values = {snapshot.task.created_at for snapshot in current_snapshots}
    if len(existing_values) != 1:
        raise ValueError("automated_cycle_recovery_created_at_conflict")
    return _timestamp(
        existing_values.pop(),
        "automated_cycle_recovery_created_at_invalid",
    )


def _explicit_self_correlation_evidence(
    connection: sqlite3.Connection,
    *,
    account_scope: str,
    evidence_cutoff: datetime,
) -> tuple[ExplicitSelfCorrelationEvidence, ...]:
    built: list[ExplicitSelfCorrelationEvidence] = []
    for attempt in list_formal_submission_attempts(
        connection,
        account_scope=account_scope,
    ):
        if (
            attempt.check_payload_json is None
            or attempt.check_observed_at is None
            or _timestamp(
                attempt.check_observed_at,
                "action_effect_formal_check_timestamp_invalid",
            )
            > evidence_cutoff
        ):
            continue
        try:
            payload = json.loads(attempt.check_payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("action_effect_formal_check_payload_invalid") from exc
        assessment = assess_formal_check_payload(payload)
        status = dict(assessment.statuses).get("SELF_CORRELATION")
        if status not in {"PASS", "FAIL"}:
            continue
        built.append(
            ExplicitSelfCorrelationEvidence(
                task_id=attempt.task_id,
                status=status,
                formal_check_state=assessment.state,
                observed_at=attempt.check_observed_at,
            )
        )
    return tuple(built)


def _eligible_improvement_parent_task_ids(
    parent_task_ids: tuple[str, ...],
    *,
    account_scope: str,
    settings_evidence: ParentSettingsEvidenceSet,
    optimization_targets: ParentOptimizationTargetSet,
    qualified_parent_task_ids: tuple[str, ...],
) -> tuple[str, ...]:
    settings_by_parent = {
        record.parent_task_id: record
        for record in settings_evidence.records
        if record.account_scope == account_scope
    }
    target_records_by_parent = {
        record.parent_task_id: record
        for record in optimization_targets.records
        if record.account_scope == account_scope
    }
    qualified_parent_ids = set(qualified_parent_task_ids)
    return tuple(
        parent_task_id
        for parent_task_id in parent_task_ids
        if (settings := settings_by_parent.get(parent_task_id)) is not None
        and (targets := target_records_by_parent.get(parent_task_id)) is not None
        and settings.settings_key == targets.settings_key
        and (parent_task_id in qualified_parent_ids or bool(targets.targets))
    )


def _required_running_run(
    connection: sqlite3.Connection,
    run_id: str,
) -> AutomatedRunRecord:
    run = get_automated_run(connection, run_id)
    if run is None:
        raise ValueError("automated_run_missing")
    if run.status != "running":
        raise ValueError("automated_run_status_invalid")
    return run


def _validate_frozen_cycle(run, snapshots: tuple[BacktestSnapshot, ...]) -> None:
    policy = BacktestSettingsPolicy.from_config_dict(
        json.loads(run.settings_policy_json)
    )
    for snapshot in snapshots:
        task = snapshot.task
        if (
            task.account_scope != run.account_scope
            or not policy.allows(
                BacktestSettings.from_platform_dict(json.loads(task.settings_json))
            )
            or parse_formula(task.formula).fingerprint != task.formula_fingerprint
            or backtest_request_fingerprint(
                account_scope=task.account_scope,
                formula=task.formula,
                settings_json=task.settings_json,
            )
            != task.request_fingerprint
        ):
            raise ValueError("automated_cycle_recovery_identity_conflict")


@dataclass(frozen=True, slots=True)
class _SupportedFormulaIdentities:
    formulas: tuple[str, ...]
    formula_fingerprints: frozenset[str]


def _supported_formula_identities(
    formulas: tuple[str, ...],
) -> _SupportedFormulaIdentities:
    supported_formulas: list[str] = []
    fingerprints: set[str] = set()
    for formula in formulas:
        try:
            candidate = exploration_candidate(parse_formula(formula).expression)
        except (FormulaSyntaxError, ValueError):
            continue
        supported_formulas.append(candidate.formula)
        fingerprints.add(candidate.fingerprint)
    return _SupportedFormulaIdentities(
        formulas=tuple(supported_formulas),
        formula_fingerprints=frozenset(fingerprints),
    )


def _cycle_seed_start(run_id: str, cycle_number: int) -> int:
    return _stable_seed(f"{run_id}:{cycle_number}")


def _stable_seed(identity: str) -> int:
    return int.from_bytes(sha256(identity.encode("utf-8")).digest()[:8], "big")


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
