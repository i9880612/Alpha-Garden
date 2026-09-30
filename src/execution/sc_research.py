from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from execution.catalog import load_generation_catalog
from execution.cycle_candidates import build_improvement_candidate_pools, supported_formula_identities
from execution.self_correlation import load_latest_submission_checks, load_self_correlation_references
from generation.candidate import CandidateChange, FormulaCandidate, mutation_candidate
from generation.parser import parse_formula
from generation.self_correlation import SELF_CORRELATION_RESEARCH_FAMILIES
from learning.frontiers import PARENT_ATTEMPT_BUDGET
from learning.quality_proximity import ParentImprovementStage, ParentImprovementStageSet, QUALIFIED_EVOLUTION_STAGE
from learning.seeds import assess_signal_seed
from persistence.backtests import (
    BACKTEST_ACTIVE_STATUSES, backtest_was_cancelled_before_submission,
    get_backtest_task, list_backtest_mutations, list_backtest_tasks,
)
from persistence.catalog import FieldCatalogContext
from persistence.seeds import list_signal_seeds
from persistence.submissions import list_platform_submitted_alphas
from selection.settings import BacktestSettingsPolicy
from worldquant.backtests import BacktestSettings, STANDARD_NON_SC_CHECK_NAMES


@dataclass(frozen=True, slots=True)
class SelfCorrelationResearchPlan:
    parent_task_id: str
    root_task_id: str
    account_scope: str
    settings_json: str
    catalog_fingerprint: str
    evidence_observed_at: str
    remaining_attempts: int
    sharpe: float
    fitness: float
    correlation: float
    required_sharpe: float
    blockers: tuple[str, ...]
    candidates: tuple[FormulaCandidate, ...]

    def canonical_json(self) -> str:
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "candidates"}
        payload["candidates"] = [dict(
            formula=candidate.formula,
            parent_formula_fingerprint=candidate.parent_formula_fingerprint,
            change=asdict(candidate.change),
        ) for candidate in self.candidates]
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def fingerprint(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, value: str) -> SelfCorrelationResearchPlan:
        payload = json.loads(value)
        parent_id = payload["parent_task_id"]
        candidates = []
        for item in payload.pop("candidates"):
            change = item["change"]
            change["parameters"] = tuple(tuple(pair) for pair in change["parameters"])
            candidates.append(mutation_candidate(
                parse_formula(item["formula"]).expression, parent_task_id=parent_id,
                parent_formula_fingerprint=item["parent_formula_fingerprint"],
                change=CandidateChange(**change),
            ))
        payload["blockers"] = tuple(payload["blockers"])
        plan = cls(**payload, candidates=tuple(candidates))
        if (len(plan.candidates) != 6 or len({c.fingerprint for c in plan.candidates}) != 6
                or any(sum(c.change.action == family for c in plan.candidates) != 2
                       for family in SELF_CORRELATION_RESEARCH_FAMILIES)
                or plan.canonical_json() != value):
            raise ValueError("sc_research_plan_invalid")
        return plan


def build_self_correlation_research_plan(
    connection: sqlite3.Connection, policy: BacktestSettingsPolicy, *,
    parent_task_id: str, observed_at: datetime, account_scope: str | None = None,
) -> SelfCorrelationResearchPlan:
    """Read facts and select six legal initial trials; no network or state writes."""
    parent = get_backtest_task(connection, parent_task_id)
    if parent is None or parent.task.status != "completed" or parent.result is None:
        raise ValueError("sc_research_parent_result_missing")
    if account_scope is not None and parent.task.account_scope != account_scope:
        raise ValueError("sc_research_parent_account_mismatch")
    account_scope = parent.task.account_scope
    settings = BacktestSettings.from_platform_dict(json.loads(parent.task.settings_json))
    if not policy.allows(settings):
        raise ValueError("sc_research_parent_settings_not_allowed")
    check = load_latest_submission_checks(connection, account_scope=account_scope, observed_at=observed_at).get(parent_task_id)
    statuses = dict(check.assessment.statuses) if check is not None else {}
    if (check is None or not check.repair_allowed or statuses.get("SELF_CORRELATION") != "FAIL"
            or any(statuses.get(name) != "PASS" for name in STANDARD_NON_SC_CHECK_NAMES)
            or not assess_signal_seed(parent).eligible):
        raise ValueError("sc_research_parent_not_only_sc_failed")
    submitted = list_platform_submitted_alphas(connection, account_scope=account_scope)
    references = load_self_correlation_references(
        connection, parents=(parent,), submitted_alphas=submitted,
        account_scope=account_scope, observed_at=observed_at,
    )
    if len(references) != 1 or references[0].correlation is None or references[0].required_sharpe is None:
        raise ValueError("sc_research_blocker_evidence_pending")
    reference = references[0]
    tasks = tuple(t for t in list_backtest_tasks(connection) if t.account_scope == account_scope)
    mutations = list_backtest_mutations(connection)
    by_id = {t.task_id: t for t in tasks}
    children = tuple(by_id[m.child_task_id] for m in mutations
                     if m.parent_task_id == parent_task_id and m.child_task_id in by_id
                     and not backtest_was_cancelled_before_submission(by_id[m.child_task_id]))
    remaining = PARENT_ATTEMPT_BUDGET - sum(
        t.submission_started_at is not None or t.status in BACKTEST_ACTIVE_STATUSES for t in children)
    if remaining < 6:
        raise ValueError("sc_research_parent_budget_insufficient")
    roots = {s.root_task_id for s in list_signal_seeds(connection)}
    ancestors = {m.child_task_id: m.parent_task_id for m in mutations}
    root_id = parent_task_id
    visited = set()
    while root_id not in roots and root_id in ancestors:
        if root_id in visited:
            raise ValueError("sc_research_lineage_invalid")
        visited.add(root_id)
        root_id = ancestors[root_id]
    if root_id not in roots and (root_id != parent_task_id or parent_task_id in ancestors):
        raise ValueError("sc_research_lineage_missing")
    # This explicit SC target is qualified by the current blocker evidence and
    # its own remaining budget. The Sharpe/Fitness frontier ranks a different
    # objective; refreshing unbound historical repairs must not revoke a valid
    # present SC target or manufacture old conflict bindings.
    catalog = load_generation_catalog(connection, FieldCatalogContext(
        policy.instrument_type, policy.region, policy.universe, policy.delay), account_scope=account_scope)
    fields = tuple(f.field_id for f in catalog.fields if f.field_type == "MATRIX")
    research_fields = tuple(f.field_id for f in catalog.fields if f.field_type in {"MATRIX", "VECTOR"})
    groups = tuple(f.field_id for f in catalog.fields if f.field_type == "GROUP")
    pool = build_improvement_candidate_pools(
        catalog, policy, eligible_parent_task_ids=(parent_task_id,),
        improvement_stages=ParentImprovementStageSet((ParentImprovementStage(parent_task_id, QUALIFIED_EVOLUTION_STAGE, None),)),
        completed_by_task_id={parent_task_id: parent}, field_candidates=fields, group_candidates=groups,
        internal_field_candidates=research_fields,
        reserved_tasks=tasks, mutations=mutations, self_correlation_references=references,
        excluded_formula_fingerprints=supported_formula_identities(tuple(r.formula for r in submitted)).formula_fingerprints,
        rotation_key=f"{parent_task_id}|sc-research",
    )[0]
    if any(u.attempted_request_count or u.unsubmitted_request_count for u in pool.family_usage
           if u.family in SELF_CORRELATION_RESEARCH_FAMILIES):
        raise ValueError("sc_research_initial_trials_already_started")
    chosen = []
    for family in SELF_CORRELATION_RESEARCH_FAMILIES:
        available = [item.candidate for item in pool.candidates if item.family == family]
        if len(available) < 2:
            raise ValueError(f"sc_research_candidates_insufficient:{family}")
        chosen.append(available[:2])
    candidates = tuple(chosen[family][trial] for trial in range(2) for family in range(3))
    return SelfCorrelationResearchPlan(
        parent_task_id, root_id, account_scope, parent.task.settings_json, catalog.fingerprint,
        check.observed_at, remaining, parent.result.sharpe, parent.result.fitness,
        reference.correlation, reference.required_sharpe,
        tuple(b.reference_id for b in reference.blockers), candidates,
    )


def preview_self_correlation_research_plan(
    database_path: str | Path, policy: BacktestSettingsPolicy, *, parent_task_id: str,
    observed_at: datetime,
) -> SelfCorrelationResearchPlan:
    with sqlite3.connect(Path(database_path).resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='backtest_mutation_references'").fetchone() is None:
            raise ValueError("sc_research_mutation_binding_storage_update_required")
        return build_self_correlation_research_plan(
            connection, policy, parent_task_id=parent_task_id, observed_at=observed_at)
