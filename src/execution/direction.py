from __future__ import annotations

import json
from datetime import datetime

from execution.cycle_candidates import CycleCandidate
from generation.catalog import GenerationCatalog
from generation.direction import reverse_direction_candidate
from generation.parser import FormulaSyntaxError, parse_formula
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import validate_formula
from learning.direction import negative_direction_is_testable
from persistence.backtests import (
    backtest_was_cancelled_before_submission,
    BacktestMutationRecord,
    BacktestSnapshot,
    BacktestTaskRecord,
)
from selection.formulas import formula_selection_candidate
from selection.settings import BacktestSettingsPolicy
from worldquant.backtests import BacktestSettings


def build_direction_candidates(
    catalog: GenerationCatalog,
    policy: BacktestSettingsPolicy,
    *,
    completed: tuple[BacktestSnapshot, ...],
    mutations: tuple[BacktestMutationRecord, ...],
    reserved_tasks: tuple[BacktestTaskRecord, ...],
    excluded_formula_fingerprints: frozenset[str],
    account_scope: str,
    evidence_cutoff: datetime,
    max_candidates: int,
) -> tuple[CycleCandidate, ...]:
    """Discover whole-signal direction questions from completed local facts."""
    if max_candidates == 0:
        return ()
    child_ids = {mutation.child_task_id for mutation in mutations}
    used = {
        (task.account_scope, task.formula_fingerprint, task.settings_json)
        for task in reserved_tasks if not backtest_was_cancelled_before_submission(task)
    }
    proposed: list[CycleCandidate] = []
    # Stable ordering provides reproducibility, not a quality ranking.
    for parent in sorted(completed, key=lambda item: item.task.task_id):
        task = parent.task
        if (
            task.account_scope != account_scope
            or task.task_id in child_ids
            or not negative_direction_is_testable(parent)
        ):
            continue
        if datetime.fromisoformat(task.finished_at) > evidence_cutoff:
            continue
        settings = BacktestSettings.from_platform_dict(json.loads(task.settings_json))
        if not policy.allows(settings):
            continue
        try:
            parsed = parse_formula(task.formula)
        except FormulaSyntaxError:
            # A historical formula unsupported by the current parser is not a candidate.
            continue
        candidate = reverse_direction_candidate(
            parsed.expression, parent_task_id=task.task_id
        )
        identity = (account_scope, candidate.fingerprint, task.settings_json)
        if (
            identity in used
            or candidate.fingerprint in excluded_formula_fingerprints
            or not validate_formula(candidate.expression, catalog).is_valid
            or find_coarse_unit_issues(candidate.expression, catalog)
            or formula_selection_candidate(
                candidate.expression, catalog, settings
            ).rejection_reasons
        ):
            continue
        used.add(identity)
        proposed.append(CycleCandidate(candidate, settings))
        if len(proposed) == max_candidates:
            break
    return tuple(proposed)
