from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from collections.abc import Mapping

from evaluation.backtests import evaluate_backtest
from persistence.backtests import BacktestSnapshot
from persistence.submissions import PlatformSubmittedAlphaRecord, normalize_submitted_formula
from worldquant.backtests import (
    STANDARD_NON_SC_CHECK_NAMES,
    STANDARD_REGULAR_CHECK_NAMES,
)


@dataclass(frozen=True, slots=True)
class SelfCorrelationCheckEvidence:
    task_id: str
    attempt_status: str
    observed_at: str
    statuses: tuple[tuple[str, str], ...]
    reference_id: str | None
    correlation: float | None


@dataclass(frozen=True, slots=True)
class SelfCorrelationReference:
    parent_task_id: str
    formula: str
    platform_alpha_id: str | None = None
    correlation: float | None = None


def select_self_correlation_reference(
    snapshot: BacktestSnapshot,
    check: SelfCorrelationCheckEvidence | None,
    references: tuple[PlatformSubmittedAlphaRecord, ...],
    *,
    observed_at: datetime,
) -> SelfCorrelationReference | None:
    if check is not None and check.task_id != snapshot.task.task_id:
        raise ValueError("self_correlation_check_identity_mismatch")
    if observed_at.utcoffset() is None:
        raise ValueError("self_correlation_observed_at_invalid")
    task = snapshot.task
    if (
        task.status != "completed"
        or snapshot.result is None
        or task.platform_alpha_id is None
        or task.finished_at is None
        or not snapshot.yearly_stats
        or _time(task.finished_at) > observed_at
    ):
        return None
    evaluation = evaluate_backtest(snapshot)
    if (
        not evaluation.non_sc_check_set_complete
        or not STANDARD_NON_SC_CHECK_NAMES <= set(evaluation.passed_checks)
    ):
        return None
    statuses = dict(check.statuses) if check is not None else {}
    checked_conflict = (
        check is not None
        and check.attempt_status == "ineligible"
        and bool(check.reference_id)
        and not isinstance(check.correlation, bool)
        and isinstance(check.correlation, (int, float))
        and math.isfinite(check.correlation)
        and 0 < check.correlation <= 1
        and _time(task.finished_at) <= _time(check.observed_at) <= observed_at
        and len(statuses) == len(check.statuses)
        and statuses.keys() == STANDARD_REGULAR_CHECK_NAMES
        and statuses.get("SELF_CORRELATION") == "FAIL"
        and all(statuses.get(name) == "PASS" for name in STANDARD_NON_SC_CHECK_NAMES)
    )
    parent_reference = None
    for reference in references:
        if (
            reference.account_scope != task.account_scope
            or reference.status != "ACTIVE"
            or _time(reference.observed_at) > observed_at
        ):
            continue
        settings = reference.raw_payload.get("settings")
        if not isinstance(settings, Mapping):
            continue
        comparable_settings = {
            key: value
            for key, value in settings.items()
            if key not in {"startDate", "endDate"}
        }
        if comparable_settings != json.loads(task.settings_json):
            continue
        # Preserve an existing verified conflict, including for historical recovery.
        if (checked_conflict and reference.platform_alpha_id == check.reference_id
                and reference.platform_alpha_id != task.platform_alpha_id):
            return SelfCorrelationReference(
                task.task_id, reference.formula, reference.platform_alpha_id, check.correlation,
            )
        if reference.normalized_formula == normalize_submitted_formula(task.formula):
            # This is structural self-reference, not an observed official SC value.
            parent_reference = SelfCorrelationReference(
                task.task_id, task.formula, task.platform_alpha_id,
            )
    return parent_reference


def _time(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value)
    if timestamp.utcoffset() is None:
        raise ValueError("self_correlation_evidence_time_invalid")
    return timestamp
