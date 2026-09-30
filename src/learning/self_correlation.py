from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from collections.abc import Mapping

from evaluation.backtests import evaluate_backtest
from evaluation.correlation import CORRELATION_CUTOFF, correlation_check, submitted_sharpe
from persistence.backtests import BacktestSnapshot
from persistence.submissions import PlatformSubmittedAlphaRecord, normalize_submitted_formula
from worldquant.backtests import (
    BacktestSettings,
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
    blockers: tuple[SelfCorrelationBlocker, ...] = ()
    required_sharpe: float | None = None


@dataclass(frozen=True, slots=True)
class SelfCorrelationBlocker:
    reference_id: str
    correlation: float
    required_sharpe: float


@dataclass(frozen=True, slots=True)
class SelfCorrelationAssessment:
    state: str
    blockers: tuple[SelfCorrelationBlocker, ...]
    unresolved_count: int
    required_sharpe: float | None


def assess_self_correlation_peers(
    candidate_sharpe: float | None,
    peers: tuple[tuple[str, float], ...] | None,
    references: tuple[PlatformSubmittedAlphaRecord, ...],
) -> SelfCorrelationAssessment:
    """Assess all reported pairs; callers supply the account and time scoped facts."""
    if peers is None:
        return SelfCorrelationAssessment("pending", (), 1, None)
    by_id = {reference.platform_alpha_id: reference for reference in references}
    blockers = []
    unresolved = 0
    required = []
    for reference_id, correlation in peers:
        reference = by_id.get(reference_id)
        sharpe = submitted_sharpe(reference.raw_payload) if reference is not None else None
        state = correlation_check(correlation, candidate_sharpe, sharpe)
        if state == "pending":
            unresolved += 1
        elif correlation >= CORRELATION_CUTOFF:
            assert sharpe is not None
            threshold = float(Decimal(str(sharpe)) * Decimal("1.10"))
            required.append(threshold)
            if state == "failed":
                blockers.append(SelfCorrelationBlocker(reference_id, correlation, threshold))
    return SelfCorrelationAssessment(
        "failed" if blockers else "pending" if unresolved else "passed",
        tuple(sorted(blockers, key=lambda item: (-item.required_sharpe, -item.correlation, item.reference_id))),
        unresolved, None if unresolved else max(required, default=0.0),
    )


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
    parent_settings = json.loads(task.settings_json)
    parent_reference = None
    for reference in references:
        reported_peer = checked_conflict and reference.platform_alpha_id == check.reference_id
        if (
            reference.account_scope != task.account_scope
            or (reference.status != "ACTIVE"
                and not (reported_peer and reference.status == "DECOMMISSIONED"))
            or _time(reference.observed_at) > observed_at
        ):
            continue
        settings = reference.raw_payload.get("settings")
        if not isinstance(settings, Mapping):
            continue
        if settings.get("simulationMode", "FULL") != "FULL":
            continue
        comparable_settings = {
            key: value
            for key, value in settings.items()
            if key not in {"startDate", "endDate", "simulationMode"}
        }
        try:
            BacktestSettings.from_platform_dict(comparable_settings)
        except ValueError:
            continue
        # A reported conflict remains real across neutralization/truncation.
        # Keep every other setting equal; this only selects the research peer.
        if (reported_peer
                and reference.platform_alpha_id != task.platform_alpha_id
                and comparable_settings.keys() == parent_settings.keys()
                and all(value == parent_settings[key]
                        for key, value in comparable_settings.items()
                        if key not in {"neutralization", "truncation"})):
            return SelfCorrelationReference(
                task.task_id, reference.formula, reference.platform_alpha_id, check.correlation,
            )
        if (reference.status == "ACTIVE" and comparable_settings == parent_settings
                and reference.normalized_formula == normalize_submitted_formula(task.formula)):
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
