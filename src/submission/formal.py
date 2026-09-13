from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from evaluation.backtests import evaluate_backtest
from persistence.backtests import BacktestSnapshot
from worldquant.backtests import (
    STANDARD_NON_SC_CHECK_NAMES,
    STANDARD_REGULAR_CHECK_NAMES,
)


BELOW_TARGET_GRADES = frozenset({"INFERIOR", "AVERAGE", "GOOD", "EXCELLENT"})


@dataclass(frozen=True, slots=True)
class FormalCheckAssessment:
    state: str
    statuses: tuple[tuple[str, str], ...]
    failed_checks: tuple[str, ...]
    unresolved_checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FormalSubmissionCandidate:
    task_id: str
    cycle_number: int
    family_root_task_id: str
    platform_alpha_id: str
    formula: str
    normalized_formula: str
    sharpe: float
    fitness: float
    finished_at: str


@dataclass(frozen=True, slots=True)
class ConfirmedFormalSubmission:
    platform_alpha_id: str
    formula: str
    status: str
    date_submitted: str
    hidden: bool
    raw_payload: Mapping[str, object]


def local_formal_submission_eligible(snapshot: BacktestSnapshot) -> bool:
    if (
        not isinstance(snapshot, BacktestSnapshot)
        or snapshot.task.status != "completed"
        or snapshot.task.platform_alpha_id is None
        or snapshot.result is None
        or not snapshot.yearly_stats
    ):
        return False
    evaluation = evaluate_backtest(snapshot)
    return bool(
        evaluation.non_sc_check_set_complete
        and STANDARD_NON_SC_CHECK_NAMES <= set(evaluation.passed_checks)
    )


def formal_submission_grade_rejection(
    grade: str | None, *, source: str = "queue", expected_grade: str | None = None,
) -> str | None:
    if source not in {"queue", "qualified_archive"}:
        raise ValueError("formal_submission_source_invalid")
    if expected_grade is not None and grade in BELOW_TARGET_GRADES | {"SPECTACULAR"} and grade != expected_grade:
        return f"formal_submission_grade_changed:{expected_grade}:{grade}"
    if grade == "SPECTACULAR":
        return None
    if grade in BELOW_TARGET_GRADES:
        if source == "qualified_archive":
            return None
        return "formal_submission_grade_below_target:" + grade
    return "formal_submission_grade_unavailable"


def submission_queue_eligible(snapshot: BacktestSnapshot) -> bool:
    return bool(
        local_formal_submission_eligible(snapshot)
        and snapshot.result is not None
        and formal_submission_grade_rejection(snapshot.result.grade) is None
    )


def qualified_archive_eligible(snapshot: BacktestSnapshot, check_payload: object) -> bool:
    """Archive only fully checked candidates outside the Spectacular queue."""
    return bool(
        local_formal_submission_eligible(snapshot)
        and snapshot.result is not None
        and formal_submission_grade_rejection(snapshot.result.grade) is not None
        and assess_formal_check_payload(check_payload).state == "passed"
    )


def select_next_family_candidate(
    candidates: Sequence[FormalSubmissionCandidate],
) -> FormalSubmissionCandidate | None:
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("formal_submission_candidates_invalid")
    candidates_by_family: dict[str, list[FormalSubmissionCandidate]] = {}
    task_ids: set[str] = set()
    for candidate in candidates:
        _validate_candidate(candidate)
        if candidate.task_id in task_ids:
            raise ValueError("formal_submission_candidate_duplicate")
        task_ids.add(candidate.task_id)
        candidates_by_family.setdefault(candidate.family_root_task_id, []).append(
            candidate
        )
    if not candidates_by_family:
        return None
    selected_family = min(
        candidates_by_family.values(),
        key=_family_order,
    )
    return min(selected_family, key=_within_family_order)


def assess_formal_check_payload(payload: object) -> FormalCheckAssessment:
    if not isinstance(payload, Mapping):
        return _pending_assessment((), STANDARD_REGULAR_CHECK_NAMES)
    metrics = payload.get("is")
    if not isinstance(metrics, Mapping):
        return _pending_assessment((), STANDARD_REGULAR_CHECK_NAMES)
    raw_checks = metrics.get("checks")
    if not isinstance(raw_checks, list) or not raw_checks:
        return _pending_assessment((), STANDARD_REGULAR_CHECK_NAMES)

    statuses: dict[str, str] = {}
    for raw_check in raw_checks:
        if not isinstance(raw_check, Mapping):
            return _pending_assessment((), STANDARD_REGULAR_CHECK_NAMES)
        raw_name = raw_check.get("name")
        raw_status = raw_check.get("result")
        if (
            not isinstance(raw_name, str)
            or not raw_name.strip()
            or not isinstance(raw_status, str)
            or not raw_status.strip()
        ):
            return _pending_assessment((), STANDARD_REGULAR_CHECK_NAMES)
        name = raw_name.strip()
        if name in statuses:
            return _pending_assessment((), STANDARD_REGULAR_CHECK_NAMES)
        statuses[name] = raw_status.strip().upper()

    ordered = tuple(sorted(statuses.items()))
    failed = tuple(
        sorted(name for name, status in statuses.items() if status == "FAIL")
    )
    if failed:
        return FormalCheckAssessment(
            state="failed",
            statuses=ordered,
            failed_checks=failed,
            unresolved_checks=(),
        )
    unresolved = tuple(
        sorted(
            (STANDARD_REGULAR_CHECK_NAMES - statuses.keys())
            | {name for name, status in statuses.items() if status != "PASS"}
        )
    )
    if unresolved:
        return _pending_assessment(ordered, unresolved)
    return FormalCheckAssessment(
        state="passed",
        statuses=ordered,
        failed_checks=(),
        unresolved_checks=(),
    )


def confirm_formal_submission(
    payload: object,
    *,
    expected_alpha_id: str,
    expected_normalized_formula: str,
    normalize_formula,
) -> ConfirmedFormalSubmission | None:
    if not isinstance(payload, Mapping):
        raise ValueError("formal_submission_confirmation_payload_invalid")
    if payload.get("id") != expected_alpha_id:
        raise ValueError("formal_submission_confirmation_identity_invalid")
    raw_status = payload.get("status")
    if not isinstance(raw_status, str) or not raw_status.strip():
        raise ValueError("formal_submission_confirmation_status_invalid")
    status = raw_status.strip().upper()
    regular = payload.get("regular")
    formula = regular.get("code") if isinstance(regular, Mapping) else None
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formal_submission_confirmation_formula_invalid")
    if normalize_formula(formula) != expected_normalized_formula:
        raise ValueError("formal_submission_confirmation_formula_mismatch")
    if status == "UNSUBMITTED":
        return None
    if status != "ACTIVE":
        raise ValueError("formal_submission_confirmation_status_invalid")
    raw_date = payload.get("dateSubmitted")
    if not isinstance(raw_date, str) or not raw_date.strip():
        raise ValueError("formal_submission_confirmation_date_invalid")
    try:
        submitted_at = datetime.fromisoformat(raw_date.strip())
    except ValueError as exc:
        raise ValueError("formal_submission_confirmation_date_invalid") from exc
    if submitted_at.utcoffset() is None:
        raise ValueError("formal_submission_confirmation_date_invalid")
    hidden = payload.get("hidden")
    if not isinstance(hidden, bool):
        raise ValueError("formal_submission_confirmation_hidden_invalid")
    return ConfirmedFormalSubmission(
        platform_alpha_id=expected_alpha_id,
        formula=formula,
        status=status,
        date_submitted=raw_date.strip(),
        hidden=hidden,
        raw_payload=dict(payload),
    )


def _pending_assessment(
    statuses: tuple[tuple[str, str], ...],
    unresolved: object,
) -> FormalCheckAssessment:
    return FormalCheckAssessment(
        state="pending",
        statuses=statuses,
        failed_checks=(),
        unresolved_checks=tuple(sorted(unresolved)),
    )


def _family_order(
    candidates: Sequence[FormalSubmissionCandidate],
) -> tuple[object, ...]:
    highest_sharpe = max(candidate.sharpe for candidate in candidates)
    highest_fitness = max(
        candidate.fitness
        for candidate in candidates
        if candidate.sharpe == highest_sharpe
    )
    return (
        -highest_sharpe,
        -highest_fitness,
        candidates[0].family_root_task_id,
    )


def _within_family_order(
    candidate: FormalSubmissionCandidate,
) -> tuple[object, ...]:
    return (
        candidate.sharpe,
        -candidate.fitness,
        candidate.finished_at,
        candidate.task_id,
    )


def _validate_candidate(candidate: FormalSubmissionCandidate) -> None:
    if not isinstance(candidate, FormalSubmissionCandidate):
        raise ValueError("formal_submission_candidate_invalid")
    if (
        isinstance(candidate.cycle_number, bool)
        or not isinstance(candidate.cycle_number, int)
        or candidate.cycle_number <= 0
    ):
        raise ValueError("formal_submission_candidate_invalid")
    for value in (
        candidate.task_id,
        candidate.family_root_task_id,
        candidate.platform_alpha_id,
        candidate.formula,
        candidate.normalized_formula,
        candidate.finished_at,
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("formal_submission_candidate_invalid")
