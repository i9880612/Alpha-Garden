from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from evaluation.backtests import evaluate_backtest
from persistence.backtests import (
    BacktestCheckRecord,
    BacktestMutationRecord,
    BacktestSnapshot,
)


_SELF_CORRELATION = "SELF_CORRELATION"


@dataclass(frozen=True, slots=True)
class LearningEvidenceRecord:
    task_id: str
    account_scope: str
    formula: str
    formula_fingerprint: str
    settings_key: str
    finished_at: str
    check_details_captured: bool
    checks: tuple[BacktestCheckRecord, ...]
    outcome: str | None
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    failure_categories: tuple[str, ...]
    pending_checks: tuple[str, ...]
    missing_checks: tuple[str, ...]
    unexpected_checks: tuple[str, ...]
    check_set_complete: bool
    non_sc_check_set_complete: bool
    sharpe: float
    fitness: float
    turnover: float
    returns: float
    drawdown: float
    margin: float
    book_size: float | None
    pnl: float | None


@dataclass(frozen=True, slots=True)
class SettingsEvidenceSummary:
    account_scope: str
    settings_key: str
    total_count: int
    labeled_count: int
    passed_count: int
    failed_count: int
    pending_count: int
    comparison_available: bool
    outcome_contrast_available: bool


@dataclass(frozen=True, slots=True)
class LearningEvidenceSet:
    records: tuple[LearningEvidenceRecord, ...]
    settings: tuple[SettingsEvidenceSummary, ...]


@dataclass(frozen=True, slots=True)
class MutationLearningEvidenceRecord:
    child_task_id: str
    parent_task_id: str
    child_formula: str
    parent_formula: str
    parent_account_scope: str
    child_account_scope: str
    parent_settings_key: str
    child_settings_key: str
    comparison_available: bool
    action: str
    location: str
    before: str
    after: str
    parent_outcome: str | None
    child_outcome: str | None
    repaired_checks: tuple[str, ...]
    remaining_failed_checks: tuple[str, ...]
    introduced_failed_checks: tuple[str, ...]
    introduced_pending_checks: tuple[str, ...]
    child_missing_checks: tuple[str, ...]
    child_unexpected_checks: tuple[str, ...]
    child_non_sc_check_set_complete: bool
    parent_pending_checks: tuple[str, ...]
    child_pending_checks: tuple[str, ...]
    sharpe_delta: float | None
    fitness_delta: float | None
    turnover_delta: float | None
    returns_delta: float | None
    drawdown_delta: float | None
    margin_delta: float | None


@dataclass(frozen=True, slots=True)
class MutationLearningEvidenceSet:
    records: tuple[MutationLearningEvidenceRecord, ...]


def learning_settings_key(settings_json: str) -> str:
    if not isinstance(settings_json, str) or not settings_json.strip():
        raise ValueError("learning_settings_json_missing")
    return sha256(settings_json.encode("utf-8")).hexdigest()


def _learning_outcome(
    failed_checks: tuple[str, ...],
    pending_checks: tuple[str, ...],
    non_sc_check_set_complete: bool,
) -> str | None:
    quality_failed = set(failed_checks) - {_SELF_CORRELATION}
    quality_pending = set(pending_checks) - {_SELF_CORRELATION}
    if quality_failed:
        return "failed"
    if quality_pending or not non_sc_check_set_complete:
        return None
    return "passed"


def build_learning_evidence(
    snapshots: Iterable[BacktestSnapshot],
) -> LearningEvidenceSet:
    records: list[LearningEvidenceRecord] = []
    task_ids: set[str] = set()

    for snapshot in snapshots:
        evaluation = evaluate_backtest(snapshot)
        task = snapshot.task
        result = snapshot.result
        if result is None:
            raise ValueError("learning_completed_backtest_required")
        if task.task_id in task_ids:
            raise ValueError("learning_task_duplicated")
        if task.finished_at is None:
            raise ValueError("learning_finished_at_missing")
        task_ids.add(task.task_id)

        records.append(
            LearningEvidenceRecord(
                task_id=task.task_id,
                account_scope=task.account_scope,
                formula=task.formula,
                formula_fingerprint=task.formula_fingerprint,
                settings_key=learning_settings_key(task.settings_json),
                finished_at=task.finished_at,
                check_details_captured=result.check_details_captured,
                checks=result.checks,
                outcome=_learning_outcome(
                    evaluation.failed_checks,
                    evaluation.pending_checks,
                    evaluation.non_sc_check_set_complete,
                ),
                passed_checks=evaluation.passed_checks,
                failed_checks=evaluation.failed_checks,
                failure_categories=tuple(
                    sorted(
                        {
                            failure.category
                            for failure in evaluation.failures
                            if failure.check_name != _SELF_CORRELATION
                        }
                    )
                ),
                pending_checks=evaluation.pending_checks,
                missing_checks=evaluation.missing_checks,
                unexpected_checks=evaluation.unexpected_checks,
                check_set_complete=evaluation.check_set_complete,
                non_sc_check_set_complete=(
                    evaluation.non_sc_check_set_complete
                ),
                sharpe=result.sharpe,
                fitness=result.fitness,
                turnover=result.turnover,
                returns=result.returns,
                drawdown=result.drawdown,
                margin=result.margin,
                book_size=result.book_size,
                pnl=result.pnl,
            )
        )

    ordered = tuple(sorted(records, key=lambda item: (item.finished_at, item.task_id)))
    grouped: defaultdict[
        tuple[str, str], list[LearningEvidenceRecord]
    ] = defaultdict(list)
    for record in ordered:
        grouped[(record.account_scope, record.settings_key)].append(record)

    summaries: list[SettingsEvidenceSummary] = []
    for (account_scope, settings_key), group in sorted(grouped.items()):
        passed_count = sum(record.outcome == "passed" for record in group)
        failed_count = sum(record.outcome == "failed" for record in group)
        labeled_count = passed_count + failed_count
        summaries.append(
            SettingsEvidenceSummary(
                account_scope=account_scope,
                settings_key=settings_key,
                total_count=len(group),
                labeled_count=labeled_count,
                passed_count=passed_count,
                failed_count=failed_count,
                pending_count=len(group) - labeled_count,
                comparison_available=labeled_count >= 2,
                outcome_contrast_available=passed_count > 0 and failed_count > 0,
            )
        )
    return LearningEvidenceSet(records=ordered, settings=tuple(summaries))


def build_mutation_learning_evidence(
    evidence: LearningEvidenceSet,
    mutations: Iterable[BacktestMutationRecord],
) -> MutationLearningEvidenceSet:
    records_by_task = {record.task_id: record for record in evidence.records}
    built: list[MutationLearningEvidenceRecord] = []
    child_task_ids: set[str] = set()
    for mutation in mutations:
        if mutation.child_task_id in child_task_ids:
            raise ValueError("learning_mutation_child_duplicated")
        child_task_ids.add(mutation.child_task_id)
        child = records_by_task.get(mutation.child_task_id)
        parent = records_by_task.get(mutation.parent_task_id)
        if child is None or parent is None:
            continue
        comparison_available = (
            child.account_scope == parent.account_scope
            and child.settings_key == parent.settings_key
        )
        parent_failed = set(parent.failed_checks)
        child_failed = set(child.failed_checks)
        child_passed = set(child.passed_checks)
        parent_pending = set(parent.pending_checks)
        child_pending = set(child.pending_checks)
        parent_observed = set(parent.passed_checks) | parent_failed | parent_pending
        child_observed = child_passed | child_failed | child_pending
        built.append(
            MutationLearningEvidenceRecord(
                child_task_id=child.task_id,
                parent_task_id=parent.task_id,
                child_formula=child.formula,
                parent_formula=parent.formula,
                parent_account_scope=parent.account_scope,
                child_account_scope=child.account_scope,
                parent_settings_key=parent.settings_key,
                child_settings_key=child.settings_key,
                comparison_available=comparison_available,
                action=mutation.action,
                location=mutation.location,
                before=mutation.before,
                after=mutation.after,
                parent_outcome=parent.outcome,
                child_outcome=child.outcome,
                repaired_checks=(
                    tuple(sorted(parent_failed & child_passed))
                    if comparison_available
                    else ()
                ),
                remaining_failed_checks=(
                    tuple(sorted(parent_failed & child_failed))
                    if comparison_available
                    else ()
                ),
                introduced_failed_checks=(
                    tuple(sorted(child_failed - parent_failed))
                    if comparison_available
                    else ()
                ),
                introduced_pending_checks=(
                    tuple(sorted(child_pending - parent_pending))
                    if comparison_available
                    else ()
                ),
                child_missing_checks=(
                    tuple(
                        sorted(
                            set(child.missing_checks)
                            | (parent_observed - child_observed)
                        )
                    )
                    if comparison_available
                    else ()
                ),
                child_unexpected_checks=(
                    child.unexpected_checks if comparison_available else ()
                ),
                child_non_sc_check_set_complete=(
                    child.non_sc_check_set_complete
                    if comparison_available
                    else False
                ),
                parent_pending_checks=parent.pending_checks,
                child_pending_checks=child.pending_checks,
                sharpe_delta=(
                    child.sharpe - parent.sharpe
                    if comparison_available
                    else None
                ),
                fitness_delta=(
                    child.fitness - parent.fitness
                    if comparison_available
                    else None
                ),
                turnover_delta=(
                    child.turnover - parent.turnover
                    if comparison_available
                    else None
                ),
                returns_delta=(
                    child.returns - parent.returns
                    if comparison_available
                    else None
                ),
                drawdown_delta=(
                    child.drawdown - parent.drawdown
                    if comparison_available
                    else None
                ),
                margin_delta=(
                    child.margin - parent.margin
                    if comparison_available
                    else None
                ),
            )
        )
    return MutationLearningEvidenceSet(
        records=tuple(
            sorted(
                built,
                key=lambda item: (item.child_task_id, item.parent_task_id),
            )
        )
    )
