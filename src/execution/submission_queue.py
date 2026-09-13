from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass

from execution.qualified_archive import consume_qualified_alpha_archive
from execution.qualified_candidates import load_exhausted_qualified_candidates

from persistence.backtests import (
    BacktestMutationRecord,
    BacktestSnapshot,
    get_backtest_task,
    list_backtest_mutations,
)
from persistence.runs import get_automated_run_backtest_by_task
from persistence.submission_checks import get_submission_check
from persistence.submission_queue import (
    FormalSubmissionQueueRecord,
    create_formal_submission_queue_item,
    delete_formal_submission_queue_item,
    list_formal_submission_queue,
)
from persistence.submissions import (
    FORMAL_SUBMISSION_ACTIVE_STATUSES,
    FormalSubmissionAttemptRecord,
    create_formal_submission_attempt,
    list_formal_submission_attempts,
    list_platform_submitted_alphas,
    normalize_submitted_formula,
)
from submission.formal import (
    BELOW_TARGET_GRADES,
    FormalSubmissionCandidate,
    submission_queue_eligible,
    assess_formal_check_payload,
    select_next_family_candidate,
)


@dataclass(frozen=True, slots=True)
class SubmissionQueueSynchronization:
    enqueued_task_ids: tuple[str, ...]
    replaced_task_ids: tuple[str, ...]


def synchronize_submission_queue(
    connection: sqlite3.Connection,
    *,
    candidate_task_ids: tuple[str, ...],
    enqueued_at: str,
) -> SubmissionQueueSynchronization:
    requested = _candidate_task_ids(candidate_task_ids)
    if not requested:
        return SubmissionQueueSynchronization((), ())
    parent_by_child = _parent_map(list_backtest_mutations(connection))
    submitted_formulas_by_account: dict[str, set[str]] = {}
    attempts = list_formal_submission_attempts(connection)
    consumed_formulas: set[tuple[str, str]] = set()
    for attempt in attempts:
        snapshot = _required_snapshot(connection, attempt.task_id)
        consumed_formulas.add(
            (
                snapshot.task.account_scope,
                normalize_submitted_formula(snapshot.task.formula),
            )
        )

    existing_queue = list_formal_submission_queue(connection)
    queued_by_formula = {
        (record.account_scope, record.normalized_formula): record
        for record in existing_queue
    }
    selected_by_formula: dict[
        tuple[str, str], tuple[FormalSubmissionQueueRecord, BacktestSnapshot]
    ] = {}
    for task_id in requested:
        snapshot = _required_snapshot(connection, task_id)
        if not submission_queue_eligible(snapshot):
            continue
        check = get_submission_check(connection, task_id)
        if (check is None or check.payload_json is None
                or assess_formal_check_payload(json.loads(check.payload_json)).state != "passed"):
            continue
        assert snapshot.result is not None
        link = get_automated_run_backtest_by_task(connection, task_id)
        if link is None:
            raise ValueError("formal_submission_queue_run_link_missing")
        normalized_formula = normalize_submitted_formula(snapshot.task.formula)
        formula_identity = (snapshot.task.account_scope, normalized_formula)
        if formula_identity in consumed_formulas:
            continue
        submitted = submitted_formulas_by_account.get(snapshot.task.account_scope)
        if submitted is None:
            submitted = {
                record.normalized_formula
                for record in list_platform_submitted_alphas(
                    connection,
                    account_scope=snapshot.task.account_scope,
                )
            }
            submitted_formulas_by_account[snapshot.task.account_scope] = submitted
        if normalized_formula in submitted:
            continue
        record = FormalSubmissionQueueRecord(
            task_id=task_id,
            account_scope=snapshot.task.account_scope,
            run_id=link.run_id,
            cycle_number=link.cycle_number,
            family_root_task_id=_family_root_task_id(task_id, parent_by_child),
            normalized_formula=normalized_formula,
            enqueued_at=enqueued_at,
        )
        current = selected_by_formula.get(formula_identity)
        if current is None or _snapshot_quality(snapshot) > _snapshot_quality(
            current[1]
        ):
            selected_by_formula[formula_identity] = (record, snapshot)

    enqueued: list[str] = []
    replaced: list[str] = []
    for formula_identity, (record, snapshot) in sorted(
        selected_by_formula.items(),
        key=lambda item: item[1][0].task_id,
    ):
        existing = queued_by_formula.get(formula_identity)
        if existing is not None and existing.task_id != record.task_id:
            existing_snapshot = _required_snapshot(connection, existing.task_id)
            if _snapshot_quality(existing_snapshot) >= _snapshot_quality(snapshot):
                continue
            delete_formal_submission_queue_item(connection, existing.task_id)
            replaced.append(existing.task_id)
        created = create_formal_submission_queue_item(connection, record)
        if existing is None or existing.task_id != created.task_id:
            enqueued.append(created.task_id)
    return SubmissionQueueSynchronization(tuple(enqueued), tuple(replaced))


def list_submission_candidates(
    connection: sqlite3.Connection, *, account_scope: str,
    source: str = "queue", run_id: str | None = None,
    grade: str | None = None,
) -> tuple[FormalSubmissionQueueRecord, ...]:
    if grade is not None and (source != "qualified_archive" or grade not in BELOW_TARGET_GRADES):
        raise ValueError("formal_submission_grade_filter_invalid")
    if source == "queue":
        return list_formal_submission_queue(connection, account_scope=account_scope, run_id=run_id)
    if source != "qualified_archive" or run_id is not None:
        raise ValueError("formal_submission_source_invalid")
    parents = _parent_map(list_backtest_mutations(connection))
    candidates = []
    for snapshot in load_exhausted_qualified_candidates(connection, account_scope=account_scope):
        if grade is not None and snapshot.result.grade != grade:
            continue
        task = snapshot.task
        link = get_automated_run_backtest_by_task(connection, task.task_id)
        if link is None:
            raise ValueError("formal_submission_queue_run_link_missing")
        assert task.finished_at is not None
        candidates.append(FormalSubmissionQueueRecord(
            task_id=task.task_id, account_scope=account_scope, run_id=link.run_id,
            cycle_number=link.cycle_number, family_root_task_id=_family_root_task_id(task.task_id, parents),
            normalized_formula=normalize_submitted_formula(task.formula), enqueued_at=task.finished_at,
        ))
    return tuple(candidates)


def claim_next_submission_queue_item(
    connection: sqlite3.Connection,
    *,
    account_scope: str,
    observed_at: str,
    submission_mode: str,
    run_id: str | None = None,
    allowed_task_ids: frozenset[str] | None = None,
    source: str = "queue",
) -> FormalSubmissionAttemptRecord | None:
    if source == "qualified_archive" and submission_mode != "manual":
        raise ValueError("formal_submission_source_invalid")
    queue = list_submission_candidates(
        connection,
        account_scope=account_scope,
        run_id=run_id,
        source=source,
    )
    if allowed_task_ids is not None:
        if not isinstance(allowed_task_ids, frozenset) or any(
            not isinstance(task_id, str) or not task_id.strip()
            for task_id in allowed_task_ids
        ):
            raise ValueError("formal_submission_authority_scope_invalid")
        queue = tuple(record for record in queue if record.task_id in allowed_task_ids)
    if not queue:
        return None
    submitted_formulas = {
        record.normalized_formula
        for record in list_platform_submitted_alphas(
            connection,
            account_scope=account_scope,
        )
    }
    attempts = list_formal_submission_attempts(connection)
    attempt_snapshots = tuple(
        (attempt, _required_snapshot(connection, attempt.task_id))
        for attempt in attempts
    )
    active = tuple(
        attempt
        for attempt, snapshot in attempt_snapshots
        if attempt.status in FORMAL_SUBMISSION_ACTIVE_STATUSES
        and snapshot.task.account_scope == account_scope
    )
    if active:
        raise ValueError("formal_submission_active_attempt_exists")
    consumed_formulas = {
        normalize_submitted_formula(snapshot.task.formula)
        for _attempt, snapshot in attempt_snapshots
        if snapshot.task.account_scope == account_scope
    }
    attempted_task_ids = {attempt.task_id for attempt in attempts}

    candidates: list[FormalSubmissionCandidate] = []
    queued_by_task_id = {record.task_id: record for record in queue}
    for record in queue:
        snapshot = _required_snapshot(connection, record.task_id)
        if (
            record.task_id in attempted_task_ids
            or record.normalized_formula in consumed_formulas
        ):
            if source == "queue":
                delete_formal_submission_queue_item(connection, record.task_id)
            continue
        if source == "queue" and not submission_queue_eligible(snapshot):
            delete_formal_submission_queue_item(connection, record.task_id)
            continue
        assert snapshot.result is not None
        assert snapshot.task.platform_alpha_id is not None
        assert snapshot.task.finished_at is not None
        if record.normalized_formula in submitted_formulas:
            if source == "queue":
                delete_formal_submission_queue_item(connection, record.task_id)
            continue
        candidates.append(
            FormalSubmissionCandidate(
                task_id=record.task_id,
                cycle_number=record.cycle_number,
                family_root_task_id=record.family_root_task_id,
                platform_alpha_id=snapshot.task.platform_alpha_id,
                formula=snapshot.task.formula,
                normalized_formula=record.normalized_formula,
                sharpe=snapshot.result.sharpe,
                fitness=snapshot.result.fitness,
                finished_at=snapshot.task.finished_at,
            )
        )
    candidate = select_next_family_candidate(candidates)
    if candidate is None:
        return None
    queue_record = queued_by_task_id[candidate.task_id]
    attempt = create_formal_submission_attempt(
        connection,
        FormalSubmissionAttemptRecord(
            task_id=candidate.task_id,
            run_id=queue_record.run_id,
            cycle_number=queue_record.cycle_number,
            family_root_task_id=queue_record.family_root_task_id,
            submission_mode=submission_mode,
            source=source,
            status="detail_pending",
            check_attempt_count=0,
            check_payload_json=None,
            check_observed_at=None,
            retry_not_before=None,
            submission_claimed_at=None,
            submit_http_status=None,
            submit_response_json=None,
            confirmation_observed_at=None,
            failure_code=None,
            created_at=observed_at,
            updated_at=observed_at,
        ),
    )
    if source == "queue":
        delete_formal_submission_queue_item(connection, queue_record.task_id)
    consume_qualified_alpha_archive(connection, task_id=candidate.task_id)
    return attempt


def _candidate_task_ids(candidate_task_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(candidate_task_ids, tuple) or any(
        not isinstance(task_id, str) or not task_id.strip()
        for task_id in candidate_task_ids
    ):
        raise ValueError("formal_submission_queue_candidate_tasks_invalid")
    if len(set(candidate_task_ids)) != len(candidate_task_ids):
        raise ValueError("formal_submission_queue_candidate_task_duplicate")
    return candidate_task_ids


def _required_snapshot(
    connection: sqlite3.Connection,
    task_id: str,
) -> BacktestSnapshot:
    snapshot = get_backtest_task(connection, task_id)
    if snapshot is None:
        raise ValueError("formal_submission_queue_task_missing")
    return snapshot


def _snapshot_quality(snapshot: BacktestSnapshot) -> tuple[object, ...]:
    if snapshot.result is None or snapshot.task.finished_at is None:
        raise ValueError("formal_submission_queue_result_missing")
    return (
        snapshot.result.sharpe,
        snapshot.result.fitness,
        snapshot.task.finished_at,
        snapshot.task.task_id,
    )


def _parent_map(
    mutations: tuple[BacktestMutationRecord, ...],
) -> dict[str, str]:
    parent_by_child: dict[str, str] = {}
    for mutation in mutations:
        if mutation.child_task_id in parent_by_child:
            raise ValueError("formal_submission_lineage_child_duplicate")
        parent_by_child[mutation.child_task_id] = mutation.parent_task_id
    return parent_by_child


def _family_root_task_id(task_id: str, parent_by_child: dict[str, str]) -> str:
    current = task_id
    visited: set[str] = set()
    while current in parent_by_child:
        if current in visited:
            raise ValueError("formal_submission_lineage_cycle")
        visited.add(current)
        current = parent_by_child[current]
    return current
