from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime

from learning.self_correlation import (
    SelfCorrelationCheckEvidence, SelfCorrelationReference,
    assess_self_correlation_peers, select_self_correlation_reference,
)
from persistence.backtests import BacktestSnapshot
from persistence.submission_checks import list_submission_checks
from persistence.submissions import list_formal_submission_attempts, PlatformSubmittedAlphaRecord
from submission.formal import FormalCheckAssessment, assess_formal_check_payload
from worldquant.self_correlation import self_correlation_peers


@dataclass(frozen=True, slots=True)
class RecordedSubmissionCheck:
    task_id: str
    observed_at: str
    assessment: FormalCheckAssessment
    peers: tuple[tuple[str, float], ...] | None
    repair_allowed: bool


def load_latest_submission_checks(
    connection: sqlite3.Connection, *, account_scope: str, observed_at: datetime,
) -> dict[str, RecordedSubmissionCheck]:
    """Latest observation wins, including errors; equal-time conflicts stay pending."""
    observations = [
        (attempt.task_id, attempt.check_observed_at, attempt.check_payload_json, attempt.status)
        for attempt in list_formal_submission_attempts(connection, account_scope=account_scope)
        if attempt.check_observed_at is not None
    ]
    observations.extend((check.task_id, check.observed_at, check.payload_json, None)
                        for check in list_submission_checks(connection, account_scope=account_scope))
    latest = {}
    for task_id, timestamp, payload_json, attempt_status in observations:
        time = datetime.fromisoformat(timestamp)
        if time.utcoffset() is None:
            raise ValueError("self_correlation_evidence_time_invalid")
        if time > observed_at:
            continue
        previous = latest.get(task_id)
        if previous is not None and time < datetime.fromisoformat(previous.observed_at):
            continue
        payload = json.loads(payload_json) if payload_json is not None else None
        assessment = assess_formal_check_payload(payload)
        current = RecordedSubmissionCheck(task_id, timestamp, assessment, self_correlation_peers(payload),
            assessment.state == "failed" and attempt_status in {None, "ineligible"})
        if (previous is not None and time == datetime.fromisoformat(previous.observed_at)
                and (current.assessment, current.peers, current.repair_allowed)
                != (previous.assessment, previous.peers, previous.repair_allowed)):
            current = RecordedSubmissionCheck(task_id, timestamp, assess_formal_check_payload(None), None, False)
        latest[task_id] = current
    return latest


def load_self_correlation_references(
    connection: sqlite3.Connection, *, parents: tuple[BacktestSnapshot, ...],
    submitted_alphas: tuple[PlatformSubmittedAlphaRecord, ...], account_scope: str,
    observed_at: datetime,
) -> tuple[SelfCorrelationReference, ...]:
    """Plan from recorded blockers only; never request a check or change facts."""
    submitted = tuple(ref for ref in submitted_alphas if ref.account_scope == account_scope
                      and datetime.fromisoformat(ref.observed_at) <= observed_at)
    if not submitted:
        return ()
    checks = load_latest_submission_checks(connection, account_scope=account_scope, observed_at=observed_at)
    selected = []
    for parent in parents:
        if parent.task.account_scope != account_scope:
            continue
        check = checks.get(parent.task.task_id)
        assessment = assess_self_correlation_peers(
            parent.result.sharpe if parent.result is not None else None,
            check.peers if check is not None else None, submitted,
        )
        reference = None
        if check is not None and check.repair_allowed and assessment.state == "failed" and not assessment.unresolved_count:
            for blocker in assessment.blockers:
                evidence = SelfCorrelationCheckEvidence(
                    parent.task.task_id,
                    "ineligible" if check.assessment.state == "failed" else check.assessment.state,
                    check.observed_at, check.assessment.statuses, blocker.reference_id, blocker.correlation,
                )
                reference = select_self_correlation_reference(parent, evidence, submitted, observed_at=observed_at)
                if reference is not None and reference.correlation is not None:
                    reference = replace(reference, blockers=assessment.blockers, required_sharpe=assessment.required_sharpe)
                    break
        if reference is None or reference.correlation is None:
            reference = select_self_correlation_reference(parent, None, submitted, observed_at=observed_at)
        if reference is not None:
            selected.append(reference)
    return tuple(selected)
