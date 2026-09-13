from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from learning.self_correlation import (
    SelfCorrelationCheckEvidence,
    SelfCorrelationReference,
    select_self_correlation_reference,
)
from persistence.backtests import BacktestSnapshot
from persistence.submission_checks import get_submission_check
from persistence.submissions import (
    list_formal_submission_attempts,
    PlatformSubmittedAlphaRecord,
)
from submission.formal import assess_formal_check_payload
from worldquant.self_correlation import maximum_self_correlation_reference


def load_self_correlation_references(
    connection: sqlite3.Connection,
    *,
    parents: tuple[BacktestSnapshot, ...],
    submitted_alphas: tuple[PlatformSubmittedAlphaRecord, ...],
    account_scope: str,
    observed_at: datetime,
) -> tuple[SelfCorrelationReference, ...]:
    """Read recorded evidence only; cycle planning never requests platform checks."""
    attempts = {
        attempt.task_id: attempt
        for attempt in list_formal_submission_attempts(
            connection, account_scope=account_scope
        )
    }
    selected = []
    for parent in parents:
        if parent.task.account_scope != account_scope:
            continue
        attempt = attempts.get(parent.task.task_id)
        check = get_submission_check(connection, parent.task.task_id)
        observations = []
        if attempt is not None and attempt.check_payload_json is not None and attempt.check_observed_at is not None:
            observations.append((attempt.check_observed_at, attempt.check_payload_json, attempt.status))
        if check is not None and check.payload_json is not None:
            assessment = assess_formal_check_payload(json.loads(check.payload_json))
            observations.append((check.observed_at, check.payload_json,
                                 "ineligible" if assessment.state == "failed" else assessment.state))
        evidence = None
        if observations:
            check_time, payload_json, check_status = max(observations, key=lambda item: datetime.fromisoformat(item[0]))
            payload = json.loads(payload_json)
            reference_id = maximum_self_correlation_reference(payload)
            if reference_id is not None:
                evidence = SelfCorrelationCheckEvidence(
                    task_id=parent.task.task_id,
                    attempt_status=check_status,
                    observed_at=check_time,
                    statuses=assess_formal_check_payload(payload).statuses,
                    reference_id=reference_id,
                    # The protocol reader checked max/value/limit and peer identity.
                    correlation=payload["is"]["selfCorrelated"]["max"],
                )
        reference = select_self_correlation_reference(
            parent,
            evidence,
            submitted_alphas,
            observed_at=observed_at,
        )
        if reference is not None:
            selected.append(reference)
    return tuple(selected)
