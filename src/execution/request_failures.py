from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from execution.runs import (
    fail_automated_run,
    record_automated_request_failure,
    uses_continuous_recovery,
)
from persistence.runs import AutomatedRunRecord
from worldquant.client import WorldQuantRequestError


_MAX_RETRY_DELAY_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class AutomatedRequestFailure:
    run: AutomatedRunRecord
    retry_after_seconds: float | None


def handle_automated_request_failure(
    database_path: str | Path,
    run: AutomatedRunRecord,
    error: WorldQuantRequestError,
    *,
    observed_at: str,
    non_retryable_reason: str,
) -> AutomatedRequestFailure:
    error_code = error.code
    if error.transport_error_type is not None:
        error_code = f"{error.code}:{error.transport_error_type}"
    if not error.retryable:
        stopped = fail_automated_run(
            database_path,
            run.run_id,
            failed_at=observed_at,
            reason=f"{non_retryable_reason}:{error.code}",
            request_failure_code=error_code,
            request_status_code=error.status_code,
            request_retry_after_seconds=error.retry_after_seconds,
        )
        return AutomatedRequestFailure(stopped, None)

    # Clamp before exponentiation: an unattended outage can last indefinitely.
    local_retry_delay = min(
        _MAX_RETRY_DELAY_SECONDS,
        float(2 ** min(run.request_failure_count, 6)),
    )
    if uses_continuous_recovery(run):
        local_retry_delay = min(120.0, 3.0 * local_retry_delay)
    retry_delay = max(
        local_retry_delay,
        error.retry_after_seconds or 0.0,
    )
    updated = record_automated_request_failure(
        database_path,
        run.run_id,
        observed_at=observed_at,
        retry_after_seconds=retry_delay,
        error_code=error_code,
        status_code=error.status_code,
    )
    return AutomatedRequestFailure(
        run=updated,
        retry_after_seconds=(
            None if updated.status == "failed" else retry_delay
        ),
    )


def remaining_automated_request_retry_seconds(
    run: AutomatedRunRecord,
    observed_at: datetime,
) -> float | None:
    if run.retry_not_before is None:
        return None
    retry_at = _timestamp(run.retry_not_before)
    remaining = (retry_at - observed_at).total_seconds()
    return remaining if remaining > 0 else None


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("automated_run_observed_at_invalid") from exc
    if parsed.utcoffset() is None:
        raise ValueError("automated_run_observed_at_invalid")
    return parsed
