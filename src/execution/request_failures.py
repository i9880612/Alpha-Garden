from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from execution.runs import (
    fail_automated_run,
    record_automated_request_failure,
)
from persistence.runs import AutomatedRunRecord
from execution.pnl import PnlAccountRequestError
from worldquant.client import WorldQuantRequestError


# Retryable faults (network drops, 5xx, 429) back off 5, 10, 20 ... seconds up to 5 minutes.
# A finite run stops only after max_request_failures consecutive faults, so the limit
# maps to an outage length: 30 consecutive faults wait about 2 hours before stopping.
_FIRST_RETRY_DELAY_SECONDS = 5.0
_MAX_RETRY_DELAY_SECONDS = 300.0


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

    local_retry_delay = request_retry_delay_seconds(run.request_failure_count)
    if (isinstance(error, PnlAccountRequestError) and error.status_code == 429
            and error.retry_after_seconds is None):
        local_retry_delay = max(60.0, local_retry_delay)
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


def request_retry_delay_seconds(previous_failures: int) -> float:
    """Local wait after the (previous_failures + 1)-th consecutive retryable fault."""
    # Clamp before exponentiation: an unattended outage can last indefinitely.
    return min(
        _MAX_RETRY_DELAY_SECONDS,
        _FIRST_RETRY_DELAY_SECONDS * float(2 ** min(previous_failures, 10)),
    )


def request_outage_tolerance_seconds(max_request_failures: int) -> float:
    """Minimum outage a finite run waits through before the failure limit stops it."""
    return sum(request_retry_delay_seconds(count) for count in range(max_request_failures - 1))


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
