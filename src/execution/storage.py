"""Read-only capacity inspection and conservative retention preview."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from persistence.database import read_database
from persistence.storage import PnlRetentionFacts, StorageMeasurements, iter_pnl_retention_facts, measure_storage


@dataclass(frozen=True, slots=True)
class StorageInspection:
    measurements: StorageMeasurements
    database_file_bytes: int
    wal_file_bytes: int


@dataclass(frozen=True, slots=True)
class PrunePreview:
    observed_at: str
    retention_days: int
    candidate_count: int
    candidate_payload_bytes: int
    retained_counts: tuple[tuple[str, int], ...]


def inspect_storage(database_path: str | Path) -> StorageInspection:
    path = Path(database_path)
    with read_database(path) as connection:
        measurements = measure_storage(connection)
        wal = Path(str(path) + "-wal")
        try:
            wal_bytes = wal.stat().st_size
        except FileNotFoundError:
            wal_bytes = 0
        return StorageInspection(measurements, path.stat().st_size, wal_bytes)


def preview_pnl_prune(
    database_path: str | Path, *, retention_days: int = 90, observed_at: datetime | None = None,
) -> PrunePreview:
    if type(retention_days) is not int or retention_days < 30:
        raise ValueError("pnl_retention_days_must_be_at_least_30")
    now = observed_at if observed_at is not None else datetime.now(timezone.utc)
    if now.utcoffset() is None:
        raise ValueError("storage_observation_timezone_required")
    cutoff = now - timedelta(days=retention_days)
    retained = Counter()
    count = payload_bytes = 0
    with read_database(database_path) as connection:
        for facts in iter_pnl_retention_facts(connection):
            reason = _retention_reason(facts, cutoff)
            if reason is not None:
                retained[reason] += 1
            else:
                count += 1
                payload_bytes += facts.payload_bytes
    return PrunePreview(now.isoformat(), retention_days, count, payload_bytes, tuple(sorted(retained.items())))


def _retention_reason(facts: PnlRetentionFacts, cutoff: datetime) -> str | None:
    if facts.submitted:
        return "submitted_reference"
    if facts.recovery_reference:
        return "recovery_reference"
    if facts.protected_task_count:
        return "research_or_submission_dependency"
    if facts.payload_bytes is None:
        return "pending_evidence"
    if not facts.task_count:
        return "unlinked_evidence"
    if facts.completed_count != facts.task_count or facts.failed_check_count != facts.task_count:
        return "no_definite_rejection"
    if facts.unknown_task_time or facts.latest_task_observation is None:
        return "unknown_timestamp"
    try:
        timestamps = [datetime.fromisoformat(value) for value in (facts.observed_at, facts.latest_task_observation)]
    except ValueError:
        return "unknown_timestamp"
    if any(value.utcoffset() is None for value in timestamps):
        return "unknown_timestamp"
    if any(value >= cutoff for value in timestamps):
        return "retention_window"
    return None
