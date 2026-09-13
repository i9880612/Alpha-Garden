from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class PlatformSubmittedAlphaRecord:
    account_scope: str
    platform_alpha_id: str
    formula: str
    status: str
    date_submitted: str
    hidden: bool
    raw_payload: Mapping[str, Any]
    observed_at: str

    @property
    def normalized_formula(self) -> str:
        return normalize_submitted_formula(self.formula)


FORMAL_SUBMISSION_ACTIVE_STATUSES = frozenset(
    {
        "detail_pending",
        "check_pending",
        "ready",
        "submitting",
        "confirmation_pending",
        "submission_unknown",
    }
)
FORMAL_SUBMISSION_UNRESOLVED_STATUSES = frozenset(
    {"submitting", "confirmation_pending", "submission_unknown"}
)
FORMAL_SUBMISSION_TERMINAL_STATUSES = frozenset({"ineligible", "submitted", "failed"})
FORMAL_SUBMISSION_MODES = frozenset({"automatic", "manual"})


@dataclass(frozen=True, slots=True)
class FormalSubmissionAttemptRecord:
    task_id: str
    run_id: str
    cycle_number: int
    family_root_task_id: str
    submission_mode: str
    status: str
    check_attempt_count: int
    check_payload_json: str | None
    check_observed_at: str | None
    retry_not_before: str | None
    submission_claimed_at: str | None
    submit_http_status: int | None
    submit_response_json: str | None
    confirmation_observed_at: str | None
    failure_code: str | None
    created_at: str
    updated_at: str
    source: str = "queue"


def initialize_submission_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS platform_submitted_alphas (
            account_scope TEXT NOT NULL CHECK (
                length(trim(account_scope)) > 0
            ),
            platform_alpha_id TEXT NOT NULL CHECK (
                length(trim(platform_alpha_id)) > 0
            ),
            formula TEXT NOT NULL CHECK (length(trim(formula)) > 0),
            normalized_formula TEXT NOT NULL CHECK (
                length(normalized_formula) > 0
            ),
            status TEXT NOT NULL CHECK (
                length(trim(status)) > 0 AND upper(status) != 'UNSUBMITTED'
            ),
            date_submitted TEXT NOT NULL CHECK (
                length(trim(date_submitted)) > 0
            ),
            hidden INTEGER NOT NULL CHECK (hidden IN (0, 1)),
            raw_payload_json TEXT NOT NULL,
            observed_at TEXT NOT NULL CHECK (
                length(trim(observed_at)) > 0
            ),
            PRIMARY KEY (account_scope, platform_alpha_id)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_platform_submitted_alphas_formula
        ON platform_submitted_alphas (account_scope, normalized_formula)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS formal_submission_attempts (
            task_id TEXT PRIMARY KEY REFERENCES backtest_tasks(task_id),
            run_id TEXT NOT NULL REFERENCES automated_runs(run_id),
            cycle_number INTEGER NOT NULL CHECK (cycle_number > 0),
            family_root_task_id TEXT NOT NULL REFERENCES backtest_tasks(task_id),
            submission_mode TEXT NOT NULL CHECK (
                submission_mode IN ('automatic', 'manual')
            ),
            status TEXT NOT NULL CHECK (
                status IN (
                    'detail_pending', 'check_pending', 'ready', 'ineligible', 'submitting',
                    'confirmation_pending', 'submission_unknown',
                    'submitted', 'failed'
                )
            ),
            check_attempt_count INTEGER NOT NULL CHECK (check_attempt_count >= 0),
            check_payload_json TEXT,
            check_observed_at TEXT,
            retry_not_before TEXT,
            submission_claimed_at TEXT,
            submit_http_status INTEGER CHECK (
                submit_http_status IS NULL
                OR submit_http_status BETWEEN 100 AND 599
            ),
            submit_response_json TEXT,
            confirmation_observed_at TEXT,
            failure_code TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'queue' CHECK (
                source = 'queue' OR (source = 'qualified_archive' AND submission_mode = 'manual')
            ),
            CHECK (
                (check_payload_json IS NULL) = (check_observed_at IS NULL)
            ),
            CHECK (
                submission_claimed_at IS NULL
                OR status IN (
                    'submitting', 'confirmation_pending',
                    'submission_unknown', 'submitted', 'failed'
                )
            )
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_formal_submission_attempts_run
        ON formal_submission_attempts (run_id, cycle_number, status, task_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_formal_submission_attempts_family
        ON formal_submission_attempts (family_root_task_id, status, task_id)
        """
    )


def record_platform_submitted_alphas(
    connection: sqlite3.Connection,
    records: Sequence[PlatformSubmittedAlphaRecord],
) -> None:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("platform_submitted_alphas_invalid")
    prepared: list[tuple[PlatformSubmittedAlphaRecord, str]] = []
    identities: set[tuple[str, str]] = set()
    for record in records:
        raw_payload_json = _validate_record(record)
        identity = (record.account_scope, record.platform_alpha_id)
        if identity in identities:
            raise ValueError("platform_submitted_alpha_duplicate")
        identities.add(identity)
        prepared.append((record, raw_payload_json))

    for record, raw_payload_json in prepared:
        write = connection.execute(
            """
            INSERT INTO platform_submitted_alphas (
                account_scope, platform_alpha_id, formula, normalized_formula,
                status, date_submitted, hidden, raw_payload_json, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (account_scope, platform_alpha_id) DO UPDATE SET
                formula = excluded.formula,
                normalized_formula = excluded.normalized_formula,
                status = excluded.status,
                hidden = excluded.hidden,
                raw_payload_json = excluded.raw_payload_json,
                observed_at = excluded.observed_at
            WHERE
                platform_submitted_alphas.normalized_formula
                    = excluded.normalized_formula
                AND platform_submitted_alphas.date_submitted
                    = excluded.date_submitted
            """,
            (
                record.account_scope,
                record.platform_alpha_id,
                record.formula,
                record.normalized_formula,
                record.status,
                record.date_submitted,
                int(record.hidden),
                raw_payload_json,
                record.observed_at,
            ),
        )
        if write.rowcount != 1:
            raise ValueError("platform_submitted_alpha_identity_conflict")


def list_platform_submitted_alphas(
    connection: sqlite3.Connection,
    *,
    account_scope: str,
) -> tuple[PlatformSubmittedAlphaRecord, ...]:
    _require_clean_text(
        account_scope,
        "platform_submitted_alpha_account_scope_invalid",
    )
    rows = connection.execute(
        """
        SELECT *
        FROM platform_submitted_alphas
        WHERE account_scope = ?
        ORDER BY date_submitted, platform_alpha_id
        """,
        (account_scope,),
    ).fetchall()
    return tuple(_record_from_row(row) for row in rows)


def create_formal_submission_attempt(
    connection: sqlite3.Connection,
    record: FormalSubmissionAttemptRecord,
) -> FormalSubmissionAttemptRecord:
    _validate_attempt(record)
    _validate_attempt_links(connection, record)
    existing = get_formal_submission_attempt(connection, record.task_id)
    if existing is not None:
        if existing != record:
            raise ValueError("formal_submission_attempt_identity_conflict")
        return existing
    connection.execute(
        """
        INSERT INTO formal_submission_attempts (
            task_id, run_id, cycle_number, family_root_task_id,
            submission_mode, status,
            check_attempt_count, check_payload_json, check_observed_at,
            retry_not_before, submission_claimed_at, submit_http_status,
            submit_response_json, confirmation_observed_at, failure_code,
            created_at, updated_at, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _attempt_values(record),
    )
    return record


def replace_formal_submission_attempt(
    connection: sqlite3.Connection,
    record: FormalSubmissionAttemptRecord,
    *,
    expected_status: str,
    expected_updated_at: str,
) -> FormalSubmissionAttemptRecord:
    _validate_attempt(record)
    existing = get_formal_submission_attempt(connection, record.task_id)
    if existing is None:
        raise ValueError("formal_submission_attempt_missing")
    if _attempt_identity(existing) != _attempt_identity(record):
        raise ValueError("formal_submission_attempt_identity_conflict")
    if existing == record:
        return existing
    cursor = connection.execute(
        """
        UPDATE formal_submission_attempts SET
            status = ?, check_attempt_count = ?, check_payload_json = ?,
            check_observed_at = ?, retry_not_before = ?,
            submission_claimed_at = ?, submit_http_status = ?,
            submit_response_json = ?, confirmation_observed_at = ?,
            failure_code = ?, updated_at = ?
        WHERE task_id = ? AND status = ? AND updated_at = ?
        """,
        (
            record.status,
            record.check_attempt_count,
            record.check_payload_json,
            record.check_observed_at,
            record.retry_not_before,
            record.submission_claimed_at,
            record.submit_http_status,
            record.submit_response_json,
            record.confirmation_observed_at,
            record.failure_code,
            record.updated_at,
            record.task_id,
            expected_status,
            expected_updated_at,
        ),
    )
    if cursor.rowcount != 1:
        raise ValueError("formal_submission_attempt_transition_conflict")
    return record


def get_formal_submission_attempt(
    connection: sqlite3.Connection,
    task_id: str,
) -> FormalSubmissionAttemptRecord | None:
    row = connection.execute(
        "SELECT * FROM formal_submission_attempts WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    return _attempt_from_row(row) if row is not None else None


def list_formal_submission_attempts(
    connection: sqlite3.Connection,
    *,
    run_id: str | None = None,
    account_scope: str | None = None,
) -> tuple[FormalSubmissionAttemptRecord, ...]:
    clauses: list[str] = []
    parameters: list[str] = []
    if run_id is not None:
        _require_clean_text(run_id, "formal_submission_run_id_invalid")
        clauses.append("attempts.run_id = ?")
        parameters.append(run_id)
    if account_scope is not None:
        _require_clean_text(
            account_scope,
            "formal_submission_account_scope_invalid",
        )
        clauses.append("tasks.account_scope = ?")
        parameters.append(account_scope)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = connection.execute(
        f"""
        SELECT attempts.*
        FROM formal_submission_attempts AS attempts
        JOIN backtest_tasks AS tasks ON tasks.task_id = attempts.task_id
        {where}
        ORDER BY attempts.cycle_number, attempts.created_at, attempts.task_id
        """,
        parameters,
    ).fetchall()
    return tuple(_attempt_from_row(row) for row in rows)


def account_has_active_formal_submission(
    connection: sqlite3.Connection,
    account_scope: str,
) -> bool:
    _require_clean_text(
        account_scope,
        "formal_submission_account_scope_invalid",
    )
    placeholders = ", ".join("?" for _ in FORMAL_SUBMISSION_ACTIVE_STATUSES)
    row = connection.execute(
        f"""
        SELECT 1
        FROM formal_submission_attempts AS attempts
        JOIN backtest_tasks AS tasks ON tasks.task_id = attempts.task_id
        WHERE tasks.account_scope = ?
          AND attempts.status IN ({placeholders})
        LIMIT 1
        """,
        (account_scope, *sorted(FORMAL_SUBMISSION_ACTIVE_STATUSES)),
    ).fetchone()
    return row is not None


def automated_run_has_unresolved_formal_submission(
    connection: sqlite3.Connection,
    run_id: str,
) -> bool:
    _require_clean_text(run_id, "formal_submission_run_id_invalid")
    row = connection.execute(
        """
        SELECT 1
        FROM formal_submission_attempts
        WHERE run_id = ?
          AND submission_mode = 'automatic'
          AND status IN (
              'submitting', 'confirmation_pending', 'submission_unknown'
          )
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    return row is not None


def automated_run_has_active_formal_submission(
    connection: sqlite3.Connection,
    run_id: str,
) -> bool:
    _require_clean_text(run_id, "formal_submission_run_id_invalid")
    placeholders = ", ".join("?" for _ in FORMAL_SUBMISSION_ACTIVE_STATUSES)
    row = connection.execute(
        f"""
        SELECT 1
        FROM formal_submission_attempts
        WHERE run_id = ?
          AND submission_mode = 'automatic'
          AND status IN ({placeholders})
        LIMIT 1
        """,
        (run_id, *sorted(FORMAL_SUBMISSION_ACTIVE_STATUSES)),
    ).fetchone()
    return row is not None


def normalize_submitted_formula(formula: str) -> str:
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("platform_submitted_alpha_formula_invalid")
    return re.sub(r"\s+", "", formula).lower()


def canonical_submission_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("formal_submission_payload_invalid") from exc


def _validate_record(record: PlatformSubmittedAlphaRecord) -> str:
    if not isinstance(record, PlatformSubmittedAlphaRecord):
        raise ValueError("platform_submitted_alpha_invalid")
    _require_clean_text(
        record.account_scope,
        "platform_submitted_alpha_account_scope_invalid",
    )
    _require_clean_text(
        record.platform_alpha_id,
        "platform_submitted_alpha_id_invalid",
    )
    if record.formula != record.formula.strip():
        raise ValueError("platform_submitted_alpha_formula_invalid")
    normalize_submitted_formula(record.formula)
    _require_clean_text(record.status, "platform_submitted_alpha_status_invalid")
    if record.status.upper() == "UNSUBMITTED":
        raise ValueError("platform_submitted_alpha_status_invalid")
    _timestamp(record.date_submitted, "platform_submitted_alpha_date_invalid")
    if not isinstance(record.hidden, bool):
        raise ValueError("platform_submitted_alpha_hidden_invalid")
    _timestamp(record.observed_at, "platform_submitted_alpha_observed_at_invalid")
    if not isinstance(record.raw_payload, Mapping):
        raise ValueError("platform_submitted_alpha_raw_payload_invalid")
    raw_payload = dict(record.raw_payload)
    regular = raw_payload.get("regular")
    if (
        raw_payload.get("id") != record.platform_alpha_id
        or raw_payload.get("status") != record.status
        or raw_payload.get("dateSubmitted") != record.date_submitted
        or raw_payload.get("hidden") is not record.hidden
        or not isinstance(regular, Mapping)
        or regular.get("code") != record.formula
    ):
        raise ValueError("platform_submitted_alpha_raw_payload_mismatch")
    try:
        return json.dumps(
            raw_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("platform_submitted_alpha_raw_payload_invalid") from exc


def _record_from_row(row: sqlite3.Row) -> PlatformSubmittedAlphaRecord:
    try:
        raw_payload = json.loads(row["raw_payload_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("platform_submitted_alpha_raw_payload_invalid") from exc
    record = PlatformSubmittedAlphaRecord(
        account_scope=row["account_scope"],
        platform_alpha_id=row["platform_alpha_id"],
        formula=row["formula"],
        status=row["status"],
        date_submitted=row["date_submitted"],
        hidden=bool(row["hidden"]),
        raw_payload=raw_payload,
        observed_at=row["observed_at"],
    )
    _validate_record(record)
    if row["normalized_formula"] != record.normalized_formula:
        raise ValueError("platform_submitted_alpha_formula_identity_invalid")
    return record


def _attempt_from_row(row: sqlite3.Row) -> FormalSubmissionAttemptRecord:
    record = FormalSubmissionAttemptRecord(
        task_id=row["task_id"],
        run_id=row["run_id"],
        cycle_number=row["cycle_number"],
        family_root_task_id=row["family_root_task_id"],
        submission_mode=row["submission_mode"],
        status=row["status"],
        check_attempt_count=row["check_attempt_count"],
        check_payload_json=row["check_payload_json"],
        check_observed_at=row["check_observed_at"],
        retry_not_before=row["retry_not_before"],
        submission_claimed_at=row["submission_claimed_at"],
        submit_http_status=row["submit_http_status"],
        submit_response_json=row["submit_response_json"],
        confirmation_observed_at=row["confirmation_observed_at"],
        failure_code=row["failure_code"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        source=row["source"],
    )
    _validate_attempt(record)
    return record


def _validate_attempt(record: FormalSubmissionAttemptRecord) -> None:
    if not isinstance(record, FormalSubmissionAttemptRecord):
        raise ValueError("formal_submission_attempt_invalid")
    for value, error in (
        (record.task_id, "formal_submission_task_id_invalid"),
        (record.run_id, "formal_submission_run_id_invalid"),
        (record.family_root_task_id, "formal_submission_family_root_invalid"),
    ):
        _require_clean_text(value, error)
    if record.submission_mode not in FORMAL_SUBMISSION_MODES:
        raise ValueError("formal_submission_mode_invalid")
    if record.source not in {"queue", "qualified_archive"} or (
        record.source == "qualified_archive" and record.submission_mode != "manual"
    ):
        raise ValueError("formal_submission_source_invalid")
    if (
        isinstance(record.cycle_number, bool)
        or not isinstance(record.cycle_number, int)
        or record.cycle_number <= 0
    ):
        raise ValueError("formal_submission_cycle_invalid")
    statuses = FORMAL_SUBMISSION_ACTIVE_STATUSES | FORMAL_SUBMISSION_TERMINAL_STATUSES
    if record.status not in statuses:
        raise ValueError("formal_submission_status_invalid")
    if (
        isinstance(record.check_attempt_count, bool)
        or not isinstance(record.check_attempt_count, int)
        or record.check_attempt_count < 0
    ):
        raise ValueError("formal_submission_check_count_invalid")
    if (record.check_payload_json is None) != (record.check_observed_at is None):
        raise ValueError("formal_submission_check_evidence_invalid")
    if (record.check_payload_json is None) != (record.check_attempt_count == 0):
        raise ValueError("formal_submission_check_evidence_invalid")
    if record.check_payload_json is not None:
        _canonical_json_text(
            record.check_payload_json,
            "formal_submission_check_payload_invalid",
        )
        _timestamp(
            record.check_observed_at,
            "formal_submission_check_observed_at_invalid",
        )
    if record.retry_not_before is not None:
        _timestamp(
            record.retry_not_before,
            "formal_submission_retry_not_before_invalid",
        )
        if record.status not in {"detail_pending", "check_pending", "submission_unknown"}:
            raise ValueError("formal_submission_state_invalid")
    if record.submission_claimed_at is not None:
        _timestamp(
            record.submission_claimed_at,
            "formal_submission_claimed_at_invalid",
        )
    if record.submit_http_status is not None and (
        isinstance(record.submit_http_status, bool)
        or not isinstance(record.submit_http_status, int)
        or not 100 <= record.submit_http_status <= 599
    ):
        raise ValueError("formal_submission_http_status_invalid")
    if record.submit_response_json is not None:
        _canonical_json_text(
            record.submit_response_json,
            "formal_submission_response_invalid",
        )
    if (record.submit_http_status is None) != (record.submit_response_json is None):
        raise ValueError("formal_submission_response_invalid")
    if record.confirmation_observed_at is not None:
        _timestamp(
            record.confirmation_observed_at,
            "formal_submission_confirmation_observed_at_invalid",
        )
        if record.status != "submitted":
            raise ValueError("formal_submission_state_invalid")
    if record.failure_code is not None:
        _require_clean_text(
            record.failure_code,
            "formal_submission_failure_code_invalid",
        )
    created = _timestamp(record.created_at, "formal_submission_created_at_invalid")
    updated = _timestamp(record.updated_at, "formal_submission_updated_at_invalid")
    if updated < created:
        raise ValueError("formal_submission_updated_at_invalid")
    if record.status == "detail_pending":
        if (
            record.check_attempt_count != 0
            or record.check_payload_json is not None
            or record.submission_claimed_at is not None
            or record.submit_http_status is not None
            or record.submit_response_json is not None
            or record.confirmation_observed_at is not None
            or record.failure_code is not None
        ):
            raise ValueError("formal_submission_state_invalid")
    elif record.status == "check_pending":
        if record.submission_claimed_at is not None or record.failure_code is not None:
            raise ValueError("formal_submission_state_invalid")
    elif record.status in {"ready", "ineligible"}:
        grade_rejected_before_check = (
            record.status == "ineligible"
            and ((record.failure_code or "").startswith("formal_submission_grade_below_target:")
                 or record.source == "qualified_archive"
                 and (record.failure_code or "").startswith("formal_submission_grade_changed:"))
        )
        if (
            (record.check_payload_json is None and not grade_rejected_before_check)
            or record.retry_not_before is not None
            or record.submission_claimed_at is not None
            or (record.status == "ineligible") != (record.failure_code is not None)
        ):
            raise ValueError("formal_submission_state_invalid")
    elif record.status == "submitting":
        if (
            record.check_payload_json is None
            or record.submission_claimed_at is None
            or record.submit_http_status is not None
            or record.submit_response_json is not None
            or record.failure_code is not None
        ):
            raise ValueError("formal_submission_state_invalid")
    elif record.status == "confirmation_pending":
        if (
            record.check_payload_json is None
            or record.check_attempt_count <= 0
            or record.submission_claimed_at is None
            or record.submit_http_status is None
            or record.submit_response_json is None
            or record.failure_code is not None
        ):
            raise ValueError("formal_submission_state_invalid")
    elif record.status == "submission_unknown":
        if (
            record.check_payload_json is None
            or record.check_attempt_count <= 0
            or record.submission_claimed_at is None
            or record.failure_code is None
            or (record.submit_http_status is None)
            != (record.submit_response_json is None)
        ):
            raise ValueError("formal_submission_state_invalid")
    elif record.status == "submitted":
        if (
            record.confirmation_observed_at is None
            or record.failure_code is not None
            or (record.submit_http_status is None)
            != (record.submit_response_json is None)
        ):
            raise ValueError("formal_submission_state_invalid")
        if record.submission_claimed_at is None:
            if (
                record.check_attempt_count != 0
                or record.check_payload_json is not None
                or record.submit_http_status is not None
            ):
                raise ValueError("formal_submission_state_invalid")
        elif record.check_payload_json is None or record.check_attempt_count <= 0:
            raise ValueError("formal_submission_state_invalid")
    elif record.status == "failed" and record.failure_code is None:
        raise ValueError("formal_submission_state_invalid")


def _validate_attempt_links(
    connection: sqlite3.Connection,
    record: FormalSubmissionAttemptRecord,
) -> None:
    row = connection.execute(
        """
        SELECT tasks.status, tasks.account_scope, links.cycle_number
        FROM backtest_tasks AS tasks
        JOIN automated_run_backtests AS links ON links.task_id = tasks.task_id
        WHERE tasks.task_id = ? AND links.run_id = ?
        """,
        (record.task_id, record.run_id),
    ).fetchone()
    root = connection.execute(
        "SELECT account_scope FROM backtest_tasks WHERE task_id = ?",
        (record.family_root_task_id,),
    ).fetchone()
    if (
        row is None
        or row["status"] != "completed"
        or row["cycle_number"] != record.cycle_number
        or root is None
        or root["account_scope"] != row["account_scope"]
    ):
        raise ValueError("formal_submission_attempt_source_invalid")


def _attempt_identity(record: FormalSubmissionAttemptRecord) -> tuple[object, ...]:
    return (
        record.task_id,
        record.run_id,
        record.cycle_number,
        record.family_root_task_id,
        record.submission_mode,
        record.created_at,
        record.source,
    )


def _attempt_values(record: FormalSubmissionAttemptRecord) -> tuple[object, ...]:
    return (
        record.task_id,
        record.run_id,
        record.cycle_number,
        record.family_root_task_id,
        record.submission_mode,
        record.status,
        record.check_attempt_count,
        record.check_payload_json,
        record.check_observed_at,
        record.retry_not_before,
        record.submission_claimed_at,
        record.submit_http_status,
        record.submit_response_json,
        record.confirmation_observed_at,
        record.failure_code,
        record.created_at,
        record.updated_at,
        record.source,
    )


def _canonical_json_text(value: str, error: str) -> None:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if canonical_submission_json(parsed) != value:
        raise ValueError(error)


def _timestamp(value: object, error: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(error) from exc
    if parsed.utcoffset() is None:
        raise ValueError(error)
    return parsed


def _require_clean_text(value: object, error: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(error)
