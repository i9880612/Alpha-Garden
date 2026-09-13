from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FieldCatalogContext:
    instrument_type: str
    region: str
    universe: str
    delay: int


@dataclass(frozen=True, slots=True)
class FieldCatalogRecord:
    context: FieldCatalogContext
    field_id: str
    dataset_id: str | None
    category: str | None
    subcategory: str | None
    field_type: str | None
    coverage: float | None
    description: str | None
    dataset_name: str | None
    category_id: str | None
    subcategory_id: str | None
    raw_payload: Mapping[str, Any]
    synced_at: str


@dataclass(frozen=True, slots=True)
class OperatorCatalogRecord:
    operator_name: str
    category: str | None
    definition: str | None
    description: str | None
    documentation: str | None
    level: str | None
    scope: tuple[str, ...]
    parameters: tuple[Mapping[str, Any], ...]
    raw_payload: Mapping[str, Any]
    synced_at: str


@dataclass(frozen=True, slots=True)
class WindowCatalogRecord:
    value: int
    horizon: str


@dataclass(frozen=True, slots=True)
class OperatorRoleRecord:
    operator_name: str
    role: str


@dataclass(frozen=True, slots=True)
class OperatorOutputRecord:
    operator_name: str
    output_kind: str


@dataclass(frozen=True, slots=True)
class PlatformCatalogSyncRecord:
    account_scope: str
    context: FieldCatalogContext
    field_count: int
    operator_count: int
    synced_at: str


def initialize_catalog_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS platform_fields (
            instrument_type TEXT NOT NULL,
            region TEXT NOT NULL,
            universe TEXT NOT NULL,
            delay INTEGER NOT NULL,
            field_id TEXT NOT NULL,
            dataset_id TEXT,
            category TEXT,
            subcategory TEXT,
            field_type TEXT,
            coverage REAL,
            description TEXT,
            dataset_name TEXT,
            category_id TEXT,
            subcategory_id TEXT,
            raw_payload_json TEXT NOT NULL,
            synced_at TEXT NOT NULL,
            PRIMARY KEY (instrument_type, region, universe, delay, field_id)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_platform_fields_dataset
        ON platform_fields (dataset_id, field_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_platform_fields_type
        ON platform_fields (field_type, field_id)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS platform_operators (
            operator_name TEXT PRIMARY KEY,
            category TEXT,
            definition TEXT,
            description TEXT,
            documentation TEXT,
            level TEXT,
            scope_json TEXT NOT NULL,
            parameters_json TEXT NOT NULL,
            raw_payload_json TEXT NOT NULL,
            synced_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_platform_operators_category
        ON platform_operators (category, operator_name)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS generation_windows (
            value INTEGER PRIMARY KEY CHECK (value > 0),
            horizon TEXT NOT NULL CHECK (length(trim(horizon)) > 0)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_generation_windows_horizon
        ON generation_windows (horizon, value)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS generation_operator_roles (
            operator_name TEXT NOT NULL,
            role TEXT NOT NULL CHECK (length(trim(role)) > 0),
            PRIMARY KEY (operator_name, role)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_generation_operator_roles_role
        ON generation_operator_roles (role, operator_name)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS generation_operator_outputs (
            operator_name TEXT PRIMARY KEY,
            output_kind TEXT NOT NULL CHECK (length(trim(output_kind)) > 0)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS platform_catalog_syncs (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            account_scope TEXT NOT NULL CHECK (length(trim(account_scope)) > 0),
            instrument_type TEXT NOT NULL,
            region TEXT NOT NULL,
            universe TEXT NOT NULL,
            delay INTEGER NOT NULL,
            field_count INTEGER NOT NULL CHECK (field_count > 0),
            operator_count INTEGER NOT NULL CHECK (operator_count > 0),
            synced_at TEXT NOT NULL CHECK (length(trim(synced_at)) > 0)
        )
        """
    )


def replace_platform_catalog(
    connection: sqlite3.Connection,
    sync: PlatformCatalogSyncRecord,
    field_records: Sequence[FieldCatalogRecord],
    operator_records: Sequence[OperatorCatalogRecord],
) -> None:
    _validate_sync(sync)
    if sync.field_count != len(field_records):
        raise ValueError("platform_catalog_field_count_mismatch")
    if sync.operator_count != len(operator_records):
        raise ValueError("platform_catalog_operator_count_mismatch")
    field_values = [_field_values(sync.context, record) for record in field_records]
    operator_values = [_operator_values(record) for record in operator_records]
    field_ids = [record.field_id for record in field_records]
    operator_names = [record.operator_name for record in operator_records]
    if len(set(field_ids)) != len(field_ids):
        raise ValueError("field_catalog_duplicate_id")
    if len(set(operator_names)) != len(operator_names):
        raise ValueError("operator_catalog_duplicate_name")
    if any(record.synced_at != sync.synced_at for record in field_records):
        raise ValueError("platform_catalog_synced_at_mismatch")
    if any(record.synced_at != sync.synced_at for record in operator_records):
        raise ValueError("platform_catalog_synced_at_mismatch")

    connection.execute("DELETE FROM platform_fields")
    connection.executemany(
        """
        INSERT INTO platform_fields (
            instrument_type, region, universe, delay, field_id,
            dataset_id, category, subcategory, field_type, coverage,
            description, dataset_name, category_id, subcategory_id,
            raw_payload_json, synced_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        field_values,
    )
    connection.execute("DELETE FROM platform_operators")
    connection.executemany(
        """
        INSERT INTO platform_operators (
            operator_name, category, definition, description, documentation,
            level, scope_json, parameters_json, raw_payload_json, synced_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        operator_values,
    )
    stored_field_count = connection.execute(
        "SELECT COUNT(*) FROM platform_fields"
    ).fetchone()[0]
    stored_operator_count = connection.execute(
        "SELECT COUNT(*) FROM platform_operators"
    ).fetchone()[0]
    if stored_field_count != sync.field_count:
        raise ValueError("platform_catalog_field_write_incomplete")
    if stored_operator_count != sync.operator_count:
        raise ValueError("platform_catalog_operator_write_incomplete")
    connection.execute("DELETE FROM platform_catalog_syncs")
    connection.execute(
        """
        INSERT INTO platform_catalog_syncs (
            singleton_id, account_scope, instrument_type, region, universe,
            delay, field_count, operator_count, synced_at
        ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sync.account_scope.strip(),
            sync.context.instrument_type,
            sync.context.region,
            sync.context.universe,
            sync.context.delay,
            sync.field_count,
            sync.operator_count,
            sync.synced_at,
        ),
    )


def get_platform_catalog_sync(
    connection: sqlite3.Connection,
) -> PlatformCatalogSyncRecord | None:
    table_exists = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'platform_catalog_syncs'
        """
    ).fetchone()
    if table_exists is None:
        return None
    row = connection.execute(
        "SELECT * FROM platform_catalog_syncs WHERE singleton_id = 1"
    ).fetchone()
    if row is None:
        return None
    record = PlatformCatalogSyncRecord(
        account_scope=row["account_scope"],
        context=FieldCatalogContext(
            instrument_type=row["instrument_type"],
            region=row["region"],
            universe=row["universe"],
            delay=row["delay"],
        ),
        field_count=row["field_count"],
        operator_count=row["operator_count"],
        synced_at=row["synced_at"],
    )
    _validate_sync(record)
    return record


def replace_field_catalog(
    connection: sqlite3.Connection,
    context: FieldCatalogContext,
    records: Sequence[FieldCatalogRecord],
) -> None:
    _validate_context(context)
    if not records:
        raise ValueError("field_catalog_empty")

    prepared = [_field_values(context, record) for record in records]
    field_ids = [record.field_id for record in records]
    if len(set(field_ids)) != len(field_ids):
        raise ValueError("field_catalog_duplicate_id")

    connection.execute(
        """
        DELETE FROM platform_fields
        WHERE instrument_type = ? AND region = ? AND universe = ? AND delay = ?
        """,
        (
            context.instrument_type,
            context.region,
            context.universe,
            context.delay,
        ),
    )
    connection.executemany(
        """
        INSERT INTO platform_fields (
            instrument_type, region, universe, delay, field_id,
            dataset_id, category, subcategory, field_type, coverage,
            description, dataset_name, category_id, subcategory_id,
            raw_payload_json, synced_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        prepared,
    )


def replace_operator_catalog(
    connection: sqlite3.Connection,
    records: Sequence[OperatorCatalogRecord],
) -> None:
    if not records:
        raise ValueError("operator_catalog_empty")

    prepared = [_operator_values(record) for record in records]
    operator_names = [record.operator_name for record in records]
    if len(set(operator_names)) != len(operator_names):
        raise ValueError("operator_catalog_duplicate_name")

    connection.execute("DELETE FROM platform_operators")
    connection.executemany(
        """
        INSERT INTO platform_operators (
            operator_name, category, definition, description, documentation,
            level, scope_json, parameters_json, raw_payload_json, synced_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        prepared,
    )


def replace_window_catalog(
    connection: sqlite3.Connection,
    records: Sequence[WindowCatalogRecord],
) -> None:
    if not records:
        raise ValueError("window_catalog_empty")

    prepared = [_window_values(record) for record in records]
    values = [record.value for record in records]
    if len(set(values)) != len(values):
        raise ValueError("window_catalog_duplicate_value")

    connection.execute("DELETE FROM generation_windows")
    connection.executemany(
        """
        INSERT INTO generation_windows (value, horizon)
        VALUES (?, ?)
        """,
        prepared,
    )


def replace_operator_roles(
    connection: sqlite3.Connection,
    records: Sequence[OperatorRoleRecord],
) -> None:
    if not records:
        raise ValueError("operator_roles_empty")

    prepared = [_operator_role_values(record) for record in records]
    if len(set(prepared)) != len(prepared):
        raise ValueError("operator_roles_duplicate")

    connection.execute("DELETE FROM generation_operator_roles")
    connection.executemany(
        """
        INSERT INTO generation_operator_roles (operator_name, role)
        VALUES (?, ?)
        """,
        prepared,
    )


def replace_operator_outputs(
    connection: sqlite3.Connection,
    records: Sequence[OperatorOutputRecord],
) -> None:
    if not records:
        raise ValueError("operator_outputs_empty")

    prepared = [_operator_output_values(record) for record in records]
    operator_names = [record.operator_name for record in records]
    if len(set(operator_names)) != len(operator_names):
        raise ValueError("operator_outputs_duplicate")

    connection.execute("DELETE FROM generation_operator_outputs")
    connection.executemany(
        """
        INSERT INTO generation_operator_outputs (operator_name, output_kind)
        VALUES (?, ?)
        """,
        prepared,
    )


def list_field_catalog(
    connection: sqlite3.Connection,
    context: FieldCatalogContext,
) -> tuple[FieldCatalogRecord, ...]:
    _validate_context(context)
    rows = connection.execute(
        """
        SELECT * FROM platform_fields
        WHERE instrument_type = ? AND region = ? AND universe = ? AND delay = ?
        ORDER BY field_id
        """,
        (
            context.instrument_type,
            context.region,
            context.universe,
            context.delay,
        ),
    ).fetchall()
    return tuple(_field_from_row(row) for row in rows)


def list_operator_catalog(
    connection: sqlite3.Connection,
) -> tuple[OperatorCatalogRecord, ...]:
    rows = connection.execute(
        "SELECT * FROM platform_operators ORDER BY operator_name"
    ).fetchall()
    return tuple(_operator_from_row(row) for row in rows)


def list_window_catalog(
    connection: sqlite3.Connection,
) -> tuple[WindowCatalogRecord, ...]:
    rows = connection.execute(
        "SELECT value, horizon FROM generation_windows ORDER BY value"
    ).fetchall()
    return tuple(
        WindowCatalogRecord(value=row["value"], horizon=row["horizon"])
        for row in rows
    )


def list_operator_roles(
    connection: sqlite3.Connection,
) -> tuple[OperatorRoleRecord, ...]:
    rows = connection.execute(
        """
        SELECT operator_name, role
        FROM generation_operator_roles
        ORDER BY operator_name, role
        """
    ).fetchall()
    return tuple(
        OperatorRoleRecord(
            operator_name=row["operator_name"],
            role=row["role"],
        )
        for row in rows
    )


def list_operator_outputs(
    connection: sqlite3.Connection,
) -> tuple[OperatorOutputRecord, ...]:
    rows = connection.execute(
        """
        SELECT operator_name, output_kind
        FROM generation_operator_outputs
        ORDER BY operator_name
        """
    ).fetchall()
    return tuple(
        OperatorOutputRecord(
            operator_name=row["operator_name"],
            output_kind=row["output_kind"],
        )
        for row in rows
    )


def _field_values(
    context: FieldCatalogContext,
    record: FieldCatalogRecord,
) -> tuple[object, ...]:
    _validate_context(record.context)
    if record.context != context:
        raise ValueError("field_catalog_context_mismatch")
    _require_text(record.field_id, "field_catalog_id_missing")
    _require_text(record.synced_at, "field_catalog_synced_at_missing")
    return (
        context.instrument_type,
        context.region,
        context.universe,
        context.delay,
        record.field_id,
        record.dataset_id,
        record.category,
        record.subcategory,
        record.field_type,
        record.coverage,
        record.description,
        record.dataset_name,
        record.category_id,
        record.subcategory_id,
        _mapping_json(record.raw_payload, "field_catalog_raw_payload_invalid"),
        record.synced_at,
    )


def _operator_values(record: OperatorCatalogRecord) -> tuple[object, ...]:
    _require_text(record.operator_name, "operator_catalog_name_missing")
    _require_text(record.synced_at, "operator_catalog_synced_at_missing")
    if any(not isinstance(item, str) or not item.strip() for item in record.scope):
        raise ValueError("operator_catalog_scope_invalid")
    if any(not isinstance(item, Mapping) for item in record.parameters):
        raise ValueError("operator_catalog_parameters_invalid")
    return (
        record.operator_name,
        record.category,
        record.definition,
        record.description,
        record.documentation,
        record.level,
        _sequence_json(record.scope),
        _sequence_json(record.parameters),
        _mapping_json(record.raw_payload, "operator_catalog_raw_payload_invalid"),
        record.synced_at,
    )


def _window_values(record: WindowCatalogRecord) -> tuple[object, ...]:
    if (
        isinstance(record.value, bool)
        or not isinstance(record.value, int)
        or record.value <= 0
    ):
        raise ValueError("window_catalog_value_invalid")
    _require_text(record.horizon, "window_catalog_horizon_missing")
    return (record.value, record.horizon.strip())


def _operator_role_values(record: OperatorRoleRecord) -> tuple[str, str]:
    _require_text(record.operator_name, "operator_role_name_missing")
    _require_text(record.role, "operator_role_missing")
    return (record.operator_name.strip(), record.role.strip())


def _operator_output_values(record: OperatorOutputRecord) -> tuple[str, str]:
    _require_text(record.operator_name, "operator_output_name_missing")
    _require_text(record.output_kind, "operator_output_kind_missing")
    return (record.operator_name.strip(), record.output_kind.strip())


def _field_from_row(row: sqlite3.Row) -> FieldCatalogRecord:
    return FieldCatalogRecord(
        context=FieldCatalogContext(
            instrument_type=row["instrument_type"],
            region=row["region"],
            universe=row["universe"],
            delay=row["delay"],
        ),
        field_id=row["field_id"],
        dataset_id=row["dataset_id"],
        category=row["category"],
        subcategory=row["subcategory"],
        field_type=row["field_type"],
        coverage=row["coverage"],
        description=row["description"],
        dataset_name=row["dataset_name"],
        category_id=row["category_id"],
        subcategory_id=row["subcategory_id"],
        raw_payload=_object_json(row["raw_payload_json"], "field_catalog_raw_payload_invalid"),
        synced_at=row["synced_at"],
    )


def _operator_from_row(row: sqlite3.Row) -> OperatorCatalogRecord:
    scope = _array_json(row["scope_json"], "operator_catalog_scope_invalid")
    if any(not isinstance(item, str) or not item.strip() for item in scope):
        raise ValueError("operator_catalog_scope_invalid")
    parameters = _array_json(
        row["parameters_json"],
        "operator_catalog_parameters_invalid",
    )
    if any(not isinstance(item, dict) for item in parameters):
        raise ValueError("operator_catalog_parameters_invalid")
    return OperatorCatalogRecord(
        operator_name=row["operator_name"],
        category=row["category"],
        definition=row["definition"],
        description=row["description"],
        documentation=row["documentation"],
        level=row["level"],
        scope=tuple(scope),
        parameters=tuple(parameters),
        raw_payload=_object_json(
            row["raw_payload_json"],
            "operator_catalog_raw_payload_invalid",
        ),
        synced_at=row["synced_at"],
    )


def _validate_context(context: FieldCatalogContext) -> None:
    _require_text(context.instrument_type, "field_catalog_instrument_type_missing")
    _require_text(context.region, "field_catalog_region_missing")
    _require_text(context.universe, "field_catalog_universe_missing")
    if (
        isinstance(context.delay, bool)
        or not isinstance(context.delay, int)
        or context.delay < 0
    ):
        raise ValueError("field_catalog_delay_invalid")


def _validate_sync(sync: PlatformCatalogSyncRecord) -> None:
    if not isinstance(sync, PlatformCatalogSyncRecord):
        raise ValueError("platform_catalog_sync_invalid")
    _require_text(sync.account_scope, "platform_catalog_account_scope_missing")
    _validate_context(sync.context)
    if (
        isinstance(sync.field_count, bool)
        or not isinstance(sync.field_count, int)
        or sync.field_count <= 0
    ):
        raise ValueError("platform_catalog_field_count_invalid")
    if (
        isinstance(sync.operator_count, bool)
        or not isinstance(sync.operator_count, int)
        or sync.operator_count <= 0
    ):
        raise ValueError("platform_catalog_operator_count_invalid")
    _require_text(sync.synced_at, "platform_catalog_synced_at_missing")


def _require_text(value: object, error: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)


def _mapping_json(value: Mapping[str, Any], error: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(error)
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sequence_json(value: Sequence[object]) -> str:
    return json.dumps(
        list(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _object_json(value: str, error: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if not isinstance(parsed, dict):
        raise ValueError(error)
    return parsed


def _array_json(value: str, error: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if not isinstance(parsed, list):
        raise ValueError(error)
    return parsed
