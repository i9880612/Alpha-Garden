from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping

from generation.catalog import (
    CatalogContext,
    FieldDefinition,
    GenerationCatalog,
    OperatorDefinition,
    OperatorParameter,
    WindowDefinition,
)
from persistence.catalog import (
    FieldCatalogContext,
    get_platform_catalog_sync,
    list_field_catalog,
    list_operator_outputs,
    list_operator_roles,
    list_operator_catalog,
    list_window_catalog,
)


def load_generation_catalog(
    connection: sqlite3.Connection,
    context: FieldCatalogContext,
    *,
    account_scope: str,
) -> GenerationCatalog:
    if not isinstance(account_scope, str) or not account_scope.strip():
        raise ValueError("generation_catalog_account_scope_missing")
    sync = get_platform_catalog_sync(connection)
    if sync is None:
        raise ValueError("generation_catalog_sync_missing")
    if sync.account_scope != account_scope.strip():
        raise ValueError("generation_catalog_account_scope_mismatch")
    if sync.context != context:
        raise ValueError("generation_catalog_context_mismatch")
    field_records = list_field_catalog(connection, context)
    operator_records = list_operator_catalog(connection)
    operator_output_records = list_operator_outputs(connection)
    operator_role_records = list_operator_roles(connection)
    window_records = list_window_catalog(connection)
    if len(field_records) != sync.field_count:
        raise ValueError("generation_catalog_field_count_mismatch")
    if len(operator_records) != sync.operator_count:
        raise ValueError("generation_catalog_operator_count_mismatch")
    if any(record.synced_at != sync.synced_at for record in field_records):
        raise ValueError("generation_catalog_sync_identity_mismatch")
    if any(record.synced_at != sync.synced_at for record in operator_records):
        raise ValueError("generation_catalog_sync_identity_mismatch")
    roles_by_operator: dict[str, list[str]] = defaultdict(list)
    for record in operator_role_records:
        roles_by_operator[record.operator_name].append(record.role)
    outputs_by_operator = {
        record.operator_name: record.output_kind
        for record in operator_output_records
    }
    fields = tuple(
        FieldDefinition(
            field_id=record.field_id,
            dataset_id=record.dataset_id,
            category=record.category,
            subcategory=record.subcategory,
            field_type=record.field_type,
            coverage=record.coverage,
        )
        for record in field_records
    )
    operators = tuple(
        OperatorDefinition(
            name=record.operator_name,
            category=record.category,
            scope=record.scope,
            roles=tuple(roles_by_operator[record.operator_name]),
            output_kind=outputs_by_operator.get(record.operator_name),
            parameters=_operator_parameters(
                record.operator_name,
                record.definition,
                record.parameters,
            ),
        )
        for record in operator_records
        if "REGULAR" in record.scope
    )
    catalog = GenerationCatalog(
        context=CatalogContext(
            instrument_type=context.instrument_type,
            region=context.region,
            universe=context.universe,
            delay=context.delay,
        ),
        fields=fields,
        operators=operators,
        windows=tuple(
            WindowDefinition(value=record.value, horizon=record.horizon)
            for record in window_records
        ),
    )
    return catalog


def _operator_parameters(
    operator_name: str,
    definition: str | None,
    payloads: tuple[Mapping[str, object], ...],
) -> tuple[OperatorParameter, ...]:
    if (
        operator_name == "ts_step"
        and not payloads
        and definition is not None
        and definition.strip() == "ts_step(1)"
    ):
        return (OperatorParameter("step", "int"),)
    return tuple(
        _operator_parameter(operator_name, payload)
        for payload in payloads
    )


def _operator_parameter(
    operator_name: str,
    payload: Mapping[str, object],
) -> OperatorParameter:
    name = payload.get("name")
    kind = payload.get("kind")
    optional = payload.get("optional", False)
    variadic = payload.get("variadic", False)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"operator_parameter_name_invalid:{operator_name}")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"operator_parameter_kind_invalid:{operator_name}:{name}")
    if not isinstance(optional, bool):
        raise ValueError(f"operator_parameter_optional_invalid:{operator_name}:{name}")
    if not isinstance(variadic, bool):
        raise ValueError(f"operator_parameter_variadic_invalid:{operator_name}:{name}")
    return OperatorParameter(
        name=name,
        kind=kind,
        optional=optional,
        variadic=variadic,
    )
