from __future__ import annotations

import sqlite3

from persistence.catalog import (
    FieldCatalogContext,
    FieldCatalogRecord,
    OperatorCatalogRecord,
    OperatorOutputRecord,
    OperatorRoleRecord,
    PlatformCatalogSyncRecord,
    WindowCatalogRecord,
    initialize_catalog_schema,
    replace_operator_outputs,
    replace_operator_roles,
    replace_platform_catalog,
    replace_window_catalog,
)


def initialize_test_generation_catalog(
    connection: sqlite3.Connection,
    *,
    account_scope: str = "group-account",
) -> None:
    initialize_catalog_schema(connection)
    context = FieldCatalogContext("EQUITY", "USA", "TOP3000", 1)
    synced_at = "2026-08-30T00:00:00+08:00"
    fields = tuple(
        FieldCatalogRecord(
            context=context,
            field_id=field_id,
            dataset_id="dataset",
            category="sample",
            subcategory=None,
            field_type="MATRIX",
            coverage=1.0,
            description=field_id,
            dataset_name="sample",
            category_id="sample",
            subcategory_id=None,
            raw_payload={"id": field_id},
            synced_at=synced_at,
        )
        for field_id in ("close", "high", "open", "volume")
    )
    operator = OperatorCatalogRecord(
        operator_name="rank",
        category="Cross Sectional",
        definition="rank(x)",
        description="rank",
        documentation=None,
        level="ALL",
        scope=("REGULAR",),
        parameters=({"name": "x", "kind": "expr"},),
        raw_payload={"name": "rank"},
        synced_at=synced_at,
    )
    replace_operator_roles(
        connection,
        (OperatorRoleRecord("rank", "cross_sectional_normalization"),),
    )
    replace_operator_outputs(
        connection,
        (OperatorOutputRecord("rank", "signal"),),
    )
    replace_window_catalog(
        connection,
        (WindowCatalogRecord(22, "month"),),
    )
    replace_platform_catalog(
        connection,
        PlatformCatalogSyncRecord(
            account_scope=account_scope,
            context=context,
            field_count=len(fields),
            operator_count=1,
            synced_at=synced_at,
        ),
        fields,
        (operator,),
    )
