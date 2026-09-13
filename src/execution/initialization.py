from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from execution.catalog_sync import (
    CatalogRefreshPolicy,
    CatalogRefreshResult,
    ClientFactory,
    Clock,
    Waiter,
    refresh_generation_catalog,
    require_catalog_refresh_allowed,
)
from execution.catalog import load_generation_catalog
from generation.project_catalog import (
    OPERATOR_OUTPUTS,
    OPERATOR_ROLES,
    STANDARD_WINDOWS,
)
from persistence.catalog import (
    FieldCatalogContext,
    OperatorOutputRecord,
    OperatorRoleRecord,
    WindowCatalogRecord,
    get_platform_catalog_sync,
    list_operator_outputs,
    list_operator_roles,
    list_window_catalog,
    replace_operator_outputs,
    replace_operator_roles,
    replace_window_catalog,
)
from persistence.database import open_database
from persistence.schema import PROJECT_TABLE_NAMES, initialize_database_schema
from selection.settings import load_backtest_settings_policy
from worldquant.config import load_worldquant_connection_settings


_INCOMPLETE_CATALOG_ERRORS = frozenset(
    {
        "generation_catalog_field_count_mismatch",
        "generation_catalog_operator_count_mismatch",
        "generation_catalog_sync_identity_mismatch",
    }
)


@dataclass(frozen=True, slots=True)
class ProjectInitializationResult:
    database_path: Path
    database_created: bool
    project_table_count: int
    semantics_initialized: bool
    catalog_refreshed: bool
    window_count: int
    operator_role_count: int
    operator_output_count: int
    catalog: CatalogRefreshResult


def initialize_project(
    database_path: str | Path,
    settings_policy_path: str | Path,
    environment_path: str | Path,
    *,
    refresh_policy: CatalogRefreshPolicy | None = None,
    clock: Clock | None = None,
    waiter: Waiter | None = None,
    client_factory: ClientFactory | None = None,
) -> ProjectInitializationResult:
    path = Path(database_path)
    if path.exists() and not path.is_file():
        raise ValueError("project_database_path_invalid")
    if not path.parent.is_dir():
        raise ValueError("project_database_parent_missing")
    database_created = not path.is_file()

    with open_database(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        initialize_database_schema(connection)
        semantics_initialized = _initialize_project_semantics(connection)

    catalog = _load_existing_platform_catalog(
        path, settings_policy_path, environment_path
    )
    catalog_refreshed = catalog is None
    if catalog is None:
        catalog = refresh_generation_catalog(
            path,
            settings_policy_path,
            environment_path,
            refresh_policy=refresh_policy,
            clock=clock,
            waiter=waiter,
            client_factory=client_factory,
        )
    return ProjectInitializationResult(
        database_path=path,
        database_created=database_created,
        project_table_count=len(PROJECT_TABLE_NAMES),
        semantics_initialized=semantics_initialized,
        catalog_refreshed=catalog_refreshed,
        window_count=len(STANDARD_WINDOWS),
        operator_role_count=len(OPERATOR_ROLES),
        operator_output_count=len(OPERATOR_OUTPUTS),
        catalog=catalog,
    )


def _load_existing_platform_catalog(
    database_path: Path,
    settings_policy_path: str | Path,
    environment_path: str | Path,
) -> CatalogRefreshResult | None:
    settings_policy = load_backtest_settings_policy(settings_policy_path)
    connection_settings = load_worldquant_connection_settings(environment_path)
    context = FieldCatalogContext(
        instrument_type=settings_policy.instrument_type,
        region=settings_policy.region,
        universe=settings_policy.universe,
        delay=settings_policy.delay,
    )
    with open_database(database_path) as connection:
        sync = get_platform_catalog_sync(connection)
        if (
            sync is None
            or sync.account_scope != connection_settings.account_scope
            or sync.context != context
        ):
            return None
        try:
            catalog = load_generation_catalog(
                connection,
                context,
                account_scope=connection_settings.account_scope,
            )
        except ValueError as exc:
            if str(exc) not in _INCOMPLETE_CATALOG_ERRORS:
                raise
            return None
    return CatalogRefreshResult(
        account_scope=sync.account_scope,
        context=sync.context,
        field_count=sync.field_count,
        operator_count=sync.operator_count,
        catalog_fingerprint=catalog.fingerprint,
        synced_at=sync.synced_at,
    )


def _initialize_project_semantics(connection) -> bool:
    expected_windows = tuple(
        WindowCatalogRecord(value, horizon)
        for value, horizon in STANDARD_WINDOWS
    )
    expected_roles = tuple(
        OperatorRoleRecord(operator_name, role)
        for operator_name, role in OPERATOR_ROLES
    )
    expected_outputs = tuple(
        OperatorOutputRecord(operator_name, output_kind)
        for operator_name, output_kind in OPERATOR_OUTPUTS
    )
    current = (
        list_window_catalog(connection),
        list_operator_roles(connection),
        list_operator_outputs(connection),
    )
    expected = (expected_windows, expected_roles, expected_outputs)
    if all(not records for records in current):
        require_catalog_refresh_allowed(connection)
        replace_window_catalog(connection, expected_windows)
        replace_operator_roles(connection, expected_roles)
        replace_operator_outputs(connection, expected_outputs)
        if (
            list_window_catalog(connection),
            list_operator_roles(connection),
            list_operator_outputs(connection),
        ) != expected:
            raise ValueError("project_generation_semantics_write_incomplete")
        return True
    if current != expected:
        labels = ("windows", "operator_roles", "operator_outputs")
        mismatched = [
            label
            for label, actual, required in zip(labels, current, expected, strict=True)
            if actual != required
        ]
        raise ValueError(
            "project_generation_semantics_conflict:" + ",".join(mismatched)
        )
    return False
