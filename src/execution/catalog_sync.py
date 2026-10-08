from __future__ import annotations

import math
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from execution.catalog import load_generation_catalog
from execution.process_lock import exclusive_run_process
from persistence.catalog import (
    FieldCatalogContext,
    FieldCatalogRecord,
    OperatorCatalogRecord,
    PlatformCatalogSyncRecord,
    initialize_catalog_schema,
    list_operator_outputs,
    list_window_catalog,
    replace_platform_catalog,
)
from persistence.database import open_database
from persistence.catalog_downloads import (
    CatalogDownloadScope, CatalogFieldPage, list_catalog_field_pages,
    save_catalog_field_page, clear_catalog_field_pages,
)
from persistence.runs import (
    list_active_automated_runs,
    list_submission_reconciliation_runs,
)
from selection.settings import load_backtest_settings_policy
from worldquant.catalog import CatalogContext, DataField, Operator
from worldquant.client import WorldQuantClient, WorldQuantRequestError
from worldquant.config import (
    WorldQuantConnectionSettings,
    load_worldquant_connection_settings,
)


Clock = Callable[[], datetime]
Waiter = Callable[[float], None]
ClientFactory = Callable[[WorldQuantConnectionSettings], WorldQuantClient]
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class CatalogRefreshPolicy:
    page_size: int = 50
    page_interval_seconds: float = 1.2
    max_attempts: int = 5
    retry_base_seconds: float = 1.0
    retry_backoff_max_seconds: float = 30.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.page_size, bool)
            or not isinstance(self.page_size, int)
            or not 1 <= self.page_size <= 50
        ):
            raise ValueError("catalog_refresh_page_size_invalid")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts <= 0
        ):
            raise ValueError("catalog_refresh_max_attempts_invalid")
        for value, error, allow_zero in (
            (
                self.page_interval_seconds,
                "catalog_refresh_page_interval_invalid",
                True,
            ),
            (self.retry_base_seconds, "catalog_refresh_retry_base_invalid", False),
            (
                self.retry_backoff_max_seconds,
                "catalog_refresh_retry_backoff_max_invalid",
                False,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (not allow_zero and value == 0)
            ):
                raise ValueError(error)
        if self.retry_base_seconds > self.retry_backoff_max_seconds:
            raise ValueError("catalog_refresh_retry_range_invalid")


@dataclass(frozen=True, slots=True)
class CatalogRefreshResult:
    account_scope: str
    context: FieldCatalogContext
    field_count: int
    operator_count: int
    catalog_fingerprint: str
    synced_at: str


def refresh_generation_catalog(
    database_path: str | Path,
    settings_policy_path: str | Path,
    environment_path: str | Path,
    *,
    refresh_policy: CatalogRefreshPolicy | None = None,
    clock: Clock | None = None,
    waiter: Waiter | None = None,
    client_factory: ClientFactory | None = None,
) -> CatalogRefreshResult:
    path = Path(database_path)
    if not path.is_file():
        raise ValueError("catalog_database_missing")
    settings_policy = load_backtest_settings_policy(settings_policy_path)
    connection_settings = load_worldquant_connection_settings(environment_path)
    context = FieldCatalogContext(
        instrument_type=settings_policy.instrument_type,
        region=settings_policy.region,
        universe=settings_policy.universe,
        delay=settings_policy.delay,
    )
    scope = CatalogDownloadScope(connection_settings.account_scope,
                                 connection_settings.base_url.rstrip("/"), context)
    if refresh_policy is not None and not isinstance(
        refresh_policy, CatalogRefreshPolicy
    ):
        raise ValueError("catalog_refresh_policy_invalid")
    policy = refresh_policy or CatalogRefreshPolicy()
    current_time = clock or _utc_now
    wait = waiter or time.sleep

    with exclusive_run_process(path):
        with open_database(path) as connection:
            require_catalog_refresh_allowed(connection)
            _require_local_generation_semantics(connection)
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                  ("platform_field_download_pages",)).fetchone() is None:
                raise ValueError("catalog_download_storage_missing:run init --upgrade-catalog-storage")

        make_client = client_factory or _default_client_factory
        client = make_client(connection_settings)
        client.authenticate()
        fields = _fetch_all_fields(path, client, scope, policy=policy, waiter=wait, clock=current_time)
        operators = _with_retry(
            client.fetch_operators,
            policy=policy,
            waiter=wait,
        )
        synced_at = _aware_time(current_time()).isoformat()
        field_records = tuple(replace(field, synced_at=synced_at) for field in fields)
        operator_records = tuple(
            _operator_record(operator, synced_at=synced_at)
            for operator in operators
        )
        sync = PlatformCatalogSyncRecord(
            account_scope=connection_settings.account_scope,
            context=context,
            field_count=len(field_records),
            operator_count=len(operator_records),
            synced_at=synced_at,
        )

        with open_database(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            initialize_catalog_schema(connection)
            require_catalog_refresh_allowed(connection)
            _require_local_generation_semantics(connection)
            replace_platform_catalog(connection, sync, field_records, operator_records)
            catalog = load_generation_catalog(connection, context, account_scope=connection_settings.account_scope)
            clear_catalog_field_pages(connection, scope)

    return CatalogRefreshResult(
        account_scope=connection_settings.account_scope,
        context=context,
        field_count=len(field_records),
        operator_count=len(operator_records),
        catalog_fingerprint=catalog.fingerprint,
        synced_at=synced_at,
    )


def _fetch_all_fields(
    database_path: Path,
    client: WorldQuantClient,
    scope: CatalogDownloadScope,
    *,
    policy: CatalogRefreshPolicy,
    waiter: Waiter,
    clock: Clock,
) -> tuple[FieldCatalogRecord, ...]:
    context = CatalogContext(scope.context.instrument_type, scope.context.region,
                             scope.context.universe, scope.context.delay)
    fields: list[FieldCatalogRecord] = []
    identifiers: set[str] = set()
    expected_total: int | None = None
    with open_database(database_path) as connection:
        saved_pages = list_catalog_field_pages(connection, scope)
    for saved in saved_pages:
        _validate_field_page(saved, offset=len(fields), expected_total=expected_total, identifiers=identifiers)
        fields.extend(saved.fields)
        identifiers.update(field.field_id for field in saved.fields)
        expected_total = saved.total_count
    logger = logging.getLogger("execution.progress")
    if saved_pages:
        last = saved_pages[-1]
        boundary = _with_retry(lambda: client.fetch_data_field_page(
            context=context, limit=last.page_size, offset=last.offset), policy=policy, waiter=waiter)
        if (boundary.total_count != expected_total
                or tuple(field.field_id for field in boundary.fields) != tuple(field.field_id for field in last.fields)):
            with open_database(database_path) as connection:
                clear_catalog_field_pages(connection, scope)
            raise ValueError("catalog_refresh_checkpoint_changed:rerun init to download from offset 0")
        logger.info("字段断点已核对：%d/%d；从 offset=%d 接续。", len(fields), expected_total, len(fields))
        if len(fields) < expected_total and policy.page_interval_seconds:
            waiter(float(policy.page_interval_seconds))
    while expected_total is None or len(fields) < expected_total:
        page = _with_retry(
            lambda: client.fetch_data_field_page(
                context=context,
                limit=policy.page_size,
                offset=len(fields),
            ),
            policy=policy,
            waiter=waiter,
        )
        observed_at = _aware_time(clock()).isoformat()
        saved = CatalogFieldPage(len(fields), policy.page_size, page.total_count,
            tuple(_field_record(field, context=scope.context, synced_at=observed_at) for field in page.fields))
        try:
            _validate_field_page(saved, offset=len(fields), expected_total=expected_total, identifiers=identifiers)
        except ValueError:
            # An inconsistent listing cannot be completed by mixing its pages.
            # Leave the published catalog intact; the next attempt starts fresh.
            with open_database(database_path) as connection:
                clear_catalog_field_pages(connection, scope)
            raise
        with open_database(database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            require_catalog_refresh_allowed(connection)
            save_catalog_field_page(connection, scope, saved)
        expected_total = page.total_count
        identifiers.update(field.field_id for field in saved.fields)
        fields.extend(saved.fields)
        logger.info("字段分页已保存：%d/%d；下次从 offset=%d 继续。", len(fields), expected_total, len(fields))
        if len(fields) < expected_total and policy.page_interval_seconds:
            waiter(float(policy.page_interval_seconds))
    assert expected_total is not None
    if len(fields) != expected_total:
        raise ValueError("catalog_refresh_field_count_mismatch")
    return tuple(fields)


def _validate_field_page(page: CatalogFieldPage, *, offset: int, expected_total: int | None,
                         identifiers: set[str]) -> None:
    if page.offset != offset:
        raise ValueError("catalog_refresh_field_page_gap")
    if expected_total is not None and page.total_count != expected_total:
        raise ValueError("catalog_refresh_field_total_changed")
    if not page.fields:
        raise ValueError("catalog_refresh_field_page_empty")
    page_identifiers = {field.field_id for field in page.fields}
    if len(page_identifiers) != len(page.fields) or identifiers.intersection(page_identifiers):
        raise ValueError("catalog_refresh_field_duplicate")
    if len(page.fields) > page.page_size or offset + len(page.fields) > page.total_count:
        raise ValueError("catalog_refresh_field_count_exceeded")


def _with_retry(
    operation: Callable[[], _T],
    *,
    policy: CatalogRefreshPolicy,
    waiter: Waiter,
) -> _T:
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except WorldQuantRequestError as exc:
            if not exc.retryable or attempt == policy.max_attempts:
                raise
            delay = exc.retry_after_seconds
            if delay is None:
                delay = min(
                    policy.retry_base_seconds * (2 ** (attempt - 1)),
                    policy.retry_backoff_max_seconds,
                )
            waiter(float(delay))
    raise AssertionError("catalog_refresh_retry_unreachable")


def _field_record(
    field: DataField,
    *,
    context: FieldCatalogContext,
    synced_at: str,
) -> FieldCatalogRecord:
    return FieldCatalogRecord(
        context=context,
        field_id=field.field_id,
        dataset_id=field.dataset_id,
        category=field.category,
        subcategory=field.subcategory,
        field_type=field.field_type,
        coverage=field.coverage,
        description=field.description,
        dataset_name=field.dataset_name,
        category_id=field.category_id,
        subcategory_id=field.subcategory_id,
        raw_payload=field.raw_payload,
        synced_at=synced_at,
    )


def _operator_record(
    operator: Operator,
    *,
    synced_at: str,
) -> OperatorCatalogRecord:
    return OperatorCatalogRecord(
        operator_name=operator.name,
        category=operator.category,
        definition=operator.definition,
        description=operator.description,
        documentation=operator.documentation,
        level=operator.level,
        scope=operator.scope,
        parameters=operator.parameters,
        raw_payload=operator.raw_payload,
        synced_at=synced_at,
    )


def require_catalog_refresh_allowed(connection) -> None:
    if list_active_automated_runs(connection) or list_submission_reconciliation_runs(
        connection
    ):
        raise ValueError("catalog_refresh_bound_run_exists")


def _require_local_generation_semantics(connection) -> None:
    if not list_window_catalog(connection):
        raise ValueError("catalog_refresh_window_catalog_empty")
    if not list_operator_outputs(connection):
        raise ValueError("catalog_refresh_operator_outputs_empty")


def _default_client_factory(
    settings: WorldQuantConnectionSettings,
) -> WorldQuantClient:
    return WorldQuantClient(
        base_url=settings.base_url,
        credentials=settings.credentials,
    )


def _aware_time(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("catalog_refresh_clock_invalid")
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)
