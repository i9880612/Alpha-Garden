"""Unpublished field pages; a committed page is also its durable checkpoint."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from persistence.catalog import FieldCatalogContext, FieldCatalogRecord


@dataclass(frozen=True, slots=True)
class CatalogDownloadScope:
    account_scope: str
    base_url: str
    context: FieldCatalogContext

    @property
    def key(self) -> tuple:
        return (self.account_scope, self.base_url, self.context.instrument_type,
                self.context.region, self.context.universe, self.context.delay)


@dataclass(frozen=True, slots=True)
class CatalogFieldPage:
    offset: int
    page_size: int
    total_count: int
    fields: tuple[FieldCatalogRecord, ...]


def initialize_catalog_download_schema(connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS platform_field_download_pages (
        account_scope TEXT NOT NULL,
        base_url TEXT NOT NULL,
        instrument_type TEXT NOT NULL,
        region TEXT NOT NULL,
        universe TEXT NOT NULL,
        delay INTEGER NOT NULL,
        page_offset INTEGER NOT NULL CHECK (page_offset >= 0),
        page_size INTEGER NOT NULL CHECK (page_size BETWEEN 1 AND 50),
        total_count INTEGER NOT NULL CHECK (total_count > 0),
        fields_json TEXT NOT NULL,
        PRIMARY KEY (account_scope, base_url, instrument_type, region, universe, delay, page_offset)
    )""")


def list_catalog_field_pages(connection, scope: CatalogDownloadScope) -> tuple[CatalogFieldPage, ...]:
    rows = connection.execute("""SELECT * FROM platform_field_download_pages
        WHERE account_scope=? AND base_url=? AND instrument_type=? AND region=? AND universe=? AND delay=?
        ORDER BY page_offset""", scope.key).fetchall()
    return tuple(CatalogFieldPage(row["page_offset"], row["page_size"], row["total_count"],
        tuple(FieldCatalogRecord(context=scope.context, **field) for field in json.loads(row["fields_json"])))
        for row in rows)


def catalog_field_download_pending(connection, scope: CatalogDownloadScope) -> bool:
    return connection.execute("""SELECT 1 FROM platform_field_download_pages
        WHERE account_scope=? AND base_url=? AND instrument_type=? AND region=? AND universe=? AND delay=?
        LIMIT 1""", scope.key).fetchone() is not None


def save_catalog_field_page(connection, scope: CatalogDownloadScope, page: CatalogFieldPage) -> None:
    """Append one validated page; the caller owns the transaction."""
    if (not scope.account_scope or not scope.base_url or not page.fields
            or len(page.fields) > page.page_size or page.offset + len(page.fields) > page.total_count
            or len({field.field_id for field in page.fields}) != len(page.fields)
            or any(field.context != scope.context for field in page.fields)):
        raise ValueError("catalog_download_page_invalid")
    fields = []
    for field in page.fields:
        value = asdict(field)
        del value["context"]
        fields.append(value)
    payload = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    connection.execute("""INSERT INTO platform_field_download_pages
        (account_scope, base_url, instrument_type, region, universe, delay,
         page_offset, page_size, total_count, fields_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (*scope.key, page.offset, page.page_size, page.total_count, payload))


def clear_catalog_field_pages(connection, scope: CatalogDownloadScope) -> None:
    connection.execute("""DELETE FROM platform_field_download_pages
        WHERE account_scope=? AND base_url=? AND instrument_type=? AND region=? AND universe=? AND delay=?""",
        scope.key)
