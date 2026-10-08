import json
import sqlite3
from dataclasses import replace
from datetime import datetime
from unittest.mock import patch

import pytest

from execution.catalog_sync import CatalogRefreshPolicy, refresh_generation_catalog
from execution.initialization import initialize_project
from execution.process_lock import exclusive_run_process
from persistence.catalog import FieldCatalogContext, get_platform_catalog_sync, list_field_catalog
from persistence.catalog_downloads import CatalogDownloadScope, list_catalog_field_pages, save_catalog_field_page
from persistence.database import open_database
from tests.execution import test_catalog_sync as fixture
from tests.execution import test_initialization as init_fixture
from worldquant.catalog import DataFieldPage
from worldquant.client import WorldQuantRequestError


POLICY = CatalogRefreshPolicy(page_size=2, page_interval_seconds=0, max_attempts=1)
NOW = datetime.fromisoformat("2026-10-08T12:00:00+08:00")


@pytest.fixture
def catalog():
    case = fixture.CatalogRefreshTests()
    case.setUp()
    try:
        yield case
    finally:
        case.doCleanups()


def page(total, *ids):
    return DataFieldPage(total, tuple(fixture.CatalogRefreshTests._field(field_id) for field_id in ids))


def scope(case):
    return CatalogDownloadScope("group-account", "https://api.worldquantbrain.com", case.context)


def refresh(case, client, *, policy=POLICY):
    return refresh_generation_catalog(case.database_path, case.settings_path, case.environment_path,
        refresh_policy=policy, clock=lambda: NOW, waiter=lambda _: None, client_factory=lambda _: client)


def interrupted(case, *, operators=False):
    error = WorldQuantRequestError("temporary", status_code=503, retryable=True, outcome_unknown=False)
    client = fixture.FakeCatalogClient({0: page(5, "a", "b"), 2: page(5, "c", "d"),
        4: page(5, "e") if operators else error}, [error] if operators else [])
    with pytest.raises(WorldQuantRequestError):
        refresh(case, client)
    case._assert_old_catalog_preserved()


def test_pages_survive_new_client_and_resume_from_boundary_with_changed_page_size(catalog):
    interrupted(catalog)
    with open_database(catalog.database_path) as connection:
        saved = list_catalog_field_pages(connection, scope(catalog))
    assert [row.offset for row in saved] == [0, 2]
    assert [[field.field_id for field in row.fields] for row in saved] == [["a", "b"], ["c", "d"]]
    client = fixture.FakeCatalogClient({2: page(5, "c", "d"), 4: page(5, "e")},
                                      [(catalog._operator("rank"),)])
    result = refresh(catalog, client, policy=replace(POLICY, page_size=1))
    assert client.field_calls == [(2, 2), (1, 4)]
    assert result.field_count == 5
    with open_database(catalog.database_path) as connection:
        fields = list_field_catalog(connection, catalog.context)
        assert [field.field_id for field in fields] == list("abcde")
        assert all(field.dataset_id == "dataset" and field.coverage == 1.0 for field in fields)
        assert fields[0].raw_payload == catalog._field("a").raw_payload
        assert list_catalog_field_pages(connection, scope(catalog)) == ()


def test_operator_failure_does_not_discard_finished_field_download(catalog):
    interrupted(catalog, operators=True)
    client = fixture.FakeCatalogClient({4: page(5, "e")}, [(catalog._operator("rank"),)])
    assert refresh(catalog, client).field_count == 5
    assert client.field_calls == [(2, 4)]


@pytest.mark.parametrize("boundary", [page(6, "c", "d"), page(5, "c", "changed")])
def test_changed_boundary_discards_only_incomplete_download_then_allows_restart(catalog, boundary):
    interrupted(catalog)
    client = fixture.FakeCatalogClient({2: boundary}, [])
    with pytest.raises(ValueError, match="catalog_refresh_checkpoint_changed"):
        refresh(catalog, client)
    catalog._assert_old_catalog_preserved()
    assert client.operator_calls == 0
    with open_database(catalog.database_path) as connection:
        assert list_catalog_field_pages(connection, scope(catalog)) == ()
    restarted = fixture.FakeCatalogClient({0: page(1, "replacement")}, [(catalog._operator("rank"),)])
    assert refresh(catalog, restarted).field_count == 1
    assert restarted.field_calls == [(2, 0)]


@pytest.mark.parametrize("change", ["account", "platform", "universe", "delay"])
def test_download_checkpoints_are_isolated_and_reusable_after_scope_switch(catalog, change):
    interrupted(catalog)
    environment = catalog.environment_path.read_text(encoding="utf-8")
    settings = catalog.settings_path.read_text(encoding="utf-8")
    if change in {"account", "platform"}:
        old, new = (("group-account", "other-account") if change == "account" else
                    ("https://api.worldquantbrain.com", "https://other.example.com"))
        catalog.environment_path.write_text(environment.replace(old, new), encoding="utf-8")
    else:
        value = json.loads(settings)
        value["catalogContext"][change] = "TOP1000" if change == "universe" else 0
        catalog.settings_path.write_text(json.dumps(value), encoding="utf-8")
    client = fixture.FakeCatalogClient({0: page(1, "other")}, [(catalog._operator("rank"),)])
    refresh(catalog, client)
    assert client.field_calls == [(2, 0)]
    with open_database(catalog.database_path) as connection:
        assert len(list_catalog_field_pages(connection, scope(catalog))) == 2
    catalog.environment_path.write_text(environment, encoding="utf-8")
    catalog.settings_path.write_text(settings, encoding="utf-8")
    resumed = fixture.FakeCatalogClient({2: page(5, "c", "d"), 4: page(5, "e")}, [(catalog._operator("rank"),)])
    assert refresh(catalog, resumed).field_count == 5
    assert resumed.field_calls == [(2, 2), (2, 4)]


def test_page_write_failure_rolls_back_data_and_checkpoint_together(catalog):
    def fail_second_page(connection, identity, value):
        save_catalog_field_page(connection, identity, value)
        if value.offset == 2:
            raise sqlite3.OperationalError("synthetic disk failure")
    client = fixture.FakeCatalogClient({0: page(3, "a", "b"), 2: page(3, "c")}, [])
    with patch("execution.catalog_sync.save_catalog_field_page", side_effect=fail_second_page):
        with pytest.raises(sqlite3.OperationalError, match="synthetic disk failure"):
            refresh(catalog, client)
    catalog._assert_old_catalog_preserved()
    with open_database(catalog.database_path) as connection:
        assert [p.offset for p in list_catalog_field_pages(connection, scope(catalog))] == [0]
    resumed = fixture.FakeCatalogClient({0: page(3, "a", "b"), 2: page(3, "c")}, [(catalog._operator("rank"),)])
    assert refresh(catalog, resumed).field_count == 3


def test_publish_failure_keeps_staged_fields_and_previous_catalog(catalog):
    client = fixture.FakeCatalogClient({0: page(1, "a")}, [(catalog._operator("rank"),)])
    with patch("execution.catalog_sync.load_generation_catalog", side_effect=ValueError("invalid semantics")):
        with pytest.raises(ValueError, match="invalid semantics"):
            refresh(catalog, client)
    catalog._assert_old_catalog_preserved()
    with open_database(catalog.database_path) as connection:
        assert len(list_catalog_field_pages(connection, scope(catalog))) == 1
    resumed = fixture.FakeCatalogClient({0: page(1, "a")}, [(catalog._operator("rank"),)])
    assert refresh(catalog, resumed).field_count == 1


def test_existing_writer_lock_prevents_catalog_requests(catalog):
    with exclusive_run_process(catalog.database_path):
        with pytest.raises(ValueError, match="已有命令"):
            refresh_generation_catalog(catalog.database_path, catalog.settings_path, catalog.environment_path,
                client_factory=lambda _: pytest.fail("must not create a platform client"))


def test_init_resumes_committed_pages_after_keyboard_interrupt():
    case = init_fixture.ProjectInitializationTests()
    case.setUp()
    try:
        client = fixture.FakeCatalogClient({0: page(3, "a", "b"), 2: page(3, "c")},
                                          [(fixture.CatalogRefreshTests._operator("rank"),)])
        def stop(_):
            raise KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            initialize_project(case.database_path, case.settings_path, case.environment_path,
                refresh_policy=replace(POLICY, page_interval_seconds=1), clock=lambda: NOW,
                waiter=stop, client_factory=lambda _: client)
        with open_database(case.database_path) as connection:
            assert get_platform_catalog_sync(connection) is None
            assert list_field_catalog(connection, FieldCatalogContext("EQUITY", "USA", "TOP3000", 1)) == ()
        resumed = fixture.FakeCatalogClient({0: page(3, "a", "b"), 2: page(3, "c")},
                                           [(fixture.CatalogRefreshTests._operator("rank"),)])
        result = initialize_project(case.database_path, case.settings_path, case.environment_path,
            refresh_policy=POLICY, clock=lambda: NOW, waiter=lambda _: None, client_factory=lambda _: resumed)
        assert result.catalog.field_count == 3
        assert resumed.field_calls == [(2, 0), (2, 2)]
        reused = initialize_project(case.database_path, case.settings_path, case.environment_path,
            client_factory=lambda _: pytest.fail("complete catalog must be reused"))
        assert not reused.catalog_refreshed
    finally:
        case.doCleanups()
