from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from execution.runner import (
    AutomatedRunCompletion,
    Clock,
    Waiter,
    run_automated_run,
    validate_automated_run_poll_interval,
)
from execution.cycle_backtests import failed_automated_run_has_local_work
from execution.catalog import load_generation_catalog
from execution.process_lock import exclusive_run_process
from execution.run_recovery import settle_stopped_run_results
from execution.runs import (
    AutomatedRunLimits,
    prepare_automated_run,
    retire_previous_automated_runs,
    validate_automated_run_limits,
    resume_request_failed_run,
)
from persistence.catalog import FieldCatalogContext
from persistence.database import open_database
from persistence.runs import (
    AutomatedRunRecord,
    automated_run_has_submission_unknown,
    get_automated_run,
)
from selection.settings import BacktestSettingsPolicy, load_backtest_settings_policy
from worldquant.client import WorldQuantClient
from worldquant.config import (
    WorldQuantConnectionSettings,
    load_worldquant_connection_settings,
)


ClientFactory = Callable[[WorldQuantConnectionSettings], WorldQuantClient]
RunCreatedObserver = Callable[[AutomatedRunRecord], None]


def launch_automated_run(
    database_path: str | Path,
    policy_path: str | Path,
    environment_path: str | Path,
    *,
    limits: AutomatedRunLimits,
    poll_interval_seconds: float = 1.0,
    clock: Clock | None = None,
    waiter: Waiter | None = None,
    client_factory: ClientFactory | None = None,
    run_created: RunCreatedObserver | None = None,
) -> AutomatedRunCompletion:
    _validate_launch_authority(limits)
    validate_automated_run_poll_interval(poll_interval_seconds)
    connection_settings = load_worldquant_connection_settings(environment_path)

    current_time = clock or _utc_now
    with exclusive_run_process(database_path):
        created_at = _aware_time(current_time()).isoformat()
        # Bad new inputs must not stop a previously valid plan.
        policy = load_backtest_settings_policy(policy_path)
        with open_database(database_path) as connection:
            load_generation_catalog(
                connection,
                FieldCatalogContext(policy.instrument_type, policy.region,
                                    policy.universe, policy.delay),
                account_scope=connection_settings.account_scope,
            )
        retire_previous_automated_runs(
            database_path, account_scope=connection_settings.account_scope,
            observed_at=created_at,
        )
        settle_stopped_run_results(
            database_path, account_scope=connection_settings.account_scope,
            observed_at=created_at,
        )
        run = prepare_automated_run(
            database_path,
            policy_path,
            account_scope=connection_settings.account_scope,
            limits=limits,
            created_at=created_at,
        )
        if run_created is not None:
            run_created(run)
        make_client = client_factory or _default_client_factory
        client = make_client(connection_settings)
        return run_automated_run(
            database_path,
            client,
            run.run_id,
            poll_interval_seconds=poll_interval_seconds,
            clock=current_time,
            waiter=waiter,
        )


def resume_automated_run(
    database_path: str | Path,
    environment_path: str | Path,
    run_id: str,
    *,
    poll_interval_seconds: float = 1.0,
    clock: Clock | None = None,
    waiter: Waiter | None = None,
    client_factory: ClientFactory | None = None,
) -> AutomatedRunCompletion:
    validate_automated_run_poll_interval(poll_interval_seconds)
    connection_settings = load_worldquant_connection_settings(environment_path)
    with exclusive_run_process(database_path):
        _require_run_account_scope(
            database_path,
            run_id,
            connection_settings.account_scope,
        )
        resume_request_failed_run(database_path, run_id)
        _require_run_account_scope(database_path, run_id, connection_settings.account_scope)
        make_client = client_factory or _default_client_factory
        client = make_client(connection_settings)
        return run_automated_run(
            database_path,
            client,
            run_id,
            poll_interval_seconds=poll_interval_seconds,
            clock=clock,
            waiter=waiter,
        )


def _validate_launch_authority(limits: AutomatedRunLimits) -> None:
    validate_automated_run_limits(limits)
    if not limits.real_backtests_authorized:
        raise ValueError("automated_run_backtests_not_authorized")


def _default_client_factory(
    settings: WorldQuantConnectionSettings,
) -> WorldQuantClient:
    return WorldQuantClient(
        base_url=settings.base_url,
        credentials=settings.credentials,
    )


def _require_run_account_scope(
    database_path: str | Path,
    run_id: str,
    account_scope: str,
) -> None:
    with open_database(database_path) as connection:
        run = get_automated_run(connection, run_id)
        if run is None:
            raise ValueError("automated_run_missing")
        if run.account_scope != account_scope:
            raise ValueError("automated_run_account_scope_mismatch")
        if run.status == "completed":
            raise ValueError("automated_run_already_completed")
        if run.status == "failed":
            if (
                run.stop_reason != "submission_reconciliation_required"
                and run.stop_reason != "request_failure_limit_reached"
                and not automated_run_has_submission_unknown(connection, run.run_id)
                and not failed_automated_run_has_local_work(connection, run)
            ):
                raise ValueError("automated_run_already_failed")
        if run.status in {"created", "running"}:
            settings_policy = BacktestSettingsPolicy.from_config_dict(
                json.loads(run.settings_policy_json)
            )
            load_generation_catalog(
                connection,
                FieldCatalogContext(
                    instrument_type=settings_policy.instrument_type,
                    region=settings_policy.region,
                    universe=settings_policy.universe,
                    delay=settings_policy.delay,
                ),
                account_scope=run.account_scope,
            )

def _aware_time(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("automated_run_clock_invalid")
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)
