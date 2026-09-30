from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from execution.runs import AutomatedRunLimits, validate_automated_run_limits


_CONFIG_KEYS = {
    "generationCount": "generation_count",
    "backtestCount": "backtest_count",
    "explorationPercent": "exploration_percent",
    "selfCorrelationPercent": "self_correlation_percent",
    "mutationPercent": "mutation_percent",
    "directionValidationPercent": "direction_validation_percent",
    "maxPendingSeconds": "max_pending_seconds",
    "maxConsecutiveFailures": "max_consecutive_failures",
    "maxRequestFailures": "max_request_failures",
    "maxInFlightBacktests": "max_in_flight_backtests",
    "explorationSeedAttemptMultiplier": "exploration_seed_attempt_multiplier",
}


def load_automated_run_limits(
    path: str | Path,
    *,
    cycles: int,
    automatic_submissions_enabled: bool = False,
    optimization_only: bool = False,
    self_correlation_parent_task_id: str | None = None,
    self_correlation_plan_key: str | None = None,
) -> AutomatedRunLimits:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        limits = automated_run_limits_from_config(
            payload,
            cycles=cycles,
            automatic_submissions_enabled=automatic_submissions_enabled,
            optimization_only=optimization_only,
            self_correlation_parent_task_id=self_correlation_parent_task_id,
            self_correlation_plan_key=self_correlation_plan_key,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("automated_run_config_file_invalid") from exc
    return limits


def automated_run_limits_from_config(
    payload: object,
    *,
    cycles: int,
    automatic_submissions_enabled: bool = False,
    optimization_only: bool = False,
    self_correlation_parent_task_id: str | None = None,
    self_correlation_plan_key: str | None = None,
) -> AutomatedRunLimits:
    if not isinstance(payload, Mapping) or set(payload) != set(_CONFIG_KEYS):
        raise ValueError("automated_run_config_invalid")
    values: dict[str, int] = {}
    for config_name, field_name in _CONFIG_KEYS.items():
        value = payload[config_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("automated_run_config_invalid")
        values[field_name] = value
    backtest_count = values["backtest_count"]
    max_backtests = 0 if cycles == -1 else cycles * backtest_count
    if not isinstance(automatic_submissions_enabled, bool):
        raise ValueError("automated_run_automatic_submission_setting_invalid")
    limits = AutomatedRunLimits(
        **values,
        max_cycles=cycles,
        max_backtests=max_backtests,
        real_backtests_authorized=True,
        automatic_submissions_enabled=automatic_submissions_enabled,
        optimization_only=optimization_only,
    )
    validate_automated_run_limits(limits)
    if self_correlation_parent_task_id is not None:
        if cycles != 3:
            raise ValueError("sc_research_requires_three_cycles")
        limits = replace(
            limits, generation_count=2, backtest_count=2, max_backtests=6,
            max_in_flight_backtests=min(2, limits.max_in_flight_backtests),
            self_correlation_parent_task_id=self_correlation_parent_task_id,
            self_correlation_plan_key=self_correlation_plan_key,
        )
    elif self_correlation_plan_key is not None:
        raise ValueError("automated_run_sc_plan_key_invalid")
    validate_automated_run_limits(limits)
    return limits
