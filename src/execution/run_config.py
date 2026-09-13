from __future__ import annotations

import json
from collections.abc import Mapping
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
) -> AutomatedRunLimits:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        limits = _limits_from_config(
            payload,
            cycles=cycles,
            automatic_submissions_enabled=automatic_submissions_enabled,
            optimization_only=optimization_only,
        )
        validate_automated_run_limits(limits)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("automated_run_config_file_invalid") from exc
    return limits


def _limits_from_config(
    payload: object,
    *,
    cycles: int,
    automatic_submissions_enabled: bool,
    optimization_only: bool,
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
    return AutomatedRunLimits(
        **values,
        max_cycles=cycles,
        max_backtests=max_backtests,
        real_backtests_authorized=True,
        automatic_submissions_enabled=automatic_submissions_enabled,
        optimization_only=optimization_only,
    )
