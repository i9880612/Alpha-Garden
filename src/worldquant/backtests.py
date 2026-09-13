from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from urllib.parse import urljoin, urlsplit


class WorldQuantProtocolError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class WorldQuantDetailMetricUnavailable(WorldQuantProtocolError):
    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name
        super().__init__(f"worldquant_detail_metric_unavailable:{metric_name}")


class WorldQuantZeroCapitalResult(WorldQuantProtocolError):
    def __init__(self) -> None:
        super().__init__("worldquant_detail_zero_capital")


STANDARD_REGULAR_CHECK_NAMES = frozenset(
    {
        "CONCENTRATED_WEIGHT",
        "HIGH_TURNOVER",
        "LOW_FITNESS",
        "LOW_SHARPE",
        "LOW_SUB_UNIVERSE_SHARPE",
        "LOW_TURNOVER",
        "MATCHES_COMPETITION",
        "SELF_CORRELATION",
    }
)
SELF_CORRELATION_CHECK_NAME = "SELF_CORRELATION"
STANDARD_NON_SC_CHECK_NAMES = (
    STANDARD_REGULAR_CHECK_NAMES - {SELF_CORRELATION_CHECK_NAME}
)


@dataclass(frozen=True, slots=True)
class BacktestSettings:
    instrument_type: str
    region: str
    universe: str
    delay: int
    decay: int
    neutralization: str
    truncation: float
    pasteurization: str
    unit_handling: str
    nan_handling: str
    language: str
    visualization: bool
    max_trade: str
    max_position: str

    def __post_init__(self) -> None:
        for value in (
            self.instrument_type,
            self.region,
            self.universe,
            self.neutralization,
            self.pasteurization,
            self.unit_handling,
            self.nan_handling,
            self.language,
            self.max_trade,
            self.max_position,
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("worldquant_backtest_setting_text_invalid")
        for value, name in ((self.delay, "delay"), (self.decay, "decay")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"worldquant_backtest_setting_{name}_invalid")
        if (
            isinstance(self.truncation, bool)
            or not isinstance(self.truncation, (int, float))
            or not math.isfinite(self.truncation)
            or not 0 <= self.truncation <= 1
        ):
            raise ValueError("worldquant_backtest_setting_truncation_invalid")
        if not isinstance(self.visualization, bool):
            raise ValueError("worldquant_backtest_setting_visualization_invalid")

    def as_platform_dict(self) -> dict[str, object]:
        return {
            "instrumentType": self.instrument_type,
            "region": self.region,
            "universe": self.universe,
            "delay": self.delay,
            "decay": self.decay,
            "neutralization": self.neutralization,
            "truncation": self.truncation,
            "pasteurization": self.pasteurization,
            "unitHandling": self.unit_handling,
            "nanHandling": self.nan_handling,
            "language": self.language,
            "visualization": self.visualization,
            "maxTrade": self.max_trade,
            "maxPosition": self.max_position,
        }

    @classmethod
    def from_platform_dict(
        cls,
        value: Mapping[str, object],
    ) -> "BacktestSettings":
        required = {
            "instrumentType",
            "region",
            "universe",
            "delay",
            "decay",
            "neutralization",
            "truncation",
            "pasteurization",
            "unitHandling",
            "nanHandling",
            "language",
            "visualization",
            "maxTrade",
            "maxPosition",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("worldquant_backtest_settings_shape_invalid")
        return cls(
            instrument_type=value["instrumentType"],
            region=value["region"],
            universe=value["universe"],
            delay=value["delay"],
            decay=value["decay"],
            neutralization=value["neutralization"],
            truncation=value["truncation"],
            pasteurization=value["pasteurization"],
            unit_handling=value["unitHandling"],
            nan_handling=value["nanHandling"],
            language=value["language"],
            visualization=value["visualization"],
            max_trade=value["maxTrade"],
            max_position=value["maxPosition"],
        )


@dataclass(frozen=True, slots=True)
class BacktestSubmissionObservation:
    state: str
    remote_id: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class BacktestPollObservation:
    state: str
    platform_alpha_id: str | None
    platform_status: str
    progress: float | None
    failure_message: str | None = None


@dataclass(frozen=True, slots=True)
class BacktestCheck:
    name: str
    status: str
    threshold: float | None
    actual: float | None
    platform_date: str | None


@dataclass(frozen=True, slots=True)
class BacktestDetail:
    platform_alpha_id: str
    sharpe: float
    fitness: float
    turnover: float
    returns: float
    drawdown: float
    margin: float
    book_size: float | None
    pnl: float | None
    checks: tuple[BacktestCheck, ...]
    long_count: int | None = None
    short_count: int | None = None
    grade: str | None = None

    @property
    def check_set_complete(self) -> bool:
        return frozenset(check.name for check in self.checks) == (
            STANDARD_REGULAR_CHECK_NAMES
        )

    @property
    def has_failed_check(self) -> bool:
        return any(check.status.strip().upper() == "FAIL" for check in self.checks)


@dataclass(frozen=True, slots=True)
class BacktestYearlyStat:
    year: int
    pnl: float | None
    book_size: float | None
    turnover: float | None
    sharpe: float | None
    returns: float | None
    drawdown: float | None
    margin: float | None
    fitness: float | None
    long_count: int | None
    short_count: int | None
    stage: str


@dataclass(frozen=True, slots=True)
class BacktestYearlyStatsObservation:
    state: str
    stats: tuple[BacktestYearlyStat, ...]
    retry_after_seconds: float | None

    def __post_init__(self) -> None:
        if self.state not in {"pending", "ready"}:
            raise ValueError("worldquant_yearly_stats_observation_state_invalid")
        if not isinstance(self.stats, tuple) or any(
            not isinstance(stat, BacktestYearlyStat) for stat in self.stats
        ):
            raise ValueError("worldquant_yearly_stats_observation_stats_invalid")
        if self.retry_after_seconds is not None and (
            isinstance(self.retry_after_seconds, bool)
            or not isinstance(self.retry_after_seconds, (int, float))
            or not math.isfinite(self.retry_after_seconds)
            or self.retry_after_seconds <= 0
        ):
            raise ValueError("worldquant_yearly_stats_observation_retry_after_invalid")
        if self.state == "pending" and self.stats:
            raise ValueError("worldquant_yearly_stats_observation_stats_invalid")
        if self.state == "ready" and self.retry_after_seconds is not None:
            raise ValueError("worldquant_yearly_stats_observation_retry_after_invalid")


def build_backtest_request(
    formula: str,
    settings: BacktestSettings,
) -> dict[str, object]:
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("worldquant_backtest_formula_missing")
    return {
        "type": "REGULAR",
        "settings": settings.as_platform_dict(),
        "regular": formula,
    }


def backtest_settings_match(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    return isinstance(actual, Mapping) and isinstance(expected, Mapping) and all(
        name in actual and _same_json_value(actual[name], value)
        for name, value in expected.items()
    )


def parse_submission_response(
    *,
    status_code: int,
    headers: Mapping[str, str],
    base_url: str,
) -> BacktestSubmissionObservation:
    if status_code != 201:
        raise WorldQuantProtocolError("worldquant_submission_status_unexpected")
    location = _header_value(headers, "location")
    if location is None or not location.strip():
        return BacktestSubmissionObservation(
            state="unknown",
            remote_id=None,
            reason="accepted_without_remote_location",
        )
    remote_id = _remote_simulation_url(base_url, location)
    if remote_id is None:
        return BacktestSubmissionObservation(
            state="unknown",
            remote_id=None,
            reason="accepted_with_invalid_remote_location",
        )
    return BacktestSubmissionObservation(
        state="accepted",
        remote_id=remote_id,
        reason=None,
    )


def parse_poll_response(payload: Mapping[str, object]) -> BacktestPollObservation:
    if not isinstance(payload, Mapping):
        raise WorldQuantProtocolError("worldquant_poll_payload_invalid")
    alpha_id = _optional_text(
        payload.get("alpha"),
        "worldquant_poll_alpha_id_invalid",
    )
    raw_status = payload.get("status")
    if raw_status is None:
        if alpha_id is not None:
            raise WorldQuantProtocolError("worldquant_poll_status_missing")
        progress = _required_progress(payload)
        return BacktestPollObservation(
            state="pending",
            platform_alpha_id=None,
            platform_status="RUNNING",
            progress=progress,
        )
    if not isinstance(raw_status, str) or not raw_status.strip():
        raise WorldQuantProtocolError("worldquant_poll_status_invalid")
    status = raw_status.strip().upper()
    if status in {"COMPLETE", "COMPLETED"}:
        if alpha_id is None:
            raise WorldQuantProtocolError("worldquant_poll_completed_alpha_missing")
        return BacktestPollObservation(
            state="completed",
            platform_alpha_id=alpha_id,
            platform_status="COMPLETE",
            progress=None,
        )
    if status in {"RUNNING", "PENDING"}:
        if alpha_id is not None:
            raise WorldQuantProtocolError("worldquant_poll_state_conflict")
        return BacktestPollObservation(
            state="pending",
            platform_alpha_id=None,
            platform_status=status,
            progress=_optional_progress(payload),
        )
    if status in {"FAILED", "ERROR", "CANCELLED", "CANCELED"}:
        if alpha_id is not None:
            raise WorldQuantProtocolError("worldquant_poll_state_conflict")
        normalized = "CANCELLED" if status == "CANCELED" else status
        return BacktestPollObservation(
            state="failed",
            platform_alpha_id=None,
            platform_status=normalized,
            progress=_optional_progress(payload),
        )
    if status == "WARNING":
        if alpha_id is None:
            raise WorldQuantProtocolError("worldquant_poll_warning_alpha_missing")
        message = _required_text(
            payload.get("message"),
            "worldquant_poll_warning_message_missing",
        )
        return BacktestPollObservation(
            state="failed",
            platform_alpha_id=alpha_id,
            platform_status="WARNING",
            progress=None,
            failure_message=message,
        )
    raise WorldQuantProtocolError("worldquant_poll_status_unknown")


def parse_backtest_detail(
    payload: Mapping[str, object],
    *,
    expected_alpha_id: str,
    expected_formula: str,
    expected_settings: Mapping[str, object],
) -> BacktestDetail:
    if not isinstance(payload, Mapping):
        raise WorldQuantProtocolError("worldquant_detail_payload_invalid")
    _require_text(expected_alpha_id, "worldquant_detail_expected_alpha_missing")
    _require_text(expected_formula, "worldquant_detail_expected_formula_missing")
    alpha_id = _required_text(payload.get("id"), "worldquant_detail_alpha_missing")
    if alpha_id != expected_alpha_id:
        raise WorldQuantProtocolError("worldquant_detail_alpha_mismatch")
    if payload.get("type") != "REGULAR":
        raise WorldQuantProtocolError("worldquant_detail_type_invalid")

    regular = payload.get("regular")
    if not isinstance(regular, Mapping):
        raise WorldQuantProtocolError("worldquant_detail_formula_missing")
    formula = _required_text(
        regular.get("code"),
        "worldquant_detail_formula_missing",
    )
    if formula != expected_formula:
        raise WorldQuantProtocolError("worldquant_detail_formula_mismatch")

    response_settings = payload.get("settings")
    if not isinstance(response_settings, Mapping):
        raise WorldQuantProtocolError("worldquant_detail_settings_missing")
    if not backtest_settings_match(response_settings, expected_settings):
        raise WorldQuantProtocolError("worldquant_detail_settings_mismatch")

    metrics = payload.get("is")
    if not isinstance(metrics, Mapping):
        raise WorldQuantProtocolError("worldquant_detail_metrics_missing")
    checks = _parse_checks(metrics.get("checks"))
    if _is_zero_capital_result(metrics):
        raise WorldQuantZeroCapitalResult()
    return BacktestDetail(
        platform_alpha_id=alpha_id,
        sharpe=_required_number(metrics, "sharpe"),
        fitness=_required_number(metrics, "fitness"),
        turnover=_required_number(metrics, "turnover"),
        returns=_required_number(metrics, "returns"),
        drawdown=_required_number(metrics, "drawdown"),
        margin=_required_number(metrics, "margin"),
        book_size=_optional_number(metrics, "bookSize"),
        pnl=_optional_number(metrics, "pnl"),
        checks=checks,
        long_count=_optional_nonnegative_integer(metrics, "longCount"),
        short_count=_optional_nonnegative_integer(metrics, "shortCount"),
        grade=parse_alpha_grade(payload),
    )


def parse_alpha_grade(payload: Mapping[str, object]) -> str | None:
    """Keep a platform label; missing or malformed labels provide no grade evidence."""
    value = payload.get("grade")
    return value.strip().upper() if isinstance(value, str) and value.strip() else None


_YEARLY_STATS_PROPERTY_TYPES = {
    "year": "year",
    "pnl": "amount",
    "bookSize": "amount",
    "longCount": "integer",
    "shortCount": "integer",
    "turnover": "percent",
    "sharpe": "decimal",
    "returns": "percent",
    "drawdown": "percent",
    "margin": "permyriad",
    "fitness": "decimal",
    "stage": "string",
}


def parse_backtest_yearly_stats(
    payload: Mapping[str, object],
) -> tuple[BacktestYearlyStat, ...]:
    if not isinstance(payload, Mapping) or set(payload) != {"schema", "records"}:
        raise WorldQuantProtocolError("worldquant_yearly_stats_payload_invalid")
    schema = payload["schema"]
    records = payload["records"]
    if not isinstance(schema, Mapping) or set(schema) != {
        "name",
        "properties",
        "title",
    }:
        raise WorldQuantProtocolError("worldquant_yearly_stats_schema_invalid")
    if schema["name"] != "yearly-stats":
        raise WorldQuantProtocolError("worldquant_yearly_stats_schema_invalid")
    _required_text(schema["title"], "worldquant_yearly_stats_schema_invalid")
    properties = schema["properties"]
    if not isinstance(properties, list) or len(properties) != len(
        _YEARLY_STATS_PROPERTY_TYPES
    ):
        raise WorldQuantProtocolError("worldquant_yearly_stats_schema_invalid")

    names: list[str] = []
    for property_value in properties:
        if not isinstance(property_value, Mapping) or set(property_value) != {
            "name",
            "title",
            "type",
        }:
            raise WorldQuantProtocolError("worldquant_yearly_stats_schema_invalid")
        name = property_value["name"]
        if (
            not isinstance(name, str)
            or _YEARLY_STATS_PROPERTY_TYPES.get(name) != property_value["type"]
            or name in names
        ):
            raise WorldQuantProtocolError("worldquant_yearly_stats_schema_invalid")
        _required_text(
            property_value["title"],
            "worldquant_yearly_stats_schema_invalid",
        )
        names.append(name)
    if set(names) != set(_YEARLY_STATS_PROPERTY_TYPES):
        raise WorldQuantProtocolError("worldquant_yearly_stats_schema_invalid")
    if not isinstance(records, list):
        raise WorldQuantProtocolError("worldquant_yearly_stats_records_invalid")

    stats: list[BacktestYearlyStat] = []
    identities: set[tuple[str, int]] = set()
    for row in records:
        if not isinstance(row, list) or len(row) != len(names):
            raise WorldQuantProtocolError("worldquant_yearly_stats_record_invalid")
        values = dict(zip(names, row, strict=True))
        year = _yearly_stats_year(values["year"])
        raw_stage = values["stage"]
        stage = _required_text(
            raw_stage,
            "worldquant_yearly_stats_stage_invalid",
        )
        if stage != raw_stage:
            raise WorldQuantProtocolError("worldquant_yearly_stats_stage_invalid")
        identity = (stage, year)
        if identity in identities:
            raise WorldQuantProtocolError("worldquant_yearly_stats_record_duplicate")
        identities.add(identity)
        stats.append(
            BacktestYearlyStat(
                year=year,
                pnl=_optional_yearly_stats_number(values["pnl"], "pnl"),
                book_size=_optional_yearly_stats_number(
                    values["bookSize"], "book_size"
                ),
                turnover=_optional_yearly_stats_number(
                    values["turnover"], "turnover"
                ),
                sharpe=_optional_yearly_stats_number(values["sharpe"], "sharpe"),
                returns=_optional_yearly_stats_number(values["returns"], "returns"),
                drawdown=_optional_yearly_stats_number(
                    values["drawdown"], "drawdown"
                ),
                margin=_optional_yearly_stats_number(values["margin"], "margin"),
                fitness=_optional_yearly_stats_number(
                    values["fitness"], "fitness"
                ),
                long_count=_optional_yearly_stats_count(
                    values["longCount"], "long_count"
                ),
                short_count=_optional_yearly_stats_count(
                    values["shortCount"], "short_count"
                ),
                stage=stage,
            )
        )
    return tuple(sorted(stats, key=lambda stat: (stat.stage, stat.year)))


def _remote_simulation_url(base_url: str, location: str) -> str | None:
    base_origin = _origin(base_url)
    if base_origin is None:
        raise ValueError("worldquant_base_url_invalid")
    try:
        remote_id = urljoin(f"{base_url.rstrip('/')}/", location)
        parsed = urlsplit(remote_id)
    except (TypeError, ValueError):
        return None
    if _origin(remote_id) != base_origin:
        return None
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        return None
    path_parts = tuple(part for part in parsed.path.split("/") if part)
    if len(path_parts) != 2 or path_parts[0] != "simulations" or not path_parts[1]:
        return None
    return remote_id


def _origin(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        return None
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    if scheme not in {"http", "https"} or hostname is None:
        return None
    default_port = 443 if scheme == "https" else 80
    return scheme, hostname.lower(), port if port is not None else default_port


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == name:
            return value if isinstance(value, str) else None
    return None


def _required_progress(payload: Mapping[str, object]) -> float:
    if "progress" not in payload:
        raise WorldQuantProtocolError("worldquant_poll_state_missing")
    progress = _number(payload["progress"], "worldquant_poll_progress_invalid")
    if not 0 <= progress <= 1:
        raise WorldQuantProtocolError("worldquant_poll_progress_invalid")
    return progress


def _optional_progress(payload: Mapping[str, object]) -> float | None:
    if "progress" not in payload:
        return None
    return _required_progress(payload)


def _parse_checks(value: object) -> tuple[BacktestCheck, ...]:
    if not isinstance(value, list) or not value:
        raise WorldQuantProtocolError("worldquant_detail_checks_invalid")
    checks: list[BacktestCheck] = []
    names: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise WorldQuantProtocolError("worldquant_detail_checks_invalid")
        name = _required_text(
            item.get("name"),
            "worldquant_detail_check_name_invalid",
        )
        status = _required_text(
            item.get("result"),
            "worldquant_detail_check_result_invalid",
        )
        if name in names:
            raise WorldQuantProtocolError("worldquant_detail_check_duplicate")
        names.add(name)
        checks.append(
            BacktestCheck(
                name=name,
                status=status,
                threshold=_optional_check_number(item, "limit"),
                actual=_optional_check_number(item, "value"),
                platform_date=_optional_check_date(item.get("date")),
            )
        )
    return tuple(sorted(checks, key=lambda check: check.name))


def _optional_check_number(item: Mapping[str, object], name: str) -> float | None:
    if name not in item or item[name] is None:
        return None
    return _number(item[name], f"worldquant_detail_check_{name}_invalid")


def _optional_check_date(value: object) -> str | None:
    if value is None:
        return None
    raw = _required_text(value, "worldquant_detail_check_date_invalid")
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise WorldQuantProtocolError("worldquant_detail_check_date_invalid") from exc
    if raw != parsed.isoformat():
        raise WorldQuantProtocolError("worldquant_detail_check_date_invalid")
    return raw


def _required_number(metrics: Mapping[str, object], name: str) -> float:
    if name not in metrics:
        raise WorldQuantProtocolError(f"worldquant_detail_metric_missing:{name}")
    if metrics[name] is None:
        raise WorldQuantDetailMetricUnavailable(name)
    return _number(metrics[name], f"worldquant_detail_metric_invalid:{name}")


def _is_zero_capital_result(metrics: Mapping[str, object]) -> bool:
    if not all(
        _is_explicit_zero(metrics.get(name))
        for name in (
            "sharpe",
            "turnover",
            "returns",
            "drawdown",
            "margin",
            "bookSize",
            "pnl",
        )
    ):
        return False
    fitness = metrics.get("fitness")
    return fitness is None or _is_explicit_zero(fitness)


def _is_explicit_zero(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and float(value) == 0.0
    )


def _optional_number(metrics: Mapping[str, object], name: str) -> float | None:
    if name not in metrics or metrics[name] is None:
        return None
    return _number(metrics[name], f"worldquant_detail_metric_invalid:{name}")


def _optional_nonnegative_integer(
    metrics: Mapping[str, object],
    name: str,
) -> int | None:
    if name not in metrics or metrics[name] is None:
        return None
    return _nonnegative_integer(
        metrics[name],
        f"worldquant_detail_metric_invalid:{name}",
    )


def _yearly_stats_year(value: object) -> int:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise WorldQuantProtocolError("worldquant_yearly_stats_year_invalid")
    year = int(value)
    if year <= 0 or str(year) != value:
        raise WorldQuantProtocolError("worldquant_yearly_stats_year_invalid")
    return year


def _optional_yearly_stats_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _number(value, f"worldquant_yearly_stats_{name}_invalid")


def _optional_yearly_stats_count(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, f"worldquant_yearly_stats_{name}_invalid")


def _nonnegative_integer(value: object, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorldQuantProtocolError(error)
    return value


def _number(value: object, error: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise WorldQuantProtocolError(error)
    return float(value)


def _optional_text(value: object, error: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, error)


def _required_text(value: object, error: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorldQuantProtocolError(error)
    return value.strip()


def _require_text(value: object, error: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)


def _same_json_value(actual: object, expected: object) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isfinite(actual) and math.isfinite(expected) and actual == expected
    return type(actual) is type(expected) and actual == expected
