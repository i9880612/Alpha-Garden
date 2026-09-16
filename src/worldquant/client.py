from __future__ import annotations

import base64
import json
import math
import socket
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.cookiejar import CookieJar
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPCookieProcessor, Request, build_opener

from worldquant.alphas import UserAlphaPage, parse_user_alpha_page
from worldquant.pnl import PnlObservation, parse_pnl
from worldquant.recordsets import RECORDSET_UNITS, RecordsetObservation, parse_recordset
from worldquant.backtests import (
    BacktestDetail,
    BacktestPollObservation,
    BacktestSettings,
    BacktestSubmissionObservation,
    BacktestYearlyStatsObservation,
    WorldQuantProtocolError,
    build_backtest_request,
    parse_backtest_detail,
    parse_backtest_yearly_stats,
    parse_poll_response,
    parse_submission_response,
)
from worldquant.catalog import (
    CatalogContext,
    DataFieldPage,
    Operator,
    parse_data_field_page,
    parse_operators,
)
from worldquant.submissions import (
    AlphaDetailObservation,
    FormalCheckObservation,
    FormalSubmissionObservation,
)


DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class WorldQuantCredentials:
    username: str | None = field(default=None, repr=False)
    password: str | None = field(default=None, repr=False)
    bearer_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        has_basic = bool(
            isinstance(self.username, str)
            and self.username.strip()
            and isinstance(self.password, str)
            and self.password
        )
        has_bearer = bool(
            isinstance(self.bearer_token, str) and self.bearer_token.strip()
        )
        if has_basic == has_bearer:
            raise ValueError("worldquant_credentials_invalid")


@dataclass(frozen=True, slots=True)
class WorldQuantResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True, slots=True)
class AuthenticationResult:
    status_code: int


class WorldQuantRequestError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        status_code: int | None = None,
        retryable: bool,
        outcome_unknown: bool,
        retry_after_seconds: float | None = None,
        transport_error_type: str | None = None,
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown
        self.retry_after_seconds = retry_after_seconds
        self.transport_error_type = transport_error_type
        super().__init__(code)


RequestExecutor = Callable[[Request, float], WorldQuantResponse]


class WorldQuantClient:
    def __init__(
        self,
        *,
        base_url: str,
        credentials: WorldQuantCredentials,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        request_executor: RequestExecutor | None = None,
    ) -> None:
        self.base_url = _validated_base_url(base_url)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("worldquant_timeout_invalid")
        self.credentials = credentials
        self.timeout_seconds = float(timeout_seconds)
        self._request_executor = request_executor or _UrllibExecutor()
        self._authenticated = False

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    def authenticate(self) -> AuthenticationResult:
        self._authenticated = False
        headers = {"Accept": "application/json"}
        if self.credentials.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self.credentials.bearer_token.strip()}"
        else:
            assert self.credentials.username is not None
            assert self.credentials.password is not None
            raw = f"{self.credentials.username.strip()}:{self.credentials.password}"
            encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {encoded}"
        response = self._send(
            Request(
                self._url("/authentication"),
                data=b"",
                headers=headers,
                method="POST",
            ),
            code="worldquant_authentication_request_failed",
            outcome_unknown=False,
        )
        if response.status_code not in {200, 201}:
            raise _http_error(
                "worldquant_authentication_rejected",
                response,
                outcome_unknown=False,
            )
        self._authenticated = True
        return AuthenticationResult(status_code=response.status_code)

    def fetch_data_field_page(
        self,
        *,
        context: CatalogContext,
        limit: int,
        offset: int,
    ) -> DataFieldPage:
        self._require_authenticated()
        if not isinstance(context, CatalogContext):
            raise ValueError("worldquant_catalog_context_invalid")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 50
        ):
            raise ValueError("worldquant_data_fields_limit_invalid")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("worldquant_data_fields_offset_invalid")
        query = urlencode(
            {
                "instrumentType": context.instrument_type,
                "region": context.region,
                "universe": context.universe,
                "delay": context.delay,
                "limit": limit,
                "offset": offset,
            }
        )
        response = self._send(
            Request(
                self._url(f"/data-fields?{query}"),
                headers={"Accept": "application/json"},
                method="GET",
            ),
            code="worldquant_data_fields_request_failed",
            outcome_unknown=False,
        )
        if response.status_code != 200:
            raise _http_error(
                "worldquant_data_fields_http_error",
                response,
                outcome_unknown=False,
            )
        return parse_data_field_page(
            _json_value(response.body, "worldquant_data_fields_json_invalid"),
            expected_context=context,
        )

    def fetch_operators(self) -> tuple[Operator, ...]:
        self._require_authenticated()
        response = self._send(
            Request(
                self._url("/operators"),
                headers={"Accept": "application/json"},
                method="GET",
            ),
            code="worldquant_operators_request_failed",
            outcome_unknown=False,
        )
        if response.status_code != 200:
            raise _http_error(
                "worldquant_operators_http_error",
                response,
                outcome_unknown=False,
            )
        return parse_operators(
            _json_value(response.body, "worldquant_operators_json_invalid")
        )

    def submit_backtest(
        self,
        *,
        formula: str,
        settings: BacktestSettings,
    ) -> BacktestSubmissionObservation:
        self._require_authenticated()
        payload = json.dumps(
            build_backtest_request(formula, settings),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        response = self._send(
            Request(
                self._url("/simulations"),
                data=payload,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                method="POST",
            ),
            code="worldquant_submission_request_failed",
            outcome_unknown=True,
        )
        if response.status_code != 201:
            raise _http_error(
                "worldquant_submission_http_error",
                response,
                outcome_unknown=response.status_code >= 500,
            )
        return parse_submission_response(
            status_code=response.status_code,
            headers=response.headers,
            base_url=self.base_url,
        )

    def poll_backtest(self, remote_id: str) -> BacktestPollObservation:
        self._require_authenticated()
        if not _is_remote_simulation(self.base_url, remote_id):
            raise ValueError("worldquant_remote_simulation_invalid")
        response = self._send(
            Request(
                remote_id,
                headers={"Accept": "application/json"},
                method="GET",
            ),
            code="worldquant_poll_request_failed",
            outcome_unknown=False,
        )
        if response.status_code != 200:
            raise _http_error(
                "worldquant_poll_http_error",
                response,
                outcome_unknown=False,
            )
        return parse_poll_response(
            _json_object(response.body, "worldquant_poll_json_invalid")
        )

    def fetch_backtest_detail(
        self,
        *,
        platform_alpha_id: str,
        expected_formula: str,
        expected_settings: Mapping[str, object],
    ) -> BacktestDetail:
        self._require_authenticated()
        if not isinstance(platform_alpha_id, str) or not platform_alpha_id.strip():
            raise ValueError("worldquant_alpha_id_missing")
        encoded_alpha_id = quote(platform_alpha_id.strip(), safe="")
        response = self._send(
            Request(
                self._url(f"/alphas/{encoded_alpha_id}"),
                headers={"Accept": "application/json"},
                method="GET",
            ),
            code="worldquant_detail_request_failed",
            outcome_unknown=False,
        )
        if response.status_code != 200:
            raise _http_error(
                "worldquant_detail_http_error",
                response,
                outcome_unknown=False,
            )
        return parse_backtest_detail(
            _json_object(response.body, "worldquant_detail_json_invalid"),
            expected_alpha_id=platform_alpha_id.strip(),
            expected_formula=expected_formula,
            expected_settings=expected_settings,
        )

    def fetch_backtest_yearly_stats(
        self,
        *,
        platform_alpha_id: str,
    ) -> BacktestYearlyStatsObservation:
        self._require_authenticated()
        if not isinstance(platform_alpha_id, str) or not platform_alpha_id.strip():
            raise ValueError("worldquant_alpha_id_missing")
        encoded_alpha_id = quote(platform_alpha_id.strip(), safe="")
        response = self._send(
            Request(
                self._url(
                    f"/alphas/{encoded_alpha_id}/recordsets/yearly-stats"
                ),
                headers={"Accept": "application/json"},
                method="GET",
            ),
            code="worldquant_yearly_stats_request_failed",
            outcome_unknown=False,
        )
        retry_after_seconds = _retry_after_seconds(response.headers)
        if response.status_code == 503 and retry_after_seconds is not None:
            return BacktestYearlyStatsObservation(
                state="pending",
                stats=(),
                retry_after_seconds=retry_after_seconds,
            )
        if response.status_code != 200:
            raise _http_error(
                "worldquant_yearly_stats_http_error",
                response,
                outcome_unknown=False,
            )
        if not response.body.strip():
            return BacktestYearlyStatsObservation(
                state="pending",
                stats=(),
                retry_after_seconds=retry_after_seconds,
            )
        return BacktestYearlyStatsObservation(
            state="ready",
            stats=parse_backtest_yearly_stats(
                _json_object(
                    response.body,
                    "worldquant_yearly_stats_json_invalid",
                )
            ),
            retry_after_seconds=None,
        )

    def fetch_recordset(self, *, platform_alpha_id: str, metric: str) -> RecordsetObservation:
        self._require_authenticated()
        if metric not in RECORDSET_UNITS:
            raise ValueError("worldquant_recordset_metric_invalid")
        if not isinstance(platform_alpha_id, str) or not platform_alpha_id.strip():
            raise ValueError("worldquant_alpha_id_missing")
        encoded = quote(platform_alpha_id.strip(), safe="")
        response = self._send(Request(self._url(f"/alphas/{encoded}/recordsets/{metric}"),
            headers={"Accept": "application/json"}, method="GET"),
            code="worldquant_recordset_request_failed", outcome_unknown=False)
        retry = _retry_after_seconds(response.headers)
        if (response.status_code == 200 and not response.body.strip()) or (response.status_code == 503 and retry is not None):
            return RecordsetObservation(None, retry)
        if response.status_code != 200:
            raise _http_error("worldquant_recordset_http_error", response, outcome_unknown=False)
        payload = _json_object(response.body, "worldquant_recordset_json_invalid")
        try:
            return RecordsetObservation(parse_recordset(payload, metric))
        except ValueError as exc:
            raise WorldQuantProtocolError("worldquant_recordset_response_invalid") from exc

    def fetch_pnl(self, *, platform_alpha_id: str) -> PnlObservation:
        self._require_authenticated()
        if not isinstance(platform_alpha_id, str) or not platform_alpha_id.strip():
            raise ValueError("worldquant_alpha_id_missing")
        encoded = quote(platform_alpha_id.strip(), safe="")
        short_retry_available = True
        while True:
            response = self._send(
                Request(
                    self._url(f"/alphas/{encoded}/recordsets/pnl"),
                    headers={"Accept": "application/json"},
                    method="GET",
                ),
                code="worldquant_pnl_request_failed",
                outcome_unknown=False,
            )
            retry = _retry_after_seconds(response.headers)
            if (response.status_code == 200 and not response.body.strip()) or (
                response.status_code == 503 and retry is not None
            ):
                # PnL can be prepared on demand. Give a short Retry-After one
                # follow-up GET before returning to the ordinary cooldown.
                if short_retry_available and retry is not None and retry <= self.timeout_seconds:
                    short_retry_available = False
                    time.sleep(retry)
                    continue
                return PnlObservation(None, retry)
            if response.status_code != 200:
                raise _http_error(
                    "worldquant_pnl_http_error", response, outcome_unknown=False
                )
            payload = _json_object(response.body, "worldquant_pnl_json_invalid")
            try:
                points = parse_pnl(payload)
            except ValueError as exc:
                raise WorldQuantProtocolError("worldquant_pnl_response_invalid") from exc
            return PnlObservation(points)

    def fetch_user_alpha_page(
        self,
        *,
        limit: int,
        offset: int,
        hidden: bool,
        status: str | None = None,
    ) -> UserAlphaPage:
        self._require_authenticated()
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise ValueError("worldquant_user_alphas_limit_invalid")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("worldquant_user_alphas_offset_invalid")
        if status is not None and (
            not isinstance(status, str)
            or not status.strip()
            or status != status.strip()
        ):
            raise ValueError("worldquant_user_alphas_status_invalid")
        if not isinstance(hidden, bool):
            raise ValueError("worldquant_user_alphas_hidden_invalid")
        parameters = {
            "limit": limit,
            "offset": offset,
            "order": "-dateCreated",
            "hidden": str(hidden).lower(),
        }
        if status is not None:
            parameters["status"] = status
        query = urlencode(parameters)
        response = self._send(
            Request(
                self._url(f"/users/self/alphas?{query}"),
                headers={"Accept": "application/json"},
                method="GET",
            ),
            code="worldquant_user_alphas_request_failed",
            outcome_unknown=False,
        )
        if response.status_code != 200:
            raise _http_error(
                "worldquant_user_alphas_http_error",
                response,
                outcome_unknown=False,
            )
        page = parse_user_alpha_page(
            _json_object(response.body, "worldquant_user_alphas_json_invalid")
        )
        if any(record.hidden is not hidden for record in page.records) or (
            status is not None
            and any(record.status != status for record in page.records)
        ):
            raise WorldQuantRequestError(
                "worldquant_user_alphas_filter_mismatch",
                retryable=True,
                outcome_unknown=False,
            )
        return page

    def fetch_formal_submission_check(
        self,
        *,
        platform_alpha_id: str,
    ) -> FormalCheckObservation:
        self._require_authenticated()
        encoded_alpha_id = _encoded_alpha_id(platform_alpha_id)
        response = self._send(
            Request(
                self._url(f"/alphas/{encoded_alpha_id}/check"),
                headers={"Accept": "application/json"},
                method="GET",
            ),
            code="worldquant_formal_check_request_failed",
            outcome_unknown=False,
        )
        if response.status_code != 200:
            raise _http_error(
                "worldquant_formal_check_http_error",
                response,
                outcome_unknown=False,
            )
        return FormalCheckObservation(
            payload=_json_value_or_text(response.body),
            retry_after_seconds=_retry_after_seconds(response.headers),
        )

    def submit_formal_alpha(
        self,
        *,
        platform_alpha_id: str,
    ) -> FormalSubmissionObservation:
        self._require_authenticated()
        encoded_alpha_id = _encoded_alpha_id(platform_alpha_id)
        response = self._send(
            Request(
                self._url(f"/alphas/{encoded_alpha_id}/submit"),
                data=b"",
                headers={"Accept": "application/json"},
                method="POST",
            ),
            code="worldquant_formal_submission_request_failed",
            outcome_unknown=True,
        )
        if response.status_code not in {200, 201, 202}:
            raise _http_error(
                "worldquant_formal_submission_http_error",
                response,
                outcome_unknown=response.status_code >= 500,
            )
        return FormalSubmissionObservation(
            status_code=response.status_code,
            payload=_optional_json_value(response.body),
        )

    def fetch_formal_submission_result(
        self,
        *,
        platform_alpha_id: str,
    ) -> FormalSubmissionObservation:
        self._require_authenticated()
        encoded_alpha_id = _encoded_alpha_id(platform_alpha_id)
        response = self._send(
            Request(
                self._url(f"/alphas/{encoded_alpha_id}/submit"),
                headers={"Accept": "application/json"},
                method="GET",
            ),
            code="worldquant_formal_submission_result_request_failed",
            outcome_unknown=False,
        )
        if response.status_code not in {200, 202, 403}:
            raise _http_error(
                "worldquant_formal_submission_result_http_error",
                response,
                outcome_unknown=False,
            )
        return FormalSubmissionObservation(
            status_code=response.status_code,
            payload=_json_value_or_text(response.body),
            retry_after_seconds=_retry_after_seconds(response.headers),
        )

    def fetch_alpha_detail(
        self,
        *,
        platform_alpha_id: str,
    ) -> AlphaDetailObservation:
        self._require_authenticated()
        encoded_alpha_id = _encoded_alpha_id(platform_alpha_id)
        response = self._send(
            Request(
                self._url(f"/alphas/{encoded_alpha_id}"),
                headers={"Accept": "application/json"},
                method="GET",
            ),
            code="worldquant_alpha_detail_request_failed",
            outcome_unknown=False,
        )
        if response.status_code != 200:
            raise _http_error(
                "worldquant_alpha_detail_http_error",
                response,
                outcome_unknown=False,
            )
        return AlphaDetailObservation(
            payload=_json_object(
                response.body,
                "worldquant_alpha_detail_json_invalid",
            )
        )

    def _send(
        self,
        request: Request,
        *,
        code: str,
        outcome_unknown: bool,
    ) -> WorldQuantResponse:
        for attempt in range(2):
            try:
                response = self._request_executor(request, self.timeout_seconds)
            except WorldQuantRequestError:
                raise
            except (OSError, TimeoutError, URLError, socket.timeout) as exc:
                raise WorldQuantRequestError(
                    code, retryable=True, outcome_unknown=outcome_unknown,
                    transport_error_type=type(getattr(exc, "reason", exc)).__name__,
                ) from exc
            if response.status_code != 401 or not self._authenticated or attempt:
                return response
            if request.get_method() != "GET":
                self._authenticated = False
                # A new POST must acquire a fresh execution claim after login.
                raise WorldQuantRequestError(
                    "worldquant_authentication_expired", status_code=401,
                    retryable=True, outcome_unknown=False,
                )
            # Retry reads once; never extend a POST's submission claim window.
            self.authenticate()
            request.remove_header("Cookie")
        raise AssertionError("worldquant_authentication_retry_invalid")

    def _require_authenticated(self) -> None:
        if not self._authenticated:
            raise RuntimeError("worldquant_authentication_required")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"


class _UrllibExecutor:
    def __init__(self) -> None:
        self._opener = build_opener(HTTPCookieProcessor(CookieJar()))

    def __call__(self, request: Request, timeout: float) -> WorldQuantResponse:
        try:
            with self._opener.open(request, timeout=timeout) as response:
                return WorldQuantResponse(
                    status_code=response.status,
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except HTTPError as error:
            return WorldQuantResponse(
                status_code=error.code,
                headers=dict(error.headers.items()) if error.headers else {},
                body=error.read(),
            )


def _http_error(
    code: str,
    response: WorldQuantResponse,
    *,
    outcome_unknown: bool,
) -> WorldQuantRequestError:
    return WorldQuantRequestError(
        code,
        status_code=response.status_code,
        retryable=response.status_code == 429 or response.status_code >= 500,
        outcome_unknown=outcome_unknown,
        retry_after_seconds=_retry_after_seconds(response.headers),
    )


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    raw_value: str | None = None
    for name, value in headers.items():
        if isinstance(name, str) and name.lower() == "retry-after":
            raw_value = value if isinstance(value, str) else None
            break
    if raw_value is None:
        return None
    try:
        parsed = float(raw_value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _json_object(body: bytes, error: str) -> Mapping[str, object]:
    value = _json_value(body, error)
    if not isinstance(value, dict):
        raise WorldQuantRequestError(
            error,
            retryable=True,
            outcome_unknown=False,
        )
    return value


def _json_value(body: bytes, error: str) -> object:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorldQuantRequestError(
            error,
            retryable=True,
            outcome_unknown=False,
        ) from exc
    return value


def _optional_json_value(body: bytes) -> object | None:
    if not body.strip():
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorldQuantRequestError(
            "worldquant_formal_submission_json_invalid",
            retryable=False,
            outcome_unknown=True,
        ) from exc


def _json_value_or_text(body: bytes) -> object | None:
    if not body.strip():
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body.decode("utf-8", errors="replace")


def _encoded_alpha_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("worldquant_alpha_id_missing")
    return quote(value.strip(), safe="")


def _validated_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("worldquant_base_url_invalid")
    normalized = value.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
        parsed.port
    except ValueError as exc:
        raise ValueError("worldquant_base_url_invalid") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("worldquant_base_url_invalid")
    return normalized


def _is_remote_simulation(base_url: str, remote_id: str) -> bool:
    if not isinstance(remote_id, str) or not remote_id.strip():
        return False
    try:
        base = urlsplit(base_url)
        remote = urlsplit(remote_id.strip())
        base_port = base.port or 443
        remote_port = remote.port or 443
    except ValueError:
        return False
    parts = tuple(part for part in remote.path.split("/") if part)
    return bool(
        remote.scheme == "https"
        and remote.hostname == base.hostname
        and remote_port == base_port
        and len(parts) == 2
        and parts[0] == "simulations"
        and parts[1]
        and not remote.query
        and not remote.fragment
        and remote.username is None
        and remote.password is None
    )
