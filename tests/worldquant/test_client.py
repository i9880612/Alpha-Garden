from __future__ import annotations

import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from worldquant.backtests import BacktestSettings
from worldquant.client import (
    WorldQuantClient,
    WorldQuantCredentials,
    WorldQuantRequestError,
    WorldQuantResponse,
)
from worldquant.catalog import CatalogContext


class FakeExecutor:
    def __init__(self, *responses: WorldQuantResponse | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[Request] = []

    def __call__(self, request: Request, timeout: float) -> WorldQuantResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class WorldQuantClientTests(unittest.TestCase):
    def test_expired_session_reauthenticates_and_retries_the_read_once(self):
        executor = FakeExecutor(
            self._response(201, {}), self._response(401, {}),
            self._response(201, {}), self._response(200, {"status": "RUNNING"}),
        )
        client = self._client(executor)
        client.authenticate()
        observation = client.poll_backtest("https://api.worldquantbrain.com/simulations/one")
        self.assertEqual(observation.state, "pending")
        self.assertEqual([urlsplit(r.full_url).path for r in executor.requests],
                         ["/authentication", "/simulations/one", "/authentication", "/simulations/one"])

    def test_authentication_refresh_is_bounded_and_does_not_hide_rejection(self):
        for refresh_status, final_status in ((401, None), (201, 401)):
            with self.subTest(refresh_status=refresh_status):
                responses = [self._response(201, {}), self._response(401, {}),
                             self._response(refresh_status, {})]
                if final_status is not None:
                    responses.append(self._response(final_status, {}))
                executor = FakeExecutor(*responses)
                client = self._client(executor)
                client.authenticate()
                with self.assertRaises(WorldQuantRequestError) as error:
                    client.poll_backtest("https://api.worldquantbrain.com/simulations/one")
                self.assertEqual(error.exception.status_code, 401)
                self.assertFalse(error.exception.outcome_unknown)
                self.assertEqual(len(executor.requests), len(responses))

    def test_expired_post_requires_login_before_a_new_send_and_keeps_unknown_outcome(self):
        executor = FakeExecutor(
            self._response(201, {}), self._response(401, {}),
            self._response(201, {}), TimeoutError("response lost"),
        )
        client = self._client(executor)
        client.authenticate()
        with self.assertRaises(WorldQuantRequestError) as expired:
            client.submit_backtest(formula="rank(close)", settings=self.settings)
        self.assertEqual(expired.exception.code, "worldquant_authentication_expired")
        self.assertFalse(expired.exception.outcome_unknown)
        self.assertFalse(client.authenticated)
        self.assertEqual(len(executor.requests), 2)
        client.authenticate()
        with self.assertRaises(WorldQuantRequestError) as error:
            client.submit_backtest(formula="rank(close)", settings=self.settings)
        self.assertTrue(error.exception.outcome_unknown)
        self.assertEqual([urlsplit(r.full_url).path for r in executor.requests],
                         ["/authentication", "/simulations", "/authentication", "/simulations"])

    def test_refresh_removes_expired_cookie_before_replaying_request(self):
        requests = []
        def executor(request, timeout):
            requests.append(request)
            if request.full_url.endswith("/authentication"):
                return self._response(201, {})
            if len(requests) == 2:
                request.add_unredirected_header("Cookie", "session=expired")
                return self._response(401, {})
            self.assertFalse(request.has_header("Cookie"))
            return self._response(200, {"status": "RUNNING"})
        client = self._client(executor)
        client.authenticate()
        self.assertEqual(client.poll_backtest(
            "https://api.worldquantbrain.com/simulations/one").state, "pending")
        self.assertEqual(len(requests), 4)

    def test_malformed_pnl_is_a_protocol_error_not_a_program_failure(self):
        from worldquant.backtests import WorldQuantProtocolError
        executor = FakeExecutor(self._response(201, {}), self._response(200, {}))
        client = self._client(executor)
        client.authenticate()
        with self.assertRaises(WorldQuantProtocolError) as error:
            client.fetch_pnl(platform_alpha_id="alpha")
        self.assertEqual(error.exception.code, "worldquant_pnl_response_invalid")

    def test_pnl_short_retry_fetches_prepared_series_without_new_authentication(self):
        for status in (200, 503):
            with self.subTest(status=status):
                executor = FakeExecutor(
                    self._response(201, {}),
                    WorldQuantResponse(status, {"Retry-After": "1"}, b""),
                    self._response(200, {
                        "schema": {"properties": [{"name": "date"}, {"name": "pnl"}]},
                        "records": [["2020-01-01", 1.0]],
                    }),
                )
                client = self._client(executor)
                client.authenticate()
                with patch("worldquant.client.time.sleep") as wait:
                    observation = client.fetch_pnl(platform_alpha_id="alpha")
                wait.assert_called_once_with(1.0)
                self.assertEqual(observation.points, (("2020-01-01", 1.0),))
                self.assertEqual([r.get_method() for r in executor.requests], ["POST", "GET", "GET"])

    def test_pnl_repeated_pending_returns_without_unbounded_polling(self):
        executor = FakeExecutor(
            self._response(201, {}),
            WorldQuantResponse(200, {"Retry-After": "1"}, b""),
            WorldQuantResponse(200, {"Retry-After": "2"}, b""),
        )
        client = self._client(executor)
        client.authenticate()
        with patch("worldquant.client.time.sleep") as wait:
            observation = client.fetch_pnl(platform_alpha_id="alpha")
        wait.assert_called_once_with(1.0)
        self.assertIsNone(observation.points)
        self.assertEqual(observation.retry_after_seconds, 2.0)
        self.assertEqual(len(executor.requests), 3)

    def test_pnl_long_wait_or_rate_limit_does_not_start_short_retry(self):
        for status in (200, 429, 503):
            with self.subTest(status=status):
                executor = FakeExecutor(self._response(201, {}),
                    WorldQuantResponse(status, {"Retry-After": "1200"}, b""))
                client = self._client(executor)
                client.authenticate()
                with patch("worldquant.client.time.sleep") as wait:
                    if status == 429:
                        with self.assertRaises(WorldQuantRequestError):
                            client.fetch_pnl(platform_alpha_id="alpha")
                    else:
                        observation = client.fetch_pnl(platform_alpha_id="alpha")
                        self.assertIsNone(observation.points)
                        self.assertEqual(observation.retry_after_seconds, 1200.0)
                wait.assert_not_called()
                self.assertEqual(len(executor.requests), 2)

    def test_pnl_followup_transport_failure_remains_external_error(self):
        executor = FakeExecutor(self._response(201, {}),
            WorldQuantResponse(200, {"Retry-After": "1"}, b""), TimeoutError())
        client = self._client(executor)
        client.authenticate()
        with patch("worldquant.client.time.sleep"), self.assertRaises(WorldQuantRequestError) as raised:
            client.fetch_pnl(platform_alpha_id="alpha")
        self.assertFalse(raised.exception.outcome_unknown)
        self.assertEqual(raised.exception.code, "worldquant_pnl_request_failed")

    def test_pnl_empty_rate_limit_and_ready_stay_distinct(self):
        executor = FakeExecutor(
            self._response(201, {}),
            WorldQuantResponse(200, {}, b""),
            WorldQuantResponse(429, {"Retry-After": "1200"}, b""),
            self._response(
                200,
                {
                    "schema": {"properties": [{"name": "date"}, {"name": "pnl"}]},
                    "records": [["2020-01-01", 1.0]],
                },
            ),
        )
        client = self._client(executor)
        client.authenticate()
        self.assertIsNone(client.fetch_pnl(platform_alpha_id="alpha").points)
        with self.assertRaises(WorldQuantRequestError) as raised:
            client.fetch_pnl(platform_alpha_id="alpha")
        self.assertEqual(raised.exception.status_code, 429)
        self.assertFalse(raised.exception.outcome_unknown)
        self.assertEqual(
            client.fetch_pnl(platform_alpha_id="alpha").points, (("2020-01-01", 1.0),)
        )
        self.assertTrue(
            all(request.get_method() == "GET" for request in executor.requests[1:])
        )

    def test_poll_transport_error_keeps_type_without_exposing_exception_text(self) -> None:
        executor = FakeExecutor(
            self._response(201, {}),
            TimeoutError("sensitive-url-or-credential"),
        )
        client = self._client(executor)
        client.authenticate()
        with self.assertRaises(WorldQuantRequestError) as raised:
            client.poll_backtest("https://api.worldquantbrain.com/simulations/example")
        error = raised.exception
        self.assertEqual(error.code, "worldquant_poll_request_failed")
        self.assertEqual(error.transport_error_type, "TimeoutError")
        self.assertNotIn("sensitive", str(error))
        self.assertIsNone(error.status_code)

    def setUp(self) -> None:
        self.settings = BacktestSettings(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
            decay=4,
            neutralization="SECTOR",
            truncation=0.08,
            pasteurization="ON",
            unit_handling="VERIFY",
            nan_handling="OFF",
            language="FASTEXPR",
            visualization=False,
            max_trade="OFF",
            max_position="OFF",
        )

    def test_explicit_authentication_keeps_credentials_out_of_result(self) -> None:
        executor = FakeExecutor(self._response(201, {"status": "authenticated"}))
        client = self._client(executor)

        result = client.authenticate()

        self.assertEqual(result.status_code, 201)
        request = executor.requests[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertTrue(request.full_url.endswith("/authentication"))
        self.assertIn("Authorization", request.headers)
        self.assertNotIn("user@example.com", repr(result))
        self.assertNotIn("secret", repr(client.credentials))

    def test_platform_methods_do_not_authenticate_implicitly(self) -> None:
        executor = FakeExecutor()
        client = self._client(executor)

        with self.assertRaisesRegex(RuntimeError, "worldquant_authentication_required"):
            client.submit_backtest(formula="rank(close)", settings=self.settings)

        self.assertEqual(executor.requests, [])

    def test_catalog_reads_use_authenticated_get_requests(self) -> None:
        field_payload = {
            "count": 1,
            "results": [
                {
                    "id": "close",
                    "dataset": {"id": "dataset", "name": "Dataset"},
                    "category": {"id": "pv", "name": "Price Volume"},
                    "subcategory": {"id": "price", "name": "Price"},
                    "type": "MATRIX",
                    "coverage": 1.0,
                    "description": "close",
                    "region": "USA",
                    "universe": "TOP3000",
                    "delay": 1,
                }
            ],
        }
        operator_payload = [
            {
                "name": "rank",
                "definition": "rank(x)",
                "scope": ["REGULAR"],
            }
        ]
        executor = FakeExecutor(
            self._response(200, {}),
            self._response(200, field_payload),
            self._response(200, operator_payload),
        )
        client = self._client(executor)
        client.authenticate()

        page = client.fetch_data_field_page(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            limit=50,
            offset=100,
        )
        operators = client.fetch_operators()

        self.assertEqual(page.fields[0].field_id, "close")
        self.assertEqual(operators[0].name, "rank")
        field_request = executor.requests[1]
        query = parse_qs(urlsplit(field_request.full_url).query)
        self.assertEqual(field_request.get_method(), "GET")
        self.assertEqual(
            query,
            {
                "instrumentType": ["EQUITY"],
                "region": ["USA"],
                "universe": ["TOP3000"],
                "delay": ["1"],
                "limit": ["50"],
                "offset": ["100"],
            },
        )
        self.assertTrue(executor.requests[2].full_url.endswith("/operators"))

    def test_catalog_http_error_preserves_retry_after(self) -> None:
        executor = FakeExecutor(
            self._response(200, {}),
            self._response(429, {}, headers={"Retry-After": "2.5"}),
        )
        client = self._client(executor)
        client.authenticate()

        with self.assertRaises(WorldQuantRequestError) as raised:
            client.fetch_operators()

        self.assertEqual(raised.exception.code, "worldquant_operators_http_error")
        self.assertTrue(raised.exception.retryable)
        self.assertFalse(raised.exception.outcome_unknown)
        self.assertEqual(raised.exception.retry_after_seconds, 2.5)

    def test_submit_sends_exact_request_and_parses_remote_identity(self) -> None:
        executor = FakeExecutor(
            self._response(200, {}),
            self._response(
                201,
                {},
                headers={"Location": "/simulations/simulation-1"},
            ),
        )
        client = self._client(executor)
        client.authenticate()

        observation = client.submit_backtest(
            formula="rank(close)",
            settings=self.settings,
        )

        request = executor.requests[1]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["regular"], "rank(close)")
        self.assertEqual(payload["settings"], self.settings.as_platform_dict())
        self.assertEqual(observation.state, "accepted")
        self.assertTrue(observation.remote_id.endswith("/simulations/simulation-1"))

    def test_submit_network_failure_is_outcome_unknown(self) -> None:
        executor = FakeExecutor(self._response(200, {}), OSError("connection lost"))
        client = self._client(executor)
        client.authenticate()

        with self.assertRaises(WorldQuantRequestError) as raised:
            client.submit_backtest(formula="rank(close)", settings=self.settings)

        self.assertEqual(raised.exception.code, "worldquant_submission_request_failed")
        self.assertTrue(raised.exception.retryable)
        self.assertTrue(raised.exception.outcome_unknown)

    def test_rate_limit_is_retryable_but_not_outcome_unknown(self) -> None:
        executor = FakeExecutor(
            self._response(200, {}),
            self._response(429, {}, headers={"Retry-After": "2.5"}),
        )
        client = self._client(executor)
        client.authenticate()

        with self.assertRaises(WorldQuantRequestError) as raised:
            client.submit_backtest(formula="rank(close)", settings=self.settings)

        self.assertEqual(raised.exception.status_code, 429)
        self.assertTrue(raised.exception.retryable)
        self.assertFalse(raised.exception.outcome_unknown)
        self.assertEqual(raised.exception.retry_after_seconds, 2.5)

    def test_poll_and_detail_use_authenticated_read_requests(self) -> None:
        executor = FakeExecutor(
            self._response(200, {}),
            self._response(200, {"progress": 0.35}),
            self._response(200, self._detail_payload()),
        )
        client = self._client(executor)
        client.authenticate()

        poll = client.poll_backtest(
            "https://api.worldquantbrain.com/simulations/simulation-1"
        )
        detail = client.fetch_backtest_detail(
            platform_alpha_id="alpha-1",
            expected_formula="rank(close)",
            expected_settings=self.settings.as_platform_dict(),
        )

        self.assertEqual(poll.state, "pending")
        self.assertEqual(detail.platform_alpha_id, "alpha-1")
        self.assertEqual(executor.requests[1].get_method(), "GET")
        self.assertTrue(executor.requests[2].full_url.endswith("/alphas/alpha-1"))

    def test_user_alpha_page_uses_filtered_read_request(self) -> None:
        payload = {
            "count": 1,
            "next": None,
            "previous": None,
            "results": [
                {
                    "id": "alpha-1",
                    "type": "REGULAR",
                    "status": "UNSUBMITTED",
                    "hidden": False,
                    "dateCreated": "2026-09-04T04:40:42-04:00",
                    "regular": {"code": "rank(close)"},
                    "settings": self.settings.as_platform_dict(),
                }
            ],
        }
        executor = FakeExecutor(self._response(200, {}), self._response(200, payload))
        client = self._client(executor)
        client.authenticate()

        page = client.fetch_user_alpha_page(
            limit=100,
            offset=200,
            status="UNSUBMITTED",
            hidden=False,
        )

        self.assertEqual(page.records[0].platform_alpha_id, "alpha-1")
        request = executor.requests[1]
        self.assertEqual(request.get_method(), "GET")
        self.assertTrue(urlsplit(request.full_url).path.endswith("/users/self/alphas"))
        self.assertEqual(
            parse_qs(urlsplit(request.full_url).query),
            {
                "limit": ["100"],
                "offset": ["200"],
                "status": ["UNSUBMITTED"],
                "order": ["-dateCreated"],
                "hidden": ["false"],
            },
        )

    def test_user_alpha_page_can_read_all_statuses(self) -> None:
        payload = {
            "count": 0,
            "next": None,
            "previous": None,
            "results": [],
        }
        executor = FakeExecutor(self._response(200, {}), self._response(200, payload))
        client = self._client(executor)
        client.authenticate()

        page = client.fetch_user_alpha_page(
            limit=100,
            offset=0,
            hidden=True,
        )

        self.assertEqual(page.records, ())
        query = parse_qs(urlsplit(executor.requests[1].full_url).query)
        self.assertNotIn("status", query)
        self.assertEqual(
            query,
            {
                "limit": ["100"],
                "offset": ["0"],
                "order": ["-dateCreated"],
                "hidden": ["true"],
            },
        )

    def test_user_alpha_page_error_is_read_only_failure(self) -> None:
        executor = FakeExecutor(self._response(200, {}), self._response(503, {}))
        client = self._client(executor)
        client.authenticate()

        with self.assertRaises(WorldQuantRequestError) as raised:
            client.fetch_user_alpha_page(
                limit=100,
                offset=0,
                status="UNSUBMITTED",
                hidden=False,
            )

        self.assertEqual(raised.exception.code, "worldquant_user_alphas_http_error")
        self.assertFalse(raised.exception.outcome_unknown)
        self.assertEqual(executor.requests[1].get_method(), "GET")

    def test_yearly_stats_uses_authenticated_read_request_and_preserves_pending_retry(
        self,
    ) -> None:
        executor = FakeExecutor(
            self._response(200, {}),
            WorldQuantResponse(
                status_code=200,
                headers={"Retry-After": "1.5"},
                body=b" \n",
            ),
            self._response(503, {}, headers={"Retry-After": "2.5"}),
            self._response(200, self._yearly_stats_payload()),
        )
        client = self._client(executor)
        client.authenticate()

        empty = client.fetch_backtest_yearly_stats(platform_alpha_id="alpha-1")
        unavailable = client.fetch_backtest_yearly_stats(platform_alpha_id="alpha-1")
        ready = client.fetch_backtest_yearly_stats(platform_alpha_id="alpha-1")

        self.assertEqual(
            (empty.state, empty.stats, empty.retry_after_seconds),
            ("pending", (), 1.5),
        )
        self.assertEqual(
            (unavailable.state, unavailable.stats, unavailable.retry_after_seconds),
            ("pending", (), 2.5),
        )
        self.assertEqual(ready.state, "ready")
        self.assertEqual(len(ready.stats), 1)
        request = executor.requests[3]
        self.assertEqual(request.get_method(), "GET")
        self.assertTrue(
            request.full_url.endswith("/alphas/alpha-1/recordsets/yearly-stats")
        )

    def test_yearly_stats_uses_http_error_without_valid_503_retry_after(
        self,
    ) -> None:
        executor = FakeExecutor(
            self._response(200, {}),
            self._response(503, {}, headers={"Retry-After": "0"}),
        )
        client = self._client(executor)
        client.authenticate()

        with self.assertRaises(WorldQuantRequestError) as raised:
            client.fetch_backtest_yearly_stats(platform_alpha_id="alpha-1")

        self.assertEqual(raised.exception.code, "worldquant_yearly_stats_http_error")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertTrue(raised.exception.retryable)
        self.assertFalse(raised.exception.outcome_unknown)

    def test_formal_submission_protocol_uses_check_post_and_confirmation_reads(
        self,
    ) -> None:
        check_payload = {
            "is": {"checks": [{"name": "SELF_CORRELATION", "result": "PASS"}]}
        }
        detail_payload = {
            "id": "alpha/1",
            "status": "ACTIVE",
            "dateSubmitted": "2026-09-04T01:00:00+00:00",
            "hidden": False,
            "regular": {"code": "rank(close)"},
        }
        executor = FakeExecutor(
            self._response(200, {}),
            self._response(200, check_payload, headers={"Retry-After": "2"}),
            WorldQuantResponse(status_code=201, headers={}, body=b""),
            self._response(200, detail_payload),
        )
        client = self._client(executor)
        client.authenticate()

        check = client.fetch_formal_submission_check(platform_alpha_id="alpha/1")
        submitted = client.submit_formal_alpha(platform_alpha_id="alpha/1")
        detail = client.fetch_alpha_detail(platform_alpha_id="alpha/1")

        self.assertEqual(check.payload, check_payload)
        self.assertEqual(check.retry_after_seconds, 2.0)
        self.assertEqual((submitted.status_code, submitted.payload), (201, None))
        self.assertEqual(detail.payload, detail_payload)
        self.assertTrue(executor.requests[1].full_url.endswith("/alphas/alpha%2F1/check"))
        self.assertEqual(executor.requests[1].get_method(), "GET")
        self.assertTrue(executor.requests[2].full_url.endswith("/alphas/alpha%2F1/submit"))
        self.assertEqual(executor.requests[2].get_method(), "POST")
        self.assertTrue(executor.requests[3].full_url.endswith("/alphas/alpha%2F1"))

    def test_formal_submission_result_read_preserves_rejection_and_wait(self):
        payload = {"is": {"checks": [{"name": "SELF_CORRELATION", "result": "FAIL"}]}}
        executor = FakeExecutor(
            self._response(200, {}),
            self._response(403, payload),
            WorldQuantResponse(200, {"Retry-After": "4"}, b""),
            self._response(503, {}, headers={"Retry-After": "7"}),
        )
        client = self._client(executor)
        client.authenticate()
        rejected = client.fetch_formal_submission_result(platform_alpha_id="alpha/1")
        pending = client.fetch_formal_submission_result(platform_alpha_id="alpha/1")
        self.assertEqual((rejected.status_code, rejected.payload), (403, payload))
        self.assertIsNone(rejected.retry_after_seconds)
        self.assertIsNone(pending.payload)
        self.assertEqual(pending.retry_after_seconds, 4)
        with self.assertRaises(WorldQuantRequestError) as raised:
            client.fetch_formal_submission_result(platform_alpha_id="alpha/1")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertTrue(raised.exception.retryable)
        self.assertFalse(raised.exception.outcome_unknown)
        for request in executor.requests[1:]:
            self.assertEqual(request.get_method(), "GET")
            self.assertTrue(request.full_url.endswith("/alphas/alpha%2F1/submit"))

    def test_formal_submission_network_failure_is_outcome_unknown(self) -> None:
        executor = FakeExecutor(self._response(200, {}), OSError("connection lost"))
        client = self._client(executor)
        client.authenticate()

        with self.assertRaises(WorldQuantRequestError) as raised:
            client.submit_formal_alpha(platform_alpha_id="alpha-1")

        self.assertEqual(
            raised.exception.code,
            "worldquant_formal_submission_request_failed",
        )
        self.assertTrue(raised.exception.outcome_unknown)

    def test_pending_formal_check_preserves_empty_body_and_retry_delay(self) -> None:
        executor = FakeExecutor(
            self._response(200, {}),
            WorldQuantResponse(
                status_code=200,
                headers={"Retry-After": "3"},
                body=b"",
            ),
        )
        client = self._client(executor)
        client.authenticate()

        observation = client.fetch_formal_submission_check(
            platform_alpha_id="alpha-1"
        )

        self.assertIsNone(observation.payload)
        self.assertEqual(observation.retry_after_seconds, 3.0)

    def test_formal_submission_explicit_rejection_is_not_outcome_unknown(self) -> None:
        executor = FakeExecutor(
            self._response(200, {}),
            self._response(400, {"detail": "rejected"}),
        )
        client = self._client(executor)
        client.authenticate()

        with self.assertRaises(WorldQuantRequestError) as raised:
            client.submit_formal_alpha(platform_alpha_id="alpha-1")

        self.assertEqual(raised.exception.status_code, 400)
        self.assertFalse(raised.exception.outcome_unknown)

    def test_authentication_rejection_does_not_mark_outcome_unknown(self) -> None:
        executor = FakeExecutor(self._response(401, {"detail": "invalid"}))
        client = self._client(executor)

        with self.assertRaises(WorldQuantRequestError) as raised:
            client.authenticate()

        self.assertEqual(raised.exception.code, "worldquant_authentication_rejected")
        self.assertEqual(raised.exception.status_code, 401)
        self.assertFalse(raised.exception.outcome_unknown)

    @staticmethod
    def _response(
        status_code: int,
        payload: object,
        *,
        headers: dict[str, str] | None = None,
    ) -> WorldQuantResponse:
        return WorldQuantResponse(
            status_code=status_code,
            headers=headers or {},
            body=json.dumps(payload).encode("utf-8"),
        )

    @staticmethod
    def _client(executor: FakeExecutor) -> WorldQuantClient:
        return WorldQuantClient(
            base_url="https://api.worldquantbrain.com",
            credentials=WorldQuantCredentials(
                username="user@example.com",
                password="secret",
            ),
            request_executor=executor,
        )

    def _detail_payload(self) -> dict[str, object]:
        return {
            "id": "alpha-1",
            "type": "REGULAR",
            "settings": self.settings.as_platform_dict(),
            "regular": {"code": "rank(close)"},
            "is": {
                "sharpe": 1.3,
                "fitness": 1.1,
                "turnover": 0.12,
                "returns": 0.08,
                "drawdown": 0.04,
                "margin": 0.001,
                "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
            },
        }

    @staticmethod
    def _yearly_stats_payload() -> dict[str, object]:
        return {
            "schema": {
                "name": "yearly-stats",
                "title": "Yearly statistics",
                "properties": [
                    {"name": "year", "title": "Year", "type": "year"},
                    {"name": "pnl", "title": "Pnl", "type": "amount"},
                    {
                        "name": "bookSize",
                        "title": "Book size",
                        "type": "amount",
                    },
                    {
                        "name": "longCount",
                        "title": "Long count",
                        "type": "integer",
                    },
                    {
                        "name": "shortCount",
                        "title": "Short count",
                        "type": "integer",
                    },
                    {
                        "name": "turnover",
                        "title": "Turnover",
                        "type": "percent",
                    },
                    {
                        "name": "sharpe",
                        "title": "Sharpe",
                        "type": "decimal",
                    },
                    {
                        "name": "returns",
                        "title": "Returns",
                        "type": "percent",
                    },
                    {
                        "name": "drawdown",
                        "title": "Drawdown",
                        "type": "percent",
                    },
                    {
                        "name": "margin",
                        "title": "Margin",
                        "type": "permyriad",
                    },
                    {
                        "name": "fitness",
                        "title": "Fitness",
                        "type": "decimal",
                    },
                    {"name": "stage", "title": "Stage", "type": "string"},
                ],
            },
            "records": [
                [
                    "2022",
                    120_000,
                    20_000_000,
                    201,
                    199,
                    0.17,
                    1.1,
                    0.04,
                    0.06,
                    0.0009,
                    0.7,
                    "IS",
                ]
            ],
        }


if __name__ == "__main__":
    unittest.main()
