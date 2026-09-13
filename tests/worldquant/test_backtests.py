from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from worldquant.backtests import (
    BacktestCheck,
    BacktestSettings,
    BacktestYearlyStat,
    WorldQuantDetailMetricUnavailable,
    WorldQuantProtocolError,
    WorldQuantZeroCapitalResult,
    build_backtest_request,
    parse_backtest_detail,
    parse_backtest_yearly_stats,
    parse_poll_response,
    parse_submission_response,
)


class WorldQuantBacktestProtocolTests(unittest.TestCase):
    def test_grade_is_platform_evidence_and_is_never_inferred_from_fitness(self):
        for value, expected in [("SPECTACULAR", "SPECTACULAR"), ("GOOD", "GOOD"),
                                ("FUTURE_GRADE", "FUTURE_GRADE"), (None, None),
                                ("", None), (123, None)]:
            with self.subTest(grade=value):
                payload = self._detail_payload()
                payload["grade"] = value
                payload["is"]["fitness"] = 9.0
                detail = parse_backtest_detail(
                    payload, expected_alpha_id="alpha-1", expected_formula="rank(close)",
                    expected_settings=self.settings.as_platform_dict(),
                )
                self.assertEqual(detail.grade, expected)
                self.assertEqual(detail.fitness, 9.0)
        payload.pop("grade")
        self.assertIsNone(parse_backtest_detail(
            payload, expected_alpha_id="alpha-1", expected_formula="rank(close)",
            expected_settings=self.settings.as_platform_dict(),
        ).grade)

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

    def test_builds_observed_regular_simulation_request(self) -> None:
        payload = build_backtest_request("rank(close)", self.settings)

        self.assertEqual(payload["type"], "REGULAR")
        self.assertEqual(payload["regular"], "rank(close)")
        self.assertEqual(payload["settings"], self.settings.as_platform_dict())
        self.assertEqual(
            BacktestSettings.from_platform_dict(payload["settings"]),
            self.settings,
        )
        self.assertEqual(payload["settings"]["maxTrade"], "OFF")
        self.assertEqual(payload["settings"]["maxPosition"], "OFF")

    def test_complete_settings_reject_missing_maximum_controls(self) -> None:
        for missing in ("maxTrade", "maxPosition"):
            with self.subTest(missing=missing):
                payload = self.settings.as_platform_dict()
                del payload[missing]

                with self.assertRaisesRegex(
                    ValueError,
                    "worldquant_backtest_settings_shape_invalid",
                ):
                    BacktestSettings.from_platform_dict(payload)

    def test_submission_requires_201_and_same_origin_simulation_location(self) -> None:
        accepted = parse_submission_response(
            status_code=201,
            headers={"Location": "/simulations/simulation-1"},
            base_url="https://api.worldquantbrain.com",
        )
        unknown = parse_submission_response(
            status_code=201,
            headers={"Location": "https://other.example/simulations/1"},
            base_url="https://api.worldquantbrain.com",
        )

        self.assertEqual(accepted.state, "accepted")
        self.assertEqual(
            accepted.remote_id,
            "https://api.worldquantbrain.com/simulations/simulation-1",
        )
        self.assertEqual(unknown.state, "unknown")
        self.assertIsNone(unknown.remote_id)
        with self.assertRaisesRegex(
            WorldQuantProtocolError,
            "worldquant_submission_status_unexpected",
        ):
            parse_submission_response(
                status_code=429,
                headers={},
                base_url="https://api.worldquantbrain.com",
            )

    def test_poll_supports_observed_progress_complete_and_terminal_failure(self) -> None:
        pending = parse_poll_response({"progress": 0.35})
        completed = parse_poll_response(
            {
                "status": "COMPLETE",
                "alpha": "alpha-1",
                "progress": 1,
            }
        )
        failed = parse_poll_response({"status": "ERROR"})
        warning = parse_poll_response(
            {
                "status": "WARNING",
                "alpha": "alpha-warning",
                "message": "Incompatible unit",
            }
        )

        self.assertEqual((pending.state, pending.progress), ("pending", 0.35))
        self.assertEqual(
            (completed.state, completed.platform_alpha_id),
            ("completed", "alpha-1"),
        )
        self.assertEqual(
            (failed.state, failed.platform_status),
            ("failed", "ERROR"),
        )
        self.assertEqual(
            (
                warning.state,
                warning.platform_status,
                warning.platform_alpha_id,
                warning.failure_message,
            ),
            ("failed", "WARNING", "alpha-warning", "Incompatible unit"),
        )

    def test_poll_does_not_guess_missing_or_unknown_state(self) -> None:
        for payload in (
            {},
            {"alpha": "alpha-1"},
            {"status": "WAITING"},
            {"status": "WARNING", "message": "invalid"},
            {"status": "WARNING", "alpha": "alpha-1"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(WorldQuantProtocolError):
                    parse_poll_response(payload)

    def test_detail_extracts_metrics_and_checks_from_observed_shape(self) -> None:
        detail = parse_backtest_detail(
            self._detail_payload(),
            expected_alpha_id="alpha-1",
            expected_formula="rank(close)",
            expected_settings=self.settings.as_platform_dict(),
        )

        self.assertEqual(detail.platform_alpha_id, "alpha-1")
        self.assertEqual(detail.sharpe, 1.3)
        self.assertEqual(detail.book_size, 20_000_000.0)
        self.assertEqual((detail.long_count, detail.short_count), (1_523, 1_477))
        self.assertFalse(detail.check_set_complete)
        self.assertTrue(detail.has_failed_check)
        self.assertEqual(
            detail.checks,
            (
                BacktestCheck(
                    name="CONCENTRATED_WEIGHT",
                    status="FAIL",
                    threshold=0.1,
                    actual=0.160449,
                    platform_date="2021-12-30",
                ),
                BacktestCheck(
                    name="LOW_SHARPE",
                    status="PASS",
                    threshold=1.25,
                    actual=1.3,
                    platform_date=None,
                ),
                BacktestCheck(
                    name="MATCHES_COMPETITION",
                    status="PASS",
                    threshold=None,
                    actual=None,
                    platform_date=None,
                ),
                BacktestCheck(
                    name="SELF_CORRELATION",
                    status="PENDING",
                    threshold=None,
                    actual=None,
                    platform_date=None,
                ),
            ),
        )

    def test_detail_preserves_independent_missing_position_counts(self) -> None:
        for field in ("longCount", "shortCount"):
            for representation in ("missing", "null"):
                with self.subTest(field=field, representation=representation):
                    payload = self._detail_payload()
                    if representation == "missing":
                        del payload["is"][field]
                    else:
                        payload["is"][field] = None

                    detail = parse_backtest_detail(
                        payload,
                        expected_alpha_id="alpha-1",
                        expected_formula="rank(close)",
                        expected_settings=self.settings.as_platform_dict(),
                    )

                    self.assertIsNone(
                        getattr(
                            detail,
                            "long_count" if field == "longCount" else "short_count",
                        )
                    )

    def test_detail_rejects_invalid_position_counts(self) -> None:
        for field, value in (
            ("longCount", True),
            ("longCount", -1),
            ("shortCount", 1.5),
        ):
            with self.subTest(field=field, value=value):
                payload = self._detail_payload()
                payload["is"][field] = value

                with self.assertRaisesRegex(
                    WorldQuantProtocolError,
                    f"worldquant_detail_metric_invalid:{field}",
                ):
                    parse_backtest_detail(
                        payload,
                        expected_alpha_id="alpha-1",
                        expected_formula="rank(close)",
                        expected_settings=self.settings.as_platform_dict(),
                    )

    def test_yearly_stats_maps_schema_order_and_sorts_by_stage_then_year(self) -> None:
        payload = self._yearly_stats_payload()
        payload["schema"]["properties"] = list(
            reversed(payload["schema"]["properties"])
        )
        payload["records"] = [
            list(reversed(row)) for row in payload["records"]
        ]
        payload["records"].append(
            [
                "OS",
                0.7,
                0.0009,
                0.06,
                0.04,
                1.1,
                0.17,
                199,
                201,
                20_000_000,
                120_000,
                "2022",
            ]
        )

        stats = parse_backtest_yearly_stats(payload)

        self.assertEqual(
            stats,
            (
                BacktestYearlyStat(
                    year=2021,
                    pnl=100_000.0,
                    book_size=20_000_000.0,
                    turnover=0.12,
                    sharpe=1.1,
                    returns=0.04,
                    drawdown=0.06,
                    margin=0.0009,
                    fitness=0.7,
                    long_count=201,
                    short_count=199,
                    stage="IS",
                ),
                BacktestYearlyStat(
                    year=2022,
                    pnl=120_000.0,
                    book_size=20_000_000.0,
                    turnover=0.17,
                    sharpe=1.1,
                    returns=0.04,
                    drawdown=0.06,
                    margin=0.0009,
                    fitness=0.7,
                    long_count=201,
                    short_count=199,
                    stage="IS",
                ),
                BacktestYearlyStat(
                    year=2022,
                    pnl=120_000.0,
                    book_size=20_000_000.0,
                    turnover=0.17,
                    sharpe=1.1,
                    returns=0.04,
                    drawdown=0.06,
                    margin=0.0009,
                    fitness=0.7,
                    long_count=201,
                    short_count=199,
                    stage="OS",
                ),
            ),
        )

    def test_yearly_stats_rejects_schema_rows_and_identity_conflicts(self) -> None:
        invalid_payloads: list[dict[str, object]] = []

        wrong_property_type = self._yearly_stats_payload()
        wrong_property_type["schema"]["properties"][0]["type"] = "integer"
        invalid_payloads.append(wrong_property_type)

        missing_property_title = self._yearly_stats_payload()
        del missing_property_title["schema"]["properties"][0]["title"]
        invalid_payloads.append(missing_property_title)

        empty_property_title = self._yearly_stats_payload()
        empty_property_title["schema"]["properties"][0]["title"] = ""
        invalid_payloads.append(empty_property_title)

        wrong_row_length = self._yearly_stats_payload()
        wrong_row_length["records"][0].pop()
        invalid_payloads.append(wrong_row_length)

        duplicate = self._yearly_stats_payload()
        duplicate["records"].append(list(duplicate["records"][0]))
        invalid_payloads.append(duplicate)

        invalid_year = self._yearly_stats_payload()
        invalid_year["records"][0][0] = "2021.0"
        invalid_payloads.append(invalid_year)

        nonpositive_year = self._yearly_stats_payload()
        nonpositive_year["records"][0][0] = "0"
        invalid_payloads.append(nonpositive_year)

        invalid_count = self._yearly_stats_payload()
        invalid_count["records"][0][3] = -1
        invalid_payloads.append(invalid_count)

        padded_stage = self._yearly_stats_payload()
        padded_stage["records"][0][-1] = " IS "
        invalid_payloads.append(padded_stage)

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(WorldQuantProtocolError):
                    parse_backtest_yearly_stats(payload)

    def test_yearly_stats_accepts_an_empty_ready_recordset_and_null_metrics(self) -> None:
        empty = self._yearly_stats_payload()
        empty["records"] = []
        self.assertEqual(parse_backtest_yearly_stats(empty), ())

        nullable = self._yearly_stats_payload()
        nullable["records"] = [nullable["records"][0]]
        nullable["records"][0][1:11] = [None] * 10
        stat = parse_backtest_yearly_stats(nullable)[0]
        self.assertEqual(
            (
                stat.pnl,
                stat.book_size,
                stat.long_count,
                stat.short_count,
                stat.turnover,
                stat.sharpe,
                stat.returns,
                stat.drawdown,
                stat.margin,
                stat.fitness,
            ),
            (None,) * 10,
        )

    def test_detail_preserves_zero_and_negative_check_values(self) -> None:
        payload = self._detail_payload()
        payload["is"]["checks"] = [
            {"name": "ZERO", "result": "FAIL", "limit": 0, "value": 0},
            {"name": "NEGATIVE", "result": "FAIL", "limit": -1, "value": -2},
        ]

        detail = parse_backtest_detail(
            payload,
            expected_alpha_id="alpha-1",
            expected_formula="rank(close)",
            expected_settings=self.settings.as_platform_dict(),
        )

        self.assertEqual(detail.checks[0].threshold, -1.0)
        self.assertEqual(detail.checks[0].actual, -2.0)
        self.assertEqual(detail.checks[1].threshold, 0.0)
        self.assertEqual(detail.checks[1].actual, 0.0)

    def test_detail_rejects_invalid_check_details_and_duplicate_names(self) -> None:
        invalid_checks = (
            [],
            [{"name": "LOW_SHARPE", "result": "PASS", "limit": True}],
            [{"name": "LOW_SHARPE", "result": "PASS", "value": "1.3"}],
            [{"name": "LOW_SHARPE", "result": "PASS", "date": "2021/12/30"}],
            [
                {"name": "LOW_SHARPE", "result": "PASS"},
                {"name": "LOW_SHARPE", "result": "FAIL"},
            ],
        )
        for checks in invalid_checks:
            with self.subTest(checks=checks):
                payload = self._detail_payload()
                payload["is"]["checks"] = checks
                with self.assertRaises(WorldQuantProtocolError):
                    parse_backtest_detail(
                        payload,
                        expected_alpha_id="alpha-1",
                        expected_formula="rank(close)",
                        expected_settings=self.settings.as_platform_dict(),
                    )

    def test_detail_rejects_wrong_identity_formula_settings_and_metrics(self) -> None:
        changes = (
            ("alpha", "alpha-other", "worldquant_detail_alpha_mismatch"),
            ("formula", "rank(open)", "worldquant_detail_formula_mismatch"),
            ("settings", "TOP1000", "worldquant_detail_settings_mismatch"),
            ("metric", "invalid", "worldquant_detail_metric_invalid:fitness"),
        )
        for kind, value, error in changes:
            with self.subTest(kind=kind):
                payload = self._detail_payload()
                if kind == "alpha":
                    payload["id"] = value
                elif kind == "formula":
                    payload["regular"]["code"] = value
                elif kind == "settings":
                    payload["settings"]["universe"] = value
                else:
                    payload["is"]["fitness"] = value
                with self.assertRaisesRegex(WorldQuantProtocolError, error):
                    parse_backtest_detail(
                        payload,
                        expected_alpha_id="alpha-1",
                        expected_formula="rank(close)",
                        expected_settings=self.settings.as_platform_dict(),
                    )

    def test_detail_rejects_changed_maximum_controls(self) -> None:
        for field in ("maxTrade", "maxPosition"):
            with self.subTest(field=field):
                payload = self._detail_payload()
                payload["settings"][field] = "ON"

                with self.assertRaisesRegex(
                    WorldQuantProtocolError,
                    "worldquant_detail_settings_mismatch",
                ):
                    parse_backtest_detail(
                        payload,
                        expected_alpha_id="alpha-1",
                        expected_formula="rank(close)",
                        expected_settings=self.settings.as_platform_dict(),
                    )

    def test_detail_distinguishes_explicitly_unavailable_metric(self) -> None:
        payload = self._detail_payload()
        payload["is"]["fitness"] = None

        with self.assertRaises(WorldQuantDetailMetricUnavailable) as raised:
            parse_backtest_detail(
                payload,
                expected_alpha_id="alpha-1",
                expected_formula="rank(close)",
                expected_settings=self.settings.as_platform_dict(),
            )

        self.assertEqual(raised.exception.metric_name, "fitness")
        self.assertEqual(
            raised.exception.code,
            "worldquant_detail_metric_unavailable:fitness",
        )

    def test_detail_distinguishes_zero_capital_from_missing_metric(self) -> None:
        payload = self._detail_payload()
        payload["is"].update(
            {
                "pnl": 0,
                "bookSize": 0,
                "turnover": 0.0,
                "returns": 0.0,
                "drawdown": 0.0,
                "margin": 0.0,
                "sharpe": 0.0,
                "fitness": None,
            }
        )

        with self.assertRaises(WorldQuantZeroCapitalResult) as raised:
            parse_backtest_detail(
                payload,
                expected_alpha_id="alpha-1",
                expected_formula="rank(close)",
                expected_settings=self.settings.as_platform_dict(),
            )

        self.assertEqual(raised.exception.code, "worldquant_detail_zero_capital")

    def _detail_payload(self) -> dict[str, object]:
        response_settings = self.settings.as_platform_dict()
        response_settings.update(
            {"startDate": "2019-01-01", "endDate": "2023-12-31"}
        )
        return {
            "id": "alpha-1",
            "type": "REGULAR",
            "settings": response_settings,
            "regular": {"code": "rank(close)", "operatorCount": 1},
            "status": "UNSUBMITTED",
            "is": {
                "pnl": 100_000,
                "bookSize": 20_000_000,
                "turnover": 0.12,
                "returns": 0.08,
                "drawdown": 0.04,
                "margin": 0.001,
                "sharpe": 1.3,
                "fitness": 1.1,
                "longCount": 1_523,
                "shortCount": 1_477,
                "checks": [
                    {
                        "name": "LOW_SHARPE",
                        "result": "PASS",
                        "limit": 1.25,
                        "value": 1.3,
                    },
                    {
                        "name": "CONCENTRATED_WEIGHT",
                        "result": "FAIL",
                        "limit": 0.1,
                        "value": 0.160449,
                        "date": "2021-12-30",
                    },
                    {"name": "SELF_CORRELATION", "result": "PENDING"},
                    {
                        "name": "MATCHES_COMPETITION",
                        "result": "PASS",
                        "competitions": [],
                    },
                ],
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
                ],
                [
                    "2021",
                    100_000,
                    20_000_000,
                    201,
                    199,
                    0.12,
                    1.1,
                    0.04,
                    0.06,
                    0.0009,
                    0.7,
                    "IS",
                ],
            ],
        }


if __name__ == "__main__":
    unittest.main()
