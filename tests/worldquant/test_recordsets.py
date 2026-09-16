import json
import unittest

from worldquant.backtests import WorldQuantProtocolError
from worldquant.client import WorldQuantClient, WorldQuantCredentials, WorldQuantResponse
from worldquant.recordsets import RECORDSET_UNITS, parse_recordset
from tests.worldquant.test_client import FakeExecutor


def payload(metric, rows):
    return {"schema": {"properties": [{"name": metric, "type": RECORDSET_UNITS[metric]}, {"name": "date", "type": "date"}]}, "records": rows}


class RecordsetTests(unittest.TestCase):
    def test_gaps_negatives_and_percent_ratios_are_preserved(self):
        self.assertEqual(parse_recordset(payload("sharpe", [[None, "2020-01-01"], [-1.2, "2020-01-02"]]), "sharpe"),
                         (("2020-01-01", None), ("2020-01-02", -1.2)))
        self.assertEqual(parse_recordset(payload("turnover", [[1.8858, "2020-01-01"]]), "turnover")[0][1], 1.8858)
        self.assertEqual(parse_recordset(payload("pnl", []), "pnl"), ())

    def test_invalid_values_order_units_and_columns_are_rejected(self):
        for rows in ([[True, "2020-01-01"]], [[float("nan"), "2020-01-01"]], [[1, "not-a-date"]],
                     [[1, "2020-01-02"], [2, "2020-01-01"]], [[1, "2020-01-01"], [2, "2020-01-01"]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                parse_recordset(payload("pnl", rows), "pnl")
        wrong = payload("sharpe", [])
        with self.assertRaises(ValueError):
            parse_recordset(wrong, "pnl")
        wrong["schema"]["properties"][0]["type"] = "percent"
        with self.assertRaises(ValueError):
            parse_recordset(wrong, "sharpe")

    def test_client_reads_exact_recordset_and_returns_pending_without_waiting(self):
        executor = FakeExecutor(WorldQuantResponse(201, {}, b"{}"),
            WorldQuantResponse(200, {}, json.dumps(payload("sharpe", [[None, "2020-01-01"]])).encode()),
            WorldQuantResponse(503, {"Retry-After": "5"}, b""), WorldQuantResponse(200, {}, b"{}"))
        client = WorldQuantClient(base_url="https://example.invalid", credentials=WorldQuantCredentials(bearer_token="test"), request_executor=executor)
        client.authenticate()
        self.assertEqual(client.fetch_recordset(platform_alpha_id="test-alpha", metric="sharpe").points, (("2020-01-01", None),))
        pending = client.fetch_recordset(platform_alpha_id="test-alpha", metric="turnover")
        self.assertIsNone(pending.points)
        self.assertEqual(pending.retry_after_seconds, 5)
        self.assertEqual(len(executor.requests), 3)
        self.assertTrue(executor.requests[1].full_url.endswith("/alphas/test-alpha/recordsets/sharpe"))
        self.assertEqual(executor.requests[1].get_method(), "GET")
        with self.assertRaises(WorldQuantProtocolError):
            client.fetch_recordset(platform_alpha_id="test-alpha", metric="pnl")


if __name__ == "__main__":
    unittest.main()
