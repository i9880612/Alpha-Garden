import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

from execution.console_series import ConsoleSeries
from persistence.database import open_database
from tests.execution.console_fixture import make_console_reader
from tests.execution.test_qualified_archive import complete_candidate, prepare_candidate
from worldquant.client import WorldQuantRequestError
from worldquant.recordsets import RecordsetObservation


class ConsoleSeriesTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.reader = make_console_reader(folder.name)
        with open_database(self.reader.paths.database) as connection:
            self.candidate = complete_candidate(connection, prepare_candidate(connection))
        self.series = ConsoleSeries(self.reader)

    def test_requests_only_selected_metric_and_caches_it_without_database_writes(self):
        before = self.reader.paths.database.read_bytes()
        with patch("execution.console_series.WorldQuantClient") as factory:
            client = factory.return_value
            client.authenticated = True
            client.fetch_recordset.side_effect = [RecordsetObservation((("2020-01-01", -10.0),)),
                WorldQuantRequestError("worldquant_recordset_request_failed", retryable=True, outcome_unknown=False),
                RecordsetObservation((("2020-01-01", None), ("2020-01-02", 1.2)))]
            task_id = self.candidate.task.task_id
            value = self.series.fetch(task_id, "pnl")
            self.assertEqual(value["metric"], "pnl")
            self.assertEqual(value["state"], "ready")
            client.fetch_recordset.assert_called_once_with(platform_alpha_id=value["alpha_id"], metric="pnl")
            self.assertEqual(self.series.fetch(task_id, "pnl")["points"], value["points"])
            self.assertEqual(client.fetch_recordset.call_count, 1)
            self.assertEqual(self.series.fetch(task_id, "sharpe")["state"], "error")
            self.assertEqual(self.series.fetch(task_id, "turnover")["points"][0][1], None)
            self.assertEqual([call.kwargs["metric"] for call in client.fetch_recordset.call_args_list], ["pnl", "sharpe", "turnover"])
            self.assertEqual(client.fetch_recordset.call_count, 3)
        self.assertEqual(before, self.reader.paths.database.read_bytes())

    def test_rate_limit_blocks_other_metrics_and_honors_retry_after(self):
        with patch("execution.console_series.WorldQuantClient") as factory:
            client = factory.return_value
            client.authenticated = True
            client.fetch_recordset.side_effect = WorldQuantRequestError("worldquant_recordset_http_error", status_code=429,
                retryable=True, outcome_unknown=False, retry_after_seconds=120)
            result = self.series.fetch(self.candidate.task.task_id, "pnl")
            self.assertEqual(client.fetch_recordset.call_count, 1)
            self.assertEqual(result["refresh_after_seconds"], 120)
            self.assertEqual(result["http_status"], 429)
            self.series.fetch(self.candidate.task.task_id, "pnl")
            with self.assertRaisesRegex(ValueError, "console_series_cooldown"):
                self.series.fetch(self.candidate.task.task_id, "sharpe")
            self.assertEqual(client.fetch_recordset.call_count, 1)

    def test_read_only_foreign_and_missing_tasks_never_access_platform(self):
        with patch("execution.console_series.WorldQuantClient") as factory:
            with self.assertRaisesRegex(ValueError, "console_read_only"):
                ConsoleSeries(self.reader, read_only=True).fetch(self.candidate.task.task_id, "pnl")
            with self.assertRaisesRegex(ValueError, "console_formula_not_found"):
                self.series.fetch("missing", "pnl")
            with self.assertRaisesRegex(ValueError, "console_series_metric_invalid"):
                self.series.fetch(self.candidate.task.task_id, "all")
            with open_database(self.reader.paths.database) as connection:
                connection.execute("UPDATE backtest_tasks SET account_scope='foreign' WHERE task_id=?", (self.candidate.task.task_id,))
                connection.commit()
            with self.assertRaisesRegex(ValueError, "console_formula_not_found"):
                self.series.fetch(self.candidate.task.task_id, "pnl")
            factory.assert_not_called()

    def test_overlapping_requests_share_cached_result(self):
        entered, release = Event(), Event()
        def observe(**kwargs):
            entered.set()
            self.assertTrue(release.wait(5))
            return RecordsetObservation((("2020-01-01", 10.0),))
        with patch("execution.console_series.WorldQuantClient") as factory, ThreadPoolExecutor(2) as pool:
            client = factory.return_value
            client.authenticated = True
            client.fetch_recordset.side_effect = observe
            first = pool.submit(self.series.fetch, self.candidate.task.task_id, "pnl")
            self.assertTrue(entered.wait(5))
            second = pool.submit(self.series.fetch, self.candidate.task.task_id, "pnl")
            release.set()
            self.assertEqual(first.result(5)["points"], second.result(5)["points"])
            self.assertEqual(client.fetch_recordset.call_count, 1)

    def test_pending_cache_expires_at_retry_delay(self):
        with patch("execution.console_series.WorldQuantClient") as factory, patch("execution.console_series.time.monotonic", return_value=100) as now:
            client = factory.return_value
            client.authenticated = True
            client.fetch_recordset.side_effect = [RecordsetObservation(None, retry_after_seconds=1),
                                                 RecordsetObservation((("2020-01-01", 10.0),))]
            self.assertEqual(self.series.fetch(self.candidate.task.task_id, "pnl")["state"], "pending")
            now.return_value = 100.5
            self.assertEqual(self.series.fetch(self.candidate.task.task_id, "pnl")["refresh_after_seconds"], 0.5)
            self.assertEqual(client.fetch_recordset.call_count, 1)
            now.return_value = 101
            self.assertEqual(self.series.fetch(self.candidate.task.task_id, "pnl")["state"], "ready")


if __name__ == "__main__":
    unittest.main()
