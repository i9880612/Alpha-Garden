import json
import math
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import patch

from execution.quality_diagnosis import QualityDiagnosis
from persistence.database import open_database
from persistence.pnl import save_pnl_series
from persistence.runs import create_automated_run
from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
from persistence.submissions import record_platform_submitted_alphas
from tests.execution.console_fixture import make_console_reader
from tests.execution.test_qualified_archive import complete_candidate, prepare_candidate
from tests.learning.test_seed_correlation import series, reference
from tests.persistence import test_runs


def saved_reference(alpha, account="group-account", sharpe=1.5):
    ref = reference(alpha, account, sharpe)
    return replace(ref, raw_payload={**ref.raw_payload, "id": alpha, "status": ref.status,
        "dateSubmitted": ref.date_submitted, "hidden": False, "regular": {"code": ref.formula}})


class QualityDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.reader = make_console_reader(self.folder.name)
        self.analysis = QualityDiagnosis(self.reader)

    def candidate(self, connection, **kwargs):
        return complete_candidate(connection, prepare_candidate(connection), **kwargs)

    def check(self, connection, task, checks=None, error=None):
        save_submission_check(connection, SubmissionCheckRecord(task.task.task_id,
            "2026-09-02T00:00:00+00:00", json.dumps({"is": {"checks": checks}}) if checks is not None else None, error, attempt_count=2))

    def test_account_configuration_source_run_and_date_filters_exclude_unsent(self):
        with open_database(self.reader.paths.database) as c:
            first = self.candidate(c)
            child = complete_candidate(c, prepare_candidate(c, parent=first))
            other_setting = self.candidate(c)
            foreign = self.candidate(c)
            prepare_candidate(c, start=False)
            pending = prepare_candidate(c)
            c.execute("UPDATE backtest_tasks SET settings_json=? WHERE task_id=?", ('{"delay":0}', other_setting.task.task_id))
            c.execute("UPDATE backtest_tasks SET account_scope='other' WHERE task_id=?", (foreign.task.task_id,))
            run = replace(test_runs.AutomatedRunPersistenceTests._record("run-quality"), optimization_only=True)
            create_automated_run(c, run)
            c.execute("INSERT INTO automated_run_backtests VALUES (?, ?, 1)", (run.run_id, child.task.task_id))
            today = datetime.now().astimezone().isoformat()
            c.execute("UPDATE backtest_tasks SET finished_at=? WHERE task_id=?", (today, child.task.task_id))
            c.execute("UPDATE backtest_tasks SET finished_at=? WHERE task_id=?", ((datetime.now().astimezone()-timedelta(days=60)).isoformat(), first.task.task_id))
        data = self.analysis.quality(days="all")
        self.assertEqual((data["total"], data["completed"], data["inflight"]), (3, 2, 1))
        self.assertEqual({r["task_id"] for r in data["records"]}, {first.task.task_id, child.task.task_id})
        self.assertEqual(self.analysis.quality(days="all", source="mutation")["completed"], 1)
        self.assertEqual(self.analysis.quality(days="all", mode="normal")["total"], 0)
        self.assertEqual(self.analysis.quality(days="all", mode="optimization")["records"][0]["task_id"], child.task.task_id)
        self.assertEqual(self.analysis.quality(days="all", run_id=run.run_id)["completed"], 1)
        self.assertEqual(self.analysis.quality(days="all", run_id="missing")["total"], 0)
        self.assertEqual(self.analysis.quality(days="7", source="mutation")["completed"], 1)
        setting = next(x["id"] for x in data["configurations"] if x["settings"]["delay"] == 0)
        self.assertEqual(self.analysis.quality(days="all", configuration=setting)["records"][0]["task_id"], other_setting.task.task_id)
        self.assertNotIn(pending.task.task_id, [r["task_id"] for r in data["records"]])

    def test_latest_pending_error_missing_and_snapshot_checks_keep_correct_denominators(self):
        with open_database(self.reader.paths.database) as c:
            passed, failed, pending, error, snapshot = [self.candidate(c) for _ in range(5)]
            self.check(c, passed, [{"name":"LOW_SHARPE","result":"PASS","limit":1.25}, {"name":"LOW_FITNESS","result":"PASS","limit":1}, {"name":"SELF_CORRELATION","result":"PASS"}])
            self.check(c, failed, [{"name":"LOW_SHARPE","result":"PASS","limit":1.25}, {"name":"LOW_FITNESS","result":"FAIL","limit":1}, {"name":"SELF_CORRELATION","result":"FAIL"}])
            self.check(c, pending, [{"name":"LOW_SHARPE","result":"PASS"}, {"name":"LOW_FITNESS","result":"PENDING"}, {"name":"SELF_CORRELATION","result":"PENDING"}])
            self.check(c, error, error="request_timeout")
            c.execute("DELETE FROM submission_checks WHERE task_id=?", (snapshot.task.task_id,))
            c.execute("DELETE FROM backtest_checks WHERE task_id=? AND check_name='SELF_CORRELATION'", (snapshot.task.task_id,))
            c.execute("UPDATE backtest_results SET turnover=1.52 WHERE task_id=?", (failed.task.task_id,))
        before = self.reader.paths.database.read_bytes()
        with patch("worldquant.client.WorldQuantClient.authenticate", side_effect=AssertionError("Read only")):
            data = self.analysis.quality(days="all")
            self.analysis.correlation(days="all")
        self.assertEqual(before, self.reader.paths.database.read_bytes())
        self.assertEqual(data["platform"], {"passed":1, "failed":1, "pending":1, "error":1, "missing":1})
        reason = next(r for r in data["reasons"] if r["name"] == "LOW_FITNESS")
        self.assertEqual((reason["passed"], reason["failed"], reason["unresolved"]), (2, 1, 2))
        rows = {r["task_id"]: r for r in data["records"]}
        self.assertEqual(rows[failed.task.task_id]["metric_group"], "fitness")
        self.assertEqual(rows[failed.task.task_id]["turnover"], 1.52)
        self.assertEqual(rows[error.task.task_id]["metric_group"], "unknown")
        self.assertIsNone(data["thresholds"])
        self.assertNotIn("formula", json.dumps(data))

    def test_reference_lines_require_uniform_saved_limits(self):
        with open_database(self.reader.paths.database) as c:
            tasks = [self.candidate(c) for _ in range(2)]
            for t in tasks:
                self.check(c, t, [{"name":"LOW_SHARPE","result":"PASS","limit":1.25}, {"name":"LOW_FITNESS","result":"PASS","limit":1}])
        self.assertEqual(self.analysis.quality(days="all")["thresholds"], [1.25, 1])
        with open_database(self.reader.paths.database) as c:
            c.execute("UPDATE submission_checks SET payload_json=replace(payload_json,'1.25','1.58') WHERE task_id=?", (tasks[1].task.task_id,))
        self.assertIsNone(self.analysis.quality(days="all")["thresholds"])

    def test_local_correlation_ten_percent_boundary_missing_evidence_and_account_isolation(self):
        wave = [math.sin(i) for i in range(300)]
        with open_database(self.reader.paths.database) as c:
            passed = self.candidate(c, sharpe=1.65)
            failed = self.candidate(c, sharpe=1.649999)
            short = self.candidate(c)
            c.execute("DELETE FROM platform_pnl_series")
            for t in (passed, failed):
                save_pnl_series(c, series(t.task.platform_alpha_id, wave, "group-account"))
            save_pnl_series(c, series(short.task.platform_alpha_id, wave[:20], "group-account"))
            record_platform_submitted_alphas(c, (saved_reference("ref", "group-account", 1.5), saved_reference("foreign", "other", 9)))
            save_pnl_series(c, series("ref", wave, "group-account"))
        data = self.analysis.correlation(days="all")
        self.assertEqual(data["counts"], {"passed":1,"failed":1,"pending":1})
        self.assertEqual(data["reference_count"], 1)
        self.assertEqual(data["improvement_passed"], 1)
        high = next(r for r in data["records"] if r["task_id"] == passed.task.task_id)
        self.assertEqual((high["compared"], high["required"], high["high_pairs"], high["improved_pairs"]), (1,1,1,1))
        with open_database(self.reader.paths.database) as c:
            record_platform_submitted_alphas(c, (saved_reference("missing"),))
        data = self.analysis.correlation(days="all")
        self.assertEqual(data["counts"], {"failed":1,"pending":2})
        self.assertEqual(data["improvement_passed"], 0)

    def test_no_references_and_empty_scope_are_not_low_correlation_passes(self):
        with open_database(self.reader.paths.database) as c:
            self.candidate(c)
        data = self.analysis.correlation(days="all")
        self.assertEqual(data["counts"], {"no_references":1})
        self.assertEqual(self.analysis.correlation(days="all", run_id="missing")["records"], [])
        for filters in ({"days":"-1"}, {"mode":"invalid"}, {"source":"invalid"}, {"configuration":"invalid"}):
            with self.assertRaises(ValueError):
                self.analysis.quality(**filters)
