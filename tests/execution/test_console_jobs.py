import logging
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from execution.console_jobs import ConsoleJobs
from execution.process_lock import exclusive_run_process
from execution.progress import LOGGER_NAME, live_progress
from execution.submission_runner import SubmissionQueueCompletion
from persistence.console import read_console_database
from tests.execution.console_fixture import make_console_reader

class ConsoleJobsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.reader = make_console_reader(self.folder.name)
        self.jobs = ConsoleJobs(self.reader)
        self.addCleanup(self.jobs.close)

    def finished(self):
        self.jobs._thread.join(timeout=3)
        self.assertFalse(self.jobs.snapshot()["busy"])
        return self.jobs.snapshot()["items"][0]

    def test_duplicate_requests_start_once_busy_then_pause_releases_slot(self):
        entered = threading.Event()
        def running(*args, **kwargs):
            kwargs["run_created"](SimpleNamespace(run_id="run-one"))
            logging.getLogger(LOGGER_NAME).info("本地测试正在执行")
            entered.set()
            kwargs["waiter"](60)
        request = {"kind": "run", "mode": "optimization", "cycles": 2}
        initial_version = self.jobs.version()
        with patch("execution.console_jobs.launch_automated_run", side_effect=running) as launch:
            job_id = self.jobs.start(request, "same-request-identity")
            self.assertTrue(entered.wait(2))
            running_version = self.jobs.version()
            self.assertGreater(running_version[0], initial_version[0])
            self.assertTrue(running_version[1])
            self.jobs.snapshot()
            self.assertEqual(self.jobs.version(), running_version)
            self.assertEqual(self.jobs.start(request, "same-request-identity"), job_id)
            with self.assertRaisesRegex(ValueError, "console_operation_busy"):
                self.jobs.start(request, "second-request-identity")
            with self.assertRaisesRegex(ValueError, "console_settings_busy"):
                self.jobs.save_settings({})
            with self.assertRaisesRegex(ValueError, "console_request_id_conflict"):
                self.jobs.start({**request, "cycles": 3}, "same-request-identity")
            self.jobs.stop(job_id)
            result = self.finished()
            self.assertGreater(self.jobs.version()[0], running_version[0])
            self.assertFalse(self.jobs.version()[1])
            self.assertEqual(result["status"], "paused")
            self.assertEqual(result["run_id"], "run-one")
            self.assertIn("本地测试正在执行", result["logs"][0]["message"])
            launch.assert_called_once()
            limits = launch.call_args.kwargs["limits"]
            self.assertEqual(limits.max_cycles, 2)
            self.assertTrue(limits.optimization_only)
            self.assertFalse(limits.automatic_submissions_enabled)

    def test_failure_does_not_occupy_next_operation_and_success_count_excludes_active(self):
        request = {"kind": "submit", "grade": "GOOD", "count": 2}
        with patch("execution.console_jobs.submit_queued_alphas", side_effect=RuntimeError("private-password")):
            self.jobs.start(request, "failed-request-identity")
            self.assertEqual(self.finished()["status"], "failed")
            self.assertNotIn("private-password", str(self.jobs.snapshot()))
        completion = SubmissionQueueCompletion("group-account", 3, 4, 5, False, 2, 3, 1, 0, 0, 0, 1, 2)
        with patch("execution.console_jobs.submit_queued_alphas", return_value=completion) as submit:
            self.jobs.start(request, "next-request-identity")
            result = self.finished()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["result"]["new_submitted_count"], 2)
            self.assertEqual(submit.call_args.kwargs["grade"], "GOOD")
            self.assertEqual(submit.call_args.kwargs["source"], "qualified_archive")
            self.assertEqual(submit.call_args.kwargs["max_submissions"], 2)

    def test_live_progress_is_replaced_without_filling_logs_and_cleared_after_pause(self):
        observed, proceed, updated = threading.Event(), threading.Event(), threading.Event()
        revisions = []
        def running(*args, **kwargs):
            logging.getLogger(LOGGER_NAME).info("本地测试认证完成")
            live_progress("第4轮 完成0/100 在途3 待发97 | 已提交回测，等待结果")
            revisions.append(self.jobs.version())
            for _ in range(50):
                live_progress("第4轮 完成0/100 在途3 待发97 | 已提交回测，等待结果")
            revisions.append(self.jobs.version())
            observed.set()
            if not proceed.wait(3):
                raise AssertionError("test_progress_not_released")
            live_progress("第4轮 完成1/100 在途3 待发96 | 已提交回测，等待结果")
            updated.set()
            kwargs["waiter"](60)
        self.addCleanup(proceed.set)
        with patch("execution.console_jobs.launch_automated_run", side_effect=running):
            job_id = self.jobs.start({"kind": "run", "mode": "normal", "cycles": 1}, "live-progress")
            self.assertTrue(observed.wait(2))
            first = self.jobs.snapshot()["items"][0]
            self.assertEqual(revisions[0], revisions[1])
            self.assertIn("在途3 待发97", first["progress"]["message"])
            self.assertEqual(len(first["logs"]), 1)
            live_progress("另一个线程的进度")
            self.assertEqual(self.jobs.snapshot()["items"][0]["progress"], first["progress"])
            proceed.set()
            self.assertTrue(updated.wait(2))
            current = self.jobs.snapshot()["items"][0]
            self.assertIn("完成1/100", current["progress"]["message"])
            self.assertIn("完成0/100", first["progress"]["message"])
            self.assertGreater(self.jobs.version()[0], revisions[1][0])
            self.assertEqual(len(current["logs"]), 1)
            self.jobs.stop(job_id)
            paused = self.finished()
            self.assertEqual(paused["status"], "paused")
            self.assertIsNone(paused["progress"])

    def test_result_logs_clear_previous_wait_progress(self):
        snapshots = []
        def submit(*args, **kwargs):
            live_progress("等待平台结果")
            logging.getLogger(LOGGER_NAME).info("已收到结果")
            snapshots.append(self.jobs.snapshot()["items"][0])
            return SubmissionQueueCompletion("group-account", 0, 0, 0, False, 0, 0, 0, 0, 0, 0, 0, 0)
        with patch("execution.console_jobs.submit_queued_alphas", side_effect=submit):
            self.jobs.start({"kind": "submit", "grade": "GOOD", "count": 1}, "progress-result")
            result = self.finished()
        self.assertIsNone(snapshots[0]["progress"])
        self.assertEqual([line["message"] for line in result["logs"]], ["已收到结果"])
        self.assertIsNone(result["progress"])

    def test_existing_cli_lock_is_respected_before_any_platform_request(self):
        with exclusive_run_process(self.reader.paths.database), patch("execution.launch._default_client_factory") as client:
            self.jobs.start({"kind": "run", "mode": "normal", "cycles": 1}, "locked-request-identity")
            self.assertEqual(self.finished()["error"], "console_database_busy")
            client.assert_not_called()

    def test_pause_real_launch_preserves_run_and_resume_uses_same_id(self):
        def client_created(settings):
            self.jobs.stop(self.jobs.snapshot()["items"][0]["id"])
            return SimpleNamespace()
        with patch("execution.launch._default_client_factory", side_effect=client_created):
            self.jobs.start({"kind": "run", "mode": "normal", "cycles": 1}, "pause-real-launch-identity")
            paused = self.finished()
        self.assertEqual(paused["status"], "paused")
        with read_console_database(self.reader.paths.database) as connection:
            self.assertEqual(connection.execute("SELECT run_id FROM automated_runs").fetchone()[0], paused["run_id"])
            self.assertEqual(connection.execute("SELECT stop_reason FROM automated_runs").fetchone()[0], "user_paused")
        # A new reader/server sees the durable pause without retaining the job.
        row = self.reader.runs()["items"][0]
        self.assertEqual(row["status"], "paused")
        self.assertTrue(row["can_resume"])
        complete = SimpleNamespace(run=SimpleNamespace(run_id=paused["run_id"], status="completed", stop_reason="budget_exhausted"))
        with patch("execution.console_jobs.resume_automated_run", return_value=complete) as resume:
            self.jobs.start({"kind": "resume", "run_id": paused["run_id"]}, "resume-request-identity")
            self.assertEqual(self.finished()["status"], "completed")
            self.assertEqual(resume.call_args.args[2], paused["run_id"])

    def test_read_only_refuses_operation(self):
        with self.assertRaisesRegex(ValueError, "console_read_only"):
            ConsoleJobs(self.reader, read_only=True).start({"kind": "run"}, "readonly-request-identity")
