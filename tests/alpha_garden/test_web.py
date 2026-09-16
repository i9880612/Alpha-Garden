import io
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import closing, redirect_stdout
from datetime import datetime
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch
from alpha_garden.cli import run
from alpha_garden.web import ConsoleServer
from alpha_garden.web_events import ConsoleEvents
from tests.execution.console_fixture import make_console_reader
from tests.execution.test_qualified_archive import complete_candidate, prepare_candidate
from persistence.database import open_database

class ConsoleHTTPTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.reader = make_console_reader(self.folder.name)
        with open_database(self.reader.paths.database) as connection:
            self.candidate = complete_candidate(connection, prepare_candidate(connection))
        assets = Path(self.folder.name) / "assets"
        assets.mkdir()
        (assets / "index.html").write_text("<main>console</main>", encoding="utf-8")
        self.server = ConsoleServer(0, self.reader, read_only=True, assets=assets)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self.close)
        self.cookie = ""
        self.assertEqual(self.request("/api/login", "POST", {"username": "admin", "password": "123123"},
                                      {"Content-Type": "application/json"})[0], 200)

    def test_research_routes_are_authenticated_scoped_and_validate_filters(self):
        before = self.reader.paths.database.read_bytes()
        paths = ["/api/catalog/fields", "/api/catalog/operators", "/api/seeds", "/api/seeds?mode=normal", "/api/formula?task_id=" + self.candidate.task.task_id,
                 "/api/formula/children?task_id=" + self.candidate.task.task_id,
                 "/api/analysis/quality?days=all", "/api/analysis/correlation?days=all"]
        with patch("worldquant.client.WorldQuantClient.authenticate", side_effect=AssertionError("No platform access")):
            for path in paths:
                with self.subTest(path=path):
                    status, value = self.request(path)
                    self.assertEqual(status, 200, value)
                    self.assertNotIn("synthetic-console-secret", json.dumps(value))
        self.assertEqual(before, self.reader.paths.database.read_bytes())
        self.assertEqual(self.request("/api/formula?task_id=missing")[0], 404)
        self.assertEqual(self.request("/api/formula/children?task_id=missing")[0], 404)
        child_path = "/api/formula/children?task_id=" + self.candidate.task.task_id
        for path in [child_path + "&page=0", child_path + "&page_size=101", child_path + "&page=1&page=2",
                     "/api/seeds?mode=invalid", "/api/catalog/fields?page_size=101", "/api/catalog/fields?page=0",
                     "/api/analysis/quality?days=-1", "/api/analysis/quality?days=7&days=30",
                     "/api/analysis/correlation?source=invalid", "/api/analysis/correlation?configuration=invalid"]:
            self.assertEqual(self.request(path)[0], 400, path)
        self.assertEqual(self.request("/api/analysis/records")[0], 404)
        for path in ["/data", "/operators", "/seeds", "/analysis", "/optimization", "/archive", "/submitted"]:
            self.assertEqual(self.request(path), (200, "<main>console</main>"))
        self.cookie = ""
        for path in paths:
            self.assertEqual(self.request(path)[0], 401)

    def test_platform_series_routes_select_one_metric_and_require_authenticated_csrf_post(self):
        body = {"task_id": self.candidate.task.task_id}
        headers = {"Content-Type": "application/json"}
        with patch("execution.console_series.WorldQuantClient") as client:
            self.assertEqual(self.request("/api/formula/series/pnl", "POST", body, headers)[0], 403)
            headers["X-Console-Token"] = self.server.csrf_token
            self.assertEqual(self.request("/api/formula/series/pnl", "POST", body, headers)[0], 403)
            self.assertEqual(self.request("/api/formula/series/pnl")[0], 404)
            client.assert_not_called()
        self.server.series.read_only = False
        with patch.object(self.server.series, "fetch", return_value={"state": "ready"}) as fetch:
            for metric in ("pnl", "sharpe", "turnover"):
                self.assertEqual(self.request("/api/formula/series/" + metric, "POST", body, headers), (200, {"state": "ready"}))
                fetch.assert_called_with(self.candidate.task.task_id, metric)
            self.assertEqual(fetch.call_count, 3)
            self.assertEqual(self.request("/api/formula/series", "POST", body, headers)[0], 404)
            self.assertEqual(self.request("/api/formula/series/all", "POST", body, headers)[0], 404)
            self.assertEqual(self.request("/api/formula/series/pnl", "POST", {**body, "url": "https://other.invalid"}, headers)[0], 400)
        self.cookie = ""
        self.assertEqual(self.request("/api/formula/series/pnl", "POST", body, headers)[0], 401)

    def close(self):
        self.server.shutdown()
        self.server.jobs.close()
        self.server.server_close()
        self.thread.join(3)

    def test_settings_save_requires_authorization_and_only_updates_selected_local_config(self):
        from dataclasses import replace
        policy = Path(self.folder.name) / "policy.json"
        run_config = Path(self.folder.name) / "run.json"
        policy.write_bytes(self.reader.paths.settings.read_bytes())
        run_config.write_bytes(self.reader.paths.run_config.read_bytes())
        self.reader.paths = replace(self.reader.paths, settings=policy, run_config=run_config)
        current = self.request("/api/settings")[1]
        body = {"section": "run", "revision": current["revisions"]["run"], "value": current["run_config"]}
        body["value"]["backtestCount"] = 80
        headers = {"Content-Type": "application/json"}
        self.assertEqual(self.request("/api/settings", "POST", body, headers)[0], 403)
        headers["X-Console-Token"] = self.server.csrf_token
        self.assertEqual(self.request("/api/settings", "POST", body, headers)[0], 403)
        self.server.jobs.read_only = False
        database_before, policy_before = self.reader.paths.database.read_bytes(), policy.read_bytes()
        with patch("worldquant.client.WorldQuantClient") as client:
            self.assertEqual(self.request("/api/settings", "POST", {**body, "path": "arbitrary.json"}, headers)[0], 400)
            status, saved = self.request("/api/settings", "POST", body, headers)
            self.assertEqual(status, 200, saved)
            self.assertEqual(self.request("/api/settings")[1]["limits"]["backtest_count"], 80)
            self.assertEqual(self.request("/api/settings", "POST", body, headers)[0], 409)
            client.assert_not_called()
        self.assertEqual(self.reader.paths.database.read_bytes(), database_before)
        self.assertEqual(policy.read_bytes(), policy_before)
        self.cookie = ""
        self.assertEqual(self.request("/api/settings", "POST", body, headers)[0], 401)

    def request(self, path, method="GET", body=None, headers=None):
        with closing(HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)) as connection:
            connection.request(method, path, json.dumps(body) if body is not None else None,
                               {"Cookie": self.cookie, **(headers or {})})
            response = connection.getresponse()
            if response.getheader("Set-Cookie"):
                self.cookie = response.getheader("Set-Cookie").split(";", 1)[0]
            raw = response.read().decode()
            return response.status, json.loads(raw) if response.headers.get_content_type() == "application/json" else raw

    def test_read_endpoints_are_json_scoped_and_do_not_modify_database(self):
        before = self.reader.paths.database.read_bytes()
        with patch("worldquant.client.WorldQuantClient.authenticate", side_effect=AssertionError("No platform access")):
            for path in ["/api/session", "/api/dashboard/activity", "/api/dashboard/submitted-grades", "/api/dashboard/research",
                         "/api/dashboard/progress", "/api/dashboard/recent", "/api/settings", "/api/runs", "/api/formulas",
                         "/api/formulas?category=optimization", "/api/formulas?category=archive",
                         "/api/formulas?category=submitted", "/api/formulas?category=candidates&grade=GOOD",
                         "/api/formulas?execution=started", "/api/formulas?execution=finished", "/api/formulas?execution=inflight", "/api/formulas?execution=planned",
                         "/api/submissions", "/api/jobs"]:
                with self.subTest(path=path):
                    status, value = self.request(path)
                    self.assertEqual(status, 200)
                    text = json.dumps(value)
                    self.assertNotIn("synthetic-console-secret", text)
                    if path == "/api/dashboard/recent":
                        self.assertEqual(value["items"][0]["formula"], self.candidate.task.formula)
                    else:
                        self.assertNotIn(self.candidate.task.formula, text)
        self.assertEqual(before, self.reader.paths.database.read_bytes())
        self.assertTrue(self.request("/api/session")[1]["read_only"])
        self.assertEqual(self.request("/api/formulas?search=not-a-real-id")[1]["total"], 0)

    def test_dashboard_cached_counts_follow_new_results_and_account_changes(self):
        from execution.seeds import synchronize_signal_seeds
        now = datetime.now().astimezone().isoformat()
        with open_database(self.reader.paths.database) as connection:
            connection.execute("UPDATE backtest_tasks SET finished_at=?", (now,))
            synchronize_signal_seeds(connection)
        for _ in range(2):
            self.assertEqual(self.request("/api/dashboard/activity")[1]["metrics"][0]["value"], 1)
            self.assertEqual(self.request("/api/dashboard/research")[1]["optimization_parents"], 1)
        with open_database(self.reader.paths.database) as connection:
            complete_candidate(connection, prepare_candidate(connection))
            connection.execute("UPDATE backtest_tasks SET finished_at=?", (now,))
            synchronize_signal_seeds(connection)
        # The first HTTP request after the commit must reflect it; no SSE wait.
        self.assertEqual(self.request("/api/dashboard/activity")[1]["metrics"][0]["value"], 2)
        self.assertEqual(self.request("/api/dashboard/research")[1]["optimization_parents"], 2)
        environment = self.reader.paths.environment.read_text(encoding="utf-8")
        self.reader.paths.environment.write_text(environment.replace("group-account", "other"), encoding="utf-8")
        self.assertEqual(self.request("/api/dashboard/activity")[1]["metrics"][0]["value"], 0)
        self.assertEqual(self.request("/api/dashboard/research")[1]["optimization_parents"], 0)

    def test_host_origin_token_and_read_only_are_enforced(self):
        request = {"kind": "run", "mode": "normal", "cycles": 1}
        headers = {"Content-Type": "application/json", "Idempotency-Key": "test-request-identity"}
        self.assertEqual(self.request("/api/session", headers={"Host": "attacker.invalid"})[0], 403)
        self.assertEqual(self.request("/api/session", headers={"Origin": "https://attacker.invalid"})[0], 403)
        self.assertEqual(self.request("/api/session", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.request("/api/events", headers={"Origin": "https://attacker.invalid"})[0], 403)
        self.assertEqual(self.request("/api/jobs", "POST", request, headers)[0], 403)
        headers["X-Console-Token"] = self.request("/api/session")[1]["csrf_token"]
        status, value = self.request("/api/jobs", "POST", request, headers)
        self.assertEqual((status, value["error"]), (403, "console_read_only"))
        self.assertFalse(self.server.jobs.snapshot()["items"])

    def test_resume_rejects_terminal_runs_before_creating_jobs_but_keeps_recovery_paths(self):
        from execution.run_config import load_automated_run_limits
        from execution.runs import prepare_automated_run

        run = prepare_automated_run(self.reader.paths.database, self.reader.paths.settings,
            account_scope=self.reader.account_scope,
            limits=load_automated_run_limits(self.reader.paths.run_config, cycles=1),
            created_at="2026-09-14T00:00:00+00:00")
        self.server.jobs.read_only = False
        headers = {"Content-Type": "application/json", "X-Console-Token": self.server.csrf_token}
        request = {"kind": "resume", "run_id": run.run_id}
        cases = [("created", None, True), ("running", None, True),
                 ("failed", "request_failure_limit_reached", True),
                 ("failed", "submission_reconciliation_required", True),
                 ("failed", "runtime_error", False), ("completed", "max_cycles_reached", False)]
        with patch("execution.console_jobs.resume_automated_run") as resume:
            for status, reason, resumable in cases:
                with self.subTest(status=status, reason=reason):
                    with open_database(self.reader.paths.database) as connection:
                        connection.execute("UPDATE automated_runs SET status=?, stop_reason=? WHERE run_id=?",
                                           (status, reason, run.run_id))
                    before = self.reader.paths.database.read_bytes()
                    code, page = self.request("/api/runs?page_size=10")
                    self.assertEqual(code, 200)
                    row = page["items"][0]
                    self.assertEqual(row["can_resume"], resumable)
                    if not resumable:
                        self.assertIsNotNone(row["resume_blocked_reason"])
                        for index in range(2):
                            headers["Idempotency-Key"] = f"resume-{status}-attempt-{index}"
                            self.assertEqual(self.request("/api/jobs", "POST", request, headers),
                                             (409, {"error": "console_run_not_resumable"}))
                    self.assertEqual(before, self.reader.paths.database.read_bytes())
            resume.assert_not_called()
        self.assertEqual(self.server.jobs.snapshot(), {"items": [], "busy": False})

    def test_event_stream_notifies_changes_without_running_business_reads(self):
        def frame(response):
            while True:
                line = response.readline().decode("utf-8")
                if not line:
                    self.fail("Event stream closed unexpectedly")
                if line.startswith("data: "):
                    return json.loads(line[6:])
        with patch.object(self.reader, "formulas") as formulas, patch.object(self.reader, "dashboard_research") as research:
            with closing(HTTPConnection("127.0.0.1", self.server.server_port, timeout=4)) as connection:
                connection.request("GET", "/api/events", headers={"Cookie": self.cookie})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers.get_content_type(), "text/event-stream")
                first = frame(response)
                self.assertEqual(set(first), {"database", "jobs", "session"})
                with open_database(self.reader.paths.database) as writer:
                    complete_candidate(writer, prepare_candidate(writer))
                changed = frame(response)
                self.assertGreater(changed["database"], first["database"])
                response.close()
            with closing(HTTPConnection("127.0.0.1", self.server.server_port, timeout=4)) as connection:
                connection.request("GET", "/api/events", headers={"Cookie": self.cookie})
                response = connection.getresponse()
                self.assertEqual(frame(response), changed)
                response.close()
            formulas.assert_not_called()
            research.assert_not_called()
        self.assertEqual(self.request("/api/events?resource=/api/jobs")[0], 400)
        self.assertEqual(self.request("/api/events?path=https://attacker.invalid")[0], 400)

    def test_slow_http_section_does_not_block_other_reads_or_change_notifications(self):
        started, release = threading.Event(), threading.Event()
        def slow_research():
            started.set()
            release.wait(4)
            raise OSError("synthetic-console-secret")
        with patch.object(self.reader, "dashboard_research", side_effect=slow_research):
            with ThreadPoolExecutor(max_workers=1) as pool:
                slow = pool.submit(self.request, "/api/dashboard/research")
                try:
                    self.assertTrue(started.wait(1))
                    self.assertEqual(self.request("/api/jobs")[0], 200)
                    self.assertEqual(self.request("/api/dashboard/recent")[0], 200)
                    with self.assertRaises(TimeoutError):
                        slow.result(timeout=0.05)
                finally:
                    release.set()
                self.assertEqual(slow.result(1), (503, {"error": "console_data_unavailable"}))

    def test_invalid_input_cannot_reach_execution_and_static_root_is_bounded(self):
        self.server.jobs.read_only = False
        headers = {"Content-Type": "application/json", "Idempotency-Key": "test-request-identity", "X-Console-Token": self.server.csrf_token}
        with patch.object(self.server.jobs, "start") as start:
            self.assertEqual(self.request("/api/jobs", "POST", {}, {**headers, "Content-Length": "invalid"})[0], 400)
            for request in [
                {"kind": "run", "mode": "normal", "cycles": 0},
                {"kind": "run", "mode": "normal", "cycles": True},
                {"kind": "run", "mode": "normal", "cycles": 1, "automatic_submission": True},
                {"kind": "run", "mode": "optimization", "cycles": 1.2},
                {"kind": "run", "mode": [], "cycles": 1},
                {"kind": "submit", "grade": "UNKNOWN", "count": 2},
                {"kind": "submit", "grade": {}, "count": 2},
                {"kind": "submit", "grade": "GOOD", "count": 1001},
                {"kind": "submit", "task_id": "../test"},
                {"kind": "submit", "task_id": ""},
                {"kind": "submit", "task_id": "task-one", "count": 2},
                {"kind": "submit", "task_id": "task-one", "grade": "GOOD"},
                {"kind": "submit", "task_id": "task-one", "source": "queue"},
                {"kind": "resume", "run_id": "../test"},
            ]:
                self.assertEqual(self.request("/api/jobs", "POST", request, headers)[0], 400)
            start.assert_not_called()
        for path in ["/api/formulas?page=0", "/api/formulas?page_size=1000", "/api/formulas?grade=bad",
                     "/api/formulas?category=optimization&source=bad", "/api/formulas?category=optimization&source=sc&source=mutation",
                     "/api/formulas?grade=GOOD&grade=EXCELLENT"]:
            self.assertEqual(self.request(path)[0], 400)
        self.assertEqual(self.request("/api/formulas?grade=UNKNOWN")[0], 200)
        self.assertEqual(self.request("/api/formulas?category=optimization&source=exploration")[0], 200)
        self.assertEqual(self.request("/runs"), (200, "<main>console</main>"))
        self.assertEqual(self.request("/login"), (200, "<main>console</main>"))
        self.assertEqual(self.request("/../test.env")[0], 404)
        self.assertEqual(self.request("/%2e%2e/test.env")[0], 404)
        self.assertEqual(self.request("/.env")[0], 404)

    def test_login_protects_reads_streams_and_writes_and_logout_revokes_session(self):
        before = self.reader.paths.database.read_bytes()
        anonymous = {"Cookie": ""}
        self.assertEqual(self.request("/api/auth", headers=anonymous),
                         (200, {"authenticated": False, "username": None}))
        with patch.object(self.server.jobs, "start") as start, patch.object(self.reader, "formulas") as formulas:
            for path in ("/api/session", "/api/jobs", "/api/formulas", "/api/events"):
                self.assertEqual(self.request(path, headers=anonymous), (401, {"error": "console_login_required"}))
            self.assertEqual(self.request("/api/jobs", "POST", {"kind": "run"},
                             {**anonymous, "X-Console-Token": self.server.csrf_token})[0], 401)
            start.assert_not_called()
            formulas.assert_not_called()
        for username, password in (("admin", "wrong"), ("wrong", "123123"), ("admin", ""), ([], "123123")):
            self.assertEqual(self.request("/api/login", "POST", {"username": username, "password": password},
                {**anonymous, "Content-Type": "application/json"}), (401, {"error": "console_credentials_invalid"}))
        self.assertEqual(self.request("/api/login", "POST", {"username": "admin", "password": "123123"},
                         {"Origin": "https://attacker.invalid", "Content-Type": "application/json"})[0], 403)
        self.assertEqual(self.request("/api/login", "POST", {"username": "admin", "password": "123123"},
                         {"Content-Type": "text/plain"})[0], 400)
        self.assertEqual(self.request("/api/auth")[1], {"authenticated": True, "username": "admin"})
        old_cookie = self.cookie
        self.assertEqual(self.request("/api/logout", "POST", {}, {"Content-Type": "application/json"})[0], 403)
        self.assertEqual(self.request("/api/logout", "POST", {},
            {"Content-Type": "application/json", "X-Console-Token": self.server.csrf_token})[0], 200)
        self.assertEqual(self.request("/api/runs", headers={"Cookie": old_cookie})[0], 401)
        self.assertEqual(before, self.reader.paths.database.read_bytes())

    def test_logout_revokes_an_already_open_event_stream(self):
        with closing(HTTPConnection("127.0.0.1", self.server.server_port, timeout=4)) as connection:
            connection.request("GET", "/api/events", headers={"Cookie": self.cookie})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            while not response.readline().startswith(b"data: "):
                pass
            self.server.auth.logout(self.cookie)
            with open_database(self.reader.paths.database) as writer:
                complete_candidate(writer, prepare_candidate(writer))
            remaining = response.read().decode()
            self.assertIn("event: unauthorized", remaining)
            self.assertNotIn("event: versions", remaining)

    def test_missing_database_is_unavailable_not_empty_and_is_not_created(self):
        # Release the watcher's read handle before removing the Windows fixture.
        self.server.events.close()
        self.reader.paths.database.unlink()
        self.server.events = ConsoleEvents(self.reader.paths, self.server.jobs)
        self.assertFalse(self.request("/api/session")[1]["ready"])
        for section in ("activity", "submitted-grades", "research", "progress", "recent"):
            with self.subTest(section=section):
                self.assertEqual(self.request("/api/dashboard/" + section)[0], 503)
        self.assertFalse(self.reader.paths.database.exists())

    def test_slow_failing_research_does_not_block_other_dashboard_sections(self):
        started, release = threading.Event(), threading.Event()
        research_responses = []
        def slow_research():
            started.set()
            release.wait(3)
            raise OSError("synthetic research read unavailable")
        with patch.object(self.reader, "dashboard_research", side_effect=slow_research):
            worker = threading.Thread(target=lambda: research_responses.append(self.request("/api/dashboard/research")))
            worker.start()
            try:
                self.assertTrue(started.wait(1))
                for section in ("activity", "submitted-grades", "progress", "recent"):
                    self.assertEqual(self.request("/api/dashboard/" + section)[0], 200)
                self.assertTrue(worker.is_alive())
            finally:
                release.set()
                worker.join(3)
        self.assertEqual(research_responses, [(503, {"error": "console_data_unavailable"})])

    def test_web_cli_forwards_explicit_read_only_without_starting_business_commands(self):
        with patch("alpha_garden.web.serve_console") as serve, redirect_stdout(io.StringIO()):
            self.assertEqual(run(["web", "--read-only", "--port", "8789"]), 0)
        self.assertEqual(serve.call_args.kwargs["port"], 8789)
        self.assertTrue(serve.call_args.kwargs["read_only"])

    def test_http_submission_counts_success_and_consumes_only_one_candidate(self):
        from execution.console import ConsolePaths, ConsoleReader
        from tests.execution.test_submission_runner import SubmissionQueueRunnerTests, SubmissionClient, POLICY_PATH
        fixture = SubmissionQueueRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        _, formulas = fixture._completed_run(("rank(close)", "rank(open)"))
        reader = ConsoleReader(ConsolePaths(fixture.database_path, POLICY_PATH,
                               POLICY_PATH.with_name("run.default.json"), fixture.environment_path))
        self.server.reader = reader
        self.server.jobs.reader = reader
        self.server.jobs.read_only = False
        client = SubmissionClient(formulas)
        headers = {"Content-Type": "application/json", "Idempotency-Key": "http-submit-once-identity",
                   "X-Console-Token": self.server.csrf_token}
        with patch("execution.submission_runner._default_client_factory", return_value=client):
            status, first = self.request("/api/jobs", "POST", {"kind": "submit", "grade": "SPECTACULAR", "count": 1}, headers)
            self.assertEqual(status, 202)
            self.server.jobs._thread.join(10)
            self.assertFalse(self.server.jobs.snapshot()["busy"])
            status, repeated = self.request("/api/jobs", "POST", {"kind": "submit", "grade": "SPECTACULAR", "count": 1}, headers)
            self.assertEqual(first, repeated)
        result = self.request("/api/jobs")[1]["items"][0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["new_submitted_count"], 1)
        self.assertEqual(len(client.posted_ids), 1)
        self.assertEqual(self.request("/api/formulas?category=candidates&grade=SPECTACULAR")[1]["total"], 1)
        self.assertEqual(self.request("/api/submissions")[1]["items"][0]["status"], "submitted")

    def test_http_selected_optimization_submission_is_scoped_and_idempotent(self):
        from execution.console import ConsolePaths, ConsoleReader
        from tests.execution.test_submission_runner import SubmissionQueueRunnerTests, POLICY_PATH
        fixture = SubmissionQueueRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        task_ids, client = fixture._optimization_batch()
        reader = ConsoleReader(ConsolePaths(fixture.database_path, POLICY_PATH,
                               POLICY_PATH.with_name("run.default.json"), fixture.environment_path))
        self.server.reader = self.server.jobs.reader = reader
        headers = {"Content-Type": "application/json", "Idempotency-Key": "http-selected-once",
                   "X-Console-Token": self.server.csrf_token}
        request = {"kind": "submit", "task_id": task_ids[0]}
        with patch("execution.submission_runner._default_client_factory", return_value=client):
            self.assertEqual(self.request("/api/jobs", "POST", request, headers)[0], 403)
            self.server.jobs.read_only = False
            self.assertEqual(self.request("/api/jobs", "POST", request, {**headers, "X-Console-Token": "wrong"})[0], 403)
            self.assertEqual(self.request("/api/jobs", "POST", {"kind": "submit", "task_id": "missing"}, headers),
                             (404, {"error": "formal_submission_task_not_found"}))
            self.assertEqual(self.request("/api/jobs", "POST", {"kind": "submit", "task_id": task_ids[2]}, headers),
                             (409, {"error": "formal_submission_candidate_unavailable"}))
            self.assertEqual(self.server.jobs.snapshot()["items"], [])
            self.assertEqual(client.calls, [])
            status, first = self.request("/api/jobs", "POST", request, headers)
            self.assertEqual(status, 202)
            self.server.jobs._thread.join(10)
            self.assertFalse(self.server.jobs.snapshot()["busy"])
            status, repeated = self.request("/api/jobs", "POST", request, headers)
            self.assertEqual((status, repeated), (202, first))
        result = self.request("/api/jobs")[1]["items"][0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["new_submitted_count"], 1)
        self.assertEqual(len(client.posted_ids), 1)
        remaining = self.request("/api/formulas?category=optimization")[1]
        self.assertEqual([row["task_id"] for row in remaining["items"]], [task_ids[1]])
        self.assertEqual(self.request("/api/formulas?category=optimization&source=mutation")[1]["total"], 0)
        self.assertEqual(self.request("/api/formulas?category=optimization&source=exploration&grade=EXCELLENT")[1]["total"], 1)
        history = self.request("/api/submissions")[1]["items"][0]
        self.assertEqual((history["task_id"], history["source"], history["status"]), (task_ids[0], "optimization", "submitted"))

    def test_http_archive_row_submits_only_the_selected_item_and_preserves_source(self):
        from execution.console import ConsolePaths, ConsoleReader
        from tests.execution.test_submission_runner import SubmissionQueueRunnerTests, POLICY_PATH
        fixture = SubmissionQueueRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        task_ids, client = fixture._qualified_batch()
        reader = ConsoleReader(ConsolePaths(fixture.database_path, POLICY_PATH,
                               POLICY_PATH.with_name("run.default.json"), fixture.environment_path))
        self.server.reader = self.server.jobs.reader = reader
        self.server.jobs.read_only = False
        headers = {"Content-Type": "application/json", "Idempotency-Key": "http-archive-selection",
                   "X-Console-Token": self.server.csrf_token}
        request = {"kind": "submit", "task_id": task_ids[0], "source": "qualified_archive"}
        with patch("execution.submission_runner._default_client_factory", return_value=client):
            status, first = self.request("/api/jobs", "POST", request, headers)
            self.assertEqual(status, 202)
            self.server.jobs._thread.join(10)
            self.assertEqual(self.request("/api/jobs", "POST", request, headers), (202, first))
        self.assertEqual(self.request("/api/jobs")[1]["items"][0]["status"], "completed")
        self.assertEqual(len(client.posted_ids), 1)
        self.assertEqual(self.request("/api/formulas?category=archive")[1]["total"], 1)
        history = self.request("/api/submissions")[1]["items"][0]
        self.assertEqual((history["task_id"], history["source"], history["status"]),
                         (task_ids[0], "qualified_archive", "submitted"))

    def test_paused_submission_resumes_even_after_candidate_leaves_queue(self):
        from execution.console import ConsolePaths, ConsoleReader
        from tests.execution.test_submission_runner import SubmissionQueueRunnerTests, SubmissionClient, POLICY_PATH
        fixture = SubmissionQueueRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        _, formulas = fixture._completed_run(("rank(close)",))
        reader = ConsoleReader(ConsolePaths(fixture.database_path, POLICY_PATH,
                               POLICY_PATH.with_name("run.default.json"), fixture.environment_path))
        self.server.reader = self.server.jobs.reader = reader
        self.server.jobs.read_only = False
        server = self.server
        class PausingClient(SubmissionClient):
            paused = False
            def fetch_formal_submission_check(self, **kwargs):
                response = super().fetch_formal_submission_check(**kwargs)
                if not self.paused:
                    self.paused = True
                    server.jobs.stop(server.jobs.snapshot()["items"][0]["id"])
                return response
        client = PausingClient(formulas)
        headers = {"Content-Type": "application/json", "Idempotency-Key": "http-submit-pause-identity",
                   "X-Console-Token": self.server.csrf_token}
        request = {"kind": "submit", "grade": "SPECTACULAR", "count": 1}
        with patch("execution.submission_runner._default_client_factory", return_value=client):
            self.assertEqual(self.request("/api/jobs", "POST", request, headers)[0], 202)
            self.server.jobs._thread.join(10)
            self.assertEqual(self.server.jobs.snapshot()["items"][0]["status"], "paused")
            self.assertEqual(self.request("/api/formulas?category=candidates")[1]["total"], 0)
            self.assertEqual(client.posted_ids, [])
            headers["Idempotency-Key"] = "http-submit-resume-identity"
            self.assertEqual(self.request("/api/jobs", "POST", request, headers)[0], 202)
            self.server.jobs._thread.join(10)
        self.assertEqual(self.server.jobs.snapshot()["items"][0]["status"], "completed")
        self.assertEqual(len(client.posted_ids), 1)
