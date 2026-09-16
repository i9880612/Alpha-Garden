"""Loopback-only HTTP adapter for the existing research commands."""
from __future__ import annotations

import json
import mimetypes
import re
import secrets
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from execution.console import ConsolePaths, ConsoleReader
from execution.console_jobs import ConsoleJobs
from execution.console_series import ConsoleSeries
from execution.quality_diagnosis import QualityDiagnosis
from alpha_garden.web_events import ConsoleEvents
from alpha_garden.web_auth import ConsoleAuth
from persistence.console import read_console_database


GRADES = {"SPECTACULAR", "EXCELLENT", "GOOD", "AVERAGE", "INFERIOR"}


class ConsoleServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, reader, *, read_only=False, assets=None):
        self.reader = reader
        self.jobs = ConsoleJobs(reader, read_only=read_only)
        self.series = ConsoleSeries(reader, read_only=read_only)
        self.analysis = QualityDiagnosis(reader)
        self.csrf_token = secrets.token_urlsafe(32)
        self.auth = ConsoleAuth()
        self.assets = Path(assets).resolve() if assets else None
        self.events = None
        super().__init__(("127.0.0.1", port), ConsoleHandler)
        self.allowed_hosts = {f"127.0.0.1:{self.server_port}", f"localhost:{self.server_port}"}
        self.allowed_origins = {f"http://{host}" for host in self.allowed_hosts} | {
            "http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:4173", "http://localhost:4173"}
        self.events = ConsoleEvents(reader.paths, self.jobs)

    def server_close(self):
        if self.events is not None:
            self.events.close()
        super().server_close()


class ConsoleHandler(BaseHTTPRequestHandler):
    server: ConsoleServer
    timeout = 10

    def log_message(self, *args):
        pass

    def do_GET(self):
        self._handle(False)

    def do_POST(self):
        self._handle(True)

    def _handle(self, write):
        if self.headers.get("Host") not in self.server.allowed_hosts:
            return self._json(403, {"error": "console_host_rejected"})
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.allowed_origins:
            return self._json(403, {"error": "console_origin_rejected"})
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            return self._json(403, {"error": "console_origin_rejected"})
        try:
            parsed = urlsplit(self.path)
            if not parsed.path.startswith("/api/"):
                return self._static(parsed.path) if not write else self._json(404, {"error": "not_found"})
            authenticated = self.server.auth.authenticated(self.headers.get("Cookie"))
            if not write and parsed.path == "/api/auth":
                return self._json(200, {"authenticated": authenticated, "username": "admin" if authenticated else None})
            logging_in = write and parsed.path == "/api/login"
            if not logging_in and not authenticated:
                return self._json(401, {"error": "console_login_required"})
            if write and not logging_in and not secrets.compare_digest(self.headers.get("X-Console-Token", ""), self.server.csrf_token):
                return self._json(403, {"error": "console_token_required"})
            if not write and parsed.path == "/api/events":
                return self._events(parse_qs(parsed.query, keep_blank_values=True))
            if write:
                if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
                    return self._json(400, {"error": "console_json_required"})
                size = _positive_int(self.headers.get("Content-Length", "0"), maximum=16384)
                try:
                    payload = json.loads(self.rfile.read(size))
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError("console_body_invalid") from exc
                if not isinstance(payload, dict):
                    raise ValueError("console_body_invalid")
                if logging_in:
                    if set(payload) != {"username", "password"}:
                        raise ValueError("console_body_invalid")
                    token = self.server.auth.login(payload["username"], payload["password"])
                    if token is None:
                        return self._json(401, {"error": "console_credentials_invalid"})
                    self.server.auth.logout(self.headers.get("Cookie"))
                    return self._json(200, {"authenticated": True, "username": "admin"},
                                      {"Set-Cookie": self.server.auth.cookie(token)})
                if parsed.path == "/api/logout" and not payload:
                    self.server.auth.logout(self.headers.get("Cookie"))
                    return self._json(200, {"authenticated": False}, {"Set-Cookie": self.server.auth.cookie()})
                if parsed.path in {"/api/formula/series/pnl", "/api/formula/series/sharpe", "/api/formula/series/turnover"}:
                    if set(payload) != {"task_id"}:
                        raise ValueError("console_body_invalid")
                    return self._json(200, self.server.series.fetch(_identity(payload["task_id"]), parsed.path.rsplit("/", 1)[1]))
                if parsed.path == "/api/settings":
                    return self._json(200, self.server.jobs.save_settings(payload))
                if parsed.path == "/api/jobs":
                    request = _job_request(payload)
                    request_id = self.headers.get("Idempotency-Key", "")
                    if not re.fullmatch(r"[A-Za-z0-9_-]{16,80}", request_id):
                        raise ValueError("console_request_id_invalid")
                    job_id = self.server.jobs.start(request, request_id)
                    return self._json(202, {"job_id": job_id})
                if parsed.path == "/api/jobs/stop" and set(payload) == {"job_id"}:
                    self.server.jobs.stop(_identity(payload["job_id"]))
                    return self._json(202, {"status": "stopping"})
                return self._json(404, {"error": "not_found"})
            data = self._get(parsed.path, parse_qs(parsed.query, keep_blank_values=True))
            return self._json(200, data)
        except Exception as exc:
            return self._json(*_error_response(exc))

    def _get(self, path, query):
        reader = self.server.reader
        if path == "/api/session":
            try:
                account = reader.account_scope
                reader.settings()
                with read_console_database(reader.paths.database) as connection:
                    connection.execute("SELECT 1 FROM backtest_tasks LIMIT 1").fetchone()
                ready, error = True, None
            except (OSError, ValueError, sqlite3.Error):
                account, ready, error = None, False, "console_data_unavailable"
            return {"csrf_token": self.server.csrf_token, "read_only": self.server.jobs.read_only,
                    "account_scope": account, "ready": ready, "error": error,
                    "platform_session": "unchecked"}
        if path == "/api/settings":
            return reader.settings()
        if path in {"/api/analysis/quality", "/api/analysis/correlation"}:
            filters = {key: _query(query, key, default) for key, default in
                       (("days", "30"), ("mode", "all"), ("run_id", ""), ("source", ""), ("configuration", ""))}
            read = self.server.analysis.quality if path.endswith("quality") else self.server.analysis.correlation
            return self.server.events.read_current(path + json.dumps(filters, sort_keys=True), "database",
                                                  lambda: read(**filters), cache=True, cache_group=path)
        dashboard_reads = {
            "/api/dashboard/activity": reader.dashboard_activity,
            "/api/dashboard/submitted-grades": reader.dashboard_grades,
            "/api/dashboard/research": reader.dashboard_research,
            "/api/dashboard/progress": reader.dashboard_progress,
            "/api/dashboard/recent": reader.dashboard_recent,
        }
        if path in dashboard_reads:
            if path in {"/api/dashboard/activity", "/api/dashboard/research"}:
                return self.server.events.read_current(path, "database", dashboard_reads[path], cache=True)
            return dashboard_reads[path]()
        if path == "/api/runs":
            return reader.runs(page=_positive_int(_query(query, "page", "1"), maximum=100000),
                               page_size=_positive_int(_query(query, "page_size", "25"), maximum=100))
        if path == "/api/jobs":
            return self.server.jobs.snapshot()
        if path == "/api/formula":
            return reader.formula_detail(_identity(_query(query, "task_id")))
        if path == "/api/formula/children":
            return reader.formula_children(_identity(_query(query, "task_id")),
                page=_positive_int(_query(query, "page", "1"), maximum=100000),
                page_size=_positive_int(_query(query, "page_size", "10"), maximum=100))
        if path in {"/api/catalog/fields", "/api/catalog/operators", "/api/seeds"}:
            paging = {"page": _positive_int(_query(query, "page", "1"), maximum=100000),
                      "page_size": _positive_int(_query(query, "page_size", "10"), maximum=100)}
            search = _query(query, "search")
            if len(search) > 128:
                raise ValueError("console_search_invalid")
            if path.startswith("/api/catalog/"):
                category, dataset = _query(query, "category"), _query(query, "dataset")
                if len(category) > 128 or len(dataset) > 128:
                    raise ValueError("console_catalog_filter_invalid")
                return reader.catalog(kind=path.rsplit("/", 1)[1], search=search,
                                      category=category, dataset=dataset, **paging)
            if path == "/api/seeds":
                return reader.seeds(mode=_query(query, "mode", "roots"), search=search, **paging)
        if path in {"/api/formulas", "/api/submissions"}:
            page = _positive_int(_query(query, "page", "1"), maximum=100000)
            page_size = _positive_int(_query(query, "page_size", "25"), maximum=100)
            if path == "/api/submissions":
                return reader.submissions(page=page, page_size=page_size)
            grade = _query(query, "grade").upper()
            if grade and grade not in GRADES | {"UNKNOWN"}:
                raise ValueError("console_grade_invalid")
            search = _query(query, "search")
            if len(search) > 128:
                raise ValueError("console_search_invalid")
            return reader.formulas(category=_query(query, "category", "all"), search=search, grade=grade,
                run_id=_query(query, "run_id"), page=page, page_size=page_size, execution=_query(query, "execution", "all"), source=_query(query, "source"))
        raise ValueError("console_endpoint_not_found")

    def _events(self, query):
        if query:
            raise ValueError("console_subscription_invalid")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.close_connection = True
        stream = self.server.events.stream()
        try:
            self.wfile.write(b"retry: 3000\n\n")
            self.wfile.flush()
            for payload in stream:
                if not self.server.auth.authenticated(self.headers.get("Cookie")):
                    self.wfile.write(b"event: unauthorized\ndata: {}\n\n")
                    self.wfile.flush()
                    break
                if payload is not None:
                    self.wfile.write(f"event: versions\ndata: {json.dumps(payload)}\n\n".encode("utf-8"))
                else:
                    self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
        except OSError:
            # A closed/slow browser must not hold a writer or an event backlog.
            pass
        finally:
            stream.close()

    def _static(self, path):
        root = self.server.assets
        if root is None:
            return self._json(404, {"error": "console_frontend_not_built"})
        target = (root / unquote(path).lstrip("/")).resolve()
        if not target.is_relative_to(root):
            return self._json(404, {"error": "not_found"})
        if not target.is_file() and path in {"/", "/login", "/runs", "/formulas", "/submissions", "/settings", "/data", "/operators", "/analysis", "/seeds", "/optimization", "/archive", "/submitted"}:
            target = root / "index.html"
        if not target.is_file():
            return self._json(404, {"error": "not_found"})
        return self._send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")

    def _json(self, status, value, headers=None):
        self._send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8", headers)

    def _send(self, status, data, content_type, headers=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)
        except ConnectionError:
            pass


def _error_response(exc):
    if isinstance(exc, ValueError) and str(exc) in {
        "formal_submission_task_not_found", "formal_submission_candidate_unavailable",
        "formal_submission_already_attempted", "formal_submission_other_attempt_active",
        "formal_submission_storage_update_required",
    }:
        code = str(exc)
        return (404 if code.endswith("not_found") else 409), {"error": code}
    if isinstance(exc, ValueError) and str(exc).startswith("console_"):
        code = str(exc)
        status = (409 if code in {"console_operation_busy", "console_request_id_conflict", "console_run_not_resumable", "console_series_busy", "console_series_cooldown", "console_settings_conflict", "console_settings_busy"}
                  else 403 if code == "console_read_only" else 404 if code.endswith("not_found") else 400)
        return status, {"error": code}
    if isinstance(exc, (OSError, sqlite3.Error, ValueError)):
        return 503, {"error": "console_data_unavailable"}
    return 500, {"error": "console_internal_error"}


def _query(query, key, default=""):
    values = query.get(key, [default])
    if len(values) != 1:
        raise ValueError("console_query_invalid")
    return values[0]


def _positive_int(value, *, maximum=1000):
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError("console_number_invalid")
    return value


def _identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("console_identity_invalid")
    return value


def _job_request(payload):
    kind = payload.get("kind")
    if kind == "submit" and set(payload) == {"kind", "task_id", "source"} and payload["source"] == "qualified_archive":
        return {"kind": kind, "task_id": _identity(payload["task_id"]), "source": "qualified_archive"}
    if kind == "submit" and set(payload) == {"kind", "task_id"}:
        return {"kind": kind, "task_id": _identity(payload["task_id"])}
    if kind == "run" and set(payload) == {"kind", "cycles", "mode"} and isinstance(payload["mode"], str) and payload["mode"] in {"normal", "optimization"}:
        return {"kind": kind, "cycles": _positive_int(payload["cycles"]), "mode": payload["mode"]}
    if kind == "submit" and set(payload) == {"kind", "grade", "count"} and isinstance(payload["grade"], str) and payload["grade"] in GRADES:
        return {"kind": kind, "grade": payload["grade"], "count": _positive_int(payload["count"])}
    if kind == "resume" and set(payload) == {"kind", "run_id"}:
        return {"kind": kind, "run_id": _identity(payload["run_id"])}
    raise ValueError("console_operation_invalid")


def serve_console(paths: ConsolePaths, *, port=8787, read_only=False, assets=None):
    server = ConsoleServer(port, ConsoleReader(paths), read_only=read_only, assets=assets)
    print(f"Alpha Garden: http://127.0.0.1:{server.server_port} ({'只读' if read_only else '可运行与提交'})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("正在结束网页服务；等待当前请求完成并保留已有进度。", flush=True)
    finally:
        server.jobs.close()
        server.server_close()
