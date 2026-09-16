"""One console-owned operation at a time, with bounded, non-authoritative logs."""
from __future__ import annotations

import logging
import re
import threading
import traceback
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

from execution.launch import launch_automated_run, resume_automated_run
from execution.progress import LOGGER_NAME
from execution.run_config import load_automated_run_limits
from execution.console_settings import save_settings
from execution.runs import AutomatedRunPaused
from execution.submission_runner import submit_queued_alphas


class ConsoleJobs:
    def __init__(self, reader, *, read_only=False):
        self.reader, self.read_only = reader, read_only
        self._lock = threading.RLock()
        self._jobs = deque(maxlen=20)
        self._thread = None
        self._stop = threading.Event()
        self._revision = 0

    def version(self):
        """A small change signal; reading it does not copy the retained logs."""
        with self._lock:
            return self._revision, bool(self._thread and self._thread.is_alive())

    def snapshot(self):
        with self._lock:
            return {"items": [{**job, "logs": list(job["logs"])} for job in reversed(self._jobs)],
                    "busy": bool(self._thread and self._thread.is_alive())}

    def start(self, request, request_id):
        if self.read_only:
            raise ValueError("console_read_only")
        with self._lock:
            existing = next((job for job in self._jobs if job["request_id"] == request_id), None)
            if existing:
                if existing["request"] != request:
                    raise ValueError("console_request_id_conflict")
                return existing["id"]
            if self._thread and self._thread.is_alive():
                raise ValueError("console_operation_busy")
            # Validate before creating a worker or touching a running research plan.
            if request["kind"] == "run":
                self.reader.settings()
            account = self.reader.account_scope
            self.reader.runs()
            if request["kind"] == "submit" and "task_id" in request:
                self.reader.require_selected_submission(request["task_id"], request.get("source", "optimization"))
            if request["kind"] == "resume":
                run = self.reader.require_run(request["run_id"])
                if not run["can_resume"]:
                    raise ValueError("console_run_not_resumable")
            job = {"id": str(uuid4()), "request_id": request_id, "request": dict(request),
                   "account_scope": account, "status": "running", "created_at": _now(),
                   "finished_at": None, "run_id": request.get("run_id"), "result": None,
                   "error": None, "progress": None, "logs": deque(maxlen=500)}
            self._jobs.append(job)
            self._revision += 1
            self._stop.clear()
            self._thread = threading.Thread(target=self._work, args=(job,), name="console-operation", daemon=False)
            self._thread.start()
            return job["id"]

    def save_settings(self, request):
        if self.read_only:
            raise ValueError("console_read_only")
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise ValueError("console_settings_busy")
            return save_settings(self.reader.paths, request)

    def stop(self, job_id):
        if self.read_only:
            raise ValueError("console_read_only")
        with self._lock:
            job = self._jobs[-1] if self._jobs else None
            if not job or job["id"] != job_id:
                raise ValueError("console_job_not_found")
            if job["status"] == "running":
                self._stop.set()
                job["status"] = "stopping"
                self._revision += 1

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _clock(self):
        if self._stop.is_set():
            raise AutomatedRunPaused()
        return datetime.now(UTC)

    def _wait(self, seconds):
        if self._stop.wait(seconds):
            raise AutomatedRunPaused()

    def _work(self, job):
        owner = threading.get_ident()
        jobs = self
        class ProgressHandler(logging.Handler):
            def emit(self, record):
                if record.thread == owner:
                    with jobs._lock:
                        message = record.getMessage()[:2000]
                        if getattr(record, "transient", False):
                            if job["progress"] and job["progress"]["message"] == message:
                                return
                            job["progress"] = {"at": _now(), "message": message}
                        else:
                            job["progress"] = None
                            job["logs"].append({"at": _now(), "message": message})
                        jobs._revision += 1
        handler = ProgressHandler()
        logger = logging.getLogger(LOGGER_NAME)
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            result = self._execute(job)
            with self._lock:
                job["result"] = result
                job["status"] = "completed" if result["completed"] else "needs_attention"
                self._revision += 1
        except AutomatedRunPaused:
            with self._lock:
                job["status"] = "paused"
                self._revision += 1
        except Exception as exc:
            # Do not return arbitrary exception text (paths, payloads or credentials).
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            logging.getLogger(__name__).error("Console operation failed: %s at %s:%s",
                                             type(exc).__name__, frame.filename, frame.lineno)
            with self._lock:
                job["status"] = "failed"
                message = str(exc)
                job["error"] = (
                    "console_database_busy" if message == "已有命令正在使用此数据库，请先停止该进程。"
                    else message if re.fullmatch(r"(automated_run|worldquant|formal_submission|console)_[a-z_]+", message)
                    else type(exc).__name__
                )
                job["logs"].append({"at": _now(), "message": "操作未完成；请检查运行状态、账号配置及是否已有命令占用数据库。"})
                self._revision += 1
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            with self._lock:
                job["progress"] = None
                job["finished_at"] = _now()
                self._revision += 1

    def _execute(self, job):
        paths, request = self.reader.paths, job["request"]
        common = {"clock": self._clock, "waiter": self._wait}
        if request["kind"] == "submit":
            grade = request.get("grade")
            selection = ({"source": request.get("source", "optimization"), "selected_task_id": request["task_id"], "max_submissions": 1}
                         if "task_id" in request else
                         {"source": "queue" if grade == "SPECTACULAR" else "qualified_archive",
                          "grade": None if grade == "SPECTACULAR" else grade, "max_submissions": request["count"]})
            completion = submit_queued_alphas(paths.database, paths.environment,
                                             **selection, **common)
            return {**asdict(completion), "new_submitted_count": completion.submitted_count - completion.already_active_count,
                    "completed": completion.completed}
        if request["kind"] == "resume":
            completion = resume_automated_run(paths.database, paths.environment, request["run_id"], **common)
        else:
            limits = load_automated_run_limits(paths.run_config, cycles=request["cycles"],
                optimization_only=request["mode"] == "optimization", automatic_submissions_enabled=False)
            def created(run):
                with self._lock:
                    job["run_id"] = run.run_id
                    self._revision += 1
            completion = launch_automated_run(paths.database, paths.settings, paths.environment,
                                             limits=limits, run_created=created, **common)
        return {"run_id": completion.run.run_id, "status": completion.run.status,
                "stop_reason": completion.run.stop_reason, "completed": completion.run.status == "completed"}


def _now():
    return datetime.now(UTC).isoformat()
