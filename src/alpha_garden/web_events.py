"""Observe local changes and stream bounded, independent read results."""
from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import Future
from datetime import date


class ConsoleEvents:
    def __init__(self, paths, jobs):
        self.paths, self.jobs = paths, jobs
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._versions = {"database": 0, "jobs": 0, "session": 0}
        self._ready = False
        self._inflight = {}
        self._cached = {}
        self._connection = None
        self._identity = None
        self._previous = None
        self._thread = threading.Thread(target=self._watch, name="console-events", daemon=True)
        self._thread.start()

    def wait(self, previous=None, *, timeout=10):
        with self._condition:
            if previous is None:
                self._condition.wait_for(lambda: self._stop.is_set() or self._ready)
            else:
                self._condition.wait_for(lambda: self._stop.is_set() or self._versions != previous, timeout)
            if self._stop.is_set():
                return None
            return dict(self._versions)

    def read_current(self, path, topic, read, *, cache=False, cache_group=None):
        """Share a current read; optionally retain display data until facts change."""
        with self._condition:
            if self._stop.is_set():
                raise RuntimeError("console_events_closed")
            # Check synchronously, so a cache hit cannot miss a commit between
            # the observer's one-second notifications.
            self._observe()
            key = (path, topic, self._versions[topic])
            cached = self._cached.get(path)
            if cache and cached is not None and cached[0] == key:
                return cached[1]
            future = self._inflight.get(key)
            owner = future is None
            if owner:
                future = self._inflight[key] = Future()
        if owner:
            try:
                result = read()
                with self._condition:
                    if cache and not self._stop.is_set():
                        self._observe()
                        if key[2] == self._versions[topic]:
                            if cache_group:
                                self._cached = {p: v for p, v in self._cached.items() if not p.startswith(cache_group)}
                            self._cached[path] = (key, result)
                future.set_result(result)
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                with self._condition:
                    del self._inflight[key]
        return future.result()

    def stream(self):
        """Notify revisions only; opening a stream never executes a business read."""
        versions = self.wait()
        sent = None
        last_sent = time.monotonic()
        while versions is not None:
            if versions != sent:
                yield versions
                sent = versions
                last_sent = time.monotonic()
            elif time.monotonic() - last_sent >= 10:
                yield None
                last_sent = time.monotonic()
            versions = self.wait(versions, timeout=max(0, 10 - (time.monotonic() - last_sent)))

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join()
        with self._condition:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            self._cached.clear()

    def _watch(self):
        while not self._stop.is_set():
            with self._condition:
                self._observe()
            self._stop.wait(1)

    def _observe(self):
        """Called under the condition lock by the observer and HTTP readers."""
        try:
            stat = self.paths.database.stat()
            identity = (stat.st_dev, stat.st_ino)
            if self._connection is None or self._identity != identity:
                if self._connection is not None:
                    self._connection.close()
                self._connection = sqlite3.connect(
                    self.paths.database.resolve().as_uri() + "?mode=ro", uri=True,
                    timeout=1, isolation_level=None, check_same_thread=False)
                self._connection.execute("PRAGMA query_only=ON")
                self._identity = identity
            database = (identity, self._connection.execute("PRAGMA data_version").fetchone()[0])
        except (OSError, sqlite3.Error):
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            database = None
        configuration = tuple(_file_stamp(path) for path in
                              (self.paths.environment, self.paths.settings, self.paths.run_config))
        current = (database, configuration, self.jobs.version(), date.today())
        previous = self._previous
        changed = set()
        if previous is not None:
            if current[0] != previous[0] or current[3] != previous[3]:
                changed.add("database")
            if (current[0] is None) != (previous[0] is None):
                changed.add("session")
            if current[1] != previous[1]:
                changed.update(("database", "session"))
            if current[2] != previous[2]:
                changed.add("jobs")
        if changed:
            self._cached = {path: value for path, value in self._cached.items()
                            if value[0][1] not in changed}
            for topic in changed:
                self._versions[topic] += 1
        self._previous = current
        self._ready = True
        if changed or previous is None:
            self._condition.notify_all()


def _file_stamp(path):
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None
