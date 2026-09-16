"""Explicit, bounded platform chart reads; never writes research facts."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime

from persistence.console import read_console_database
from worldquant.backtests import WorldQuantProtocolError
from worldquant.client import WorldQuantClient, WorldQuantRequestError
from worldquant.config import load_worldquant_connection_settings
from worldquant.recordsets import RECORDSET_UNITS


class ConsoleSeries:
    def __init__(self, reader, *, read_only=False):
        self.reader, self.read_only = reader, read_only
        self._lock = threading.Lock()
        self._settings = self._client = None
        self._cache = OrderedDict()
        self._account_retry_at = 0

    def fetch(self, task_id, metric):
        if self.read_only:
            raise ValueError("console_read_only")
        if metric not in RECORDSET_UNITS:
            raise ValueError("console_series_metric_invalid")
        if not self._lock.acquire(timeout=30):
            raise ValueError("console_series_busy")
        try:
            settings = load_worldquant_connection_settings(self.reader.paths.environment)
            with read_console_database(self.reader.paths.database) as connection:
                task = connection.execute("SELECT platform_alpha_id FROM backtest_tasks WHERE task_id=? AND account_scope=?",
                                          (task_id, settings.account_scope)).fetchone()
            if task is None:
                raise ValueError("console_formula_not_found")
            if not task[0]:
                raise ValueError("console_formula_alpha_missing")
            if self._settings != settings:
                self._settings = settings
                self._client = WorldQuantClient(base_url=settings.base_url, credentials=settings.credentials, timeout_seconds=8)
                self._cache.clear()
                self._account_retry_at = 0
            alpha = task[0]
            key = (alpha, metric)
            now = time.monotonic()
            cached = self._cache.get(key)
            if cached and cached[0] > now:
                return {**cached[1], "refresh_after_seconds": cached[0] - now}
            if self._account_retry_at > now:
                raise ValueError("console_series_cooldown")
            try:
                if not self._client.authenticated:
                    self._client.authenticate()
                result = self._client.fetch_recordset(platform_alpha_id=alpha, metric=metric)
                item = {"state": "pending" if result.points is None else "ready", "points": result.points,
                        "retry_after_seconds": result.retry_after_seconds, "error": None, "http_status": None}
            except (WorldQuantRequestError, WorldQuantProtocolError) as exc:
                item = _failure(exc)
            cooldown = max(1 if item["state"] == "pending" else 30, item.get("retry_after_seconds") or 0)
            if item.get("http_status") in {401, 403, 429}:
                self._account_retry_at = time.monotonic() + cooldown
            value = {"alpha_id": alpha, "metric": metric, **item, "unit": RECORDSET_UNITS[metric],
                     "observed_at": datetime.now(UTC).isoformat(), "refresh_after_seconds": cooldown}
            self._cache[key] = (time.monotonic() + cooldown, value)
            self._cache.move_to_end(key)
            while len(self._cache) > 96:
                self._cache.popitem(last=False)
            return value
        finally:
            self._lock.release()


def _failure(exc):
    return {"state": "error", "points": None, "error": exc.code,
            "http_status": getattr(exc, "status_code", None),
            "retry_after_seconds": getattr(exc, "retry_after_seconds", None)}
