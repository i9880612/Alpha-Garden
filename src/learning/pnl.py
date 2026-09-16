from __future__ import annotations

import math
import statistics


class PnlCorrelations:
    """Prepare each curve once and reuse pair results within one decision only."""

    def __init__(self, series):
        self._points = {(row.account_scope, row.platform_alpha_id): row.points
                        for row in series if row.points is not None}
        self._prepared = {}
        self._intervals = {}
        self._results = {}

    def _prepare(self, key):
        if key not in self._prepared:
            increments = _increments(self._points[key])
            intervals = tuple(sorted(increments))
            # Equal calendars share an identity, avoiding repeated alignment work.
            intervals = self._intervals.setdefault(intervals, intervals)
            self._prepared[key] = (intervals, tuple(increments[k] for k in intervals), increments)
        return self._prepared[key]

    def correlation(self, first, second, *, minimum_intervals):
        if first not in self._points or second not in self._points:
            return None
        pair = (*sorted((first, second)), minimum_intervals)
        if pair not in self._results:
            a, b = self._prepare(first), self._prepare(second)
            if a[0] is b[0]:
                x, y = a[1], b[1]
            else:
                intervals = sorted(a[2].keys() & b[2].keys())
                x, y = tuple(a[2][k] for k in intervals), tuple(b[2][k] for k in intervals)
            self._results[pair] = _correlation(x, y, minimum_intervals)
        return self._results[pair]


def daily_pnl_correlation(first, second, *, minimum_intervals: int) -> float | None:
    """Compare increments covering identical intervals in the overlapping history.

    Both endpoints must match. A missing date cannot turn a two-session return
    into a one-session return, and cumulative levels are never correlated.
    """
    increments = [_increments(points) for points in (first, second)]
    intervals = sorted(increments[0].keys() & increments[1].keys())
    return _correlation(*(tuple(values[key] for key in intervals) for values in increments), minimum_intervals)


def _increments(points):
    return {(a[0], b[0]): b[1] - a[1] for a, b in zip(points, points[1:])}


def _correlation(first, second, minimum_intervals):
    if len(first) < minimum_intervals:
        return None
    try:
        value = statistics.correlation(first, second)
    except statistics.StatisticsError:
        return None
    return value if math.isfinite(value) else None
