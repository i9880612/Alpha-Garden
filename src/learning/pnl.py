from __future__ import annotations

import math
import statistics


def daily_pnl_correlation(first, second, *, minimum_intervals: int) -> float | None:
    """Compare increments covering identical intervals in the overlapping history.

    Both endpoints must match. A missing date cannot turn a two-session return
    into a one-session return, and cumulative levels are never correlated.
    """
    increments = [dict(((a[0], b[0]), b[1] - a[1]) for a, b in zip(points, points[1:]))
                  for points in (first, second)]
    intervals = sorted(increments[0].keys() & increments[1].keys())
    if len(intervals) < minimum_intervals:
        return None
    try:
        value = statistics.correlation(*(tuple(values[key] for key in intervals) for values in increments))
    except statistics.StatisticsError:
        return None
    return value if math.isfinite(value) else None
