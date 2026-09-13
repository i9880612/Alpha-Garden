import math
from dataclasses import replace
from datetime import date, timedelta

from learning.recovery import (
    RecoveryComparison,
    recovery_correlations,
    recovery_task_ids,
)
from persistence.pnl import PnlSeriesRecord


def series(alpha, increments, account="account"):
    values = [0.0]
    for value in increments:
        values.append(values[-1] + value)
    return PnlSeriesRecord(
        account,
        alpha,
        "2026-09-07T00:00:00+00:00",
        tuple(
            ((date(2020, 1, 1) + timedelta(days=i)).isoformat(), value)
            for i, value in enumerate(values)
        ),
    )


def test_real_series_shape_is_compared_as_increments_not_cumulative_levels():
    reference = [math.sin(i) for i in range(300)]
    child = [math.cos(i) for i in range(300)]
    comparison = RecoveryComparison(
        "task", "repair", "parent-task", "account", "child", "parent", "reference"
    )
    records = (
        series("parent", reference),
        series("reference", reference),
        series("child", child),
    )
    parent_corr, child_corr = recovery_correlations(comparison, records)
    assert parent_corr > 0.999
    assert abs(child_corr) < 0.01
    assert recovery_task_ids((comparison,), records) == frozenset({"task"})
    assert not recovery_task_ids((replace(comparison, account_scope="other"),), records)
    assert not recovery_task_ids(
        (replace(comparison, child_alpha_id="parent"),), records
    )
    assert recovery_correlations(comparison, records[:-1]) is None
    assert (
        recovery_correlations(
            comparison,
            (*records[:-1], replace(records[-1], points=records[-1].points[1:])),
        )
        is None
    )
    assert (
        recovery_correlations(
            comparison, tuple(replace(r, points=r.points[:20]) for r in records)
        )
        is None
    )
    assert (
        recovery_correlations(comparison, (*records[:-1], series("child", [0.0] * 300)))
        is None
    )


def test_submitted_parent_can_be_its_own_recovery_reference_without_duplicate_series():
    comparison = RecoveryComparison("task", "repair", "parent-task", "account", "child", "parent", "parent")
    records = (series("parent", [math.sin(i) for i in range(300)]),
               series("child", [math.cos(i) for i in range(300)]))
    parent_corr, child_corr = recovery_correlations(comparison, records)
    assert parent_corr > .999 and abs(child_corr) < .01
    assert recovery_task_ids((comparison,), records) == frozenset({"task"})
    assert recovery_correlations(comparison, records[:1]) is None
