import math
from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from learning.seed_correlation import assess_seed_correlation
from persistence.pnl import PnlSeriesRecord
from persistence.submissions import PlatformSubmittedAlphaRecord


def series(alpha, increments, account="account"):
    value, points = 0.0, []
    for i, increment in enumerate([0.0, *increments]):
        value += increment
        points.append(((date(2020, 1, 1) + timedelta(days=i)).isoformat(), value))
    return PnlSeriesRecord(account, alpha, "2026-09-09T00:00:00+00:00", tuple(points))


def reference(alpha, account="account"):
    return PlatformSubmittedAlphaRecord(account, alpha, f"rank({alpha})", "ACTIVE",
        "2026-09-01T00:00:00+00:00", False, {}, "2026-09-09T00:00:00+00:00")


def candidate(alpha="child", account="account"):
    return SimpleNamespace(task=SimpleNamespace(platform_alpha_id=alpha,
                           account_scope=account, formula=f"rank({alpha})"))


def test_all_references_required_for_pass_but_one_high_pair_proves_failure():
    x = [math.sin(i) for i in range(300)]
    y = [math.cos(i) for i in range(300)]
    refs = (reference("one"), reference("two"))
    partial = (series("child", x), series("one", y))
    result = assess_seed_correlation(candidate(), refs, partial)
    assert (result.state, result.compared, result.required) == ("pending", 1, 2)
    result = assess_seed_correlation(candidate(), refs, (*partial, series("two", x)))
    assert result.state == "failed" and result.maximum > 0.999
    assert result.reference_id == "two"
    assert assess_seed_correlation(candidate(), refs, (*partial, series("two", y))).state == "passed"
    assert assess_seed_correlation(candidate(), refs, (series("child", x), series("two", x))).state == "failed"


def test_account_isolation_submitted_self_and_negative_correlation():
    x = [math.sin(i) for i in range(300)]
    refs = (reference("one"), reference("child", "other"))
    records = (series("child", x), series("one", [-v for v in x]))
    assert assess_seed_correlation(candidate(), refs, records).state == "passed"
    assert assess_seed_correlation(candidate(), (reference("one"),), tuple(replace(s, account_scope="other") for s in records)).state == "pending"
    assert assess_seed_correlation(candidate("one"), refs, ()).state == "submitted"


@pytest.mark.parametrize("kind", ["missing", "flat", "short", "dates"])
def test_unusable_pnl_is_not_a_pass(kind):
    x = [math.sin(i) for i in range(300)]
    child, ref = series("child", x), series("one", x)
    if kind == "missing":
        child = replace(child, points=None)
    elif kind == "flat":
        child = series("child", [0.0] * 300)
    elif kind == "short":
        child, ref = replace(child, points=child.points[:20]), replace(ref, points=ref.points[:20])
    else:
        child = replace(child, points=tuple(("2021" + day[4:], v) for day, v in child.points))
    assert assess_seed_correlation(candidate(), (reference("one"),), (child, ref)).state == "pending"


def test_threshold_equality_is_rejected(monkeypatch):
    monkeypatch.setattr("learning.seed_correlation.daily_pnl_correlation", lambda *a, **kw: 0.7)
    x = [math.sin(i) for i in range(300)]
    assert assess_seed_correlation(candidate(), (reference("one"),),
        (series("child", x), series("one", x))).state == "failed"
