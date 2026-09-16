import math
import statistics
from dataclasses import replace

import pytest

from learning.pnl import PnlCorrelations, daily_pnl_correlation
from tests.learning.test_seed_correlation import series


def test_appended_dates_and_changed_start_do_not_make_cached_history_unusable():
    x = [math.sin(i) for i in range(400)]
    first = series("first", x).points
    second = series("second", x[:350]).points[20:]
    assert daily_pnl_correlation(first, second, minimum_intervals=252) > 0.999


def test_gap_does_not_mix_a_multi_interval_increment_with_a_daily_increment():
    x, y = [math.sin(i) for i in range(300)], [math.cos(i) for i in range(300)]
    first, second = series("first", x).points, series("second", y).points
    second = second[:100] + second[101:]
    observed = daily_pnl_correlation(first, second, minimum_intervals=252)
    # Missing cumulative point 100 removes intervals ending at 100 and 101.
    keep = [i for i in range(300) if i not in (99, 100)]
    expected = statistics.correlation([x[i] for i in keep], [y[i] for i in keep])
    assert abs(observed - expected) < 1e-12
    assert daily_pnl_correlation(first, second[100:], minimum_intervals=252) is None


@pytest.mark.parametrize("kind", ["aligned", "gap", "overlap", "negative", "flat", "short"])
def test_batch_correlations_preserve_daily_interval_results(kind):
    x, y = [math.sin(i) for i in range(400)], [math.cos(i) for i in range(400)]
    first, second = series("first", x), series("second", y)
    if kind == "gap":
        second = replace(second, points=second.points[:100] + second.points[101:])
    elif kind == "overlap":
        second = replace(second, points=second.points[20:350])
    elif kind == "negative":
        second = series("second", [-v for v in x])
    elif kind == "flat":
        second = series("second", [0.0] * 400)
    elif kind == "short":
        second = replace(second, points=second.points[:100])
    batch = PnlCorrelations((first, second))
    a, b = ("account", "first"), ("account", "second")
    for minimum in (252, 50, 500):
        expected = daily_pnl_correlation(first.points, second.points, minimum_intervals=minimum)
        for left, right in ((a, b), (b, a), (a, b)):
            observed = batch.correlation(left, right, minimum_intervals=minimum)
            if expected is None:
                assert observed is None
            else:
                assert observed == pytest.approx(expected, abs=1e-12)


def test_batch_correlations_isolate_accounts_and_decisions():
    x = [math.sin(i) for i in range(300)]
    records = (series("first", x), series("second", x),
               series("first", x, "other"), series("second", [-v for v in x], "other"))
    batch = PnlCorrelations(records)
    assert batch.correlation(("account", "first"), ("account", "second"), minimum_intervals=252) > 0.999
    assert batch.correlation(("other", "first"), ("other", "second"), minimum_intervals=252) < -0.999
    assert batch.correlation(("missing", "first"), ("account", "second"), minimum_intervals=252) is None
    fresh = PnlCorrelations((records[0], series("second", [-v for v in x])))
    assert fresh.correlation(("account", "first"), ("account", "second"), minimum_intervals=252) < -0.999
