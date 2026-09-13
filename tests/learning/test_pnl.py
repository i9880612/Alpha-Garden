import math
import statistics

from learning.pnl import daily_pnl_correlation
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
