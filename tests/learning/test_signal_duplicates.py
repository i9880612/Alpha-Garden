import math
from dataclasses import replace

import pytest

from learning.signal_duplicates import find_signal_duplicates
from tests.learning.test_qualified_evolution import checked
from tests.learning.test_seed_correlation import series


@pytest.mark.parametrize("other", [None, [0.0] * 300, [math.sin(i) for i in range(251)],
    [math.cos(i) for i in range(300)], [-math.sin(i) for i in range(300)],
    [math.sin(i + 0.1) for i in range(300)]])
def test_unknown_or_distinct_signals_are_not_duplicates(other):
    candidates = (checked("a", 1.5), checked("b", 1.5))
    curves = (series("a", [math.sin(i) for i in range(300)]),)
    if other is not None:
        curves += (series("b", other),)
    assert find_signal_duplicates(candidates, curves, comparison_groups={"a": (), "b": ()}) == ()


def test_shared_intervals_require_both_endpoints_and_at_least_252_increments():
    candidates = (checked("a", 1.5), checked("b", 1.5))
    first = series("a", [math.sin(i) for i in range(300)])
    second = replace(first, platform_alpha_id="b", points=first.points[48:])
    assert len(find_signal_duplicates(candidates, (first, second), comparison_groups={"a": (), "b": ()})) == 1
    short = replace(second, points=second.points[1:])
    assert find_signal_duplicates(candidates, (first, short), comparison_groups={"a": (), "b": ()}) == ()


def test_cross_lineage_scope_compares_directly_to_representatives_without_chaining():
    candidates = tuple(checked(name, 1.5) for name in ("a", "b", "c"))
    curves = tuple(series(name, [math.sin(i + phase) for i in range(300)])
                   for name, phase in (("a", 0), ("b", 0.01), ("c", 0.02)))
    duplicates = find_signal_duplicates(candidates, curves, comparison_groups={name: () for name in "abc"})
    assert [(d.task_id, d.representative_task_id) for d in duplicates] == [("b", "a")]
    assert duplicates[0].correlation >= 0.9999
    assert find_signal_duplicates(candidates, curves, comparison_groups={name: name for name in "abc"}) == ()


@pytest.mark.parametrize("change", ["account", "settings", "grade"])
def test_account_settings_and_unknown_grade_are_not_merged(change):
    first, second = checked("a", 1.5), checked("b", 1.5)
    if change == "account":
        second = replace(second, task=replace(second.task, account_scope="other"))
    elif change == "settings":
        second = replace(second, task=replace(second.task, settings_json='{"delay":0}'))
    else:
        second = replace(second, result=replace(second.result, grade=None))
    curves = tuple(series(c.task.platform_alpha_id, [math.sin(i) for i in range(300)], c.task.account_scope)
                   for c in (first, second))
    assert find_signal_duplicates((first, second), curves, comparison_groups={"a": (), "b": ()}) == ()
