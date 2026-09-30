from dataclasses import replace
from datetime import datetime, timedelta, timezone

from learning.research_progress import ResearchObservation, assess_research_progress


def observations(count):
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    return tuple(ResearchObservation(str(i), (start + timedelta(minutes=i)).isoformat(), True, False)
                 for i in range(count))


def test_cold_start_and_short_runs_do_not_trigger_plateau():
    cold = assess_research_progress(observations(500), has_initial_seed=False, max_cycles=-1, max_backtests=0)
    assert (cold.phase, cold.complete_windows) == ("cold_start", 0)
    for focused, cycles, budget in ((False, 3, 60), (True, -1, 0)):
        result = assess_research_progress(observations(200), has_initial_seed=True,
            max_cycles=cycles, max_backtests=budget, focused=focused)
        assert not result.monitoring_enabled
        assert result.complete_windows == 0


def test_only_resolved_results_in_this_run_count_towards_windows():
    records = observations(200)
    assess = lambda items: assess_research_progress(items, has_initial_seed=True, max_cycles=10, max_backtests=200)
    assert assess(records).phase == "stagnating"
    assert assess(records[:100]).phase == "observing"
    assert assess(tuple(replace(item, resolved=False) if int(item.task_id) % 2 else item
                        for item in records)).complete_windows == 1
    result = assess((*records[:-1], replace(records[-1], progress=True)))
    assert (result.phase, result.stagnant_windows) == ("progressing", 0)


def test_new_seed_starts_established_research_window_and_offsets_sort_by_instant():
    records = observations(300)
    records = (*records[:149], replace(records[149], new_seed=True, progress=True), *records[150:])
    result = assess_research_progress(records, has_initial_seed=False, max_cycles=-1, max_backtests=0)
    assert (result.complete_windows, result.stagnant_windows, result.phase) == (1, 1, "observing")
    records = observations(200)
    # Last event with a negative offset sorts before earlier UTC strings lexically.
    last = replace(records[-1], progress=True,
                   finished_at=datetime.fromisoformat(records[-1].finished_at).astimezone(timezone(timedelta(hours=-8))).isoformat())
    result = assess_research_progress((*records[:-1], last), has_initial_seed=True, max_cycles=-1, max_backtests=0)
    assert result.stagnant_windows == 0
