"""Run-scoped observations; cold starts are distinct from established research."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ResearchObservation:
    task_id: str
    finished_at: str
    resolved: bool
    progress: bool
    new_seed: bool = False


@dataclass(frozen=True, slots=True)
class ResearchProgress:
    phase: str
    monitoring_enabled: bool
    resolved_count: int
    complete_windows: int
    stagnant_windows: int
    window_size: int = 100


def assess_research_progress(
    observations: tuple[ResearchObservation, ...], *, has_initial_seed: bool,
    max_cycles: int, max_backtests: int, focused: bool = False,
) -> ResearchProgress:
    ordered = tuple(sorted(observations, key=lambda item: (datetime.fromisoformat(item.finished_at), item.task_id)))
    enabled = not focused and (max_cycles == -1 or max_backtests >= 200)
    resolved_count = sum(item.resolved for item in ordered)
    if not has_initial_seed:
        first_seed = next((index for index, item in enumerate(ordered) if item.new_seed), None)
        if first_seed is None:
            return ResearchProgress("cold_start", enabled, resolved_count, 0, 0)
        ordered = ordered[first_seed + 1:]
    if not enabled:
        return ResearchProgress("short_or_focused_run", False, resolved_count, 0, 0)
    resolved = tuple(item for item in ordered if item.resolved)
    windows = len(resolved) // 100
    stagnant = 0
    for index in range(windows):
        window = resolved[index * 100:(index + 1) * 100]
        stagnant = 0 if any(item.progress for item in window) else stagnant + 1
    phase = "stagnating" if stagnant >= 2 else "observing" if windows < 2 else "progressing"
    return ResearchProgress(phase, True, resolved_count, windows, stagnant)
