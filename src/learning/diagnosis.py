"""Name a parent's main problem from its backtest, so mutations can aim at it.

The prescriptions start as priors written from how the platform scores an alpha
(Fitness = Sharpe * sqrt(|Returns| / max(Turnover, 0.125))) and from the first runs'
mutation results, then follow recent outcomes: for each problem, an action counts as a
cure when the child beats the parent's Fitness and moves the metric behind the problem
the right way (turnover down for high turnover, and so on). Per-parent learned action
preferences still take precedence, and a fixed share of slots ignores the prescription
so untried fixes keep collecting evidence.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from generation.decay_ladder import SIGNAL_DECAY_LADDER
from generation.internal_edits import INTERNAL_FIELD_REPLACEMENT
from generation.polishing import POLISHING_FAMILIES
from generation.transformations import (
    COMPLEMENTARY_SIGNAL_REFRAME,
    DISTRIBUTION_STABILIZATION,
    GROUP_RELATIVE_REFRAME,
    TEMPORAL_CHANGE_REFRAME,
    TEMPORAL_PERSISTENCE_REFRAME,
)

HIGH_TURNOVER = "high_turnover"
LOW_TURNOVER = "low_turnover"
LOW_RETURNS = "low_returns"
LOW_SHARPE = "low_sharpe"

# Above this the turnover term caps Fitness; the platform fails alphas above 70%.
HIGH_TURNOVER_LEVEL = 0.30
# The platform fails alphas trading under 1% a day.
LOW_TURNOVER_LEVEL = 0.015
STRONG_SHARPE = 1.0
# Smoothing windows that cut turnover without washing out the signal: the first run's
# 22-250 day smoothing cut turnover 45% -> 16% but halved returns.
SHORT_SMOOTHING_WINDOW = 10
# One slot in this many ignores the prescription and keeps the usual rotation.
EXPLORATION_SLOT_EVERY = 5

PRESCRIPTIONS = {
    # Add decay a rung at a time, smooth the signal a little, or lengthen its windows.
    HIGH_TURNOVER: (SIGNAL_DECAY_LADDER, TEMPORAL_PERSISTENCE_REFRAME, *POLISHING_FAMILIES),
    # Too slow to trade enough: look at its change, or add a faster companion.
    LOW_TURNOVER: (TEMPORAL_CHANGE_REFRAME, COMPLEMENTARY_SIGNAL_REFRAME),
    # Good Sharpe, small edge: compare within industries, try sibling fields, add a signal.
    LOW_RETURNS: (GROUP_RELATIVE_REFRAME, INTERNAL_FIELD_REPLACEMENT, COMPLEMENTARY_SIGNAL_REFRAME),
    # Weak or noisy: compare within industries, try sibling fields, tame outliers.
    LOW_SHARPE: (GROUP_RELATIVE_REFRAME, INTERNAL_FIELD_REPLACEMENT, DISTRIBUTION_STABILIZATION,
                 COMPLEMENTARY_SIGNAL_REFRAME),
}

PROBLEM_LABELS = {
    HIGH_TURNOVER: "换手过高",
    LOW_TURNOVER: "换手过低",
    LOW_RETURNS: "收益偏低",
    LOW_SHARPE: "夏普不足",
}

_PERFORMANCE_CHECKS = frozenset({
    "LOW_SHARPE", "LOW_FITNESS", "HIGH_TURNOVER", "LOW_TURNOVER", "LOW_SUB_UNIVERSE_SHARPE",
    "CONCENTRATED_WEIGHT",
})


@dataclass(frozen=True, slots=True)
class ParentDiagnosis:
    problem: str
    families: tuple[str, ...]

    @property
    def label(self) -> str:
        return PROBLEM_LABELS[self.problem]


def diagnose_parent(
    *,
    sharpe: float | None,
    fitness: float | None,
    turnover: float | None,
    failed_checks: frozenset[str],
    prescriptions: Mapping[str, tuple[str, ...]] | None = None,
) -> ParentDiagnosis | None:
    """None when the parent has no performance problem to aim at (e.g. only SC).

    ``prescriptions`` (from learn_prescriptions) replaces the starting priors.
    """
    if sharpe is None or fitness is None or turnover is None:
        return None
    if not failed_checks & _PERFORMANCE_CHECKS:
        return None
    if "HIGH_TURNOVER" in failed_checks or turnover > HIGH_TURNOVER_LEVEL:
        problem = HIGH_TURNOVER
    elif "LOW_TURNOVER" in failed_checks or turnover < LOW_TURNOVER_LEVEL:
        problem = LOW_TURNOVER
    elif abs(sharpe) >= STRONG_SHARPE and abs(fitness) < 1.0:
        problem = LOW_RETURNS
    else:
        problem = LOW_SHARPE
    families = (prescriptions or {}).get(problem) or PRESCRIPTIONS[problem]
    return ParentDiagnosis(problem, families)


def follows_prescription(rotation_key: str, parent_task_id: str, slot: int) -> bool:
    """Deterministic: most slots follow the prescription, one in EXPLORATION_SLOT_EVERY not."""
    digest = sha256(f"{rotation_key}|{parent_task_id}|{slot}|prescription".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % EXPLORATION_SLOT_EVERY != 0


# Learning from recent outcomes. The window and the outcome cap keep the work per cycle
# bounded however long the system runs; old evidence ages out on its own.
LEARNING_WINDOW_DAYS = 30
MAX_LEARNING_OUTCOMES = 2000
_PRIOR_WEIGHT = 2.0
_PRESCRIBED_PRIOR = 0.5
_UNPRESCRIBED_PRIOR = 0.25
MIN_CURE_SCORE = 0.3
MAX_PRESCRIBED_FAMILIES = 4


class BacktestMetrics(Protocol):
    sharpe: float
    fitness: float
    turnover: float
    returns: float
    checks: tuple


@dataclass(frozen=True, slots=True)
class MutationOutcome:
    problem: str
    action: str
    cured: bool


def mutation_outcome(parent: BacktestMetrics, child: BacktestMetrics,
                     action: str) -> MutationOutcome | None:
    """How one mutation did against its parent's diagnosed problem (None: no problem)."""
    diagnosis = diagnose_parent(
        sharpe=parent.sharpe, fitness=parent.fitness, turnover=parent.turnover,
        failed_checks=frozenset(check.name for check in parent.checks
                                if check.status.strip().upper() == "FAIL"),
    )
    if diagnosis is None:
        return None
    moved = {
        HIGH_TURNOVER: child.turnover < parent.turnover,
        LOW_TURNOVER: child.turnover > parent.turnover,
        LOW_RETURNS: abs(child.returns) > abs(parent.returns),
        LOW_SHARPE: abs(child.sharpe) > abs(parent.sharpe),
    }[diagnosis.problem]
    return MutationOutcome(diagnosis.problem, action,
                           moved and abs(child.fitness) > abs(parent.fitness))


def learn_prescriptions(outcomes: Iterable[MutationOutcome]) -> dict[str, tuple[str, ...]]:
    """Per problem, the actions most likely to cure it, best first.

    Each action's cure rate is smoothed toward a prior (higher for the starting
    prescription), so a few unlucky tries do not erase a sensible prior and a few lucky
    ones do not crown a stranger. Actions below MIN_CURE_SCORE drop out; if nothing is
    left the starting prescription stands.
    """
    tried: Counter[tuple[str, str]] = Counter()
    cured: Counter[tuple[str, str]] = Counter()
    for outcome in outcomes:
        tried[(outcome.problem, outcome.action)] += 1
        cured[(outcome.problem, outcome.action)] += outcome.cured
    learned: dict[str, tuple[str, ...]] = {}
    for problem, prior_families in PRESCRIPTIONS.items():
        actions = set(prior_families) | {action for (p, action) in tried if p == problem}

        def score(action: str) -> float:
            prior = _PRESCRIBED_PRIOR if action in prior_families else _UNPRESCRIBED_PRIOR
            return ((cured[(problem, action)] + prior * _PRIOR_WEIGHT)
                    / (tried[(problem, action)] + _PRIOR_WEIGHT))

        ranked = sorted(actions, key=lambda action: (
            -score(action),
            prior_families.index(action) if action in prior_families else len(prior_families),
            action,
        ))
        kept = tuple(action for action in ranked if score(action) >= MIN_CURE_SCORE)
        learned[problem] = kept[:MAX_PRESCRIBED_FAMILIES] or prior_families
    return learned
