from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest

from learning.diagnosis import (
    EXPLORATION_SLOT_EVERY,
    HIGH_TURNOVER,
    LOW_RETURNS,
    LOW_SHARPE,
    LOW_TURNOVER,
    PRESCRIPTIONS,
    diagnose_parent,
    follows_prescription,
)


@pytest.mark.parametrize("metrics, failed, problem", [
    # Round-2 parents: vwap reversal S 1.36 F 0.75 T 52%; fnd6_drc/cap S 1.34 F 0.90 T 1%.
    ((1.36, 0.75, 0.52), {"LOW_FITNESS"}, HIGH_TURNOVER),
    ((1.0, 0.5, 0.20), {"HIGH_TURNOVER"}, HIGH_TURNOVER),
    ((1.34, 0.90, 0.01), {"LOW_FITNESS"}, LOW_TURNOVER),
    ((1.40, 0.80, 0.08), {"LOW_FITNESS"}, LOW_RETURNS),
    ((-1.40, -0.80, 0.08), {"LOW_FITNESS"}, LOW_RETURNS),  # flipped parents count by size
    ((0.90, 0.60, 0.08), {"LOW_SHARPE", "LOW_FITNESS"}, LOW_SHARPE),
])
def test_parents_are_diagnosed_by_what_holds_fitness_back(metrics, failed, problem):
    sharpe, fitness, turnover = metrics
    diagnosis = diagnose_parent(sharpe=sharpe, fitness=fitness, turnover=turnover,
                                failed_checks=frozenset(failed))
    assert diagnosis.problem == problem
    assert diagnosis.families == PRESCRIPTIONS[problem] and diagnosis.label


def test_no_diagnosis_without_a_performance_problem_or_metrics():
    assert diagnose_parent(sharpe=1.6, fitness=1.2, turnover=0.2,
                           failed_checks=frozenset({"SELF_CORRELATION"})) is None
    assert diagnose_parent(sharpe=None, fitness=1.0, turnover=0.2,
                           failed_checks=frozenset({"LOW_SHARPE"})) is None


def test_a_fixed_share_of_slots_explores_beyond_the_prescription():
    follows = [follows_prescription("run|1", f"parent-{p}", slot)
               for p in range(40) for slot in range(10)]
    share = 1 - sum(follows) / len(follows)
    assert abs(share - 1 / EXPLORATION_SLOT_EVERY) < 0.06
    assert follows == [follows_prescription("run|1", f"parent-{p}", slot)
                       for p in range(40) for slot in range(10)]


from dataclasses import dataclass, field

from learning.diagnosis import (
    MIN_CURE_SCORE,
    learn_prescriptions,
    mutation_outcome,
    MutationOutcome,
)


@dataclass
class _Check:
    name: str
    status: str


@dataclass
class _Result:
    sharpe: float
    fitness: float
    turnover: float
    returns: float
    checks: tuple = field(default_factory=lambda: (_Check("LOW_FITNESS", "FAIL"),))


def test_a_cure_needs_better_fitness_and_the_problem_metric_moving_the_right_way():
    parent = _Result(1.36, 0.75, 0.52, 0.155)
    # Round 1's heavy smoothing: turnover fell but fitness fell too -> not a cure.
    assert not mutation_outcome(parent, _Result(1.0, 0.6, 0.16, 0.07), "smooth").cured
    assert mutation_outcome(parent, _Result(1.4, 1.05, 0.25, 0.13), "smooth").cured
    # Better fitness without the turnover falling is not a cure for high turnover.
    assert not mutation_outcome(parent, _Result(1.6, 0.9, 0.60, 0.20), "other").cured
    healthy = _Result(1.6, 1.2, 0.2, 0.1, checks=(_Check("SELF_CORRELATION", "FAIL"),))
    assert mutation_outcome(healthy, parent, "x") is None


def test_learned_prescriptions_start_from_priors_and_follow_evidence():
    assert learn_prescriptions(()) == PRESCRIPTIONS
    failing = PRESCRIPTIONS[HIGH_TURNOVER][0]
    outcomes = ([MutationOutcome(HIGH_TURNOVER, failing, False)] * 6
                + [MutationOutcome(HIGH_TURNOVER, "stranger", True)] * 4
                + [MutationOutcome(HIGH_TURNOVER, "unlucky", False)] * 2)
    learned = learn_prescriptions(outcomes)[HIGH_TURNOVER]
    assert learned[0] == "stranger"
    assert failing not in learned and "unlucky" not in learned
    assert learn_prescriptions(outcomes)[LOW_SHARPE] == PRESCRIPTIONS[LOW_SHARPE]
    # One unlucky try does not erase a sensible prior.
    once = learn_prescriptions([MutationOutcome(HIGH_TURNOVER, failing, False)])
    assert failing in once[HIGH_TURNOVER] and MIN_CURE_SCORE < 0.5


def test_diagnosis_uses_learned_prescriptions_when_given():
    diagnosis = diagnose_parent(sharpe=1.4, fitness=0.7, turnover=0.5,
                                failed_checks=frozenset({"LOW_FITNESS"}),
                                prescriptions={HIGH_TURNOVER: ("stranger",)})
    assert diagnosis.families == ("stranger",)
