import math
from types import SimpleNamespace

from submission.opportunities import select_submission_opportunities
from tests.learning.test_seed_correlation import candidate, reference, series


def formula(name, sharpe, grade="GOOD"):
    snapshot = candidate(name, sharpe=sharpe)
    snapshot.task.task_id = name
    snapshot.task.finished_at = "2026-09-01T00:00:00+00:00"
    snapshot.result = SimpleNamespace(sharpe=sharpe, fitness=1.5, grade=grade)
    return snapshot


def curves(*names):
    return tuple(series(name, [math.sin(i) for i in range(300)]) for name in names)


def test_preserves_earlier_step_instead_of_blocking_the_active_successor():
    a, b, c = formula("a", 1.5), formula("b", 1.6), formula("c", 1.7)
    data = curves("a", "b", "c")
    assert select_submission_opportunities((a, b), (c,), (), data) == ("a",)
    assert select_submission_opportunities((a, b, c), (), (), data) == ("a",)
    assert select_submission_opportunities((b, c), (c,), (reference("a"),), data) == ("c",)


def test_equal_quality_or_less_than_ten_percent_does_not_create_a_ladder():
    a, b = formula("a", 1.5), formula("b", 1.6)
    assert select_submission_opportunities((a, b), (), (), curves("a", "b")) == ("b",)
    b.result.sharpe = 1.5
    assert select_submission_opportunities((b, a), (), (), curves("a", "b")) == ("a",)


def test_independent_candidate_survives_while_high_correlation_peers_are_sequenced():
    a, b, independent = formula("a", 1.5), formula("b", 1.7), formula("independent", 1.8)
    data = (*curves("a", "b"), series("independent", [math.cos(i) for i in range(300)]))
    assert set(select_submission_opportunities((a, b, independent), (), (), data)) == {"a", "independent"}


def test_missing_pair_and_missing_reference_sharpe_do_not_imply_permission():
    a, b = formula("a", 1.5), formula("b", 1.7)
    assert select_submission_opportunities((a,), (b,), (), curves("a")) == ()
    assert select_submission_opportunities((a, b), (), (), curves("a")) == ()
    assert select_submission_opportunities((b,), (), (reference("a", sharpe=None),), curves("a", "b")) == ()


def test_does_not_protect_an_opportunity_already_blocked_by_submissions():
    a, old = formula("a", 1.7), formula("old", 1.6)
    assert select_submission_opportunities((a,), (old,), (reference("submitted", sharpe=1.5),),
                                          curves("a", "old", "submitted")) == ("a",)


def test_two_references_must_both_allow_the_candidate():
    a = formula("a", 1.65)
    assert select_submission_opportunities((a,), (),
        (reference("r1", sharpe=1.5), reference("r2", sharpe=1.6)), curves("a", "r1", "r2")) == ()
