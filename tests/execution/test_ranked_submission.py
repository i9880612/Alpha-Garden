from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution import submission_queue
from execution.submission_queue import RankedSubmissionCandidate, list_ranked_submission_candidates
from execution import submission_runner


def _record(task_id):
    return SimpleNamespace(task_id=task_id)


def _snapshot(grade, fitness, sharpe=2.0):
    return SimpleNamespace(result=SimpleNamespace(grade=grade, fitness=fitness, sharpe=sharpe))


def test_ranked_candidates_mix_sources_best_first_with_floors():
    by_source = {"queue": ["spectacular"], "qualified_archive": ["good_low", "good_high", "average", "excellent_archived"],
                 "optimization": ["excellent_opt", "good_opt", "good_high"]}
    snapshots = {"spectacular": _snapshot("SPECTACULAR", 2.4), "good_low": _snapshot("GOOD", 1.6),
                 "good_high": _snapshot("GOOD", 1.9), "average": _snapshot("AVERAGE", 1.4),
                 "excellent_archived": _snapshot("EXCELLENT", 2.0), "excellent_opt": _snapshot("EXCELLENT", 2.2),
                 "good_opt": _snapshot("GOOD", 1.95)}
    with patch.object(submission_queue, "list_submission_candidates",
                      side_effect=lambda connection, *, account_scope, source: tuple(map(_record, by_source[source]))), \
            patch.object(submission_queue, "_required_snapshot", side_effect=lambda connection, task_id: snapshots[task_id]):
        ranked = list_ranked_submission_candidates(None, account_scope="a")
        strict = list_ranked_submission_candidates(None, account_scope="a", minimum_grade="EXCELLENT")
    assert [(item.record.task_id, item.source) for item in ranked] == [
        ("spectacular", "queue"), ("excellent_opt", "optimization"), ("excellent_archived", "qualified_archive"),
        ("good_high", "qualified_archive"), ("good_low", "qualified_archive")]
    # Still-optimizing GOOD formulas wait; AVERAGE never qualifies at the default floor.
    assert [item.record.task_id for item in strict] == ["spectacular", "excellent_opt", "excellent_archived"]


def _completion(submitted=1, unresolved=0):
    return submission_runner.SubmissionQueueCompletion(
        account_scope="a", initial_queue_count=1, step_count=1, platform_request_count=1,
        authentication_performed=False, submission_claimed_count=1, submitted_count=submitted,
        already_active_count=0, ineligible_count=1 - submitted, failed_count=0, unresolved_count=unresolved,
        remaining_queue_count=0, max_submissions=1)


def _candidate(task_id, source, grade):
    return RankedSubmissionCandidate(_record(task_id), source, grade)


def test_ranked_submission_rereads_after_each_step_and_skips_failed_checks():
    rounds = iter([
        (_candidate("first", "optimization", "EXCELLENT"), _candidate("second", "queue", "SPECTACULAR")),
        (_candidate("first", "optimization", "EXCELLENT"), _candidate("second", "queue", "SPECTACULAR")),
        (_candidate("third", "qualified_archive", "GOOD"),),
        (),
    ])
    calls = []
    outcomes = iter([_completion(submitted=0), _completion(), _completion()])

    def submit(database, environment, **kwargs):
        calls.append((kwargs["source"], kwargs.get("selected_task_id"), kwargs.get("only_task_ids")))
        return next(outcomes)

    with patch.object(submission_runner, "load_worldquant_connection_settings", return_value=SimpleNamespace(account_scope="a")), \
            patch.object(submission_runner, "open_database"), \
            patch.object(submission_runner, "list_ranked_submission_candidates", side_effect=lambda *a, **k: next(rounds)), \
            patch.object(submission_runner, "submit_queued_alphas", side_effect=submit):
        completion = submission_runner.submit_ranked_alphas("db", "env", max_submissions=2,
                                                            client_factory=lambda settings: object())
    # The failed check is skipped, not retried; each step goes through its own source.
    assert calls == [("optimization", "first", None), ("queue", None, frozenset({"second"})),
                     ("qualified_archive", "third", None)]
    assert completion.submitted_count == 2 and completion.ineligible_count == 1


def test_a_strong_near_copy_kept_out_of_optimization_stays_submittable(tmp_path):
    import random
    from execution.seeds import load_signal_frontiers, synchronize_signal_seeds
    from execution.submission_queue import list_submission_candidates
    from persistence.database import open_database
    from persistence.pnl import save_pnl_series
    from persistence.schema import initialize_database_schema
    from tests.execution.test_qualified_archive import complete_candidate, prepare_candidate
    from tests.learning.test_seed_correlation import series

    database = tmp_path / "ranked.sqlite3"
    with open_database(database) as connection:
        initialize_database_schema(connection)
        seed = complete_candidate(connection, prepare_candidate(connection), sharpe=1.5, grade="GOOD")
        synchronize_signal_seeds(connection)
        copy = complete_candidate(connection, prepare_candidate(connection), sharpe=2.0, grade="EXCELLENT")
        rng = random.Random("shared")
        increments = [rng.gauss(0, 1) for _ in range(300)]
        for snapshot in (seed, copy):
            connection.execute("DELETE FROM platform_pnl_series WHERE platform_alpha_id=?",
                               (snapshot.task.platform_alpha_id,))
            save_pnl_series(connection, series(snapshot.task.platform_alpha_id, increments, "group-account"))
        frontier = load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids
        assert copy.task.task_id not in frontier  # a near-copy of the seed lineage is not optimized
        # These test tasks were made outside a run; real candidates always carry a run link.
        with patch.object(submission_queue, "get_automated_run_backtest_by_task",
                          return_value=SimpleNamespace(run_id="run", cycle_number=1)):
            offered = {item.task_id for item in list_submission_candidates(
                connection, account_scope="group-account", source="optimization")}
    assert copy.task.task_id in offered
