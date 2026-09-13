from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.backtests import (
    apply_backtest_detail, cancel_unsubmitted_backtest_task, fail_backtest_task,
    prepare_backtest_task, record_backtest_yearly_stats, record_submission_accepted,
)
from execution.qualified_archive import synchronize_qualified_alpha_archive
from execution.seeds import load_signal_frontiers, synchronize_signal_seeds
from execution.submission_queue import claim_next_submission_queue_item
from persistence.backtests import BacktestMutationRecord, create_backtest_mutation, get_backtest_task
from persistence.database import open_database
from persistence.qualified_archive import list_qualified_alpha_archive
from persistence.schema import initialize_database_schema
from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
from persistence.submissions import PlatformSubmittedAlphaRecord, record_platform_submitted_alphas
from worldquant.backtests import BacktestCheck, BacktestDetail, BacktestYearlyStat, STANDARD_REGULAR_CHECK_NAMES


CREATED = "2026-09-01T00:00:00+00:00"
STARTED = "2026-09-01T00:01:00+00:00"
FINISHED = "2026-09-01T00:02:00+00:00"
ARCHIVED = "2026-09-02T00:00:00+00:00"


def prepare_candidate(connection, *, parent=None, start=True):
    number = connection.execute("SELECT COUNT(*) FROM backtest_tasks").fetchone()[0] + 2
    snapshot = prepare_backtest_task(
        connection, account_scope="group-account", formula=f"ts_mean(close,{number})",
        settings={"delay": 1}, created_at=CREATED,
    )
    if parent is not None:
        create_backtest_mutation(connection, BacktestMutationRecord(
            snapshot.task.task_id, parent.task.task_id, "structural", "formula",
            parent.task.formula, snapshot.task.formula,
        ))
    if start:
        snapshot = record_submission_accepted(
            connection, snapshot.task.task_id, remote_id=f"simulation-{number}", observed_at=STARTED,
        )
    return snapshot


def complete_candidate(connection, snapshot, *, sharpe=1.5, grade="GOOD", qualified=True):
    checks = tuple(BacktestCheck(
        name, "PASS" if qualified or name not in {"LOW_SHARPE", "LOW_FITNESS"} else "FAIL",
        None, None, None,
    ) for name in (
        "LOW_SHARPE", "LOW_FITNESS", "LOW_SUB_UNIVERSE_SHARPE", "CONCENTRATED_WEIGHT",
        "LOW_TURNOVER", "HIGH_TURNOVER", "MATCHES_COMPETITION", "SELF_CORRELATION",
    ))
    apply_backtest_detail(connection, snapshot.task.task_id, BacktestDetail(
        platform_alpha_id=f"alpha-{snapshot.task.task_id}", sharpe=sharpe, fitness=1.6,
        turnover=0.12, returns=0.08, drawdown=0.04, margin=0.001,
        book_size=20_000_000, pnl=100_000, checks=checks, grade=grade,
    ), observed_at=FINISHED)
    result = record_backtest_yearly_stats(connection, snapshot.task.task_id, (
        BacktestYearlyStat(2025, 100_000, 20_000_000, 0.12, sharpe, 0.08, 0.04,
                          0.001, 1.6, 100, 100, "IS"),
    ), observed_at=FINISHED)
    save_submission_check(connection, SubmissionCheckRecord(snapshot.task.task_id, FINISHED,
        json.dumps({"is": {"checks": [{"name": check.name, "result": check.status} for check in checks]}}),
        None,
    ))
    return result


def exhausted_parent(connection, *, attempts=20, grade="GOOD"):
    parent = complete_candidate(connection, prepare_candidate(connection), grade=grade)
    synchronize_signal_seeds(connection)
    for _ in range(attempts):
        complete_candidate(connection, prepare_candidate(connection, parent=parent), sharpe=1.3)
    return parent


class QualifiedAlphaArchiveTests(unittest.TestCase):
    def test_requires_all_official_checks_and_removes_invalid_existing_membership(self):
        with open_database(self.database_path) as connection:
            parent = exhausted_parent(connection)
            task_id = parent.task.task_id
            for mode in ("missing", "error", "sc_fail", "sc_pending", "sc_missing", "other_fail"):
                with self.subTest(mode=mode):
                    connection.execute("DELETE FROM submission_checks WHERE task_id=?", (task_id,))
                    if mode != "missing":
                        checks = [{"name": name, "result": (
                            "FAIL" if (mode == "sc_fail" and name == "SELF_CORRELATION")
                            or (mode == "other_fail" and name == "LOW_FITNESS") else
                            "PENDING" if mode == "sc_pending" and name == "SELF_CORRELATION" else "PASS"
                        )} for name in STANDARD_REGULAR_CHECK_NAMES
                            if not (mode == "sc_missing" and name == "SELF_CORRELATION")]
                        save_submission_check(connection, SubmissionCheckRecord(task_id, ARCHIVED,
                            None if mode == "error" else json.dumps({"is": {"checks": checks}}),
                            "request_failed" if mode == "error" else None))
                    self.assertEqual(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED), ())
                    self.assertEqual(list_qualified_alpha_archive(connection), ())
            passed = json.dumps({"is": {"checks": [{"name": name, "result": "PASS"}
                                                       for name in STANDARD_REGULAR_CHECK_NAMES]}})
            save_submission_check(connection, SubmissionCheckRecord(task_id,
                "2026-09-03T00:00:00+00:00", passed, None, attempt_count=2))
            created = synchronize_qualified_alpha_archive(connection, observed_at="2026-09-03T00:00:00+00:00")
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0].grade, "GOOD")
            self.assertEqual((created[0].platform_alpha_id, created[0].sharpe,
                              created[0].fitness, created[0].turnover),
                             (parent.task.platform_alpha_id, 1.5, 1.6, 0.12))
            self.assertEqual(connection.execute("SELECT task_id FROM qualified_alpha_archive WHERE grade='GOOD'").fetchone()[0], task_id)
            connection.execute("DELETE FROM submission_checks WHERE task_id=?", (task_id,))
            self.assertEqual(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED), ())
            self.assertEqual(list_qualified_alpha_archive(connection), ())
            self.assertEqual(get_backtest_task(connection, task_id), parent)

    def test_complete_check_cannot_replace_missing_annual_evidence(self):
        with open_database(self.database_path) as connection:
            parent = exhausted_parent(connection)
            connection.execute("DELETE FROM backtest_yearly_stats WHERE task_id=?", (parent.task.task_id,))
            self.assertEqual(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED), ())

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database_path = Path(directory.name) / "archive.sqlite3"
        with open_database(self.database_path) as connection:
            initialize_database_schema(connection)

    def test_nineteen_attempts_unsent_cancel_and_pending_twentieth_do_not_archive(self):
        with open_database(self.database_path) as connection:
            parent = exhausted_parent(connection, attempts=19)
            cancelled = prepare_candidate(connection, parent=parent, start=False)
            cancel_unsubmitted_backtest_task(connection, cancelled.task.task_id, observed_at=FINISHED)
            self.assertEqual(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED), ())
            frontier = load_signal_frontiers(connection).records[0]
            self.assertEqual(frontier.qualified_parent_remaining_attempts, 1)
            pending = prepare_candidate(connection, parent=parent)
            self.assertEqual(load_signal_frontiers(connection).active_branch_task_ids, ())
            self.assertEqual(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED), ())
            fail_backtest_task(connection, pending.task.task_id, failure_code="request_failed",
                               failure_message="test failure", observed_at=FINISHED)
            records = synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED)
            self.assertEqual(tuple(record.task_id for record in records), (parent.task.task_id,))
            self.assertEqual(get_backtest_task(connection, parent.task.task_id), parent)
        with open_database(self.database_path) as connection:
            self.assertEqual(synchronize_qualified_alpha_archive(
                connection, observed_at="2026-09-03T00:00:00+00:00"), ())
            self.assertEqual(list_qualified_alpha_archive(connection), records)
            self.assertEqual(load_signal_frontiers(connection).active_branch_task_ids, ())
            self.assertIsNone(claim_next_submission_queue_item(
                connection, account_scope="group-account", observed_at=ARCHIVED, submission_mode="manual",
            ))

    def test_better_twentieth_child_takes_over_and_both_formulas_are_retained(self):
        with open_database(self.database_path) as connection:
            parent = exhausted_parent(connection, attempts=19)
            child = complete_candidate(connection, prepare_candidate(connection, parent=parent),
                                       sharpe=1.8, grade="EXCELLENT")
            self.assertEqual(tuple(r.task_id for r in synchronize_qualified_alpha_archive(
                connection, observed_at=ARCHIVED)), (parent.task.task_id,))
            frontier = load_signal_frontiers(connection).records[0]
            self.assertEqual(frontier.qualified_parent_task_id, child.task.task_id)
            self.assertEqual(frontier.branches[0].remaining_attempts, 20)
            for _ in range(20):
                complete_candidate(connection, prepare_candidate(connection, parent=child), sharpe=1.6)
            self.assertEqual(tuple(record.task_id for record in synchronize_qualified_alpha_archive(
                connection, observed_at=ARCHIVED)), (child.task.task_id,))
            self.assertEqual(load_signal_frontiers(connection).active_branch_task_ids, ())
            self.assertEqual({r.task_id for r in list_qualified_alpha_archive(connection)},
                             {parent.task.task_id, child.task.task_id})

    def test_all_checked_sibling_formulas_are_retained_when_their_own_budgets_expire(self):
        with open_database(self.database_path) as connection:
            root = complete_candidate(connection, prepare_candidate(connection), sharpe=1.1, qualified=False)
            synchronize_signal_seeds(connection)
            candidates = tuple(complete_candidate(connection, prepare_candidate(connection, parent=root),
                sharpe=1.4 + index / 100, grade="AVERAGE") for index in range(13))
            ids = {s.task.task_id for s in candidates}
            self.assertEqual(set(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids), ids)
            for candidate in candidates:
                for _ in range(20):
                    complete_candidate(connection, prepare_candidate(connection, parent=candidate),
                                       sharpe=0.5, grade="INFERIOR", qualified=False)
            self.assertEqual({r.task_id for r in synchronize_qualified_alpha_archive(
                connection, observed_at=ARCHIVED)}, ids)
            self.assertEqual(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids, ())
            self.assertEqual(len(list_qualified_alpha_archive(connection)), 13)
            self.assertEqual(tuple(get_backtest_task(connection, s.task.task_id) for s in candidates), candidates)
            self.assertEqual(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED), ())

    def test_replaced_parent_is_archived_before_budget_expiry_after_inflight_children_finish(self):
        with open_database(self.database_path) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection))
            synchronize_signal_seeds(connection)
            pending = prepare_candidate(connection, parent=parent)
            child = complete_candidate(connection, prepare_candidate(connection, parent=parent), sharpe=1.8)
            self.assertEqual(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED), ())
            self.assertEqual(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids,
                             (child.task.task_id,))
            complete_candidate(connection, pending, sharpe=0.5, qualified=False)
            self.assertEqual(tuple(r.task_id for r in synchronize_qualified_alpha_archive(
                connection, observed_at=ARCHIVED)), (parent.task.task_id,))
            self.assertEqual(get_backtest_task(connection, parent.task.task_id), parent)

    def test_archiving_does_not_suppress_late_improvement_elsewhere_in_same_lineage(self):
        with open_database(self.database_path) as connection:
            root = complete_candidate(connection, prepare_candidate(connection), sharpe=1.1, qualified=False)
            synchronize_signal_seeds(connection)
            late = prepare_candidate(connection, parent=root)
            parent = complete_candidate(connection, prepare_candidate(connection, parent=root))
            for _ in range(20):
                complete_candidate(connection, prepare_candidate(connection, parent=parent), sharpe=1.3)
            archived = synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED)
            self.assertEqual(tuple(record.task_id for record in archived), (parent.task.task_id,))
            complete_candidate(connection, late, sharpe=1.8, grade="EXCELLENT")
            self.assertEqual(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED), ())
            frontier = load_signal_frontiers(connection).records[0]
            self.assertEqual(frontier.branches[0].task_id, late.task.task_id)
            self.assertEqual(frontier.branches[0].remaining_attempts, 20)
            self.assertEqual(list_qualified_alpha_archive(connection), archived)

    def test_historical_unknown_is_retained_without_inventing_grade(self):
        with open_database(self.database_path) as connection:
            parent = exhausted_parent(connection, grade=None)
            self.assertEqual(len(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED)), 1)
            self.assertIsNone(get_backtest_task(connection, parent.task.task_id).result.grade)
            self.assertIsNone(list_qualified_alpha_archive(connection)[0].grade)

    def test_spectacular_and_already_submitted_parents_are_not_archived(self):
        with open_database(self.database_path) as connection:
            exhausted_parent(connection, grade="SPECTACULAR")
            self.assertEqual(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED), ())
            parent = exhausted_parent(connection)
            record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
                "group-account", parent.task.platform_alpha_id, parent.task.formula, "ACTIVE",
                FINISHED, False, {"id": parent.task.platform_alpha_id, "status": "ACTIVE",
                    "hidden": False, "dateSubmitted": FINISHED, "regular": {"code": parent.task.formula}},
                ARCHIVED,
            ),))
            self.assertEqual(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED), ())

    def test_archive_rollback_preserves_history_and_can_be_retried(self):
        with open_database(self.database_path) as connection:
            parent = exhausted_parent(connection)
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            with open_database(self.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                self.assertEqual(len(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED)), 1)
                raise RuntimeError("rollback")
        with open_database(self.database_path) as connection:
            self.assertEqual(list_qualified_alpha_archive(connection), ())
            self.assertEqual(get_backtest_task(connection, parent.task.task_id), parent)
            self.assertEqual(len(synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED)), 1)


if __name__ == "__main__":
    unittest.main()
