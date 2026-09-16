from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from execution.backtests import cancel_unsubmitted_backtest_task
from execution.qualified_candidates import load_submittable_qualified_candidates, load_submission_opportunity_ids, load_qualified_evolution
from execution.qualified_archive import synchronize_qualified_alpha_archive
from execution.seeds import load_signal_frontiers, synchronize_signal_seeds
from persistence.database import open_database
from persistence.qualified_archive import archive_qualified_alpha
from persistence.pnl import list_pnl_series, save_pnl_series
from persistence.schema import initialize_database_schema
from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
from persistence.submissions import PlatformSubmittedAlphaRecord, record_platform_submitted_alphas
from tests.execution.test_qualified_archive import (
    ARCHIVED, FINISHED, complete_candidate, exhausted_parent, prepare_candidate,
)
from worldquant.backtests import STANDARD_REGULAR_CHECK_NAMES
from tests.learning.test_seed_correlation import series


class ExhaustedQualifiedCandidatesTests(unittest.TestCase):
    def test_replaced_history_keeps_ten_percent_step_and_replans_after_submission(self):
        with open_database(self.database_path) as connection:
            a = complete_candidate(connection, prepare_candidate(connection), sharpe=1.5)
            b = complete_candidate(connection, prepare_candidate(connection, parent=a), sharpe=1.6)
            c = complete_candidate(connection, prepare_candidate(connection, parent=b), sharpe=1.7)
            for snapshot in (a, b, c):
                connection.execute("DELETE FROM platform_pnl_series WHERE platform_alpha_id=?", (snapshot.task.platform_alpha_id,))
                save_pnl_series(connection, series(snapshot.task.platform_alpha_id,
                    [math.sin(i) for i in range(300)], account="group-account"))
            synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED)
            before = connection.total_changes
            self.assertEqual([s.task.task_id for s in load_submittable_qualified_candidates(
                connection, account_scope="group-account")], [a.task.task_id])
            self.assertEqual(connection.total_changes, before)
            self.assertEqual({r.task_id: r.remaining_attempts for r in load_qualified_evolution(connection)},
                             {a.task.task_id: 19, b.task.task_id: 19, c.task.task_id: 20})
            record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
                "group-account", a.task.platform_alpha_id, a.task.formula, "ACTIVE", ARCHIVED, False,
                {"id": a.task.platform_alpha_id, "status": "ACTIVE", "dateSubmitted": ARCHIVED,
                 "hidden": False, "regular": {"code": a.task.formula}, "is": {"sharpe": 1.5}}, ARCHIVED),))
            self.assertEqual(load_submittable_qualified_candidates(connection, account_scope="group-account"), ())
            self.assertEqual(load_submission_opportunity_ids(connection, account_scope="group-account",
                                                            include_optimization=True), (c.task.task_id,))

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database_path = Path(directory.name) / "database.sqlite3"
        with open_database(self.database_path) as connection:
            initialize_database_schema(connection)

    def test_optimization_and_archive_share_measured_duplicate_retirement_without_rewriting_results(self):
        with open_database(self.database_path) as connection:
            parent = complete_candidate(connection, prepare_candidate(connection))
            synchronize_signal_seeds(connection)
            child = complete_candidate(connection, prepare_candidate(connection, parent=parent))
            # A later observation prevents UUID ordering from deciding equal-quality ties.
            connection.execute("UPDATE backtest_tasks SET finished_at=? WHERE task_id=?",
                               (ARCHIVED, child.task.task_id))
            different = complete_candidate(connection, prepare_candidate(connection, parent=parent))
            for snapshot, phase in ((parent, 0), (child, 0.001), (different, 1)):
                connection.execute("DELETE FROM platform_pnl_series WHERE account_scope=? AND platform_alpha_id=?",
                                   ("group-account", snapshot.task.platform_alpha_id))
                save_pnl_series(connection, series(snapshot.task.platform_alpha_id,
                    [math.sin(i + phase) for i in range(300)], account="group-account"))
            before = connection.total_changes
            decisions = {r.task_id: r for r in load_qualified_evolution(connection)}
            self.assertEqual(decisions[child.task.task_id].replacement_task_id, parent.task.task_id)
            self.assertEqual(decisions[parent.task.task_id].remaining_attempts, 18)
            self.assertEqual(set(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids),
                             {parent.task.task_id, different.task.task_id})
            self.assertEqual(connection.total_changes, before)
            self.assertEqual(len(list_pnl_series(connection,
                platform_alpha_ids=frozenset({parent.task.platform_alpha_id}))), 1)
            self.assertEqual(list_pnl_series(connection, platform_alpha_ids=frozenset()), ())
            original_results = connection.execute("SELECT * FROM backtest_results ORDER BY task_id").fetchall()
            self.assertEqual({r.task_id for r in synchronize_qualified_alpha_archive(
                connection, observed_at=ARCHIVED)}, {child.task.task_id})
            self.assertEqual(connection.execute("SELECT * FROM backtest_results ORDER BY task_id").fetchall(), original_results)
            self.assertEqual(load_submittable_qualified_candidates(connection, account_scope="group-account"), ())

    def test_archived_retired_known_lower_grades_with_settled_children_are_eligible(self):
        with open_database(self.database_path) as connection:
            eligible = set()
            for grade in ("INFERIOR", "AVERAGE", "GOOD", "EXCELLENT", "SPECTACULAR", None):
                parent = exhausted_parent(connection, grade=grade)
                archive_qualified_alpha(connection, task_id=parent.task.task_id, archived_at=ARCHIVED)
                if grade in {"INFERIOR", "AVERAGE", "GOOD", "EXCELLENT"}:
                    eligible.add(parent.task.task_id)
            nineteen = exhausted_parent(connection, attempts=19)
            cancelled = prepare_candidate(connection, parent=nineteen, start=False)
            cancel_unsubmitted_backtest_task(connection, cancelled.task.task_id, observed_at=FINISHED)
            pending = exhausted_parent(connection, attempts=19)
            prepare_candidate(connection, parent=pending)
            replaced = exhausted_parent(connection, attempts=0)
            complete_candidate(connection, prepare_candidate(connection, parent=replaced), sharpe=1.8)
            eligible.add(replaced.task.task_id)  # Replacement ends research; unused budget is not a submission veto.
            for parent in (nineteen, pending, replaced):
                archive_qualified_alpha(connection, task_id=parent.task.task_id, archived_at=ARCHIVED)
            exhausted_parent(connection)  # Full budget and checks alone do not bypass archive membership.
            before = connection.total_changes
            selected = load_submittable_qualified_candidates(connection, account_scope="group-account")
            self.assertEqual({s.task.task_id for s in selected}, eligible)
            self.assertEqual(load_submittable_qualified_candidates(connection, account_scope="another-account"), ())
            self.assertEqual(connection.total_changes, before)

    def test_stale_archive_cannot_bypass_sc_checks_or_submitted_identity(self):
        with open_database(self.database_path) as connection:
            parent = exhausted_parent(connection)
            synchronize_qualified_alpha_archive(connection, observed_at=ARCHIVED)
            task_id = parent.task.task_id
            for state in ("FAIL", "PENDING", "MISSING", "PASS"):
                payload = {"is": {"checks": [
                    {"name": name, "result": state if name == "SELF_CORRELATION" else "PASS"}
                    for name in STANDARD_REGULAR_CHECK_NAMES
                    if not (name == "SELF_CORRELATION" and state == "MISSING")
                ]}}
                # Replace the independent platform observation, retaining original backtest evidence.
                connection.execute("DELETE FROM submission_checks WHERE task_id=?", (task_id,))
                save_submission_check(connection, SubmissionCheckRecord(task_id, ARCHIVED, json.dumps(payload), None))
                selected = load_submittable_qualified_candidates(connection, account_scope="group-account")
                self.assertEqual(len(selected), 1 if state == "PASS" else 0)
            record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
                account_scope="group-account", platform_alpha_id=parent.task.platform_alpha_id,
                formula=parent.task.formula, status="ACTIVE", date_submitted=ARCHIVED, hidden=False,
                raw_payload={"id": parent.task.platform_alpha_id, "status": "ACTIVE", "dateSubmitted": ARCHIVED,
                             "hidden": False, "regular": {"code": parent.task.formula}},
                observed_at=ARCHIVED,
            ),))
            self.assertEqual(load_submittable_qualified_candidates(connection, account_scope="group-account"), ())
