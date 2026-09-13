from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from execution.backtests import cancel_unsubmitted_backtest_task
from execution.qualified_candidates import load_exhausted_qualified_candidates
from execution.qualified_archive import synchronize_qualified_alpha_archive
from persistence.database import open_database
from persistence.qualified_archive import archive_qualified_alpha
from persistence.schema import initialize_database_schema
from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
from persistence.submissions import PlatformSubmittedAlphaRecord, record_platform_submitted_alphas
from tests.execution.test_qualified_archive import (
    ARCHIVED, FINISHED, complete_candidate, exhausted_parent, prepare_candidate,
)
from worldquant.backtests import STANDARD_REGULAR_CHECK_NAMES


class ExhaustedQualifiedCandidatesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database_path = Path(directory.name) / "database.sqlite3"
        with open_database(self.database_path) as connection:
            initialize_database_schema(connection)

    def test_only_archived_exhausted_known_lower_grades_with_settled_children_are_eligible(self):
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
            for parent in (nineteen, pending, replaced):
                archive_qualified_alpha(connection, task_id=parent.task.task_id, archived_at=ARCHIVED)
            exhausted_parent(connection)  # Full budget and checks alone do not bypass archive membership.
            before = connection.total_changes
            selected = load_exhausted_qualified_candidates(connection, account_scope="group-account")
            self.assertEqual({s.task.task_id for s in selected}, eligible)
            self.assertEqual(load_exhausted_qualified_candidates(connection, account_scope="another-account"), ())
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
                selected = load_exhausted_qualified_candidates(connection, account_scope="group-account")
                self.assertEqual(len(selected), 1 if state == "PASS" else 0)
            record_platform_submitted_alphas(connection, (PlatformSubmittedAlphaRecord(
                account_scope="group-account", platform_alpha_id=parent.task.platform_alpha_id,
                formula=parent.task.formula, status="ACTIVE", date_submitted=ARCHIVED, hidden=False,
                raw_payload={"id": parent.task.platform_alpha_id, "status": "ACTIVE", "dateSubmitted": ARCHIVED,
                             "hidden": False, "regular": {"code": parent.task.formula}},
                observed_at=ARCHIVED,
            ),))
            self.assertEqual(load_exhausted_qualified_candidates(connection, account_scope="group-account"), ())
