import json
import unittest
from datetime import datetime

from execution.self_correlation import load_self_correlation_references
from persistence.backtests import get_backtest_task
from persistence.database import open_database
from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
from persistence.submissions import PlatformSubmittedAlphaRecord, list_formal_submission_attempts
from tests.execution import test_submission_runner as fixture
from worldquant.backtests import STANDARD_REGULAR_CHECK_NAMES


class SelfCorrelationReferenceTests(unittest.TestCase):
    def test_prequeue_sc_failure_supplies_repair_evidence_without_a_submission_attempt(self):
        setup = fixture.SubmissionQueueRunnerTests()
        setup.setUp()
        self.addCleanup(setup.doCleanups)
        tasks, _ = setup._completed_run(("rank(close)",))
        observed = "2026-09-04T00:08:00+00:00"
        payload = {"is": {
            "checks": [{"name": name, "result": "FAIL" if name == "SELF_CORRELATION" else "PASS",
                        "value": .9, "limit": .7} for name in STANDARD_REGULAR_CHECK_NAMES],
            "selfCorrelated": {"max": .9, "schema": {"properties": [{"name": "id"}, {"name": "correlation"}]},
                               "records": [["conflict-alpha", .9]]},
        }}
        with open_database(setup.database_path) as connection:
            connection.execute("DELETE FROM submission_checks")
            save_submission_check(connection, SubmissionCheckRecord(tasks[0], observed, json.dumps(payload), None))
            parent = get_backtest_task(connection, tasks[0])
            reference = PlatformSubmittedAlphaRecord("group-account", "conflict-alpha", "rank(open)",
                "ACTIVE", "2026-09-01T00:00:00+00:00", False,
                {"settings": json.loads(parent.task.settings_json)}, observed)
            selected = load_self_correlation_references(connection, parents=(parent,),
                submitted_alphas=(reference,), account_scope="group-account", observed_at=datetime.fromisoformat(observed))
            self.assertEqual(len(selected), 1)
            self.assertEqual((selected[0].parent_task_id, selected[0].formula, selected[0].correlation),
                             (tasks[0], "rank(open)", .9))
            self.assertEqual(list_formal_submission_attempts(connection), ())
            self.assertEqual(load_self_correlation_references(connection, parents=(parent,),
                submitted_alphas=(reference,), account_scope="other-account", observed_at=datetime.fromisoformat(observed)), ())
