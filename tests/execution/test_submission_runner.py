from __future__ import annotations

import sys
import json
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from unittest.mock import patch
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.execution.catalog_fixture import initialize_test_generation_catalog
from execution.backtest_batches import (
    AutomatedCandidateBacktest,
    prepare_automated_candidate_backtest_batch,
)
from execution.backtests import (
    apply_backtest_detail,
    record_backtest_yearly_stats,
    record_submission_accepted,
    record_submission_unknown,
)
from execution.real_backtests import prepare_real_backtest
from execution.cycle_backtests import settle_automated_cycle
from execution.runs import (
    AutomatedRunLimits,
    prepare_automated_run,
    start_automated_run,
)
from execution.process_lock import exclusive_run_process
from execution.submission_runner import run_submission_queue, submit_queued_alphas
from alpha_garden.console import run_progress_console
from execution.submissions import advance_submission_queue
from generation.candidate import FormulaCandidate, exploration_candidate
from generation.parser import parse_formula
from persistence.backtests import (
    BacktestMutationRecord,
    create_backtest_mutation,
    get_backtest_task,
)
from persistence.database import open_database
from persistence.runs import get_automated_run, list_automated_run_backtests
from persistence.schema import initialize_database_schema
from persistence.submission_queue import list_formal_submission_queue
from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
from persistence.submissions import (
    PlatformSubmittedAlphaRecord,
    get_formal_submission_attempt,
    list_formal_submission_attempts,
    record_platform_submitted_alphas,
)
from worldquant.backtests import (
    BacktestCheck,
    BacktestDetail,
    BacktestPollObservation,
    BacktestSettings,
    BacktestYearlyStat,
    BacktestYearlyStatsObservation,
    STANDARD_REGULAR_CHECK_NAMES,
)
from worldquant.client import WorldQuantRequestError
from worldquant.submissions import (
    AlphaDetailObservation,
    FormalCheckObservation,
    FormalSubmissionObservation,
)


POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "backtest.default.json"


class FakeTime:
    def __init__(self, value: str) -> None:
        self.value = datetime.fromisoformat(value)

    def now(self) -> datetime:
        return self.value

    def wait(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class SubmissionClient:
    def __init__(
        self,
        formulas: dict[str, str],
        *,
        existing_active_ids: frozenset[str] = frozenset(),
        failed_check_ids: frozenset[str] = frozenset(),
        unknown_post_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.authenticated = False
        self.formulas = formulas
        self.existing_active_ids = existing_active_ids
        self.failed_check_ids = failed_check_ids
        self.unknown_post_ids = unknown_post_ids
        self.posted_ids: list[str] = []
        self.calls: list[tuple[str, str | None]] = []

    def authenticate(self) -> None:
        self.calls.append(("authenticate", None))
        self.authenticated = True

    def fetch_alpha_detail(self, *, platform_alpha_id):
        self.calls.append(("detail", platform_alpha_id))
        active = (
            platform_alpha_id in self.existing_active_ids
            or platform_alpha_id in self.posted_ids
            and platform_alpha_id not in self.unknown_post_ids
        )
        payload: dict[str, object] = {
            "id": platform_alpha_id,
            "grade": "SPECTACULAR",
            "status": "ACTIVE" if active else "UNSUBMITTED",
            "regular": {"code": self.formulas[platform_alpha_id]},
        }
        if active:
            payload.update(
                {
                    "dateSubmitted": "2026-09-04T00:00:00+00:00",
                    "hidden": False,
                }
            )
        return AlphaDetailObservation(payload=payload)

    def fetch_formal_submission_check(self, *, platform_alpha_id):
        self.calls.append(("check", platform_alpha_id))
        return FormalCheckObservation(
            payload={
                "is": {
                    "checks": [
                        {
                            "name": name,
                            "result": (
                                "FAIL"
                                if platform_alpha_id in self.failed_check_ids
                                and name == "SELF_CORRELATION"
                                else "PASS"
                            ),
                        }
                        for name in sorted(STANDARD_REGULAR_CHECK_NAMES)
                    ]
                }
            },
            retry_after_seconds=None,
        )

    def submit_formal_alpha(self, *, platform_alpha_id):
        self.calls.append(("post", platform_alpha_id))
        self.posted_ids.append(platform_alpha_id)
        if platform_alpha_id in self.unknown_post_ids:
            raise WorldQuantRequestError(
                "connection_lost",
                retryable=True,
                outcome_unknown=True,
            )
        return FormalSubmissionObservation(status_code=201, payload=None)


class SubmissionQueueRunnerTests(unittest.TestCase):
    def _qualified_batch(self, *, grades=("GOOD", "EXCELLENT", "SPECTACULAR"), attempts=20):
        from execution.backtests import prepare_backtest_task, fail_backtest_task
        from execution.qualified_archive import synchronize_qualified_alpha_archive
        task_ids, formulas = self._completed_run(
            ("rank(close)", "rank(open)", "rank(volume)"), grades=grades,
        )
        with open_database(self.database_path) as connection:
            grade_by_id = {get_backtest_task(connection, task_id).task.platform_alpha_id: grade
                           for task_id, grade in zip(task_ids, grades, strict=True)}
            for task_id in task_ids:
                parent = get_backtest_task(connection, task_id)
                for _ in range(attempts):
                    number = connection.execute("SELECT COUNT(*) FROM backtest_tasks").fetchone()[0] + 10
                    child = prepare_backtest_task(
                        connection, account_scope="group-account", formula=f"ts_mean(close,{number})",
                        settings=json.loads(parent.task.settings_json), created_at="2026-09-04T00:10:00+00:00",
                    )
                    create_backtest_mutation(connection, BacktestMutationRecord(
                        child.task.task_id, task_id, "structural", "formula", parent.task.formula, child.task.formula,
                    ))
                    record_submission_accepted(connection, child.task.task_id,
                        remote_id=f"failed-{number}", observed_at="2026-09-04T00:11:00+00:00")
                    fail_backtest_task(connection, child.task.task_id, failure_code="test_failure",
                        failure_message="test failure", observed_at="2026-09-04T00:12:00+00:00")
            synchronize_qualified_alpha_archive(connection, observed_at="2026-09-04T00:13:00+00:00")
        class QualifiedClient(SubmissionClient):
            def fetch_alpha_detail(self, *, platform_alpha_id):
                detail = super().fetch_alpha_detail(platform_alpha_id=platform_alpha_id)
                return replace(detail, payload={**detail.payload, "grade": grade_by_id[platform_alpha_id]})
        return task_ids, QualifiedClient(formulas)

    def test_qualified_limit_consumes_only_claimed_archive_and_keeps_spectacular_queue(self):
        from persistence.qualified_archive import list_qualified_alpha_archive
        task_ids, client = self._qualified_batch()
        time = FakeTime("2026-09-04T01:00:00+00:00")
        first = self._submit(client, time, source="qualified_archive", max_submissions=1)
        self.assertEqual((first.initial_queue_count, first.submitted_count, first.remaining_queue_count), (2, 1, 1))
        with open_database(self.database_path) as connection:
            self.assertEqual({r.task_id for r in list_formal_submission_queue(connection)}, {task_ids[2]})
            self.assertEqual(len(list_qualified_alpha_archive(connection)), 1)
            self.assertEqual(list_formal_submission_attempts(connection)[0].source, "qualified_archive")
        second = self._submit(client, time, source="qualified_archive", max_submissions=2)
        self.assertEqual((second.submitted_count, second.remaining_queue_count), (1, 0))
        third = self._submit(client, time, source="qualified_archive")
        self.assertEqual(third.initial_queue_count, 0)
        self.assertEqual(len(client.posted_ids), 2)

    def test_grade_selection_keeps_other_grades_and_cannot_resume_the_wrong_grade(self):
        from execution.submission_queue import claim_next_submission_queue_item, list_submission_candidates
        from persistence.qualified_archive import list_qualified_alpha_archive
        task_ids, client = self._qualified_batch()
        time = FakeTime("2026-09-04T01:00:00+00:00")
        with open_database(self.database_path) as connection:
            good_id = get_backtest_task(connection, task_ids[0]).task.platform_alpha_id
            self.assertEqual([r.task_id for r in list_submission_candidates(connection,
                account_scope="group-account", source="qualified_archive", grade="GOOD")], [task_ids[0]])
            connection.execute("BEGIN IMMEDIATE")
            claim_next_submission_queue_item(connection, account_scope="group-account",
                source="qualified_archive", submission_mode="manual", observed_at=time.now().isoformat(),
                allowed_task_ids=frozenset({task_ids[0]}))
        with self.assertRaisesRegex(ValueError, "submit good"):
            self._submit(client, time, source="qualified_archive", grade="EXCELLENT")
        self.assertEqual(client.calls, [])
        result = self._submit(client, time, source="qualified_archive", grade="GOOD", max_submissions=2)
        self.assertEqual((result.submitted_count, result.remaining_queue_count), (1, 0))
        self.assertEqual(client.posted_ids, [good_id])
        with open_database(self.database_path) as connection:
            self.assertEqual({r.task_id for r in list_qualified_alpha_archive(connection)}, {task_ids[1]})
            self.assertEqual({r.task_id for r in list_formal_submission_queue(connection)}, {task_ids[2]})

    def test_changed_live_grade_is_not_submitted_under_the_requested_grade(self):
        task_ids, client = self._qualified_batch()
        original = client.fetch_alpha_detail
        def changed_detail(*, platform_alpha_id):
            detail = original(platform_alpha_id=platform_alpha_id)
            return replace(detail, payload={**detail.payload, "grade": "EXCELLENT"})
        client.fetch_alpha_detail = changed_detail
        result = self._submit(client, FakeTime("2026-09-04T01:00:00+00:00"),
                              source="qualified_archive", grade="GOOD")
        self.assertEqual((result.ineligible_count, result.submitted_count), (1, 0))
        self.assertEqual(client.posted_ids, [])
        self.assertFalse(any(stage == "check" for stage, _ in client.calls))
        with open_database(self.database_path) as connection:
            self.assertEqual(get_formal_submission_attempt(connection, task_ids[0]).failure_code,
                             "formal_submission_grade_changed:GOOD:EXCELLENT")

    def test_qualified_failed_sc_and_unknown_grade_skip_and_do_not_hold_next_candidate(self):
        task_ids, client = self._qualified_batch(grades=("GOOD", "AVERAGE", "EXCELLENT"))
        with open_database(self.database_path) as connection:
            failed_id, unknown_id, good_id = (get_backtest_task(connection, t).task.platform_alpha_id for t in task_ids)
        client.failed_check_ids = frozenset({failed_id})
        original = client.fetch_alpha_detail
        def detail_with_missing_grade(*, platform_alpha_id):
            detail = original(platform_alpha_id=platform_alpha_id)
            if platform_alpha_id == unknown_id:
                return replace(detail, payload={**detail.payload, "grade": None})
            return detail
        client.fetch_alpha_detail = detail_with_missing_grade
        result = self._submit(client, FakeTime("2026-09-04T01:00:00+00:00"),
                              source="qualified_archive", max_submissions=2)
        self.assertTrue(result.completed)
        self.assertEqual((result.ineligible_count, result.failed_count, result.submitted_count), (1, 1, 1))
        self.assertEqual(client.posted_ids, [good_id])

    def test_qualified_claim_is_atomic_source_is_immutable_and_resume_keeps_policy(self):
        from execution.submission_queue import claim_next_submission_queue_item
        from persistence.qualified_archive import list_qualified_alpha_archive
        from persistence.submissions import replace_formal_submission_attempt
        task_ids, client = self._qualified_batch()
        time = FakeTime("2026-09-04T01:00:00+00:00")
        with open_database(self.database_path) as connection:
            before = list_qualified_alpha_archive(connection)
            connection.execute("BEGIN IMMEDIATE")
            attempt = claim_next_submission_queue_item(connection, account_scope="group-account",
                observed_at=time.now().isoformat(), submission_mode="manual", source="qualified_archive")
            self.assertEqual(len(list_qualified_alpha_archive(connection)), len(before) - 1)
            connection.rollback()
            self.assertEqual(list_qualified_alpha_archive(connection), before)
            self.assertEqual(list_formal_submission_attempts(connection), ())
        advance_submission_queue(self.database_path, client, account_scope="group-account",
            observed_at=time.now().isoformat(), authorized_task_ids=frozenset(task_ids), source="qualified_archive")
        with open_database(self.database_path) as connection:
            attempt = list_formal_submission_attempts(connection)[0]
            with self.assertRaisesRegex(ValueError, "identity_conflict"):
                replace_formal_submission_attempt(connection, replace(attempt, source="queue"),
                    expected_status=attempt.status, expected_updated_at=attempt.updated_at)
        with self.assertRaisesRegex(ValueError, "submit (good|excellent)"):
            self._submit(client, time)
        self.assertEqual(client.calls, [])
        result = self._submit(client, time, source="qualified_archive")
        self.assertEqual(result.submitted_count, 2)
        self.assertEqual(len(client.posted_ids), 2)

    def test_qualified_unknown_post_reconciles_without_a_second_post(self):
        task_ids, client = self._qualified_batch()
        client.unknown_post_ids = frozenset(client.formulas)
        time = FakeTime("2026-09-04T01:00:00+00:00")
        first = self._submit(client, time, source="qualified_archive", max_submissions=1)
        self.assertEqual(first.unresolved_count, 1)
        self.assertEqual(len(client.posted_ids), 1)
        client.unknown_post_ids = frozenset()
        second = self._submit(client, time, source="qualified_archive", max_submissions=1)
        self.assertEqual(second.submitted_count, 1)
        self.assertEqual(len(client.posted_ids), 1)

    def test_claim_consumes_archive_in_same_transaction_and_failed_attempt_is_not_rearchived(self):
        from execution.submission_queue import claim_next_submission_queue_item
        from execution.qualified_archive import synchronize_qualified_alpha_archive
        from persistence.qualified_archive import archive_qualified_alpha, list_qualified_alpha_archive
        from unittest.mock import patch

        tasks, formulas = self._completed_run(("rank(close)",))
        task_id = tasks[0]
        with open_database(self.database_path) as connection:
            archive_qualified_alpha(connection, task_id=task_id, archived_at="2026-09-04T00:30:00+00:00")
        with self.assertRaisesRegex(RuntimeError, "consume failed"):
            with open_database(self.database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                with patch("execution.submission_queue.consume_qualified_alpha_archive", side_effect=RuntimeError("consume failed")):
                    claim_next_submission_queue_item(connection, account_scope="group-account",
                        observed_at="2026-09-04T01:00:00+00:00", submission_mode="manual")
        with open_database(self.database_path) as connection:
            self.assertEqual(len(list_qualified_alpha_archive(connection)), 1)
            self.assertEqual(len(list_formal_submission_queue(connection)), 1)
            self.assertEqual(list_formal_submission_attempts(connection), ())
            connection.execute("BEGIN IMMEDIATE")
            claim_next_submission_queue_item(connection, account_scope="group-account",
                observed_at="2026-09-04T01:00:00+00:00", submission_mode="manual")
            self.assertEqual(list_qualified_alpha_archive(connection), ())
            self.assertEqual(list_formal_submission_queue(connection), ())
        client = SubmissionClient(formulas, failed_check_ids=frozenset(formulas))
        result = self._submit(client, FakeTime("2026-09-04T01:00:01+00:00"))
        self.assertEqual(result.ineligible_count, 1)
        self.assertEqual(client.posted_ids, [])
        with open_database(self.database_path) as connection:
            self.assertEqual(len(list_formal_submission_attempts(connection)), 1)
            # Legacy archive pollution cannot survive even when a result's grade is below target.
            connection.execute("UPDATE backtest_results SET grade='GOOD' WHERE task_id=?", (task_id,))
            archive_qualified_alpha(connection, task_id=task_id, archived_at="2026-09-04T02:00:00+00:00")
            self.assertEqual(synchronize_qualified_alpha_archive(connection, observed_at="2026-09-04T02:00:00+00:00"), ())
            self.assertEqual(list_qualified_alpha_archive(connection), ())

    def test_success_confirmation_consumes_old_archive_membership_without_losing_attempt(self):
        from persistence.qualified_archive import archive_qualified_alpha, list_qualified_alpha_archive
        tasks, formulas = self._completed_run(("rank(close)",))
        client = SubmissionClient(formulas)
        for _ in range(4):
            advance_submission_queue(self.database_path, client, account_scope="group-account",
                observed_at="2026-09-04T01:00:00+00:00", authorized_task_ids=frozenset(tasks))
        with open_database(self.database_path) as connection:
            self.assertEqual(get_formal_submission_attempt(connection, tasks[0]).status, "confirmation_pending")
            archive_qualified_alpha(connection, task_id=tasks[0], archived_at="2026-09-04T01:00:00+00:00")
        self.assertTrue(self._submit(client, FakeTime("2026-09-04T01:00:01+00:00")).completed)
        with open_database(self.database_path) as connection:
            self.assertEqual(list_qualified_alpha_archive(connection), ())
            self.assertEqual(get_formal_submission_attempt(connection, tasks[0]).status, "submitted")

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "database.sqlite3"
        self.environment_path = Path(self.directory.name) / ".env"
        self.environment_path.write_text(
            "WQB_ACCOUNT_SCOPE=group-account\nWQB_EMAIL=test@example.com\n"
            "WQB_PASSWORD=test-only\nWQB_BASE_URL=https://api.worldquantbrain.com\n",
            encoding="utf-8",
        )
        self.settings = BacktestSettings(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
            decay=4,
            neutralization="SUBINDUSTRY",
            truncation=0.08,
            pasteurization="ON",
            unit_handling="VERIFY",
            nan_handling="OFF",
            language="FASTEXPR",
            visualization=False,
            max_trade="OFF",
            max_position="OFF",
        )
        with open_database(self.database_path) as connection:
            initialize_database_schema(connection)
            initialize_test_generation_catalog(connection)

    def test_processes_every_frozen_candidate_without_a_numeric_limit(self) -> None:
        formulas = ("rank(close)", "rank(open)", "rank(volume)")
        _, platform_formulas = self._completed_run(formulas)
        client = SubmissionClient(platform_formulas)

        completion = run_submission_queue(
            self.database_path,
            client,
            account_scope="group-account",
            clock=lambda: datetime.fromisoformat("2026-09-04T01:00:00+00:00"),
            waiter=lambda _seconds: None,
        )

        self.assertTrue(completion.completed)
        self.assertEqual(completion.initial_queue_count, 3)
        self.assertEqual(completion.submission_claimed_count, 3)
        self.assertEqual(completion.submitted_count, 3)
        self.assertEqual(len(client.posted_ids), 3)
        with open_database(self.database_path) as connection:
            self.assertEqual(list_formal_submission_queue(connection), ())
            self.assertEqual(len(list_formal_submission_attempts(connection)), 3)

    def test_submission_limit_keeps_remaining_queue_for_the_next_command(self):
        _, formulas = self._completed_run(
            ("rank(close)", "rank(open)", "rank(volume)", "rank(high)")
        )
        client = SubmissionClient(formulas)
        time = FakeTime("2026-09-04T01:00:00+00:00")
        with open_database(self.database_path) as connection:
            original_queue = list_formal_submission_queue(connection)
        output = io.StringIO()
        with redirect_stdout(output), run_progress_console():
            result = self._submit(client, time, max_submissions=2)
        self.assertTrue(result.completed)
        self.assertTrue(result.submission_limit_reached)
        self.assertEqual((result.submitted_count, result.remaining_queue_count), (2, 2))
        self.assertEqual(len(client.posted_ids), 2)
        self.assertIn("成功提交 2 条后停止", output.getvalue())
        with open_database(self.database_path) as connection:
            remaining = list_formal_submission_queue(connection)
            attempts = list_formal_submission_attempts(connection)
        self.assertEqual(len(attempts), 2)
        self.assertEqual({a.status for a in attempts}, {"submitted"})
        self.assertTrue(all(item in original_queue for item in remaining))
        self.assertTrue({a.task_id for a in attempts}.isdisjoint({q.task_id for q in remaining}))

        resumed = self._submit(client, time, max_submissions=2)
        self.assertTrue(resumed.completed)
        self.assertEqual((resumed.submitted_count, resumed.remaining_queue_count), (2, 0))
        self.assertEqual(len(client.posted_ids), 4)
        self.assertEqual(len(set(client.posted_ids)), 4)

    def test_limit_skips_rejected_checks_and_already_active_imports(self):
        task_ids, formulas = self._completed_run(
            ("rank(close)", "rank(open)", "rank(volume)", "rank(high)", "rank(rank(close))"),
            sharpes=(3.0, 2.5, 2.0, 1.5, 1.3),
        )
        alpha_by_formula = {formula: alpha for alpha, formula in formulas.items()}
        client = SubmissionClient(
            formulas, failed_check_ids=frozenset({alpha_by_formula["rank(close)"]}),
            existing_active_ids=frozenset({alpha_by_formula["rank(open)"]}),
        )
        result = self._submit(client, FakeTime("2026-09-04T01:00:00+00:00"), max_submissions=2)
        self.assertTrue(result.completed)
        self.assertTrue(result.submission_limit_reached)
        self.assertEqual((result.ineligible_count, result.already_active_count), (1, 1))
        self.assertEqual((result.submitted_count, result.submission_claimed_count), (3, 2))
        self.assertEqual(result.remaining_queue_count, 1)
        self.assertEqual(len(client.posted_ids), 2)
        self.assertNotIn(alpha_by_formula["rank(close)"], client.posted_ids)
        self.assertNotIn(alpha_by_formula["rank(open)"], client.posted_ids)
        with open_database(self.database_path) as connection:
            self.assertEqual([q.task_id for q in list_formal_submission_queue(connection)], [task_ids[-1]])
            self.assertIsNone(get_formal_submission_attempt(connection, task_ids[-1]))

    def test_limit_preserves_unknown_and_counts_later_confirmation(self):
        _, formulas = self._completed_run(
            ("rank(close)", "rank(open)", "rank(volume)"),
            sharpes=(3.0, 2.0, 1.5), max_pending_seconds=10,
        )
        unknown_alpha = next(alpha for alpha, formula in formulas.items() if formula == "rank(close)")
        client = SubmissionClient(formulas, unknown_post_ids=frozenset({unknown_alpha}))
        time = FakeTime("2026-09-04T01:00:00+00:00")
        first = self._submit(client, time, max_submissions=2)
        self.assertFalse(first.completed)
        self.assertFalse(first.submission_limit_reached)
        self.assertEqual((first.submitted_count, first.unresolved_count, first.remaining_queue_count), (0, 1, 2))
        self.assertEqual(client.posted_ids, [unknown_alpha])

        resumed_client = SubmissionClient(formulas, existing_active_ids=frozenset({unknown_alpha}))
        second = self._submit(resumed_client, time, max_submissions=2)
        self.assertTrue(second.completed)
        self.assertTrue(second.submission_limit_reached)
        self.assertEqual((second.submitted_count, second.already_active_count, second.remaining_queue_count), (2, 0, 1))
        self.assertEqual(len(resumed_client.posted_ids), 1)
        self.assertNotIn(unknown_alpha, resumed_client.posted_ids)
        self.assertNotIn(("check", unknown_alpha), resumed_client.calls)

        last = self._submit(resumed_client, time, max_submissions=2)
        self.assertTrue(last.completed)
        self.assertFalse(last.submission_limit_reached)
        self.assertEqual((last.submitted_count, last.remaining_queue_count), (1, 0))

    def test_invalid_limit_does_not_authenticate_or_retire_an_interrupted_run(self):
        old, tasks = self._interrupted_run()
        time = FakeTime("2026-09-04T02:00:00+00:00")
        client = self._recovery_client({})
        for value in (0, -1, True, False, 1.5, "2"):
            with self.subTest(value=value):
                with patch("execution.submission_runner.load_worldquant_connection_settings") as load:
                    with self.assertRaisesRegex(ValueError, "提交数量必须为正整数"):
                        self._submit(client, time, max_submissions=value)
                load.assert_not_called()
                with self.assertRaisesRegex(ValueError, "提交数量必须为正整数"):
                    run_submission_queue(self.database_path, client, account_scope="group-account", max_submissions=value)
        self.assertEqual(client.calls, [])
        with open_database(self.database_path) as connection:
            self.assertEqual(get_automated_run(connection, old.run_id).status, "running")
            self.assertEqual(get_backtest_task(connection, tasks[2].task.task_id).task.status, "created")

    def test_console_reports_current_item_and_authoritative_counts(self):
        _, formulas = self._completed_run(("rank(close)", "rank(open)"))
        rejected = next(iter(formulas))
        client = SubmissionClient(formulas, failed_check_ids=frozenset({rejected}))
        output = io.StringIO()
        with redirect_stdout(output), run_progress_console():
            completion = run_submission_queue(self.database_path, client, account_scope="group-account",
                clock=lambda: datetime.fromisoformat("2026-09-04T01:00:00+00:00"), waiter=lambda _: None)
        text = output.getvalue()
        self.assertIn("本次处理 2 条", text)
        self.assertIn("正在处理 1/2", text)
        self.assertIn("正在处理 2/2", text)
        self.assertIn("通过1 已交1 未过1 失败0 待定0", text)
        self.assertEqual(completion.submitted_count, 1)
        self.assertEqual(completion.ineligible_count, 1)
        self.assertEqual(len(client.posted_ids), 1)

    def test_progress_read_failure_does_not_change_submission_requests(self):
        _, formulas = self._completed_run(("rank(close)",))
        client = SubmissionClient(formulas)
        output = io.StringIO()
        with redirect_stdout(output), run_progress_console(), patch(
            "execution.submission_progress.list_formal_submission_attempts", side_effect=OSError("display read failed")):
            completion = run_submission_queue(self.database_path, client, account_scope="group-account",
                clock=lambda: datetime.fromisoformat("2026-09-04T01:00:00+00:00"), waiter=lambda _: None)
        self.assertEqual(completion.submitted_count, 1)
        self.assertEqual(len(client.posted_ids), 1)
        self.assertIn("进度暂不可读", output.getvalue())

    def test_console_counts_only_frozen_scope_and_keeps_unknown_separate(self):
        task_ids, formulas = self._completed_run(("rank(close)", "rank(open)"))
        alpha_id = next(alpha for alpha, formula in formulas.items() if formula == "rank(close)")
        with open_database(self.database_path) as connection:
            chosen = next(task_id for task_id in task_ids if get_backtest_task(connection, task_id).task.formula == "rank(close)")
        client = SubmissionClient(formulas, unknown_post_ids=frozenset({alpha_id}))
        fake_time = FakeTime("2026-09-04T01:00:00+00:00")
        output = io.StringIO()
        with redirect_stdout(output), run_progress_console():
            completion = run_submission_queue(self.database_path, client, account_scope="group-account",
                queued_task_ids=frozenset({chosen}), clock=fake_time.now, waiter=fake_time.wait)
        self.assertIn("本次处理 1 条", output.getvalue())
        self.assertEqual(completion.submitted_count, 0)
        self.assertEqual(completion.failed_count, 0)
        self.assertEqual(completion.unresolved_count, 1)
        self.assertEqual(client.posted_ids, [alpha_id])
        # A final explicit projection is also usable outside an interactive terminal.
        from execution.submission_progress import SubmissionProgress
        with redirect_stdout(output), run_progress_console():
            SubmissionProgress(self.database_path, "group-account", frozenset({chosen})).observe(None)
        self.assertIn("通过1 已交0 未过0 失败0 待定1", output.getvalue())

    def test_claim_atomically_moves_work_from_queue_to_attempt(self) -> None:
        task_ids, platform_formulas = self._completed_run(("rank(close)",))
        client = SubmissionClient(platform_formulas)

        advanced = advance_submission_queue(
            self.database_path,
            client,
            account_scope="group-account",
            observed_at="2026-09-04T01:00:00+00:00",
            authorized_task_ids=frozenset(task_ids),
        )

        self.assertEqual(advanced.action, "formal_submission_prepared")
        self.assertEqual(client.calls, [])
        with open_database(self.database_path) as connection:
            self.assertEqual(list_formal_submission_queue(connection), ())
            attempt = get_formal_submission_attempt(connection, task_ids[0])
        self.assertIsNotNone(attempt)
        assert attempt is not None
        self.assertEqual(attempt.status, "detail_pending")
        self.assertEqual(attempt.submission_mode, "manual")

    def test_unknown_backtest_does_not_block_independent_submissions(self) -> None:
        _, formulas = self._completed_run(("rank(close)", "rank(open)"))
        unknown = self._active_backtest("submission_unknown")
        client = SubmissionClient(formulas)
        completion = run_submission_queue(
            self.database_path,
            client,
            account_scope="group-account",
            clock=lambda: datetime.fromisoformat("2026-09-04T01:00:00+00:00"),
            waiter=lambda _seconds: None,
        )
        self.assertTrue(completion.completed)
        self.assertEqual(completion.submitted_count, 2)
        self.assertCountEqual(client.posted_ids, formulas)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, unknown.task.task_id), unknown)
            self.assertIsNone(get_formal_submission_attempt(connection, unknown.task.task_id))

    def test_created_and_pending_backtests_still_block_before_authentication(self) -> None:
        _, formulas = self._completed_run(("rank(close)",))
        for status in ("created", "pending"):
            with self.subTest(status=status):
                self._active_backtest(status)
                client = SubmissionClient(formulas)
                with self.assertRaisesRegex(ValueError, "formal_submission_active_backtest_exists"):
                    run_submission_queue(self.database_path, client, account_scope="group-account")
                self.assertEqual(client.calls, [])
                with open_database(self.database_path) as connection:
                    self.assertEqual(len(list_formal_submission_queue(connection)), 1)
                    self.assertEqual(list_formal_submission_attempts(connection), ())

    def test_backtest_started_after_checks_still_blocks_formal_post(self) -> None:
        task_ids, formulas = self._completed_run(("rank(close)",))
        owner = self

        class ConcurrentBacktestClient(SubmissionClient):
            def fetch_formal_submission_check(self, *, platform_alpha_id):
                response = super().fetch_formal_submission_check(platform_alpha_id=platform_alpha_id)
                owner._active_backtest("pending")
                return response

        client = ConcurrentBacktestClient(formulas)
        with self.assertRaisesRegex(ValueError, "formal_submission_active_backtest_exists"):
            run_submission_queue(
                self.database_path,
                client,
                account_scope="group-account",
                clock=lambda: datetime.fromisoformat("2026-09-04T01:00:00+00:00"),
                waiter=lambda _seconds: None,
            )
        self.assertEqual(client.posted_ids, [])
        with open_database(self.database_path) as connection:
            attempt = get_formal_submission_attempt(connection, task_ids[0])
            self.assertEqual(attempt.status, "ready")
            self.assertIsNone(attempt.submission_claimed_at)

    def _active_backtest(self, status):
        snapshot = prepare_real_backtest(
            self.database_path,
            account_scope="group-account",
            formula="-rank(volume)",
            settings=self.settings,
            created_at="2026-09-04T00:30:00+00:00",
        )
        with open_database(self.database_path) as connection:
            if status == "submission_unknown":
                return record_submission_unknown(
                    connection, snapshot.task.task_id, observed_at="2026-09-04T00:31:00+00:00"
                )
            if status == "pending":
                return record_submission_accepted(
                    connection,
                    snapshot.task.task_id,
                    remote_id="independent-simulation",
                    observed_at="2026-09-04T00:31:00+00:00",
                )
        return snapshot

    def test_empty_yearly_result_never_enters_submission_queue(self) -> None:
        self._completed_run(
            ("rank(close)",),
            include_yearly_stats=False,
        )

        with open_database(self.database_path) as connection:
            self.assertEqual(list_formal_submission_queue(connection), ())

    def test_same_family_is_submitted_from_low_sharpe_to_high_sharpe(self) -> None:
        formulas = ("rank(close)", "rank(open)")
        task_ids, platform_formulas = self._completed_run(
            formulas,
            sharpes=(2.0, 1.3),
            lineage=(1, 0),
        )
        client = SubmissionClient(platform_formulas)

        completion = run_submission_queue(
            self.database_path,
            client,
            account_scope="group-account",
            clock=lambda: datetime.fromisoformat("2026-09-04T01:00:00+00:00"),
            waiter=lambda _seconds: None,
        )

        self.assertTrue(completion.completed)
        self.assertEqual(
            tuple(platform_formulas[alpha_id] for alpha_id in client.posted_ids),
            ("rank(open)", "rank(close)"),
        )
        with open_database(self.database_path) as connection:
            attempts = list_formal_submission_attempts(connection)
        self.assertEqual({attempt.task_id for attempt in attempts}, set(task_ids))
        report = self.database_path.with_name("submitted-formulas.md").read_text(encoding="utf-8")
        self.assertIn("共 **2** 条", report)
        for alpha, formula in platform_formulas.items():
            self.assertIn(f"| {alpha} |", report)
            self.assertNotIn(formula, report)

    def test_existing_active_alpha_is_imported_without_posting_again(self) -> None:
        _, platform_formulas = self._completed_run(("rank(close)",))
        [platform_alpha_id] = platform_formulas
        client = SubmissionClient(
            platform_formulas,
            existing_active_ids=frozenset({platform_alpha_id}),
        )

        completion = run_submission_queue(
            self.database_path,
            client,
            account_scope="group-account",
            clock=lambda: datetime.fromisoformat("2026-09-04T01:00:00+00:00"),
            waiter=lambda _seconds: None,
        )

        self.assertTrue(completion.completed)
        self.assertEqual(completion.submitted_count, 1)
        self.assertEqual(completion.already_active_count, 1)
        self.assertEqual(completion.submission_claimed_count, 0)
        self.assertNotIn("post", {name for name, _value in client.calls})

        report = self.database_path.with_name("submitted-formulas.md").read_text(encoding="utf-8")
        self.assertIn(f"| {platform_alpha_id} |", report)

    def test_document_write_failure_does_not_undo_or_repeat_a_submission(self):
        _, formulas = self._completed_run(("rank(close)",))
        client = SubmissionClient(formulas)
        time = FakeTime("2026-09-04T01:00:00+00:00")
        with patch("execution.submitted_formulas.export_submitted_formulas", side_effect=PermissionError("locked")):
            with self.assertLogs("execution.progress", level="WARNING"):
                completion = run_submission_queue(
                    self.database_path, client, account_scope="group-account",
                    clock=time.now, waiter=time.wait,
                )
        self.assertTrue(completion.completed)
        self.assertEqual(completion.submitted_count, 1)
        with open_database(self.database_path) as connection:
            self.assertEqual(list_formal_submission_attempts(connection)[0].status, "submitted")
        run_submission_queue(self.database_path, client, account_scope="group-account",
                             clock=time.now, waiter=time.wait)
        self.assertEqual(client.posted_ids, list(formulas))

    def test_detail_formula_mismatch_is_consumed_without_posting(self) -> None:
        _, platform_formulas = self._completed_run(("rank(close)",))
        [platform_alpha_id] = platform_formulas
        client = SubmissionClient(
            {platform_alpha_id: "rank(open)"},
            existing_active_ids=frozenset({platform_alpha_id}),
        )

        completion = run_submission_queue(
            self.database_path,
            client,
            account_scope="group-account",
            clock=lambda: datetime.fromisoformat("2026-09-04T01:00:00+00:00"),
            waiter=lambda _seconds: None,
        )

        self.assertTrue(completion.completed)
        self.assertEqual(completion.failed_count, 1)
        self.assertEqual(completion.submission_claimed_count, 0)
        self.assertNotIn("post", {name for name, _value in client.calls})

    def test_failed_check_is_consumed_and_does_not_block_the_next_candidate(
        self,
    ) -> None:
        _, platform_formulas = self._completed_run(
            ("rank(close)", "rank(open)"),
            sharpes=(2.0, 1.3),
        )
        failed_alpha_id = next(
            alpha_id
            for alpha_id, formula in platform_formulas.items()
            if formula == "rank(close)"
        )
        client = SubmissionClient(
            platform_formulas,
            failed_check_ids=frozenset({failed_alpha_id}),
        )

        completion = run_submission_queue(
            self.database_path,
            client,
            account_scope="group-account",
            clock=lambda: datetime.fromisoformat("2026-09-04T01:00:00+00:00"),
            waiter=lambda _seconds: None,
        )

        self.assertTrue(completion.completed)
        self.assertEqual(completion.ineligible_count, 1)
        self.assertEqual(completion.submitted_count, 1)
        self.assertEqual(completion.remaining_queue_count, 0)

    def test_async_submission_rejection_is_recorded_and_next_candidate_continues(self):
        _, formulas = self._completed_run(("rank(close)", "rank(open)", "rank(volume)"))
        payload = {"is": {"checks": [
            {"name": "SELF_CORRELATION", "result": "FAIL", "value": 0.7324},
        ]}}
        class RejectedClient(SubmissionClient):
            rejected_alpha = None

            def submit_formal_alpha(self, *, platform_alpha_id):
                result = super().submit_formal_alpha(platform_alpha_id=platform_alpha_id)
                if self.rejected_alpha is None:
                    self.rejected_alpha = platform_alpha_id
                    self.unknown_post_ids = frozenset({platform_alpha_id})
                return result

            def fetch_formal_submission_result(self, *, platform_alpha_id):
                assert platform_alpha_id == self.rejected_alpha
                return FormalSubmissionObservation(403, payload)

        client = RejectedClient(formulas)
        time = FakeTime("2026-09-04T01:00:00+00:00")
        result = run_submission_queue(self.database_path, client, account_scope="group-account",
                                      clock=time.now, waiter=time.wait, max_submissions=1)
        self.assertTrue(result.completed)
        self.assertTrue(result.submission_limit_reached)
        self.assertEqual(result.remaining_queue_count, 1)
        self.assertEqual((result.failed_count, result.submitted_count), (1, 1))
        self.assertEqual(len(client.posted_ids), 2)
        self.assertEqual(len(set(client.posted_ids)), 2)
        with open_database(self.database_path) as connection:
            attempts = list_formal_submission_attempts(connection)
        rejected = next(a for a in attempts if a.status == "failed")
        self.assertEqual(rejected.failure_code, "formal_submission_rejected:SELF_CORRELATION")
        evidence = json.loads(rejected.submit_response_json)
        self.assertEqual(evidence["status_code"], 201)
        self.assertEqual(evidence["confirmation"]["status_code"], 403)
        self.assertEqual(evidence["confirmation"]["payload"], payload)

    def test_unresolved_submission_result_does_not_become_rejection_or_success(self):
        _, formulas = self._completed_run(("rank(close)",), max_pending_seconds=10)
        alpha = next(iter(formulas))
        outcomes = [
            FormalSubmissionObservation(403, {"detail": "Forbidden"}),
            FormalSubmissionObservation(200, None, 4),
            FormalSubmissionObservation(403, {"is": {"checks": [
                {"name": "SELF_CORRELATION", "result": "FAIL"},
            ]}}, 4),
            WorldQuantRequestError("result_unavailable", status_code=503,
                                  retryable=True, outcome_unknown=False),
        ]
        class PendingClient(SubmissionClient):
            def submit_formal_alpha(self, *, platform_alpha_id):
                result = super().submit_formal_alpha(platform_alpha_id=platform_alpha_id)
                self.unknown_post_ids = frozenset({platform_alpha_id})
                return result

            def fetch_formal_submission_result(self, *, platform_alpha_id):
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        client = PendingClient(formulas)
        time = FakeTime("2026-09-04T01:00:00+00:00")
        for outcome in outcomes:
            with self.subTest(outcome=outcome):
                result = run_submission_queue(self.database_path, client,
                    account_scope="group-account", clock=time.now, waiter=time.wait)
                self.assertFalse(result.completed)
                self.assertEqual((result.unresolved_count, result.failed_count, result.submitted_count),
                                 (1, 0, 0))
                self.assertEqual(client.posted_ids, [alpha])
                time.wait(10)

        outcome = FormalSubmissionObservation(403, {"is": {"checks": [
            {"name": "SELF_CORRELATION", "result": "FAIL"},
        ]}})
        resumed = run_submission_queue(self.database_path, client, account_scope="group-account",
                                       clock=time.now, waiter=time.wait)
        self.assertTrue(resumed.completed)
        self.assertEqual(resumed.failed_count, 1)
        self.assertEqual(client.posted_ids, [alpha])

    def test_later_command_reconciles_unknown_post_without_posting_twice(
        self,
    ) -> None:
        _, platform_formulas = self._completed_run(
            ("rank(close)",),
            max_pending_seconds=10,
        )
        [platform_alpha_id] = platform_formulas
        client = SubmissionClient(
            platform_formulas,
            unknown_post_ids=frozenset({platform_alpha_id}),
        )
        fake_time = FakeTime("2026-09-04T01:00:00+00:00")

        completion = run_submission_queue(
            self.database_path,
            client,
            account_scope="group-account",
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertFalse(completion.completed)
        self.assertEqual(completion.unresolved_count, 1)
        self.assertEqual(completion.remaining_queue_count, 0)
        self.assertEqual(client.posted_ids, [platform_alpha_id])
        self.assertFalse(self.database_path.with_name("submitted-formulas.md").exists())

        client.unknown_post_ids = frozenset()
        reconciled = run_submission_queue(
            self.database_path,
            client,
            account_scope="group-account",
            clock=fake_time.now,
            waiter=fake_time.wait,
        )

        self.assertTrue(reconciled.completed)
        self.assertEqual(reconciled.submitted_count, 1)
        self.assertEqual(client.posted_ids, [platform_alpha_id])
        report = self.database_path.with_name("submitted-formulas.md").read_text(encoding="utf-8")
        self.assertIn("共 **1** 条", report)
        self.assertIn(f"| {platform_alpha_id} |", report)
        self.assertEqual(
            sum(name == "post" for name, _value in client.calls),
            1,
        )

    def test_existing_submitted_family_member_keeps_lower_candidate_eligible(
        self,
    ) -> None:
        task_ids, platform_formulas = self._completed_run(
            ("rank(close)", "rank(open)"),
            sharpes=(2.0, 1.3),
            lineage=(1, 0),
        )
        high_alpha_id = next(
            alpha_id
            for alpha_id, formula in platform_formulas.items()
            if formula == "rank(close)"
        )
        self._record_submitted_alpha(
            platform_alpha_id=high_alpha_id,
            formula="rank(close)",
        )
        client = SubmissionClient(platform_formulas)

        completion = run_submission_queue(
            self.database_path,
            client,
            account_scope="group-account",
            clock=lambda: datetime.fromisoformat("2026-09-04T01:00:00+00:00"),
            waiter=lambda _seconds: None,
        )

        self.assertTrue(completion.completed)
        self.assertEqual(completion.initial_queue_count, 2)
        self.assertEqual(completion.submission_claimed_count, 1)
        self.assertEqual(
            tuple(platform_formulas[alpha_id] for alpha_id in client.posted_ids),
            ("rank(open)",),
        )
        [lower_alpha_id] = [
            alpha_id
            for alpha_id, formula in platform_formulas.items()
            if formula == "rank(open)"
        ]
        lower_candidate_stages = tuple(
            name for name, alpha_id in client.calls if alpha_id == lower_alpha_id
        )
        self.assertLess(
            lower_candidate_stages.index("detail"),
            lower_candidate_stages.index("check"),
        )
        self.assertLess(
            lower_candidate_stages.index("check"),
            lower_candidate_stages.index("post"),
        )
        with open_database(self.database_path) as connection:
            self.assertEqual(list_formal_submission_queue(connection), ())
            attempts = list_formal_submission_attempts(connection)
        self.assertEqual({attempt.task_id for attempt in attempts}, {task_ids[1]})

    def test_ambiguous_submitted_formula_does_not_block_another_candidate(
        self,
    ) -> None:
        alternate_settings = replace(self.settings, neutralization="INDUSTRY")
        _, platform_formulas = self._completed_run(
            ("rank(close)", "rank(close)", "rank(open)"),
            sharpes=(2.0, 1.9, 1.3),
            settings=(self.settings, alternate_settings, self.settings),
        )
        self._record_submitted_alpha(
            platform_alpha_id="external-alpha",
            formula="rank(close)",
        )
        client = SubmissionClient(platform_formulas)

        completion = run_submission_queue(
            self.database_path,
            client,
            account_scope="group-account",
            clock=lambda: datetime.fromisoformat("2026-09-04T01:00:00+00:00"),
            waiter=lambda _seconds: None,
        )

        self.assertTrue(completion.completed)
        self.assertEqual(completion.submission_claimed_count, 1)
        self.assertEqual(
            tuple(platform_formulas[alpha_id] for alpha_id in client.posted_ids),
            ("rank(open)",),
        )
        with open_database(self.database_path) as connection:
            self.assertEqual(list_formal_submission_queue(connection), ())
            self.assertEqual(len(list_formal_submission_attempts(connection)), 1)

    def test_active_automated_run_blocks_standalone_submission_before_login(
        self,
    ) -> None:
        prepare_automated_run(
            self.database_path,
            POLICY_PATH,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=3,
                backtest_count=1,
                max_cycles=1,
                max_backtests=1,
                max_pending_seconds=60,
                max_consecutive_failures=2,
                max_request_failures=3,
                max_in_flight_backtests=1,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ),
            created_at="2026-09-04T00:00:00+00:00",
        )
        client = SubmissionClient({})

        with self.assertRaisesRegex(
            ValueError,
            "formal_submission_active_run_exists",
        ):
            run_submission_queue(
                self.database_path,
                client,
                account_scope="group-account",
            )

        self.assertFalse(client.authenticated)
        self.assertEqual(client.calls, [])

    def test_submit_collects_interrupted_run_without_new_backtests_or_late_submissions(self):
        _, formulas = self._completed_run(("rank(close)",))
        old, tasks = self._interrupted_run()
        client = self._recovery_client(formulas)
        time = FakeTime("2026-09-04T02:00:00+00:00")
        with open_database(self.database_path) as connection:
            unknown = get_backtest_task(connection, tasks[1].task.task_id)

        result = self._submit(client, time, max_submissions=2)

        self.assertTrue(result.completed)
        self.assertEqual(result.initial_queue_count, 1)
        self.assertEqual(result.submitted_count, 1)
        self.assertTrue(result.authentication_performed)
        self.assertEqual(client.posted_ids, ["alpha-1"])
        self.assertEqual(sum(name == "authenticate" for name, _ in client.calls), 1)
        self.assertLess(client.calls.index(("yearly", "late-alpha")),
                        client.calls.index(("post", "alpha-1")))
        with open_database(self.database_path) as connection:
            stopped = get_automated_run(connection, old.run_id)
            self.assertEqual(stopped.status, "failed")
            self.assertEqual(stopped.stop_reason, "stopped_for_manual_submission")
            self.assertEqual(get_backtest_task(connection, tasks[0].task.task_id).task.status,
                             "completed")
            self.assertEqual(get_backtest_task(connection, tasks[1].task.task_id), unknown)
            cancelled = get_backtest_task(connection, tasks[2].task.task_id)
            self.assertEqual(cancelled.task.status, "failed")
            self.assertIsNone(cancelled.task.submission_started_at)
            self.assertEqual([q.task_id for q in list_formal_submission_queue(connection)],
                             [tasks[0].task.task_id])
            self.assertEqual(len(list_automated_run_backtests(connection, old.run_id)), 3)

    def test_submit_collection_expires_old_wait_and_processes_queue(self):
        _, formulas = self._completed_run(("rank(close)",))
        old, tasks = self._interrupted_run()
        client = self._recovery_client(formulas, pending=True)
        time = FakeTime("2026-09-04T02:00:00+00:00")

        result = self._submit(client, time)
        self.assertTrue(result.completed)
        self.assertEqual(len(client.posted_ids), 1)
        with open_database(self.database_path) as connection:
            pending = get_backtest_task(connection, tasks[0].task.task_id)
            self.assertEqual(pending.task.status, "failed")
            self.assertEqual(pending.task.failure_code, "platform_pending_timeout")
            self.assertEqual(pending.task.remote_id, "simulation-old")
            self.assertIsNone(pending.result)
            self.assertEqual(len(list_formal_submission_queue(connection)), 0)
        # Re-entering does not resubmit the old backtest or the submitted Alpha.
        resumed_client = self._recovery_client(formulas)
        resumed = self._submit(resumed_client, time)
        self.assertTrue(resumed.completed)
        self.assertEqual(resumed_client.posted_ids, [])

    def test_submit_lock_and_invalid_inputs_do_not_retire_old_run(self):
        old, tasks = self._interrupted_run()
        client = self._recovery_client({})
        time = FakeTime("2026-09-04T02:00:00+00:00")
        with exclusive_run_process(self.database_path):
            with self.assertRaisesRegex(ValueError, "已有命令正在使用"):
                self._submit(client, time)
        with self.assertRaisesRegex(ValueError, "poll_interval_invalid"):
            self._submit(client, time, poll_interval_seconds=0)
        with self.assertRaisesRegex(ValueError, "clock_invalid"):
            self._submit(client, FakeTime("2026-09-04T02:00:00"))
        self.assertEqual(client.calls, [])
        with open_database(self.database_path) as connection:
            self.assertEqual(get_automated_run(connection, old.run_id).status, "running")
            self.assertEqual(get_backtest_task(connection, tasks[2].task.task_id).task.status,
                             "created")

    def test_submit_still_reconciles_unknown_manual_post_without_reposting(self):
        _, formulas = self._completed_run(("rank(close)",))
        time = FakeTime("2026-09-04T01:00:00+00:00")
        first = self._submit(SubmissionClient(formulas, unknown_post_ids=frozenset({"alpha-1"})), time)
        self.assertEqual(first.unresolved_count, 1)
        second_client = SubmissionClient(formulas, existing_active_ids=frozenset({"alpha-1"}))
        second = self._submit(second_client, time)
        self.assertTrue(second.completed)
        self.assertEqual(second_client.posted_ids, [])
        self.assertNotIn("check", {name for name, _ in second_client.calls})

    def test_submit_keeps_unowned_pending_backtest_blocking_before_login(self):
        _, formulas = self._completed_run(("rank(close)",))
        pending = self._active_backtest("pending")
        client = SubmissionClient(formulas)
        with self.assertRaisesRegex(ValueError, "formal_submission_active_backtest_exists"):
            self._submit(client, FakeTime("2026-09-04T02:00:00+00:00"))
        self.assertEqual(client.calls, [])
        with open_database(self.database_path) as connection:
            self.assertEqual(get_backtest_task(connection, pending.task.task_id), pending)
            self.assertEqual(len(list_formal_submission_queue(connection)), 1)

    def test_submit_wrong_account_does_not_stop_interrupted_run(self):
        old, tasks = self._interrupted_run()
        self.environment_path.write_text(
            "WQB_ACCOUNT_SCOPE=other-account\nWQB_EMAIL=test@example.com\n"
            "WQB_PASSWORD=test-only\nWQB_BASE_URL=https://api.worldquantbrain.com\n", encoding="utf-8",
        )
        client = SubmissionClient({})
        with self.assertRaisesRegex(ValueError, "automated_run_account_scope_mismatch"):
            self._submit(client, FakeTime("2026-09-04T02:00:00+00:00"))
        self.assertEqual(client.calls, [])
        with open_database(self.database_path) as connection:
            self.assertEqual(get_automated_run(connection, old.run_id).status, "running")
            self.assertEqual(get_backtest_task(connection, tasks[2].task.task_id).task.status,
                             "created")

    def test_expired_post_login_is_renewed_before_reclaiming_manual_submission(self):
        _, formulas = self._completed_run(("rank(close)",))
        class ExpiringClient(SubmissionClient):
            rejected_once = False
            def submit_formal_alpha(self, *, platform_alpha_id):
                if not self.rejected_once:
                    self.rejected_once = True
                    self.authenticated = False
                    raise WorldQuantRequestError("worldquant_authentication_expired",
                        status_code=401, retryable=True, outcome_unknown=False)
                return super().submit_formal_alpha(platform_alpha_id=platform_alpha_id)
        client = ExpiringClient(formulas)
        result = self._submit(client, FakeTime("2026-09-04T01:00:00+00:00"))
        self.assertTrue(result.completed)
        self.assertEqual(client.posted_ids, ["alpha-1"])
        self.assertEqual(sum(name == "authenticate" for name, _ in client.calls), 2)

    def test_transient_detail_and_check_reads_retry_without_duplicate_posts(self):
        _, formulas = self._completed_run(("rank(close)",))
        client = self._flaky_read_client(formulas, detail_failures=1, check_failures=1,
                                         retry_after=4)
        time = FakeTime("2026-09-04T01:00:00+00:00")
        result = self._submit(client, time)
        self.assertTrue(result.completed)
        self.assertEqual(client.posted_ids, ["alpha-1"])
        self.assertEqual(client.read_failures, 2)
        self.assertGreaterEqual((time.now() - datetime.fromisoformat(
            "2026-09-04T01:00:00+00:00")).total_seconds(), 8)
        self.assertEqual(result.platform_request_count,
                         len([call for call in client.calls if call[0] != "authenticate"]))

    def test_detail_read_failure_limit_releases_slot_for_next_candidate(self):
        _, formulas = self._completed_run(("rank(close)", "rank(open)"))
        client = self._flaky_read_client(formulas, detail_failures=3)
        time = FakeTime("2026-09-04T01:00:00+00:00")
        result = self._submit(client, time)
        self.assertTrue(result.completed)
        self.assertEqual((result.failed_count, result.submitted_count), (1, 1))
        self.assertEqual(client.read_failures, 3)
        self.assertEqual(len(client.posted_ids), 1)
        with open_database(self.database_path) as connection:
            attempt = next(a for a in list_formal_submission_attempts(connection)
                           if a.status == "failed")
            self.assertEqual(attempt.failure_code,
                             "formal_submission_read_failed:worldquant_alpha_detail_request_failed")
            self.assertIsNone(attempt.submission_claimed_at)
        time.wait(3600)
        recovered = SubmissionClient(formulas)
        self.assertTrue(self._submit(recovered, time).completed)
        self.assertEqual(recovered.posted_ids, [])

    def test_nonretryable_or_unknown_reads_stop_without_auto_retry(self):
        _, formulas = self._completed_run(("rank(close)",))
        for retryable, unknown in ((False, False), (True, True)):
            with self.subTest(retryable=retryable, unknown=unknown):
                client = self._flaky_read_client(formulas, detail_failures=10,
                                                 retryable=retryable, unknown=unknown)
                with self.assertRaises(WorldQuantRequestError):
                    self._submit(client, FakeTime("2026-09-04T01:00:00+00:00"))
                self.assertEqual(client.read_failures, 1)
                self.assertEqual(client.posted_ids, [])

    def test_check_wait_timeout_preserves_evidence_and_continues_next_candidate(self):
        _, formulas = self._completed_run(("rank(close)", "rank(open)"), max_pending_seconds=10)
        time = FakeTime("2026-09-04T01:00:00+00:00")

        class PendingCheckClient(SubmissionClient):
            blocked_id = None

            def fetch_formal_submission_check(self, *, platform_alpha_id):
                normal = super().fetch_formal_submission_check(platform_alpha_id=platform_alpha_id)
                if self.blocked_id is None:
                    self.blocked_id = platform_alpha_id
                if platform_alpha_id != self.blocked_id:
                    return normal
                return FormalCheckObservation(payload={"is": {"checks": [
                    {"name": name, "result": "PENDING" if name == "SELF_CORRELATION" else "PASS"}
                    for name in sorted(STANDARD_REGULAR_CHECK_NAMES)
                ]}}, retry_after_seconds=600)

        client = PendingCheckClient(formulas)
        output = io.StringIO()
        with redirect_stdout(output), run_progress_console():
            result = self._submit(client, time)
        self.assertTrue(result.completed)
        self.assertEqual((result.failed_count, result.submitted_count, result.ineligible_count), (1, 1, 0))
        self.assertNotIn(client.blocked_id, client.posted_ids)
        self.assertEqual(len(client.posted_ids), 1)
        self.assertIn("提交前等待超时，已跳过", output.getvalue())
        self.assertLess((time.now() - datetime.fromisoformat("2026-09-04T01:00:00+00:00")).total_seconds(), 30)
        with open_database(self.database_path) as connection:
            attempt = next(a for a in list_formal_submission_attempts(connection)
                           if a.status == "failed")
            self.assertEqual(attempt.failure_code, "formal_submission_check_timeout")
            self.assertIsNone(attempt.submission_claimed_at)
            checks = json.loads(attempt.check_payload_json)["is"]["checks"]
            self.assertEqual(next(c["result"] for c in checks if c["name"] == "SELF_CORRELATION"), "PENDING")
        time.wait(3600)
        self.assertTrue(self._submit(SubmissionClient(formulas), time).completed)

    def test_resuming_expired_pending_check_does_not_restart_its_wait(self):
        _, formulas = self._completed_run(("rank(close)", "rank(open)"), max_pending_seconds=10)
        database_path = self.database_path

        class InterruptAfterPending(FakeTime):
            def wait(self, seconds):
                super().wait(seconds)
                with open_database(database_path) as connection:
                    attempts = list_formal_submission_attempts(connection)
                    if any(a.status == "check_pending" and a.check_attempt_count for a in attempts):
                        raise KeyboardInterrupt

        class PendingClient(SubmissionClient):
            def fetch_formal_submission_check(self, *, platform_alpha_id):
                super().fetch_formal_submission_check(platform_alpha_id=platform_alpha_id)
                return FormalCheckObservation(payload=None, retry_after_seconds=1)

        time = InterruptAfterPending("2026-09-04T01:00:00+00:00")
        first = PendingClient(formulas)
        with self.assertRaises(KeyboardInterrupt):
            self._submit(first, time)
        blocked = next(alpha_id for action, alpha_id in first.calls if action == "check")
        later = FakeTime((time.now() + timedelta(hours=1)).isoformat())
        second = SubmissionClient(formulas)
        result = self._submit(second, later)
        self.assertTrue(result.completed)
        self.assertEqual((result.failed_count, result.submitted_count), (1, 1))
        self.assertFalse(any(alpha_id == blocked for _, alpha_id in second.calls))

    def test_missing_alpha_is_skipped_and_next_candidate_is_submitted(self):
        _, formulas = self._completed_run(("rank(close)", "rank(open)"))

        class MissingClient(SubmissionClient):
            missing_id = None

            def fetch_alpha_detail(self, *, platform_alpha_id):
                if self.missing_id is None:
                    self.missing_id = platform_alpha_id
                if platform_alpha_id == self.missing_id:
                    self.calls.append(("missing", platform_alpha_id))
                    raise WorldQuantRequestError("worldquant_alpha_detail_http_error",
                                                 status_code=404, retryable=False, outcome_unknown=False)
                return super().fetch_alpha_detail(platform_alpha_id=platform_alpha_id)

        client = MissingClient(formulas)
        result = self._submit(client, FakeTime("2026-09-04T01:00:00+00:00"))
        self.assertTrue(result.completed)
        self.assertEqual((result.failed_count, result.submitted_count), (1, 1))
        self.assertNotIn(client.missing_id, client.posted_ids)
        self.assertEqual(client.calls.count(("missing", client.missing_id)), 1)

    def test_latest_detail_grade_is_rechecked_and_only_that_candidate_is_skipped(self):
        for grade in ("EXCELLENT", None, "FUTURE_GRADE"):
            with self.subTest(grade=grade):
                source = SubmissionQueueRunnerTests()
                source.setUp()
                self.addCleanup(source.doCleanups)
                _, formulas = source._completed_run(("rank(close)", "rank(open)"))

                class ChangedGradeClient(SubmissionClient):
                    blocked = None

                    def fetch_alpha_detail(self, *, platform_alpha_id):
                        observation = super().fetch_alpha_detail(platform_alpha_id=platform_alpha_id)
                        if self.blocked is None:
                            self.blocked = platform_alpha_id
                        if platform_alpha_id == self.blocked:
                            payload = dict(observation.payload)
                            payload["grade"] = grade
                            return AlphaDetailObservation(payload)
                        return observation

                client = ChangedGradeClient(formulas)
                with source.assertLogs("execution.progress", level="INFO") as captured:
                    result = source._submit(client, FakeTime("2026-09-04T01:00:00+00:00"), max_submissions=1)
                self.assertTrue(result.completed)
                self.assertEqual(result.submitted_count, 1)
                self.assertEqual(result.failed_count + result.ineligible_count, 1)
                self.assertNotIn(client.blocked, client.posted_ids)
                self.assertNotIn(("check", client.blocked), client.calls)
                self.assertIn("评级", " ".join(captured.output))
                with open_database(source.database_path) as connection:
                    rejected = next(a for a in list_formal_submission_attempts(connection)
                                    if a.status != "submitted")
                    self.assertEqual(rejected.status, "ineligible" if grade == "EXCELLENT" else "failed")
                    self.assertEqual(rejected.check_attempt_count, 0)
                    self.assertIsNone(rejected.check_payload_json)
                    self.assertIsNone(rejected.submission_claimed_at)

    def test_old_queue_and_pre_post_attempts_cannot_bypass_grade_gate(self):
        for steps in (0, 1, 2, 3):  # queued, detail_pending, check_pending, ready
            with self.subTest(steps=steps):
                source = SubmissionQueueRunnerTests()
                source.setUp()
                self.addCleanup(source.doCleanups)
                tasks, formulas = source._completed_run(("rank(close)", "rank(open)"), sharpes=(2, 1.3))
                client = SubmissionClient(formulas)
                for _ in range(steps):
                    advance_submission_queue(source.database_path, client, account_scope="group-account",
                        observed_at="2026-09-04T01:00:00+00:00", authorized_task_ids=frozenset(tasks))
                with open_database(source.database_path) as connection:
                    attempts = list_formal_submission_attempts(connection)
                    old_check = attempts[0].check_payload_json if attempts else None
                    blocked = get_backtest_task(connection, tasks[0]).task.platform_alpha_id
                    connection.execute("UPDATE backtest_results SET grade=NULL WHERE task_id=?", (tasks[0],))
                result = source._submit(client, FakeTime("2026-09-04T01:00:01+00:00"))
                self.assertTrue(result.completed)
                self.assertEqual(result.submitted_count, 1)
                self.assertNotIn(blocked, client.posted_ids)
                with open_database(source.database_path) as connection:
                    self.assertEqual(list_formal_submission_queue(connection), ())
                    if steps:
                        attempt = get_formal_submission_attempt(connection, tasks[0])
                        self.assertEqual(attempt.failure_code, "formal_submission_grade_unavailable")
                        self.assertEqual(attempt.check_payload_json, old_check)

    def test_missing_historical_grade_does_not_abandon_an_unknown_post(self):
        tasks, formulas = self._completed_run(("rank(close)",), max_pending_seconds=10)
        [alpha] = formulas
        client = SubmissionClient(formulas, unknown_post_ids=frozenset({alpha}))
        time = FakeTime("2026-09-04T01:00:00+00:00")
        self.assertEqual(self._submit(client, time).unresolved_count, 1)
        with open_database(self.database_path) as connection:
            connection.execute("UPDATE backtest_results SET grade=NULL WHERE task_id=?", (tasks[0],))
        client.unknown_post_ids = frozenset()
        result = self._submit(client, time)
        self.assertTrue(result.completed)
        self.assertEqual(result.submitted_count, 1)
        self.assertEqual(client.posted_ids, [alpha])

    def test_account_permission_error_keeps_candidate_and_stops_posting(self):
        _, formulas = self._completed_run(("rank(close)", "rank(open)"))

        class DeniedClient(SubmissionClient):
            def fetch_alpha_detail(self, *, platform_alpha_id):
                raise WorldQuantRequestError("worldquant_alpha_detail_http_error",
                                             status_code=403, retryable=False, outcome_unknown=False)

        client = DeniedClient(formulas)
        with self.assertRaises(WorldQuantRequestError):
            self._submit(client, FakeTime("2026-09-04T01:00:00+00:00"))
        self.assertEqual(client.posted_ids, [])
        with open_database(self.database_path) as connection:
            attempts = list_formal_submission_attempts(connection)
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0].status, "detail_pending")
            self.assertEqual(len(list_formal_submission_queue(connection)), 1)

    def test_resuming_old_ready_candidate_refreshes_checks_before_posting(self):
        task_ids, formulas = self._completed_run(("rank(close)",), max_pending_seconds=600)
        database_path = self.database_path

        class InterruptAfterCheck(FakeTime):
            def wait(self, seconds):
                super().wait(seconds)
                with open_database(database_path) as connection:
                    attempt = get_formal_submission_attempt(connection, task_ids[0])
                    if attempt is not None and attempt.status == "ready":
                        raise KeyboardInterrupt

        time = InterruptAfterCheck("2026-09-04T01:00:00+00:00")
        first = SubmissionClient(formulas)
        with self.assertRaises(KeyboardInterrupt):
            self._submit(first, time)
        self.assertEqual(first.posted_ids, [])
        later = FakeTime((time.now() + timedelta(seconds=400)).isoformat())
        second = SubmissionClient(formulas)
        self.assertTrue(self._submit(second, later).completed)
        self.assertLess(second.calls.index(("check", "alpha-1")),
                        second.calls.index(("post", "alpha-1")))
        self.assertEqual(second.posted_ids, ["alpha-1"])

    def test_check_read_errors_release_slot_for_next_candidate(self):
        _, formulas = self._completed_run(("rank(close)", "rank(open)"))
        client = self._flaky_read_client(formulas, check_failures=3)
        time = FakeTime("2026-09-04T01:00:00+00:00")
        result = self._submit(client, time)
        self.assertTrue(result.completed)
        self.assertEqual((result.failed_count, result.submitted_count), (1, 1))
        self.assertEqual(client.read_failures, 3)
        self.assertEqual(len(client.posted_ids), 1)
        with open_database(self.database_path) as connection:
            attempt = next(a for a in list_formal_submission_attempts(connection)
                           if a.status == "failed")
            self.assertEqual(attempt.failure_code,
                             "formal_submission_read_failed:worldquant_formal_check_request_failed")
            self.assertEqual(attempt.check_attempt_count, 0)
            self.assertIsNone(attempt.check_payload_json)
        time.wait(3600)
        self.assertTrue(self._submit(SubmissionClient(formulas), time).completed)

    def test_retry_after_beyond_remaining_wait_stops_without_early_request(self):
        task_ids, formulas = self._completed_run(("rank(close)",), max_pending_seconds=60)
        time = FakeTime("2026-09-04T01:00:00+00:00")

        class SlowReadClient(SubmissionClient):
            def fetch_alpha_detail(self, *, platform_alpha_id):
                self.calls.append(("detail_failed", platform_alpha_id))
                time.wait(50)
                raise WorldQuantRequestError(
                    "worldquant_alpha_detail_http_error", status_code=429,
                    retryable=True, outcome_unknown=False, retry_after_seconds=30,
                )

        client = SlowReadClient(formulas)
        with self.assertRaises(WorldQuantRequestError):
            self._submit(client, time)
        self.assertEqual(time.now(), datetime.fromisoformat("2026-09-04T01:00:50+00:00"))
        self.assertEqual(client.calls, [("authenticate", None), ("detail_failed", "alpha-1")])
        with open_database(self.database_path) as connection:
            self.assertEqual(get_formal_submission_attempt(connection, task_ids[0]).status,
                             "detail_pending")

    @staticmethod
    def _flaky_read_client(formulas, *, detail_failures=0, check_failures=0,
                           retry_after=None, retryable=True, unknown=False):
        class FlakyReadClient(SubmissionClient):
            def __init__(self):
                super().__init__(formulas)
                self.failures_left = {"detail": detail_failures, "check": check_failures}
                self.read_failures = 0

            def fail_read(self, stage, alpha_id):
                if self.failures_left[stage] > 0:
                    self.failures_left[stage] -= 1
                    self.read_failures += 1
                    self.calls.append((stage + "_failed", alpha_id))
                    raise WorldQuantRequestError(
                        "worldquant_alpha_detail_request_failed" if stage == "detail"
                        else "worldquant_formal_check_request_failed",
                        retryable=retryable, outcome_unknown=unknown,
                        retry_after_seconds=retry_after, transport_error_type="TimeoutError",
                    )

            def fetch_alpha_detail(self, *, platform_alpha_id):
                self.fail_read("detail", platform_alpha_id)
                return super().fetch_alpha_detail(platform_alpha_id=platform_alpha_id)

            def fetch_formal_submission_check(self, *, platform_alpha_id):
                self.fail_read("check", platform_alpha_id)
                return super().fetch_formal_submission_check(platform_alpha_id=platform_alpha_id)

        return FlakyReadClient()

    def _submit(self, client, time, **kwargs):
        return submit_queued_alphas(
            self.database_path, self.environment_path, clock=time.now, waiter=time.wait,
            client_factory=lambda _settings: client, **kwargs,
        )

    def _interrupted_run(self):
        run = prepare_automated_run(
            self.database_path, POLICY_PATH, account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=5, backtest_count=3, max_cycles=-1, max_backtests=0,
                max_pending_seconds=60, max_consecutive_failures=2, max_request_failures=3,
                max_in_flight_backtests=3, exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ), created_at="2026-09-04T01:00:00+00:00",
        )
        start_automated_run(self.database_path, run.run_id,
                            started_at="2026-09-04T01:00:00+00:00")
        tasks = prepare_automated_candidate_backtest_batch(
            self.database_path, run_id=run.run_id,
            candidates=tuple(AutomatedCandidateBacktest(self._candidate(f), self.settings)
                             for f in ("rank(open)", "rank(volume)", "rank(high)")),
            created_at="2026-09-04T01:00:01+00:00",
        )
        with open_database(self.database_path) as connection:
            record_submission_accepted(connection, tasks[0].task.task_id,
                                       remote_id="simulation-old",
                                       observed_at="2026-09-04T01:00:02+00:00")
            record_submission_unknown(connection, tasks[1].task.task_id,
                                      observed_at="2026-09-04T01:00:02+00:00")
        return run, tasks

    def _recovery_client(self, formulas, *, pending=False):
        yearly = self._yearly_stat(1.4)

        class RecoveryClient(SubmissionClient):
            def submit_backtest(self, **kwargs):
                raise AssertionError("manual submission must not create backtests")

            def poll_backtest(self, remote_id):
                self.calls.append(("poll", remote_id))
                return BacktestPollObservation(
                    "pending" if pending else "completed", None if pending else "late-alpha",
                    "PENDING" if pending else "COMPLETE", None,
                )

            def fetch_backtest_detail(self, *, platform_alpha_id, expected_formula, expected_settings):
                self.calls.append(("backtest_detail", platform_alpha_id))
                return BacktestDetail(
                    platform_alpha_id=platform_alpha_id, sharpe=1.4, fitness=1.1,
                    grade="SPECTACULAR",
                    turnover=0.12, returns=0.08, drawdown=0.04, margin=0.001,
                    book_size=20_000_000, pnl=100_000,
                    checks=tuple(BacktestCheck(name, "PASS", None, None, None)
                                 for name in sorted(STANDARD_REGULAR_CHECK_NAMES)),
                )

            def fetch_backtest_yearly_stats(self, *, platform_alpha_id):
                self.calls.append(("yearly", platform_alpha_id))
                return BacktestYearlyStatsObservation("ready", (yearly,), None)

        return RecoveryClient(formulas)

    def _completed_run(
        self,
        formulas: tuple[str, ...],
        *,
        sharpes: tuple[float, ...] | None = None,
        lineage: tuple[int, int] | None = None,
        max_pending_seconds: int = 60,
        include_yearly_stats: bool = True,
        settings: tuple[BacktestSettings, ...] | None = None,
        grades: tuple[str | None, ...] | None = None,
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        sharpe_values = sharpes or tuple(1.3 for _formula in formulas)
        grade_values = grades if grades is not None else tuple("SPECTACULAR" for _ in formulas)
        setting_values = settings or tuple(self.settings for _formula in formulas)
        if len(setting_values) != len(formulas):
            raise ValueError("test_settings_count_invalid")
        run = prepare_automated_run(
            self.database_path,
            POLICY_PATH,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=max(3, len(formulas)),
                backtest_count=len(formulas),
                max_cycles=1,
                max_backtests=len(formulas),
                max_pending_seconds=max_pending_seconds,
                max_consecutive_failures=2,
                max_request_failures=3,
                max_in_flight_backtests=3,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True,
            ),
            created_at="2026-09-04T00:00:00+00:00",
        )
        start_automated_run(
            self.database_path,
            run.run_id,
            started_at="2026-09-04T00:01:00+00:00",
        )
        snapshots = prepare_automated_candidate_backtest_batch(
            self.database_path,
            run_id=run.run_id,
            candidates=tuple(
                AutomatedCandidateBacktest(self._candidate(formula), setting)
                for formula, setting in zip(
                    formulas,
                    setting_values,
                    strict=True,
                )
            ),
            created_at="2026-09-04T00:02:00+00:00",
        )
        task_ids = tuple(snapshot.task.task_id for snapshot in snapshots)
        if lineage is not None:
            child_index, parent_index = lineage
            with open_database(self.database_path) as connection:
                create_backtest_mutation(
                    connection,
                    BacktestMutationRecord(
                        child_task_id=task_ids[child_index],
                        parent_task_id=task_ids[parent_index],
                        action="test_mutation",
                        location="formula",
                        before="parent",
                        after="child",
                    ),
                )

        platform_formulas: dict[str, str] = {}
        snapshot_by_task = {snapshot.task.task_id: snapshot for snapshot in snapshots}
        sharpe_by_task = dict(zip(task_ids, sharpe_values, strict=True))
        grade_by_task = dict(zip(task_ids, grade_values, strict=True))
        with open_database(self.database_path) as connection:
            links = list_automated_run_backtests(connection, run.run_id)
            for index, link in enumerate(links):
                platform_alpha_id = f"alpha-{index + 1}"
                formula = snapshot_by_task[link.task_id].task.formula
                sharpe = sharpe_by_task[link.task_id]
                platform_formulas[platform_alpha_id] = formula
                record_submission_accepted(
                    connection,
                    link.task_id,
                    remote_id=f"simulation-{index + 1}",
                    platform_alpha_id=platform_alpha_id,
                    observed_at="2026-09-04T00:03:00+00:00",
                )
                apply_backtest_detail(
                    connection,
                    link.task_id,
                    BacktestDetail(
                        platform_alpha_id=platform_alpha_id,
                        grade=grade_by_task[link.task_id],
                        sharpe=sharpe,
                        fitness=1.1,
                        turnover=0.12,
                        returns=0.08,
                        drawdown=0.04,
                        margin=0.001,
                        book_size=20_000_000,
                        pnl=100_000,
                        checks=tuple(
                            BacktestCheck(name, "PASS", None, None, None)
                            for name in sorted(STANDARD_REGULAR_CHECK_NAMES)
                        ),
                    ),
                    observed_at="2026-09-04T00:04:00+00:00",
                )
                record_backtest_yearly_stats(
                    connection,
                    link.task_id,
                    (self._yearly_stat(sharpe),) if include_yearly_stats else (),
                    observed_at="2026-09-04T00:05:00+00:00",
                )
                save_submission_check(connection, SubmissionCheckRecord(
                    link.task_id, "2026-09-04T00:05:00+00:00",
                    json.dumps({"is": {"checks": [{"name": name, "result": "PASS"}
                                                  for name in STANDARD_REGULAR_CHECK_NAMES]}}), None,
                ))
        settle_automated_cycle(
            self.database_path,
            run.run_id,
            cycle_number=1,
            observed_at="2026-09-04T00:06:00+00:00",
        )
        return task_ids, platform_formulas

    def _record_submitted_alpha(
        self,
        *,
        platform_alpha_id: str,
        formula: str,
    ) -> None:
        submitted_at = "2026-09-04T00:30:00+00:00"
        with open_database(self.database_path) as connection:
            record_platform_submitted_alphas(
                connection,
                (
                    PlatformSubmittedAlphaRecord(
                        account_scope="group-account",
                        platform_alpha_id=platform_alpha_id,
                        formula=formula,
                        status="ACTIVE",
                        date_submitted=submitted_at,
                        hidden=False,
                        raw_payload={
                            "id": platform_alpha_id,
                            "status": "ACTIVE",
                            "dateSubmitted": submitted_at,
                            "hidden": False,
                            "regular": {"code": formula},
                        },
                        observed_at=submitted_at,
                    ),
                ),
            )

    @staticmethod
    def _candidate(formula: str) -> FormulaCandidate:
        return exploration_candidate(parse_formula(formula).expression)

    @staticmethod
    def _yearly_stat(sharpe: float) -> BacktestYearlyStat:
        return BacktestYearlyStat(
            year=2025,
            pnl=100_000,
            book_size=20_000_000,
            turnover=0.12,
            sharpe=sharpe,
            returns=0.08,
            drawdown=0.04,
            margin=0.001,
            fitness=1.1,
            long_count=100,
            short_count=100,
            stage="IS",
        )


if __name__ == "__main__":
    unittest.main()
