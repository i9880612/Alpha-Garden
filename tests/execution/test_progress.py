from __future__ import annotations

import io
import tempfile
import unittest
import unicodedata
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from alpha_garden.console import run_progress_console
from execution.backtests import prepare_backtest_task
from execution.progress import RunProgress
from execution.runs import prepare_automated_run, start_automated_run
from persistence.backtests import (
    BacktestCheckRecord, BacktestMutationRecord, BacktestResultRecord,
    create_backtest_mutation,
)
from persistence.database import open_database
from persistence.runs import AutomatedRunBacktestRecord, attach_backtest_to_automated_run
from tests.execution import test_launch as launch_fixture
from worldquant.backtests import STANDARD_NON_SC_CHECK_NAMES


class RunProgressTests(unittest.TestCase):
    def setUp(self):
        self.fixture = launch_fixture.AutomatedRunLaunchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run = prepare_automated_run(
            self.fixture.database_path, self.fixture.settings_path,
            account_scope="group-account", limits=replace(
                self.fixture._limits(), backtest_count=2, max_cycles=2, max_backtests=4
            ), created_at="2026-09-06T00:00:00+00:00",
        )
        start_automated_run(self.fixture.database_path, self.run.run_id,
                            started_at="2026-09-06T00:00:00+00:00")
        self.tasks = []
        for cycle, formulas in enumerate((("rank(close)", "rank(open)"), ("rank(volume)", "ts_rank(close, 5)")), 1):
            batch = []
            with open_database(self.fixture.database_path) as connection:
                for formula in formulas:
                    snapshot = prepare_backtest_task(
                        connection, account_scope="group-account", formula=formula,
                        settings=self.fixture.settings.as_platform_dict(),
                        created_at="2026-09-06T00:00:00+00:00",
                    )
                    attach_backtest_to_automated_run(connection, AutomatedRunBacktestRecord(
                        self.run.run_id, snapshot.task.task_id, cycle
                    ))
                    batch.append(snapshot)
            self.tasks.append(sorted(batch, key=lambda item: item.task.task_id))
        self.progress = RunProgress(self.fixture.database_path, self.run.run_id)

    def test_parallel_results_keep_frozen_cycle_and_item_numbers(self):
        output = io.StringIO()
        grades = ("SPECTACULAR", None, "GOOD", "AVERAGE")
        with redirect_stdout(output), run_progress_console():
            for (cycle, index), grade in zip(((1, 1), (0, 0), (1, 0), (0, 1)), grades):
                snapshot = self._completed(self.tasks[cycle][index])
                snapshot = replace(snapshot, result=replace(snapshot.result, grade=grade))
                self.progress.backtest(snapshot.task.task_id, snapshot=snapshot)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        column_offsets = set()
        for line, position, grade in zip(lines, ("第2轮 02/2", "第1轮 01/2", "第2轮 01/2", "第1轮 02/2"), grades):
            self.assertIn(position, line)
            self.assertIn("S -1.25 F  1.00 T  2.9%", line)
            self.assertEqual(line.split()[1], "[未通过]")
            self.assertEqual(line.split()[2], f"[{grade or '未知'}]")
            self.assertNotIn("SC", line)
            self.assertNotIn("基础", line)
            column_offsets.add(sum(2 if unicodedata.east_asian_width(c) in {"W", "F"} else 1
                                   for c in line[:line.index(position)]))
        self.assertEqual(len(column_offsets), 1)

    def test_source_labels_follow_recorded_actions_for_results_and_recovery(self):
        parent = self.tasks[0][0]
        cases = (
            (parent, None, "探索"),
            (self.tasks[0][1], "single_window_mutation", "变异"),
            (self.tasks[1][0], "self_correlation_industry_neutralization", "SC治理"),
            (self.tasks[1][1], "direction_reversal", "反转"),
        )
        with open_database(self.fixture.database_path) as connection:
            for original, action, _ in cases:
                if action is not None:
                    create_backtest_mutation(connection, BacktestMutationRecord(
                        child_task_id=original.task.task_id,
                        parent_task_id=parent.task.task_id,
                        action=action, location="formula",
                        before=parent.task.formula, after=original.task.formula,
                    ))
        for original, action, source in cases:
            with self.subTest(action=action):
                completed = self._completed(original)
                completed = replace(completed, result=replace(
                    completed.result, checks=completed.result.checks + (
                        BacktestCheckRecord("SELF_CORRELATION", "FAIL", None, None, None),
                    ),
                ))
                failed = replace(original, task=replace(
                    original.task, status="failed", failure_code="platform_error",
                ))
                output = io.StringIO()
                with redirect_stdout(output), run_progress_console():
                    RunProgress(self.fixture.database_path, self.run.run_id).backtest(
                        completed.task.task_id, snapshot=completed)
                    RunProgress(self.fixture.database_path, "another-run").backtest(
                        failed.task.task_id, snapshot=failed)
                lines = output.getvalue().splitlines()
                self.assertEqual(len(lines), 2)
                self.assertIn(f"[{source}] alpha-1 S -1.25 F 1.00 T 2.9%", " ".join(lines[0].split()))
                self.assertIn(f"旧{self.run.run_id.removeprefix('run_')[:8]}", lines[1])
                self.assertIn(f"[{source}]", lines[1])
                self.assertEqual(lines[1].split()[1], "[异常]")
                self.assertEqual(lines[1].split()[2], "[未知]")
                self.assertIn("回测失败（platform_error）", lines[1])

    def test_partial_metrics_and_unknown_are_never_reported_as_completed(self):
        original = self.tasks[0][0]
        pending = replace(original, task=replace(original.task, status="pending", platform_alpha_id="alpha-1"),
                          result=self._completed(original).result)
        unknown = replace(original, task=replace(original.task, status="submission_unknown"))
        output = io.StringIO()
        with redirect_stdout(output), run_progress_console():
            self.progress.backtest(pending.task.task_id, snapshot=pending)
            self.progress.backtest(unknown.task.task_id, snapshot=unknown)
            expired = replace(original, task=replace(original.task, status="failed", failure_code="submission_outcome_timeout"))
            self.progress.backtest(expired.task.task_id, snapshot=expired)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertIn("接收超时；远端未知，不重发", output.getvalue())
        self.assertNotIn("回测完成", output.getvalue())
        self.assertNotIn("Sharpe", output.getvalue())

    def test_exhausted_response_logs_local_failure_without_fake_scores(self):
        original = self.tasks[0][0]
        failed = replace(original, task=replace(
            original.task, status="failed",
            failure_code="platform_response_retry_exhausted",
            failure_message="结果读取连续 3 次异常；本地任务失败，远端结果未确认",
        ))
        output = io.StringIO()
        with redirect_stdout(output), run_progress_console():
            self.progress.backtest(failed.task.task_id, snapshot=failed)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertIn("连续 3 次", output.getvalue())
        self.assertIn("远端结果未确认", output.getvalue())
        self.assertNotIn("未通过", output.getvalue())

    def test_unchanged_waiting_does_not_repeat_and_failure_has_no_fake_scores(self):
        original = self.tasks[0][0]
        pending = replace(original, task=replace(original.task, status="pending"))
        output = io.StringIO()
        with redirect_stdout(output), run_progress_console():
            for _ in range(10):
                self.progress.backtest(pending.task.task_id, snapshot=pending)
                self.progress.advanced(SimpleNamespace(
                    action="backtest_advanced", task_id=None, backtest_action="retry_wait"
                ))
            failed = replace(original, task=replace(original.task, status="failed", failure_code="platform_zero_capital"))
            self.progress.backtest(failed.task.task_id, snapshot=failed)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertIn("回测失败（platform_zero_capital）", output.getvalue())
        self.assertNotIn("Sharpe", output.getvalue())

    def test_old_run_is_labelled_and_reconstructed_reporter_keeps_index(self):
        snapshot = self._completed(self.tasks[1][1])
        output = io.StringIO()
        with redirect_stdout(output), run_progress_console():
            RunProgress(self.fixture.database_path, "another-run").backtest(snapshot.task.task_id, snapshot=snapshot)
            RunProgress(self.fixture.database_path, self.run.run_id).backtest(snapshot.task.task_id, snapshot=snapshot)
        self.assertIn(f"旧{self.run.run_id.removeprefix('run_')[:8]}", output.getvalue())
        self.assertEqual(output.getvalue().count("02/2"), 2)

    def test_completed_formula_is_printed_once_without_check_details(self):
        snapshot = self._completed(self.tasks[0][0])
        snapshot = replace(snapshot, result=replace(snapshot.result,
            checks=snapshot.result.checks + (BacktestCheckRecord("SELF_CORRELATION", "PENDING", None, None, None),)))
        output = io.StringIO()
        with redirect_stdout(output), run_progress_console():
            for _ in range(3):
                self.progress.backtest(snapshot.task.task_id, snapshot=snapshot)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(output.getvalue().split()[1], "[未通过]")
        self.assertNotIn("SC", output.getvalue())
        self.assertNotIn("LOW_SHARPE", output.getvalue())

    def test_check_only_recovery_does_not_replay_completed_scores(self):
        snapshot = self._completed(self.tasks[0][0])
        output = io.StringIO()
        with redirect_stdout(output), run_progress_console():
            RunProgress(self.fixture.database_path, "another-run").backtest(
                snapshot.task.task_id, snapshot=snapshot, action="submission_check_observed")
        self.assertEqual(output.getvalue(), "")

    def test_passed_base_checks_ignore_sc_in_result_display(self):
        original = self._completed(self.tasks[0][0])
        checks = tuple(BacktestCheckRecord(name, "PASS", None, None, None)
                       for name in sorted(STANDARD_NON_SC_CHECK_NAMES))
        for sc_status in ("PASS", "FAIL", "PENDING", None):
            with self.subTest(sc_status=sc_status):
                snapshot = replace(original, result=replace(original.result, grade="AVERAGE", checks=checks + (
                    (BacktestCheckRecord("SELF_CORRELATION", sc_status, None, None, None),)
                    if sc_status else ())))
                output = io.StringIO()
                with redirect_stdout(output), run_progress_console():
                    RunProgress(self.fixture.database_path, self.run.run_id).backtest(
                        snapshot.task.task_id, snapshot=snapshot)
                self.assertEqual(output.getvalue().split()[1], "[通过]")
                self.assertEqual(output.getvalue().split()[2], "[AVERAGE]")
                self.assertNotIn("SC", output.getvalue())

    def test_unresolved_base_checks_do_not_invent_a_verdict(self):
        original = self._completed(self.tasks[0][0])
        for checks in ((), (BacktestCheckRecord("LOW_SHARPE", "PENDING", None, None, None),)):
            with self.subTest(checks=checks):
                snapshot = replace(original, result=replace(original.result, checks=checks))
                output = io.StringIO()
                with redirect_stdout(output), run_progress_console():
                    RunProgress(self.fixture.database_path, self.run.run_id).backtest(
                        snapshot.task.task_id, snapshot=snapshot)
                self.assertIn("S -1.25 F  1.00 T  2.9%", output.getvalue())
                self.assertEqual(output.getvalue().split()[1], "[待定]")
                self.assertNotIn("通过", output.getvalue())
                self.assertNotIn("SC", output.getvalue())

    def test_absent_progress_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "absent.sqlite3"
            with redirect_stdout(io.StringIO()), run_progress_console():
                RunProgress(path, "absent").backtest("absent")
            self.assertFalse(path.exists())

    @staticmethod
    def _completed(snapshot):
        return replace(snapshot, task=replace(snapshot.task, status="completed", platform_alpha_id="alpha-1"),
                       result=BacktestResultRecord(
                           task_id=snapshot.task.task_id, sharpe=-1.25, fitness=1.0, turnover=0.029,
                           returns=0.08, drawdown=0.04, margin=0.001, book_size=None, pnl=None,
                           long_count=None, short_count=None, check_details_captured=True,
                           checks=(BacktestCheckRecord("LOW_SHARPE", "FAIL", None, None, None),),
                       ))
