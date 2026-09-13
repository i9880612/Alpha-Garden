from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from alpha_garden.cli import run
from execution.progress import phase
from execution.catalog_sync import CatalogRefreshResult
from execution.initialization import ProjectInitializationResult
from execution.runs import AutomatedRunLimits
from execution.submission_runner import SubmissionQueueCompletion
from persistence.catalog import FieldCatalogContext
from worldquant.client import WorldQuantRequestError


class CommandLineTests(unittest.TestCase):
    def test_qualified_submission_command_count_and_recovery_guidance(self):
        for arguments, count, grade in ((["submit", "good"], None, "GOOD"),
                                         (["submit", "GOOD", "2"], 2, "GOOD"),
                                         (["submit", "average", "3"], 3, "AVERAGE"),
                                         (["submit", "excellent"], None, "EXCELLENT"),
                                         (["submit", "inferior"], None, "INFERIOR"),
                                         (["submit", "spectacular", "2"], 2, None)):
            with self.subTest(arguments=arguments):
                with patch("alpha_garden.cli.submit_queued_alphas", side_effect=KeyboardInterrupt) as call:
                    output = io.StringIO()
                    with redirect_stdout(output):
                        self.assertEqual(run(arguments), 130)
                    call.assert_called_once_with(Path("data/alpha_garden.sqlite3"), Path(".env"),
                                                 max_submissions=count,
                                                 source="qualified_archive" if grade else "queue", grade=grade)
                    self.assertIn("再次 submit" + (" " + grade.lower() if grade else ""), output.getvalue())
        for arguments in (["good", "0"], ["good", "-1"], ["good", "1.5"], ["bad"], ["2", "3"]):
            with self.subTest(arguments=arguments), patch("alpha_garden.cli.submit_queued_alphas") as call:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    run(["submit", *arguments])
                call.assert_not_called()

    def test_run_opt_uses_shared_launch_and_keeps_submission_disabled(self):
        for arguments, cycles in ((["run", "-opt"], 1), (["run", "-opt", "2"], 2),
                                  (["run", "2", "-opt"], 2), (["run", "-opt", "-1"], -1)):
            with self.subTest(arguments=arguments), patch(
                "alpha_garden.cli.launch_automated_run", return_value=self._completion(status="completed")
            ) as launch, redirect_stdout(io.StringIO()):
                self.assertEqual(run(arguments), 0)
                limits = launch.call_args.kwargs["limits"]
                self.assertTrue(limits.optimization_only)
                self.assertEqual(limits.max_cycles, cycles)
                self.assertFalse(limits.automatic_submissions_enabled)

    def test_export_submitted_rebuilds_locally_and_respects_running_writer(self):
        from execution.process_lock import exclusive_run_process
        from persistence.database import open_database
        from persistence.submissions import initialize_submission_schema

        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "database.sqlite3"
            with open_database(database) as connection:
                initialize_submission_schema(connection)
            command = ["export-submitted", "--database", str(database)]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(run(command), 0)
            document = database.with_name("submitted-formulas.md")
            self.assertIn("共 **0** 条", document.read_text(encoding="utf-8"))
            original = document.read_bytes()
            with exclusive_run_process(database), redirect_stderr(io.StringIO()):
                self.assertEqual(run(command), 1)
            self.assertEqual(document.read_bytes(), original)

    def test_run_and_resume_enable_console_progress(self):
        def execute(*args, **kwargs):
            phase(2, 3, "当前第 1/20 条 已提交回测，等待结果")
            return self._completion(status="completed")

        for command, entry in ((["run"], "launch_automated_run"), (["resume", "saved-run"], "resume_automated_run")):
            with self.subTest(command=command):
                output = io.StringIO()
                with patch("alpha_garden.cli.load_automated_run_limits", return_value=self._limits()), \
                     patch(f"alpha_garden.cli.{entry}", side_effect=execute), redirect_stdout(output):
                    self.assertEqual(run(command), 0)
                self.assertIn("第 2 轮 阶段[3] 当前第 1/20 条", output.getvalue())

    def test_run_interruption_is_explained_without_traceback(self):
        output = io.StringIO()
        with patch("alpha_garden.cli.load_automated_run_limits", return_value=self._limits()):
            with patch("alpha_garden.cli.launch_automated_run", side_effect=KeyboardInterrupt):
                with redirect_stdout(output):
                    self.assertEqual(run(["run", "-1"]), 130)
        self.assertIn("新参数", output.getvalue())
        self.assertIn("resume", output.getvalue())

    def test_submit_interruption_preserves_recovery_guidance(self):
        output = io.StringIO()
        with patch("alpha_garden.cli.submit_queued_alphas", side_effect=KeyboardInterrupt):
            with redirect_stdout(output):
                self.assertEqual(run(["submit"]), 130)
        self.assertIn("已有请求和结果保留", output.getvalue())
        self.assertIn("再次 submit", output.getvalue())

    def test_submit_request_error_reports_safe_details_and_resume_guidance(self):
        error = WorldQuantRequestError(
            "worldquant_alpha_detail_request_failed", status_code=503,
            retryable=True, outcome_unknown=False, transport_error_type="TimeoutError",
        )
        error.__cause__ = TimeoutError("sensitive-url-or-credential")
        output = io.StringIO()
        with patch("alpha_garden.cli.submit_queued_alphas", side_effect=error):
            with redirect_stderr(output):
                self.assertEqual(run(["submit"]), 1)
        self.assertIn("HTTP 503", output.getvalue())
        self.assertIn("TimeoutError", output.getvalue())
        self.assertIn("再次 submit", output.getvalue())
        self.assertNotIn("sensitive", output.getvalue())

    def test_init_uses_project_defaults_and_reports_each_completed_step(self) -> None:
        result = ProjectInitializationResult(
            database_path=Path("data/alpha_garden.sqlite3"),
            database_created=True,
            project_table_count=19,
            semantics_initialized=True,
            catalog_refreshed=True,
            window_count=5,
            operator_role_count=28,
            operator_output_count=67,
            catalog=CatalogRefreshResult(
                account_scope="group-account",
                context=FieldCatalogContext("EQUITY", "USA", "TOP3000", 1),
                field_count=8642,
                operator_count=67,
                catalog_fingerprint="fingerprint",
                synced_at="2026-09-02T08:13:52+00:00",
            ),
        )
        output = io.StringIO()

        with patch("alpha_garden.cli.initialize_project", return_value=result) as call:
            with redirect_stdout(output):
                exit_code = run(["init"])

        self.assertEqual(exit_code, 0)
        call.assert_called_once_with(
            Path("data/alpha_garden.sqlite3"),
            Path("config/backtest.default.json"),
            Path(".env"),
        )
        rendered = output.getvalue()
        for expected in (
            "1. 数据库：已创建",
            "2. 现行表：已创建或核对 19 张",
            "5 个标准窗口",
            "28 条算子角色关系",
            "67 条算子输出映射",
            "4. WQB 平台目录",
            "已同步",
            "8642 个字段、67 个算子",
        ):
            self.assertIn(expected, rendered)

    def test_init_reports_when_existing_platform_catalog_is_reused(self) -> None:
        result = ProjectInitializationResult(
            database_path=Path("data/alpha_garden.sqlite3"),
            database_created=False,
            project_table_count=19,
            semantics_initialized=False,
            catalog_refreshed=False,
            window_count=5,
            operator_role_count=28,
            operator_output_count=67,
            catalog=CatalogRefreshResult(
                account_scope="group-account",
                context=FieldCatalogContext("EQUITY", "USA", "TOP3000", 1),
                field_count=8642,
                operator_count=67,
                catalog_fingerprint="fingerprint",
                synced_at="2026-09-02T08:13:52+00:00",
            ),
        )
        output = io.StringIO()

        with patch("alpha_garden.cli.initialize_project", return_value=result):
            with redirect_stdout(output):
                exit_code = run(["init"])

        self.assertEqual(exit_code, 0)
        self.assertIn("4. WQB 平台目录：已复用", output.getvalue())

    def test_init_accepts_explicit_paths(self) -> None:
        failure = io.StringIO()
        with patch(
            "alpha_garden.cli.initialize_project",
            side_effect=ValueError("project_database_parent_missing"),
        ) as call:
            with redirect_stderr(failure):
                exit_code = run(
                    [
                        "init",
                        "--database",
                        "state/project.sqlite3",
                        "--settings",
                        "settings.json",
                        "--env",
                        "credentials.env",
                    ]
                )

        self.assertEqual(exit_code, 1)
        call.assert_called_once_with(
            Path("state/project.sqlite3"),
            Path("settings.json"),
            Path("credentials.env"),
        )
        self.assertIn("project_database_parent_missing", failure.getvalue())

    def test_run_rejects_removed_per_value_arguments(self) -> None:
        with patch("alpha_garden.cli.launch_automated_run") as launch:
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    run(["run", "--max-backtests", "20"])

        self.assertEqual(raised.exception.code, 2)
        launch.assert_not_called()

    def test_run_uses_default_config_reports_plan_and_disables_submission(
        self,
    ) -> None:
        completion = self._completion(status="completed")
        limits = self._limits()

        def launch(database, settings, environment, *, limits, run_created):
            run_created(completion.run)
            return completion

        output = io.StringIO()
        with patch(
            "alpha_garden.cli.load_automated_run_limits",
            return_value=limits,
        ) as load:
            with patch(
                "alpha_garden.cli.launch_automated_run",
                side_effect=launch,
            ) as call:
                with redirect_stdout(output):
                    exit_code = run(["run"])

        self.assertEqual(exit_code, 0)
        load.assert_called_once_with(
            Path("config/run.default.json"),
            cycles=1,
            automatic_submissions_enabled=False,
            optimization_only=False,
        )
        positional = call.call_args.args
        self.assertEqual(
            positional,
            (
                Path("data/alpha_garden.sqlite3"),
                Path("config/backtest.default.json"),
                Path(".env"),
            ),
        )
        self.assertEqual(call.call_args.kwargs["limits"], limits)
        rendered = output.getvalue()
        self.assertIn("自动运行已创建，可恢复 ID：run-1", rendered)
        self.assertIn("每轮生成 500", rendered)
        self.assertIn("运行 1 轮/20 条真实回测", rendered)
        self.assertIn("单任务 pending 上限 600 秒", rendered)
        self.assertIn("连续 2 轮整批失败停止", rendered)
        self.assertIn("连续请求故障上限 3", rendered)
        self.assertIn("状态：completed", rendered)
        self.assertIn("正式 Alpha 提交：关闭", rendered)
        self.assertNotIn("secret", rendered)

    def test_failed_run_returns_failure_exit_code(self) -> None:
        completion = self._completion(status="failed")
        with patch(
            "alpha_garden.cli.load_automated_run_limits",
            return_value=self._limits(),
        ):
            with patch(
                "alpha_garden.cli.launch_automated_run",
                return_value=completion,
            ):
                with redirect_stdout(io.StringIO()):
                    exit_code = run(["run"])

        self.assertEqual(exit_code, 1)

    def test_invalid_run_config_fails_before_launch(self) -> None:
        failure = io.StringIO()
        with patch(
            "alpha_garden.cli.load_automated_run_limits",
            side_effect=ValueError("automated_run_config_file_invalid"),
        ) as load:
            with patch("alpha_garden.cli.launch_automated_run") as launch:
                with redirect_stderr(failure):
                    exit_code = run(
                        ["run", "--run-config", "config/custom-run.json"]
                    )

        self.assertEqual(exit_code, 1)
        load.assert_called_once_with(
            Path("config/custom-run.json"),
            cycles=1,
            automatic_submissions_enabled=False,
            optimization_only=False,
        )
        launch.assert_not_called()
        self.assertIn("automated_run_config_file_invalid", failure.getvalue())

    def test_run_passes_an_explicit_cycle_count_or_continuous_mode(self) -> None:
        for value in ("5", "-1"):
            with self.subTest(value=value):
                with patch(
                    "alpha_garden.cli.load_automated_run_limits",
                    return_value=self._limits(),
                ) as load:
                    with patch(
                        "alpha_garden.cli.launch_automated_run",
                        return_value=self._completion(status="completed"),
                    ):
                        with redirect_stdout(io.StringIO()):
                            exit_code = run(["run", value])

                self.assertEqual(exit_code, 0)
                load.assert_called_once_with(
                    Path("config/run.default.json"),
                    cycles=int(value),
                    automatic_submissions_enabled=False,
                    optimization_only=False,
                )

    def test_run_requires_an_explicit_auto_submit_switch(self) -> None:
        limits = replace(
            self._limits(),
            automatic_submissions_enabled=True,
        )
        completion = self._completion(status="completed")
        completion.run.automatic_submissions_enabled = True
        completion.formal_submission_claimed_count = 2
        completion.formal_submission_confirmed_count = 1
        completion.formal_submission_unresolved_count = 1
        output = io.StringIO()
        with patch(
            "alpha_garden.cli.load_automated_run_limits",
            return_value=limits,
        ) as load:
            with patch(
                "alpha_garden.cli.launch_automated_run",
                return_value=completion,
            ):
                with redirect_stdout(output):
                    exit_code = run(["run", "--auto-submit"])

        self.assertEqual(exit_code, 0)
        load.assert_called_once_with(
            Path("config/run.default.json"),
            cycles=1,
            automatic_submissions_enabled=True,
            optimization_only=False,
        )
        self.assertIn(
            "正式 Alpha 提交：开启，每轮处理全部当前合格候选",
            output.getvalue(),
        )
        self.assertIn(
            "正式提交结果：已领取正式提交权 2 条，已确认 1 条，待对账 1 条",
            output.getvalue(),
        )

    def test_removed_formal_submission_count_argument_is_rejected(self) -> None:
        with patch("alpha_garden.cli.launch_automated_run") as launch:
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    run(["run", "--formal-submissions", "2"])

        self.assertEqual(raised.exception.code, 2)
        launch.assert_not_called()

    def test_continuous_run_reports_no_total_backtest_limit(self) -> None:
        completion = self._completion(status="completed")
        completion.run.max_cycles = -1
        completion.run.max_backtests = 0
        limits = replace(self._limits(), max_cycles=-1, max_backtests=0)

        def launch(database, settings, environment, *, limits, run_created):
            run_created(completion.run)
            return completion

        output = io.StringIO()
        with patch(
            "alpha_garden.cli.load_automated_run_limits",
            return_value=limits,
        ):
            with patch(
                "alpha_garden.cli.launch_automated_run",
                side_effect=launch,
            ):
                with redirect_stdout(output):
                    exit_code = run(["run", "-1"])

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "持续运行，不设总回测条数上限",
            output.getvalue(),
        )
        self.assertIn("通信故障自动退避恢复", output.getvalue())
        self.assertNotIn("连续请求故障上限", output.getvalue())

    def test_run_rejects_cycle_values_without_a_defined_meaning(self) -> None:
        for value in ("0", "-2", "not-a-number"):
            with self.subTest(value=value):
                with patch("alpha_garden.cli.load_automated_run_limits") as load:
                    with patch("alpha_garden.cli.launch_automated_run") as launch:
                        with redirect_stderr(io.StringIO()):
                            with self.assertRaises(SystemExit) as raised:
                                run(["run", value])

                self.assertEqual(raised.exception.code, 2)
                load.assert_not_called()
                launch.assert_not_called()

    def test_resume_passes_only_frozen_run_identity_and_paths(self) -> None:
        completion = self._completion(status="completed")
        output = io.StringIO()
        with patch(
            "alpha_garden.cli.resume_automated_run",
            return_value=completion,
        ) as call:
            with redirect_stdout(output):
                exit_code = run(
                    [
                        "resume",
                        "run-1",
                        "--database",
                        "state/project.sqlite3",
                        "--env",
                        "credentials.env",
                    ]
                )

        self.assertEqual(exit_code, 0)
        call.assert_called_once_with(
            Path("state/project.sqlite3"),
            Path("credentials.env"),
            "run-1",
        )
        self.assertIn("运行 ID：run-1", output.getvalue())

    def test_submit_processes_the_whole_current_queue_and_reports_results(
        self,
    ) -> None:
        completion = SubmissionQueueCompletion(
            account_scope="group-account",
            initial_queue_count=3,
            step_count=12,
            platform_request_count=8,
            authentication_performed=True,
            submission_claimed_count=2,
            submitted_count=2,
            already_active_count=1,
            ineligible_count=1,
            failed_count=0,
            unresolved_count=0,
            remaining_queue_count=0,
        )
        output = io.StringIO()
        with patch(
            "alpha_garden.cli.submit_queued_alphas",
            return_value=completion,
        ) as call:
            with redirect_stdout(output):
                exit_code = run(
                    [
                        "submit",
                        "--database",
                        "state/project.sqlite3",
                        "--env",
                        "credentials.env",
                    ]
                )

        self.assertEqual(exit_code, 0)
        call.assert_called_once_with(
            Path("state/project.sqlite3"),
            Path("credentials.env"),
            max_submissions=None,
            source="queue",
            grade=None,
        )
        rendered = output.getvalue()
        self.assertIn("启动时待处理：3 条", rendered)
        self.assertIn("已确认提交 2 条", rendered)
        self.assertIn("检查不通过 1 条", rendered)
        self.assertIn("本批仍在队列 0 条", rendered)

    def test_submit_count_stops_successfully_with_candidates_left(self) -> None:
        completion = SubmissionQueueCompletion(
            account_scope="group-account", initial_queue_count=5, step_count=12,
            platform_request_count=8, authentication_performed=True,
            submission_claimed_count=2, submitted_count=2, already_active_count=0,
            ineligible_count=0, failed_count=0, unresolved_count=0,
            remaining_queue_count=3, max_submissions=2,
        )
        output = io.StringIO()
        with patch("alpha_garden.cli.submit_queued_alphas", return_value=completion) as call:
            with redirect_stdout(output):
                exit_code = run(["submit", "2"])
        self.assertEqual(exit_code, 0)
        call.assert_called_once_with(
            Path("data/alpha_garden.sqlite3"), Path(".env"), max_submissions=2, source="queue", grade=None,
        )
        self.assertIn("本次成功提交上限：2 条", output.getvalue())
        self.assertIn("已达到成功提交数量", output.getvalue())
        self.assertIn("本批仍在队列 3 条", output.getvalue())

        for outcome, expected_exit in (
            (replace(completion, submitted_count=1, remaining_queue_count=0), 0),
            (replace(completion, submitted_count=0, unresolved_count=1), 1),
        ):
            with self.subTest(outcome=outcome):
                output = io.StringIO()
                with patch("alpha_garden.cli.submit_queued_alphas", return_value=outcome):
                    with redirect_stdout(output):
                        exit_code = run(["submit", "2"])
                self.assertEqual(exit_code, expected_exit)
                self.assertNotIn("已达到成功提交数量", output.getvalue())
                if expected_exit == 0:
                    self.assertIn("成功提交数量未达到上限", output.getvalue())

    def test_invalid_submit_count_is_rejected_before_execution(self) -> None:
        for value in ("0", "-1", "-2", "1.5", "invalid"):
            with self.subTest(value=value):
                with patch("alpha_garden.cli.submit_queued_alphas") as submit:
                    with redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            run(["submit", value])
                self.assertEqual(raised.exception.code, 2)
                submit.assert_not_called()

    @staticmethod
    def _limits() -> AutomatedRunLimits:
        return AutomatedRunLimits(
            exploration_percent=30,
            self_correlation_percent=30,
            mutation_percent=40,
            direction_validation_percent=3,
            generation_count=500,
            backtest_count=20,
            max_cycles=1,
            max_backtests=20,
            max_pending_seconds=600,
            max_consecutive_failures=2,
            max_request_failures=3,
            max_in_flight_backtests=3,
            exploration_seed_attempt_multiplier=4,
            real_backtests_authorized=True,
            automatic_submissions_enabled=False,
        )

    @staticmethod
    def _completion(*, status: str):
        return SimpleNamespace(
            run=SimpleNamespace(
                run_id="run-1",
                status=status,
                stop_reason=(
                    "max_cycles_reached"
                    if status == "completed"
                    else "request_failure_limit_reached"
                ),
                current_cycle=1,
                generation_count=500,
                backtest_count=20,
                max_cycles=1,
                max_backtests=20,
                max_pending_seconds=600,
                max_consecutive_failures=2,
                max_request_failures=3,
                max_in_flight_backtests=3,
                automatic_submissions_enabled=False,
                optimization_only=False,
            ),
            step_count=5,
            platform_request_count=4,
            authentication_performed=True,
            formal_submission_claimed_count=0,
            formal_submission_confirmed_count=0,
            formal_submission_unresolved_count=0,
        )


if __name__ == "__main__":
    unittest.main()
