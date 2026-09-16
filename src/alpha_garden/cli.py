from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from alpha_garden.console import run_progress_console
from execution.initialization import ProjectInitializationResult, initialize_project
from execution.launch import launch_automated_run, resume_automated_run
from execution.process_lock import exclusive_run_process
from execution.run_config import load_automated_run_limits
from execution.runner import AutomatedRunCompletion
from execution.runs import uses_continuous_recovery
from execution.submission_runner import (
    SubmissionQueueCompletion,
    submit_queued_alphas,
)
from execution.submitted_formulas import export_submitted_formulas
from persistence.runs import AutomatedRunRecord
from submission.formal import BELOW_TARGET_GRADES
from worldquant.client import WorldQuantRequestError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-garden",
        description="初始化 Alpha Garden，或执行有界的真实回测自动运行。",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    web = commands.add_parser("web", help="启动本机网页控制台和接口；不会自动运行或提交。")
    web.add_argument("--port", type=int, default=8787)
    web.add_argument("--read-only", action="store_true", help="仅查看本地数据，禁用运行和提交操作。")
    web.add_argument("--database", type=Path, default=Path("data/alpha_garden.sqlite3"))
    web.add_argument("--settings", type=Path, default=Path("config/backtest.default.json"))
    web.add_argument("--run-config", type=Path, default=Path("config/run.default.json"))
    web.add_argument("--env", type=Path, default=Path(".env"))
    web.add_argument("--assets", type=Path, default=Path("webui/dist"))
    initialize = commands.add_parser(
        "init",
        help="初始化本地结构，并在平台目录缺失时同步字段和算子。",
        description=(
            "按固定顺序初始化：创建数据库 -> 创建全部现行表 -> "
            "写入 5 个标准窗口、28 条算子角色关系和 67 条输出映射 -> "
            "复用完整平台目录，缺失时才登录 WorldQuant BRAIN 同步。"
        ),
    )
    initialize.add_argument(
        "--database",
        type=Path,
        default=Path("data/alpha_garden.sqlite3"),
        help="SQLite 数据库路径（默认：data/alpha_garden.sqlite3）。",
    )
    initialize.add_argument(
        "--settings",
        type=Path,
        default=Path("config/backtest.default.json"),
        help="回测设置策略路径（默认：config/backtest.default.json）。",
    )
    initialize.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="WQB 账号环境文件路径（默认：.env）。",
    )
    automated_run = commands.add_parser(
        "run",
        help="按指定轮数启动真实回测自动运行（默认 1 轮，-1 持续运行）。",
        description=(
            "按本次参数创建新批次，自动收尾已停止的旧批次；"
            "继续原批次请用 resume。正式 Alpha 提交默认关闭，"
            "只有显式给出 --auto-submit 才会开启。"
        ),
    )
    automated_run.add_argument(
        "cycles",
        type=_run_cycles,
        nargs="?",
        default=1,
        help="运行轮数（默认：1；-1：持续运行，不设总回测条数上限）。",
    )
    automated_run.add_argument(
        "-opt", dest="optimization_only", action="store_true",
        help="仅变异全部官方检查通过、评级不足且仍有额度的活动父代；候选不足时按实际数量运行。",
    )
    automated_run.add_argument(
        "--auto-submit",
        action="store_true",
        help=(
            "开启本次运行的自动正式提交；默认关闭。开启后每轮处理全部"
            "当前合格候选，不设置本地数量上限。"
        ),
    )
    automated_run.add_argument(
        "--run-config",
        type=Path,
        default=Path("config/run.default.json"),
        help="完整运行边界数值配置路径（默认：config/run.default.json）。",
    )
    automated_run.add_argument(
        "--database",
        type=Path,
        default=Path("data/alpha_garden.sqlite3"),
        help="SQLite 数据库路径（默认：data/alpha_garden.sqlite3）。",
    )
    automated_run.add_argument(
        "--settings",
        type=Path,
        default=Path("config/backtest.default.json"),
        help="回测设置策略路径（默认：config/backtest.default.json）。",
    )
    automated_run.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="WQB 账号环境文件路径（默认：.env）。",
    )
    resume = commands.add_parser(
        "resume",
        help="使用已经冻结的边界恢复一个中断的自动运行。",
        description=("恢复已有运行；不允许修改设置、额度、授权或探索倍率。"),
    )
    resume.add_argument("run_id", help="需要恢复的自动运行 ID。")
    resume.add_argument(
        "--database",
        type=Path,
        default=Path("data/alpha_garden.sqlite3"),
        help="SQLite 数据库路径（默认：data/alpha_garden.sqlite3）。",
    )
    resume.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="WQB 账号环境文件路径（默认：.env）。",
    )
    submit = commands.add_parser(
        "submit",
        help="逐条复检并提交，可指定评级和成功提交数量，如 submit good 2。",
        description=(
            "冻结命令启动时的待提交候选，按同一谱系低 Sharpe（夏普比率）"
            "到高 Sharpe "
            "逐条读取平台详情、执行最新检查，并只提交全部检查明确通过的公式。"
            "指定数量时成功提交达到该数量即停止；省略时处理全部启动候选。"
            "如 submit good 2：选择合格归档中 Good 级别、当前满足相关性与改善条件的公式。"
        ),
    )
    submit.add_argument(
        "target",
        type=_submission_target,
        nargs="?",
        default=None,
        help="成功数量，或评级 inferior / average / good / excellent / spectacular；省略使用 Spectacular 队列。",
    )
    submit.add_argument("count", type=_submission_count, nargs="?", default=None,
                        help="指定评级后的成功提交数量上限，如 submit good 2。")
    submit.add_argument(
        "--database",
        type=Path,
        default=Path("data/alpha_garden.sqlite3"),
        help="SQLite 数据库路径（默认：data/alpha_garden.sqlite3）。",
    )
    submit.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="WQB 账号环境文件路径（默认：.env）。",
    )
    export = commands.add_parser(
        "export-submitted", help="从本地已确认事实重建提交成功的公式清单，不访问平台。",
    )
    export.add_argument(
        "--database", type=Path, default=Path("data/alpha_garden.sqlite3"),
        help="源数据库路径；清单写入同目录的 submitted-formulas.md。",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "web":
        from alpha_garden.web import serve_console
        from execution.console import ConsolePaths
        if not 1 <= arguments.port <= 65535:
            parser.error("端口必须在 1 到 65535 之间。")
        try:
            serve_console(ConsolePaths(arguments.database, arguments.settings, arguments.run_config, arguments.env),
                          port=arguments.port, read_only=arguments.read_only, assets=arguments.assets)
        except OSError as exc:
            print(f"网页服务启动失败：{exc}", file=sys.stderr)
            return 1
        return 0
    if arguments.command == "run":
        with run_progress_console():
            return _run_automated(arguments)
    if arguments.command == "resume":
        with run_progress_console():
            return _resume_automated(arguments)
    if arguments.command == "submit":
        if isinstance(arguments.target, int) and arguments.count is not None:
            parser.error("提交数量只能指定一次；按评级提交请使用 submit good 2。")
        return _submit_queue(arguments)
    if arguments.command == "export-submitted":
        try:
            with exclusive_run_process(arguments.database):
                path = export_submitted_formulas(arguments.database)
        except (OSError, sqlite3.Error, ValueError) as exc:
            print(f"公式清单导出失败：{exc}", file=sys.stderr)
            return 1
        print(f"已更新提交成功的公式清单：{path}")
        return 0
    if arguments.command != "init":
        raise AssertionError("command_dispatch_invalid")
    try:
        result = initialize_project(
            arguments.database,
            arguments.settings,
            arguments.env,
        )
    except (OSError, sqlite3.Error, ValueError, WorldQuantRequestError) as exc:
        print(f"初始化失败：{exc}", file=sys.stderr)
        return 1
    _print_initialization_result(result)
    return 0


def _run_automated(arguments: argparse.Namespace) -> int:
    try:
        limits = load_automated_run_limits(
            arguments.run_config,
            cycles=arguments.cycles,
            automatic_submissions_enabled=arguments.auto_submit,
            optimization_only=arguments.optimization_only,
        )
        completion = launch_automated_run(
            arguments.database,
            arguments.settings,
            arguments.env,
            limits=limits,
            run_created=_print_automated_run_created,
        )
    except KeyboardInterrupt:
        print("运行已中断。再次 run 将按新参数开批次并收尾旧任务；resume 可继续原批次。")
        return 130
    except (OSError, sqlite3.Error, ValueError, WorldQuantRequestError) as exc:
        print(f"自动运行失败：{exc}", file=sys.stderr)
        return 1
    _print_automated_run_completion(completion)
    return 0 if completion.run.status == "completed" else 1


def _resume_automated(arguments: argparse.Namespace) -> int:
    try:
        completion = resume_automated_run(
            arguments.database,
            arguments.env,
            arguments.run_id,
        )
    except KeyboardInterrupt:
        print("恢复已中断，已有请求与结果保留。")
        return 130
    except (OSError, sqlite3.Error, ValueError, WorldQuantRequestError) as exc:
        print(f"自动运行恢复失败：{exc}", file=sys.stderr)
        return 1
    _print_automated_run_completion(completion)
    return 0 if completion.run.status == "completed" else 1


def _submit_queue(arguments: argparse.Namespace) -> int:
    grade = arguments.target if arguments.target in BELOW_TARGET_GRADES else None
    source = "qualified_archive" if grade is not None else "queue"
    count = arguments.target if isinstance(arguments.target, int) else arguments.count
    command = "submit " + grade.lower() if grade is not None else "submit"
    try:
        with run_progress_console():
            completion = submit_queued_alphas(
                arguments.database,
                arguments.env,
                max_submissions=count,
                source=source,
                grade=grade,
            )
    except KeyboardInterrupt:
        print(f"提交已中断，已有请求和结果保留；再次 {command} 可继续收尾与对账。")
        return 130
    except WorldQuantRequestError as exc:
        details = [exc.code]
        if exc.status_code is not None:
            details.append(f"HTTP {exc.status_code}")
        if exc.transport_error_type is not None:
            details.append(exc.transport_error_type)
        print("正式提交暂未完成：" + "；".join(details)
              + f"。已有进度保留，可再次 {command} 接续。", file=sys.stderr)
        return 1
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"正式提交失败：{exc}", file=sys.stderr)
        return 1
    _print_submission_queue_completion(completion, qualified=grade is not None)
    return 0 if completion.completed else 1


def main() -> int:
    return run()


def _run_cycles(value: str) -> int:
    try:
        cycles = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("轮数必须是正整数或 -1。") from exc
    if cycles == 0 or cycles < -1:
        raise argparse.ArgumentTypeError("轮数必须是正整数或 -1。")
    return cycles


def _submission_target(value: str) -> str | int:
    if value.upper() in BELOW_TARGET_GRADES | {"SPECTACULAR"}:
        return value.upper()
    try:
        return _submission_count(value)
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(
            "请输入正整数，或评级 inferior / average / good / excellent / spectacular。"
        ) from exc


def _submission_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("提交数量必须为正整数。") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("提交数量必须为正整数。")
    return count


def _print_initialization_result(result: ProjectInitializationResult) -> None:
    database_action = "已创建" if result.database_created else "已复用"
    semantics_action = "已写入" if result.semantics_initialized else "已核对一致"
    catalog_action = "已同步" if result.catalog_refreshed else "已复用"
    context = result.catalog.context
    print("初始化完成")
    print(f"1. 数据库：{database_action} {result.database_path}")
    print(f"2. 现行表：已创建或核对 {result.project_table_count} 张")
    print(
        "3. 项目自有目录："
        f"{semantics_action} {result.window_count} 个标准窗口、"
        f"{result.operator_role_count} 条算子角色关系、"
        f"{result.operator_output_count} 条算子输出映射"
    )
    print(
        f"4. WQB 平台目录：{catalog_action}，"
        f"账号范围 {result.catalog.account_scope}，"
        f"{context.instrument_type}/{context.region}/{context.universe}/delay={context.delay}，"
        f"{result.catalog.field_count} 个字段、"
        f"{result.catalog.operator_count} 个算子"
    )
    print(f"   目录更新时间：{result.catalog.synced_at}")
    print(f"   目录指纹：{result.catalog.catalog_fingerprint}")


def _print_automated_run_completion(completion: AutomatedRunCompletion) -> None:
    print("自动运行结束")
    print(f"运行 ID：{completion.run.run_id}")
    print(f"状态：{completion.run.status}")
    print(f"停止原因：{completion.run.stop_reason or '无'}")
    print(f"已结算轮数：{completion.run.current_cycle}")
    print(f"执行步数：{completion.step_count}")
    print(f"平台业务请求数：{completion.platform_request_count}")
    authentication = "是" if completion.authentication_performed else "否"
    print(f"本次是否完成认证：{authentication}")
    submission = (
        "开启，每轮处理全部当前合格候选"
        if completion.run.automatic_submissions_enabled
        else "关闭"
    )
    print(f"正式 Alpha 提交：{submission}")
    if completion.run.automatic_submissions_enabled:
        print(
            "正式提交结果："
            f"已领取正式提交权 {completion.formal_submission_claimed_count} 条，"
            f"已确认 {completion.formal_submission_confirmed_count} 条，"
            f"待对账 {completion.formal_submission_unresolved_count} 条"
        )


def _print_automated_run_created(run: AutomatedRunRecord) -> None:
    print(f"自动运行已创建，可恢复 ID：{run.run_id}", flush=True)
    if run.optimization_only:
        print("运行模式：评级提升专项；不分配探索、SC治理或反转；共用父代20次额度，候选不足按实际数量运行。", flush=True)
    if run.max_cycles == -1 and run.max_backtests == 0:
        cycle_plan = "持续运行，不设总回测条数上限"
    elif run.max_cycles == -1:
        cycle_plan = f"按已冻结额度持续到 {run.max_backtests} 条真实回测"
    else:
        cycle_plan = f"运行 {run.max_cycles} 轮/{run.max_backtests} 条真实回测"
    if run.optimization_only:
        cycle_plan = cycle_plan.replace("运行 ", "最多运行 ", 1)
    if uses_continuous_recovery(run):
        recovery_plan = (
            f"任务等待上限 {run.max_pending_seconds} 秒，超时释放本地名额并保留远端身份，"
            "通信故障自动退避恢复，整批失败记录后继续"
        )
    else:
        recovery_plan = (
            f"单任务 pending 上限 {run.max_pending_seconds} 秒，"
            f"连续 {run.max_consecutive_failures} 轮整批失败停止，"
            f"连续请求故障上限 {run.max_request_failures}"
        )
    batch_plan = (f"每轮回测上限 {run.backtest_count}，" if run.optimization_only else
                  f"每轮生成 {run.generation_count}，每轮回测 {run.backtest_count}，")
    print(
        "运行计划："
        f"{batch_plan}"
        f"{cycle_plan}，"
        f"最大在途 {run.max_in_flight_backtests}；"
        f"{recovery_plan}",
        flush=True,
    )
    submission = (
        "开启，每轮处理全部当前合格候选"
        if run.automatic_submissions_enabled
        else "关闭"
    )
    print(f"正式 Alpha 提交：{submission}", flush=True)


def _print_submission_queue_completion(
    completion: SubmissionQueueCompletion,
    *, qualified: bool = False,
) -> None:
    remaining_location = "合格表" if qualified else "队列"
    print("正式提交批次结束")
    print(f"账号范围：{completion.account_scope}")
    print(f"启动时待处理：{completion.initial_queue_count} 条")
    if completion.max_submissions is not None:
        print(f"本次成功提交上限：{completion.max_submissions} 条")
    print(f"执行步数：{completion.step_count}")
    print(f"平台业务请求数：{completion.platform_request_count}")
    authentication = "是" if completion.authentication_performed else "否"
    print(f"本次是否完成认证：{authentication}")
    print(
        "处理结果："
        f"已确认提交 {completion.submitted_count} 条，"
        f"其中平台原已 ACTIVE（已提交） {completion.already_active_count} 条；"
        f"已领取正式提交权 {completion.submission_claimed_count} 条，"
        f"检查不通过 {completion.ineligible_count} 条，"
        f"明确失败 {completion.failed_count} 条，"
        f"待对账 {completion.unresolved_count} 条，"
        f"本批仍在{remaining_location} {completion.remaining_queue_count} 条"
    )
    if completion.submission_limit_reached:
        print(f"已达到成功提交数量，剩余候选保留在{remaining_location}。")
    elif completion.max_submissions is not None and completion.completed:
        print("启动时的候选已处理完，成功提交数量未达到上限。")


if __name__ == "__main__":
    raise SystemExit(main())
