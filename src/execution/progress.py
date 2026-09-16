from __future__ import annotations

import logging
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from generation.self_correlation import SELF_CORRELATION_REPAIR_FAMILIES
from generation.direction import DIRECTION_REVERSAL
from persistence.backtests import BacktestSnapshot, get_backtest_mutation, get_backtest_task
from worldquant.backtests import STANDARD_NON_SC_CHECK_NAMES, STANDARD_REGULAR_CHECK_NAMES
from persistence.runs import (
    get_automated_cycle_settlement,
    get_automated_run_backtest_by_task,
    list_automated_run_backtests,
)

if TYPE_CHECKING:
    from execution.driver import AutomatedRunAdvance


LOGGER_NAME = "execution.progress"


def phase(cycle_number: int, number: int, message: str) -> None:
    logging.getLogger(LOGGER_NAME).info(
        "第 %s 轮 阶段[%s] %s", cycle_number, number, message
    )


def live_progress(message: str) -> None:
    logging.getLogger(LOGGER_NAME).info(message, extra={"transient": True})


class RunProgress:
    """Console observations only; never a source of scheduling or result state."""

    def __init__(self, database_path: str | Path, run_id: str) -> None:
        self.database_path = database_path
        self.run_id = run_id
        # Cleared at cycle settlement; never the history of an unlimited run.
        self.waiting: dict[str, tuple[str, str]] = {}
        self.last_wait: str | None = None

    def notice(self, message: str) -> None:
        if message != self.last_wait:
            logging.getLogger(LOGGER_NAME).info(message)
            self.last_wait = message

    def backtest(
        self, task_id: str, *, action: str = "", snapshot: BacktestSnapshot | None = None
    ) -> None:
        if action in {"submission_check_observed", "submission_check_wait"}:
            return
        logger = logging.getLogger(LOGGER_NAME)
        if not logger.isEnabledFor(logging.INFO):
            return
        try:
            with closing(_read_connection(self.database_path)) as connection:
                snapshot = snapshot or get_backtest_task(connection, task_id)
                link = get_automated_run_backtest_by_task(connection, task_id)
                if snapshot is None or link is None:
                    return
                message = _backtest_message(snapshot, action)
                if message is None or self.waiting.get(task_id) == message:
                    return
                mutation = get_backtest_mutation(connection, task_id)
                source = "探索"
                if mutation is not None:
                    source = ("反转" if mutation.action == DIRECTION_REVERSAL else
                              "SC治理" if mutation.action in SELF_CORRELATION_REPAIR_FAMILIES else "变异")
                peers = list_automated_run_backtests(
                    connection, link.run_id, cycle_number=link.cycle_number
                )
                states = connection.execute(
                    "SELECT t.status, COUNT(*) AS count FROM backtest_tasks t "
                    "JOIN automated_run_backtests l ON l.task_id=t.task_id "
                    "WHERE l.run_id=? AND l.cycle_number=? GROUP BY t.status",
                    (link.run_id, link.cycle_number),
                ).fetchall()
            index = next((i for i, item in enumerate(peers, 1) if item.task_id == task_id), None)
            if index is None:
                return
            owner = "" if link.run_id == self.run_id else f" 旧{link.run_id.removeprefix('run_')[:8]}"
            identity = snapshot.task.platform_alpha_id or task_id.removeprefix("backtest_")[:10]
            label, detail = message
            if snapshot.task.status in {"completed", "failed"}:
                index_width = max(2, len(str(len(peers))))
                grade = snapshot.result.grade if snapshot.result is not None else None
                logger.info("%s %s 第%s轮 %0*d/%s %s %-10s %s%s",
                            _pad_column(f"[{label}]", 8), _pad_column(f"[{grade or '未知'}]", 13), link.cycle_number,
                            index_width, index, len(peers), _pad_column(f"[{source}]", 8),
                            identity, detail, owner)
            else:
                counts = {row["status"]: row["count"] for row in states}
                done = counts.get("completed", 0) + counts.get("failed", 0)
                active = counts.get("pending", 0) + counts.get("submission_unknown", 0)
                live_progress(f"第{link.cycle_number}轮 完成{done}/{len(peers)} 在途{active} 待发{counts.get('created', 0)} | [{source}] {identity} {detail}{owner}")
            self.last_wait = None
            self.waiting[task_id] = message
        except (OSError, sqlite3.Error):
            # A failed progress read must not change a task or fail the run.
            logger.info("进度暂时无法读取；任务仍按原流程执行。")

    def advanced(self, advance: AutomatedRunAdvance) -> None:
        if not logging.getLogger(LOGGER_NAME).isEnabledFor(logging.INFO):
            return
        if advance.action == "cycle_planned":
            phase(advance.cycle_number, 3, "开始回测，按可用并发名额推进")
            logging.getLogger(LOGGER_NAME).info("结果列：状态 | 平台评级 | 轮次/序号 | 探索/SC治理/变异/反转 | Alpha ID | S=夏普 F=适应度 T=换手率；通过/未通过/待定仅表示基础检查，评级缺失显示未知，异常见行末原因")
        if advance.task_id and advance.backtest_action not in {None, "cycle_terminal"}:
            self.backtest(advance.task_id, action=advance.backtest_action)
        if advance.action == "request_retry_scheduled":
            self.notice(
                f"请求暂未成功（{advance.run.last_request_failure_code}），"
                f"{advance.retry_after_seconds:g} 秒后重试"
            )
        elif advance.backtest_action == "submission_rate_limited":
            self.notice(
                f"发送回测遇到429限流，冷却{advance.run.last_request_retry_after_seconds:g}秒；"
                f"重试{advance.run.request_failure_count}/{advance.run.max_request_failures}，继续收取在途结果"
            )
        elif advance.backtest_action == "poll_rate_limited":
            self.notice("收取结果遇到429限流，该任务延后检查，结果保持待定")
        elif advance.backtest_action == "poll_retry_scheduled":
            self.notice("收取结果暂未成功，该任务延后检查，不影响发送限流计数")
        elif advance.backtest_action == "submission_cooldown":
            live_progress("发送回测处于限流冷却，批次继续保留")
        elif advance.backtest_action == "capacity_wait":
            live_progress("回测名额已满，等待结果并对账，不重复发送")
        elif advance.backtest_action == "retry_wait":
            live_progress("回测仍在进行，按平台要求等待后继续检查")
        elif advance.backtest_action == "submission_check_wait":
            live_progress("入队检查等待平台重试时间；不会生成或发送下一批公式")
        if advance.action == "formal_submission_advanced":
            self.notice(f"第 {advance.cycle_number} 轮 阶段[4] 自动正式提交：{advance.formal_submission_action}")
        if advance.action in {"cycle_settled", "run_stopped"} and advance.backtest_action == "cycle_terminal":
            try:
                with closing(_read_connection(self.database_path)) as connection:
                    settled = get_automated_cycle_settlement(
                        connection, self.run_id, advance.cycle_number
                    )
                    if settled is not None:
                        for link in list_automated_run_backtests(
                            connection, self.run_id, cycle_number=advance.cycle_number
                        ):
                            self.waiting.pop(link.task_id, None)
                if settled is not None:
                    phase(advance.cycle_number, 5, "结算与学习反馈完成")
            except (OSError, sqlite3.Error):
                pass


def _read_connection(database_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(database_path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _pad_column(text: str, width: int) -> str:
    display_width = sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
                        for char in text)
    return text + " " * max(0, width - display_width)


def _backtest_message(snapshot: BacktestSnapshot, action: str) -> tuple[str, str] | None:
    task = snapshot.task
    if task.status == "completed":
        result = snapshot.result
        if result is None:
            return "待定", "状态为完成，但指标缺失，不能展示得分"
        statuses = {check.name: check.status for check in result.checks}
        failed = any(statuses.get(name) == "FAIL" for name in STANDARD_NON_SC_CHECK_NAMES)
        passed = all(statuses.get(name) == "PASS" for name in STANDARD_NON_SC_CHECK_NAMES)
        passed = passed and not (set(statuses) - STANDARD_REGULAR_CHECK_NAMES)
        # Unresolved checks do not produce a made-up pass/fail verdict.
        checking = "未通过" if failed else ("通过" if passed else "待定")
        return checking, (
            f"S{result.sharpe:6.2f} F{result.fitness:6.2f} "
            f"T{result.turnover:6.1%}"
        )
    if task.status == "failed":
        if task.failure_code == "platform_submission_rate_limit_exhausted":
            return "异常", "发送失败（429重试耗尽，未被平台接受）"
        if task.failure_code == "platform_response_retry_exhausted":
            return "异常", task.failure_message or "结果读取失败；远端结果未确认"
        if task.failure_code == "submission_outcome_timeout":
            return "异常", "接收超时；远端未知，不重发"
        if task.failure_code == "platform_pending_timeout":
            return "异常", "等待超时；远端结果未决"
        detail = f"回测失败（{task.failure_code or '原因未提供'}）"
        if task.failure_message:
            detail += "：" + " ".join(task.failure_message.split())
        return "异常", detail
    if task.status == "submission_unknown":
        return "", (
            "回测请求发送中，尚未确认接收" if action == "submission_in_progress"
            else "提交结果未知，保留名额，只对账、不重发"
        )
    if task.status == "pending":
        if action == "recovery_deferred":
            return "", "旧请求收取暂未成功，保留名额，稍后重查"
        if snapshot.result is not None:
            return "", "指标已收到，等待年度数据；尚未完整完成"
        if task.platform_alpha_id is not None:
            return "", "平台计算已结束，等待完整指标与检查"
        return "", "已提交回测，等待结果"
    return None
