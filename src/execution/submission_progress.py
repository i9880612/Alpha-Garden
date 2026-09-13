from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

from execution.progress import LOGGER_NAME, live_progress
from persistence.submissions import (
    FORMAL_SUBMISSION_ACTIVE_STATUSES,
    list_formal_submission_attempts,
)
from submission.formal import assess_formal_check_payload


class SubmissionProgress:
    """A read-only projection of this command's frozen submission scope."""

    def __init__(
        self, database_path: str | Path, account_scope: str, task_ids: frozenset[str],
        *, max_submissions: int | None = None,
    ):
        self.database_path = Path(database_path).resolve()
        self.account_scope = account_scope
        self.task_ids = task_ids
        self.max_submissions = max_submissions
        self.reported: set[str] = set()
        self.current_task: str | None = None

    def started(self):
        if self.max_submissions is not None:
            logging.getLogger(LOGGER_NAME).info(
                "正式提交：本次候选 %s 条（含待恢复项），成功提交 %s 条后停止；"
                "检查失败跳过，原已提交的同步不计入数量。",
                len(self.task_ids), self.max_submissions,
            )
            return
        logging.getLogger(LOGGER_NAME).info(
            "正式提交：本次处理 %s 条（含待恢复项）", len(self.task_ids)
        )

    def observe(self, task_id: str | None):
        logger = logging.getLogger(LOGGER_NAME)
        if not logger.isEnabledFor(logging.INFO):
            return
        try:
            with closing(
                sqlite3.connect(self.database_path.as_uri() + "?mode=ro", uri=True)
            ) as connection:
                connection.row_factory = sqlite3.Row
                attempts = tuple(
                    item
                    for item in list_formal_submission_attempts(
                        connection, account_scope=self.account_scope
                    )
                    if item.task_id in self.task_ids
                )
                current = next(
                    (item for item in attempts if item.task_id == task_id), None
                )
                row = connection.execute(
                    "SELECT platform_alpha_id FROM backtest_tasks WHERE task_id=?",
                    (task_id,),
                ).fetchone()
            statuses = [item.status for item in attempts]
            submitted, rejected, failed = (
                statuses.count(state) for state in ("submitted", "ineligible", "failed")
            )
            pending = sum(
                status in FORMAL_SUBMISSION_ACTIVE_STATUSES for status in statuses
            )
            passed = sum(
                assess_formal_check_payload(json.loads(item.check_payload_json)).state
                == "passed"
                for item in attempts
                if item.check_payload_json is not None
            )
            done = submitted + rejected + failed
            summary = f"通过{passed} 已交{submitted} 未过{rejected} 失败{failed} 待定{pending}"
            if current is None:
                logger.info("提交进度 %s/%s | %s", done, len(self.task_ids), summary)
                return
            identity = (
                row["platform_alpha_id"] if row else None
            ) or current.task_id.removeprefix("backtest_")[:10]
            label = {
                "detail_pending": "读取详情",
                "check_pending": "检查中",
                "ready": "检查通过",
                "submitting": "发送中",
                "confirmation_pending": "确认中",
                "submission_unknown": "结果未知",
                "submitted": "已提交",
                "ineligible": "检查未过",
                "failed": "失败",
            }[current.status]
            if current.status == "failed":
                if current.failure_code == "formal_submission_check_timeout":
                    label = "提交前等待超时，已跳过"
                elif (current.failure_code or "").startswith("formal_submission_read_failed:"):
                    label = "提交前读取失败，已跳过"
            if current.failure_code == "formal_submission_grade_unavailable":
                label = "评级未确认，已跳过"
            elif (current.failure_code or "").startswith("formal_submission_grade_below_target:"):
                grade = current.failure_code.rsplit(":", 1)[1]
                label = f"评级 {grade}，未达 SPECTACULAR，已跳过"
            elif (current.failure_code or "").startswith("formal_submission_grade_changed:"):
                _, before, after = current.failure_code.split(":")
                label = f"评级从 {before} 变为 {after}，不符合所选等级，已跳过"
            if current.status in {"submitted", "ineligible", "failed"}:
                if current.task_id not in self.reported:
                    logger.info(
                        "提交 %s/%s %s %s | %s",
                        done,
                        len(self.task_ids),
                        identity,
                        label,
                        summary,
                    )
                    self.reported.add(current.task_id)
            else:
                if current.task_id != self.current_task:
                    logger.info(
                        "正在处理 %s/%s %s", done + 1, len(self.task_ids), identity
                    )
                live_progress(
                    f"提交 {done + 1}/{len(self.task_ids)} {identity} {label} | {summary}"
                )
            self.current_task = current.task_id
        except (OSError, sqlite3.Error, ValueError):
            logger.info("提交进度暂不可读；不改变提交状态")
