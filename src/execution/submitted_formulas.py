from __future__ import annotations

import logging
import math
import sqlite3
import tempfile
from collections.abc import Mapping
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from persistence.submissions import list_platform_submitted_alphas


def export_submitted_formulas(database_path: str | Path) -> Path:
    """Rebuild the local formula document from committed submission facts only."""
    database = Path(database_path).resolve()
    destination = database.with_name("submitted-formulas.md")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        scopes = connection.execute(
            "SELECT DISTINCT account_scope FROM platform_submitted_alphas ORDER BY account_scope"
        ).fetchall()
        records = [record for scope in scopes for record in list_platform_submitted_alphas(
            connection, account_scope=scope["account_scope"],
        )]
    records.sort(key=lambda record: (
        datetime.fromisoformat(record.date_submitted), record.account_scope,
        record.platform_alpha_id,
    ), reverse=True)
    lines = [
        "# 已确认提交的公式", "",
        f"来源：同目录数据库 `{database.name}` 的 `platform_submitted_alphas`。",
        "本文件自动生成，可从数据库重建；请勿手动维护。仅包含本地已确认的提交事实，按提交时间倒序排列。",
        "仅列编号、提交时间和简要指标。时间为北京时间（UTC+8），缺失指标标为未记录。", "",
        f"共 **{len(records)}** 条。", "",
        "| Alpha ID | 提交时间 | 夏普 | 适应度 | 换手率 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for record in records:
        metrics = record.raw_payload.get("is")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        lines.append(
            f"| {record.platform_alpha_id} | {_local_time(record.date_submitted)} | "
            f"{_metric(metrics.get('sharpe'))} | {_metric(metrics.get('fitness'))} | "
            f"{_metric(metrics.get('turnover'), percent=True)} |"
        )
    content = "\n".join(lines) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=".submitted-formulas-", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def refresh_submitted_formulas(database_path: str | Path) -> None:
    """A document I/O failure must not undo or retry a confirmed submission."""
    try:
        export_submitted_formulas(database_path)
    except (OSError, sqlite3.Error) as exc:
        logging.getLogger("execution.progress").warning(
            "已确认提交事实已保存，但公式清单更新失败（%s）；"
            "可运行 alpha-garden export-submitted --database \"%s\" 重建。",
            type(exc).__name__, database_path,
        )


def _local_time(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(
        timezone(timedelta(hours=8))
    ).strftime("%Y-%m-%d %H:%M:%S")


def _metric(value: object, *, percent: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "未记录"
    return f"{value:.1%}" if percent else f"{value:.2f}"
