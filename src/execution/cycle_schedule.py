from __future__ import annotations

import sqlite3
from execution.runs import can_plan_automated_cycle
from execution.real_backtests import account_in_flight_backtest_count
from persistence.runs import AutomatedRunRecord


def unsettled_cycle_numbers(
    connection: sqlite3.Connection,
    run_id: str,
) -> tuple[int, ...]:
    return tuple(
        row[0]
        for row in connection.execute(
            """SELECT DISTINCT links.cycle_number FROM automated_run_backtests links
        LEFT JOIN automated_cycle_settlements settled
          ON settled.run_id = links.run_id AND settled.cycle_number = links.cycle_number
        WHERE links.run_id = ? AND settled.run_id IS NULL ORDER BY links.cycle_number""",
            (run_id,),
        )
    )


def scheduled_cycle_number(
    connection: sqlite3.Connection,
    run: AutomatedRunRecord,
    *,
    allow_planning: bool = True,
) -> int:
    rows = connection.execute(
        """SELECT links.cycle_number, tasks.status
        FROM automated_run_backtests links JOIN backtest_tasks tasks USING(task_id)
        WHERE links.run_id = ? ORDER BY links.cycle_number, links.task_id""",
        (run.run_id,),
    ).fetchall()
    latest = max((row[0] for row in rows), default=0)
    unsettled = unsettled_cycle_numbers(connection, run.run_id)
    for number in unsettled:
        if all(row[1] in {"completed", "failed"} for row in rows if row[0] == number):
            return number
    for number, status in rows:
        if status == "created":
            return number
    active = [row for row in rows if row[1] in {"pending", "submission_unknown"}]
    if not active:
        return latest + 1
    if (
        allow_planning
        and not run.optimization_only
        and can_plan_automated_cycle(run, latest_cycle=latest, prepared_backtests=len(rows))
        and account_in_flight_backtest_count(connection, run.account_scope) < run.max_in_flight_backtests
        and all(status == "submission_unknown" for _, status in active)
    ):
        return latest + 1
    return active[0][0]
