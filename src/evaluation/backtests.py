from __future__ import annotations

from dataclasses import dataclass

from persistence.backtests import (
    BacktestCheckRecord,
    BacktestSnapshot,
    BacktestYearlyStatRecord,
)
from worldquant.backtests import (
    STANDARD_NON_SC_CHECK_NAMES,
    STANDARD_REGULAR_CHECK_NAMES,
)


@dataclass(frozen=True, slots=True)
class CheckFailure:
    check_name: str
    category: str


@dataclass(frozen=True, slots=True)
class BacktestEvaluation:
    task_id: str
    state: str
    check_details_captured: bool
    checks: tuple[BacktestCheckRecord, ...]
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    pending_checks: tuple[str, ...]
    missing_checks: tuple[str, ...]
    unexpected_checks: tuple[str, ...]
    check_set_complete: bool
    non_sc_check_set_complete: bool
    long_count: int | None
    short_count: int | None
    yearly_stats: tuple[BacktestYearlyStatRecord, ...]
    failures: tuple[CheckFailure, ...]
    negative_metrics: tuple[str, ...]


_CHECK_FAILURE_CATEGORIES = {
    "LOW_SHARPE": "signal_quality",
    "LOW_FITNESS": "signal_quality",
    "LOW_SUB_UNIVERSE_SHARPE": "robustness",
    "CONCENTRATED_WEIGHT": "concentration",
    "HIGH_TURNOVER": "turnover",
    "LOW_TURNOVER": "turnover",
    "SELF_CORRELATION": "self_correlation",
    "MATCHES_COMPETITION": "competition",
}
_SIGNED_METRICS = (
    "sharpe",
    "fitness",
    "returns",
    "margin",
    "pnl",
)


def evaluate_backtest(snapshot: BacktestSnapshot) -> BacktestEvaluation:
    if snapshot.task.status != "completed" or snapshot.result is None:
        raise ValueError("evaluation_completed_backtest_required")
    if snapshot.result.task_id != snapshot.task.task_id:
        raise ValueError("evaluation_result_identity_mismatch")
    if snapshot.yearly_stats is None:
        raise ValueError("evaluation_yearly_stats_not_captured")

    checks = snapshot.result.checks
    if not isinstance(checks, tuple) or not checks:
        raise ValueError("evaluation_checks_invalid")
    passed: list[str] = []
    failed: list[str] = []
    pending: list[str] = []
    names: set[str] = set()
    for check in checks:
        if not isinstance(check, BacktestCheckRecord):
            raise ValueError("evaluation_checks_invalid")
        name = check.name.strip()
        if not name or name in names:
            raise ValueError("evaluation_check_name_invalid")
        names.add(name)
        status = check.status.strip().upper()
        if status == "PASS":
            passed.append(name)
        elif status == "FAIL":
            failed.append(name)
        else:
            pending.append(name)

    missing = tuple(sorted(STANDARD_REGULAR_CHECK_NAMES - names))
    unexpected = tuple(sorted(names - STANDARD_REGULAR_CHECK_NAMES))
    check_set_complete = not missing and not unexpected
    non_sc_check_set_complete = (
        not unexpected and not (STANDARD_NON_SC_CHECK_NAMES - names)
    )
    state = (
        "failed"
        if failed
        else "pending"
        if pending or not check_set_complete
        else "passed"
    )
    failures = tuple(
        CheckFailure(
            check_name=name,
            category=_CHECK_FAILURE_CATEGORIES.get(name, "platform_check"),
        )
        for name in failed
    )
    negative_metrics = tuple(
        name
        for name in _SIGNED_METRICS
        if (value := getattr(snapshot.result, name)) is not None and value < 0
    )
    return BacktestEvaluation(
        task_id=snapshot.task.task_id,
        state=state,
        check_details_captured=snapshot.result.check_details_captured,
        checks=checks,
        passed_checks=tuple(passed),
        failed_checks=tuple(failed),
        pending_checks=tuple(pending),
        missing_checks=missing,
        unexpected_checks=unexpected,
        check_set_complete=check_set_complete,
        non_sc_check_set_complete=non_sc_check_set_complete,
        long_count=snapshot.result.long_count,
        short_count=snapshot.result.short_count,
        yearly_stats=snapshot.yearly_stats,
        failures=failures,
        negative_metrics=negative_metrics,
    )
