from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from execution.cycle_backtests import (
    advance_automated_cycle_backtests,
    remaining_automated_pending_seconds,
    settle_automated_cycle,
    synchronize_automated_cycle_submission_queue,
)
from execution.cycle_candidates import CycleCandidatePlanningStopped, OptimizationCandidatesExhausted
from execution.cycles import plan_automated_cycle
from execution.recovery import capture_next_recovery_series
from execution.seed_evidence import capture_next_seed_series
from execution.run_recovery import advance_stopped_run_check
from execution.progress import phase
from execution.cycle_schedule import scheduled_cycle_number
from execution.request_failures import (
    handle_automated_request_failure,
    remaining_automated_request_retry_seconds,
)
from execution.runs import (
    AutomatedRunPaused,
    clear_automated_request_failures,
    complete_automated_run_after_candidate_planning_stop,
    complete_optimization_run,
    decide_automated_cycle_transition,
    start_automated_run,
    uses_continuous_recovery,
    submission_rate_limited,
)
from execution.submissions import (
    FormalSubmissionAdvance,
    advance_automated_run_formal_submissions,
    fail_unclaimed_automated_submission,
)
from persistence.database import open_database
from persistence.runs import (
    AutomatedRunRecord,
    get_automated_run,
    list_automated_run_backtests,
)
from worldquant.client import WorldQuantClient, WorldQuantRequestError


@dataclass(frozen=True, slots=True)
class AutomatedRunAdvance:
    run: AutomatedRunRecord
    action: str
    cycle_number: int | None
    task_id: str | None
    backtest_action: str | None
    formal_submission_action: str | None
    platform_request_performed: bool
    retry_after_seconds: float | None = None


def advance_automated_run(
    database_path: str | Path,
    client: WorldQuantClient,
    run_id: str,
    *,
    observed_at: str,
    checkpoint: Callable[[], object] | None = None,
) -> AutomatedRunAdvance:
    observed = _timestamp(observed_at)
    run = _load_run(database_path, run_id)
    if run.stop_reason == "user_paused":
        raise AutomatedRunPaused()
    if run.status in {"completed", "failed"}:
        return _result(run, action="run_stopped")
    if run.status == "created":
        started = start_automated_run(
            database_path,
            run.run_id,
            started_at=observed_at,
        )
        return _result(started, action="run_started")
    if run.status != "running":
        raise ValueError("automated_run_status_invalid")
    if run.started_at is None or observed < _timestamp(run.started_at):
        raise ValueError("automated_run_observed_at_invalid")
    retry_after = remaining_automated_request_retry_seconds(run, observed)
    if retry_after is not None and not submission_rate_limited(run):
        return _result(
            run,
            action="retry_wait",
            cycle_number=run.current_cycle + 1,
            retry_after_seconds=_bounded_wait(
                database_path,
                run.run_id,
                observed_at=observed_at,
                requested_seconds=retry_after,
            ),
        )

    with open_database(database_path) as connection:
        cycle_number = scheduled_cycle_number(connection, run)
    if not _cycle_has_tasks(database_path, run.run_id, cycle_number):
        old_check = advance_stopped_run_check(
            database_path, client, account_scope=run.account_scope, observed_at=observed_at,
        )
        if old_check is not None:
            return _result(
                run, action="stopped_checks_advanced", cycle_number=cycle_number,
                task_id=old_check.snapshot.task.task_id, backtest_action=old_check.action,
                platform_request_performed=old_check.platform_request_performed,
                retry_after_seconds=old_check.retry_after_seconds,
            )
        seed_delay = capture_next_seed_series(
            database_path, client, account_scope=run.account_scope, observed_at=observed_at,
            admit_seeds=not run.optimization_only,
        )
        if seed_delay is not None:
            return _result(run, action="seed_evidence_captured", cycle_number=cycle_number,
                           platform_request_performed=True, retry_after_seconds=seed_delay)
        try:
            delay = None if run.optimization_only else capture_next_recovery_series(
                database_path,
                client,
                account_scope=run.account_scope,
                observed_at=observed_at,
                retry_interval_seconds=run.max_pending_seconds,
            )
        except WorldQuantRequestError as exc:
            failure = handle_automated_request_failure(
                database_path,
                run,
                exc,
                observed_at=observed_at,
                non_retryable_reason="platform_request_not_retryable",
            )
            return _result(
                failure.run,
                action="run_stopped"
                if failure.run.status == "failed"
                else "request_retry_scheduled",
                cycle_number=cycle_number,
                platform_request_performed=True,
                retry_after_seconds=failure.retry_after_seconds,
            )
        if delay is not None:
            if delay == 0:
                run = clear_automated_request_failures(database_path, run.run_id)
            return _result(
                run,
                action="recovery_evidence_captured"
                if delay == 0
                else "recovery_evidence_pending",
                cycle_number=cycle_number,
                platform_request_performed=True,
                retry_after_seconds=delay,
            )
        try:
            if checkpoint is not None:
                checkpoint()
            plan = plan_automated_cycle(
                database_path,
                run_id=run.run_id,
                created_at=observed_at,
                cycle_number=cycle_number,
            )
        except OptimizationCandidatesExhausted:
            formal_submission, response = _advance_formal_submission_phase(
                database_path, client, run, cycle_number=cycle_number, observed_at=observed_at,
            )
            if response is not None:
                return response
            assert formal_submission is not None
            phase(cycle_number, 2, "评级提升专项无可继续变异的合格候选，本次结束")
            completed = complete_optimization_run(database_path, run.run_id, completed_at=observed_at)
            return _result(completed, action="run_stopped", cycle_number=cycle_number,
                           formal_submission_action=formal_submission.action)
        except CycleCandidatePlanningStopped as exc:
            phase(
                cycle_number,
                1 if exc.stop_reason == "generation_attempt_budget_exhausted" else 2,
                f"候选规划未完成（{exc.stop_reason}），按已有任务状态收尾",
            )
            with open_database(database_path) as connection:
                active_cycle = scheduled_cycle_number(
                    connection,
                    run,
                    allow_planning=False,
                )
            if active_cycle != cycle_number:
                # Drain existing work before committing a planning terminal state.
                cycle_number = active_cycle
            else:
                return _finish_planning_stop(
                    database_path,
                    client,
                    run,
                    cycle_number,
                    observed_at,
                    exc,
                )
        else:
            return _result(
                _load_run(database_path, run.run_id),
                action="cycle_planned",
                cycle_number=plan.cycle_number,
            )

    try:
        backtest = advance_automated_cycle_backtests(
            database_path,
            client,
            run.run_id,
            observed_at=observed_at,
            cycle_number=cycle_number,
        )
    except WorldQuantRequestError as exc:
        if exc.outcome_unknown:
            raise ValueError("automated_run_unknown_request_not_isolated") from exc
        failure = handle_automated_request_failure(
            database_path,
            run,
            exc,
            observed_at=observed_at,
            non_retryable_reason="platform_request_not_retryable",
        )
        return _result(
            failure.run,
            action=(
                "run_stopped"
                if failure.run.status == "failed"
                else "request_retry_scheduled"
            ),
            cycle_number=cycle_number,
            retry_after_seconds=_bounded_wait(
                database_path,
                run.run_id,
                observed_at=observed_at,
                requested_seconds=failure.retry_after_seconds,
            ),
            platform_request_performed=True,
        )

    return _finish_backtest_advance(database_path, client, run, backtest, observed_at)


def _finish_planning_stop(database_path, client, run, cycle_number, observed_at, exc):
    formal_submission, response = _advance_formal_submission_phase(
        database_path,
        client,
        run,
        cycle_number=cycle_number,
        observed_at=observed_at,
    )
    if response is not None:
        return response
    assert formal_submission is not None
    stopped = complete_automated_run_after_candidate_planning_stop(
        database_path,
        run.run_id,
        completed_at=observed_at,
        reason=exc.stop_reason,
        diagnostic=exc.diagnostic,
    )
    return _result(
        stopped,
        action="run_stopped",
        cycle_number=cycle_number,
        formal_submission_action=formal_submission.action,
    )


def _finish_backtest_advance(database_path, client, run, backtest, observed_at):
    cycle_number = backtest.cycle_number
    if (backtest.platform_request_performed and backtest.run_status == "running"
            and (not submission_rate_limited(_load_run(database_path, run.run_id))
                 or backtest.action in {"submitted", "reconciliation_required"})):
        run = clear_automated_request_failures(database_path, run.run_id)
    else:
        run = _load_run(database_path, run.run_id)
    if backtest.run_status == "failed":
        return _result(
            run,
            action="run_stopped",
            cycle_number=cycle_number,
            task_id=(
                backtest.snapshot.task.task_id
                if backtest.snapshot is not None
                else None
            ),
            backtest_action=backtest.action,
            platform_request_performed=backtest.platform_request_performed,
        )
    if backtest.action == "cycle_terminal":
        synchronize_automated_cycle_submission_queue(
            database_path,
            run.run_id,
            cycle_number=cycle_number,
            observed_at=observed_at,
        )
        with open_database(database_path) as connection:
            transition = decide_automated_cycle_transition(
                connection,
                run,
                cycle_failed=backtest.counts.completed == 0,
            )
        formal_submission = None
        if transition.status != "failed":
            formal_submission, response = _advance_formal_submission_phase(
                database_path,
                client,
                run,
                cycle_number=cycle_number,
                observed_at=observed_at,
                backtest_action="cycle_terminal",
            )
            if response is not None:
                return response
            assert formal_submission is not None
        phase(cycle_number, 5, "结算与学习反馈中...")
        settlement = settle_automated_cycle(
            database_path,
            run.run_id,
            cycle_number=cycle_number,
            observed_at=observed_at,
        )
        return _result(
            settlement.run,
            action=(
                "run_stopped"
                if settlement.run.status in {"completed", "failed"}
                else "cycle_settled"
            ),
            cycle_number=cycle_number,
            backtest_action="cycle_terminal",
            formal_submission_action=(
                formal_submission.action if formal_submission is not None else None
            ),
        )
    return _result(
        run,
        action="backtest_advanced",
        cycle_number=cycle_number,
        task_id=(
            backtest.snapshot.task.task_id if backtest.snapshot is not None else None
        ),
        backtest_action=backtest.action,
        platform_request_performed=backtest.platform_request_performed,
        retry_after_seconds=_bounded_wait(
            database_path,
            run.run_id,
            observed_at=observed_at,
            requested_seconds=backtest.retry_after_seconds,
        ),
    )


def _advance_formal_submission_phase(
    database_path: str | Path,
    client: WorldQuantClient,
    run: AutomatedRunRecord,
    *,
    cycle_number: int,
    observed_at: str,
    backtest_action: str | None = None,
) -> tuple[FormalSubmissionAdvance | None, AutomatedRunAdvance | None]:
    try:
        formal_submission = advance_automated_run_formal_submissions(
            database_path,
            client,
            run.run_id,
            finalization_cycle_number=cycle_number,
            observed_at=observed_at,
        )
    except WorldQuantRequestError as exc:
        if exc.outcome_unknown:
            raise ValueError("formal_submission_unknown_request_not_isolated") from exc
        if not exc.retryable or (
            not uses_continuous_recovery(run)
            and run.request_failure_count + 1 >= run.max_request_failures
        ):
            fail_unclaimed_automated_submission(
                database_path,
                run.run_id,
                observed_at=observed_at,
                failure_code=f"formal_submission_request_failed:{exc.code}",
            )
        failure = handle_automated_request_failure(
            database_path,
            run,
            exc,
            observed_at=observed_at,
            non_retryable_reason="formal_submission_request_not_retryable",
        )
        return None, _result(
            failure.run,
            action=(
                "run_stopped"
                if failure.run.status == "failed"
                else "request_retry_scheduled"
            ),
            cycle_number=cycle_number,
            backtest_action=backtest_action,
            formal_submission_action="formal_submission_request_failed",
            retry_after_seconds=failure.retry_after_seconds,
            platform_request_performed=True,
        )
    if (
        formal_submission.platform_request_performed
        and formal_submission.run_status == "running"
    ):
        current_run = clear_automated_request_failures(
            database_path,
            run.run_id,
        )
    else:
        current_run = _load_run(database_path, run.run_id)
    if formal_submission.phase_terminal:
        return formal_submission, None
    return None, _result(
        current_run,
        action=(
            "run_stopped"
            if formal_submission.run_status == "failed"
            else "formal_submission_advanced"
        ),
        cycle_number=cycle_number,
        task_id=formal_submission.task_id,
        backtest_action=backtest_action,
        formal_submission_action=formal_submission.action,
        platform_request_performed=formal_submission.platform_request_performed,
        retry_after_seconds=formal_submission.retry_after_seconds,
    )


def _bounded_wait(
    database_path: str | Path,
    run_id: str,
    *,
    observed_at: str,
    requested_seconds: float | None,
) -> float | None:
    if requested_seconds is None:
        return None
    pending_seconds = remaining_automated_pending_seconds(
        database_path,
        run_id,
        observed_at=observed_at,
    )
    return (
        min(requested_seconds, pending_seconds)
        if pending_seconds is not None and pending_seconds > 0
        else requested_seconds
    )


def _load_run(database_path: str | Path, run_id: str) -> AutomatedRunRecord:
    with open_database(database_path) as connection:
        run = get_automated_run(connection, run_id)
    if run is None:
        raise ValueError("automated_run_missing")
    return run


def _cycle_has_tasks(
    database_path: str | Path,
    run_id: str,
    cycle_number: int,
) -> bool:
    with open_database(database_path) as connection:
        return any(
            link.cycle_number == cycle_number
            for link in list_automated_run_backtests(connection, run_id)
        )


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("automated_run_observed_at_invalid") from exc
    if parsed.utcoffset() is None:
        raise ValueError("automated_run_observed_at_invalid")
    return parsed


def _result(
    run: AutomatedRunRecord,
    *,
    action: str,
    cycle_number: int | None = None,
    task_id: str | None = None,
    backtest_action: str | None = None,
    formal_submission_action: str | None = None,
    platform_request_performed: bool = False,
    retry_after_seconds: float | None = None,
) -> AutomatedRunAdvance:
    return AutomatedRunAdvance(
        run=run,
        action=action,
        cycle_number=cycle_number,
        task_id=task_id,
        backtest_action=backtest_action,
        formal_submission_action=formal_submission_action,
        platform_request_performed=platform_request_performed,
        retry_after_seconds=retry_after_seconds,
    )
