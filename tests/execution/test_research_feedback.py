from datetime import datetime

import pytest

from alpha_garden.cli import run
from execution.backtests import fail_backtest_task, record_submission_accepted
from execution.cycles import plan_automated_cycle
from execution.research_feedback import load_research_feedback
from persistence.database import open_database, read_database
from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
from tests.execution import test_cycles as cycle_fixtures
from tests.execution.test_qualified_archive import complete_candidate


@pytest.fixture
def researcher():
    fixture = cycle_fixtures.AutomatedCyclePlanningTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def test_feedback_and_cli_are_read_only_and_latest_error_is_pending(researcher, capsys):
    run_id = researcher._start_run()
    plan = plan_automated_cycle(researcher.database_path, run_id=run_id,
                               created_at="2026-08-30T00:04:00+08:00")
    with open_database(researcher.database_path) as connection:
        first, second = plan.backtests
        for item in plan.backtests:
            record_submission_accepted(connection, item.task.task_id, remote_id="sim-" + item.task.task_id,
                                       observed_at="2026-09-01T00:01:00+00:00")
        complete_candidate(connection, first)
        fail_backtest_task(connection, second.task.task_id, failure_code="request_failed",
                           failure_message="isolated sample", observed_at="2026-09-01T00:02:00+00:00")
        save_submission_check(connection, SubmissionCheckRecord(first.task.task_id,
            "2026-09-02T00:00:00+00:00", None, "request_failed", attempt_count=2))
    before = researcher.database_path.read_bytes()
    with read_database(researcher.database_path) as connection:
        report = load_research_feedback(connection, run_id,
                                       observed_at=datetime.fromisoformat("2026-09-03T00:00:00+00:00"))
        assert (report.progress.phase, report.progress.resolved_count) == ("cold_start", 0)
        source, = report.sources
        assert (source.source, source.planned, source.attempted, source.qualified,
                source.rejected, source.pending, source.request_failed) == ("exploration", 2, 2, 0, 0, 1, 1)
    assert run(["research-status", run_id, "--database", str(researcher.database_path)]) == 0
    output = capsys.readouterr().out
    assert "待定 1" in output and "请求/任务错误 1" in output and "此统计只读" in output
    assert researcher.database_path.read_bytes() == before


def test_quality_feedback_uses_actual_grade_and_full_checks(researcher):
    researcher._optimization_parent()
    run_id = researcher._start_run(optimization_only=True)
    plan = plan_automated_cycle(researcher.database_path, run_id=run_id,
                               created_at="2026-08-30T00:04:00+08:00")
    with open_database(researcher.database_path) as connection:
        for index, item in enumerate(plan.backtests):
            record_submission_accepted(connection, item.task.task_id, remote_id="sim-" + item.task.task_id,
                                       observed_at="2026-09-01T00:01:00+00:00")
            complete_candidate(connection, item, sharpe=1.6 if index == 0 else 1.2,
                               grade="EXCELLENT" if index == 0 else "GOOD")
    with read_database(researcher.database_path) as connection:
        result = load_research_feedback(connection, run_id,
            observed_at=datetime.fromisoformat("2026-09-02T00:00:00+00:00"))
        assert result.sources[0].source == "quality"
        assert result.sources[0].qualified == 2
        assert dict(result.quality_outcomes) == {"conflict": 1, "safe_progress": 1}
        assert not result.progress.monitoring_enabled
