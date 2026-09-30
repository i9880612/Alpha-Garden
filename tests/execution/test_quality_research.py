import pytest

from execution.backtests import cancel_unsubmitted_backtest_task, record_submission_unknown, record_submission_accepted
from execution.qualified_candidates import load_qualified_evolution
from execution.quality_research import load_quality_budgets, reserve_quality_task
from persistence.backtests import get_backtest_task
from persistence.database import open_database
from persistence.schema import initialize_database_schema
from tests.execution.test_qualified_archive import prepare_candidate, complete_candidate, FINISHED, ARCHIVED
from tests.execution.test_research_feedback import researcher


def test_quality_allowance_survives_descendants_reservations_reopen_and_uncertain_requests(tmp_path):
    path = tmp_path / "quality.sqlite3"
    with open_database(path) as connection:
        initialize_database_schema(connection)
        root = complete_candidate(connection, prepare_candidate(connection))
        parent = root
        for index in range(79):
            child = prepare_candidate(connection, parent=parent, start=False)
            reserve_quality_task(connection, parent=parent, child_task_id=child.task.task_id,
                                 root_task_id=root.task.task_id)
            if index in (19, 39, 59):
                record_submission_accepted(connection, child.task.task_id, remote_id=f"simulation-{index}", observed_at=FINISHED)
            else:
                record_submission_unknown(connection, child.task.task_id, observed_at=FINISHED)
            # Each 20th child can take over with a tiny gain; the lineage total survives.
            if index in (19, 39, 59):
                parent = complete_candidate(connection, get_backtest_task(connection, child.task.task_id),
                                            sharpe=1.5 + (index + 1) / 2000)
        last = prepare_candidate(connection, parent=parent, start=False)
        reserve_quality_task(connection, parent=parent, child_task_id=last.task.task_id,
                             root_task_id=root.task.task_id)
        # SC/ordinary tasks have no quality purpose and cannot charge this allowance.
        prepare_candidate(connection, parent=root)
        budget = load_quality_budgets(connection, (parent,))[parent.task.task_id]
        assert (budget.attempted, budget.reserved, budget.available, budget.unspent) == (79, 1, 0, 1)
        decision = next(item for item in load_qualified_evolution(connection) if item.task_id == parent.task.task_id)
        assert not decision.retired
        assert decision.remaining_attempts == 1
    with open_database(path) as connection:
        assert load_quality_budgets(connection, (parent,))[parent.task.task_id] == budget
        before = connection.execute("SELECT count(*) FROM backtest_tasks").fetchone()[0]
    with pytest.raises(ValueError, match="quality_lineage_budget_exhausted"):
        with open_database(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            extra = prepare_candidate(connection, parent=parent, start=False)
            reserve_quality_task(connection, parent=parent, child_task_id=extra.task.task_id,
                                 root_task_id=root.task.task_id)
    with open_database(path) as connection:
        assert connection.execute("SELECT count(*) FROM backtest_tasks").fetchone()[0] == before
        cancel_unsubmitted_backtest_task(connection, last.task.task_id, observed_at=ARCHIVED)
        assert load_quality_budgets(connection, (parent,))[parent.task.task_id].available == 1
        replacement = prepare_candidate(connection, parent=parent, start=False)
        reserve_quality_task(connection, parent=parent, child_task_id=replacement.task.task_id,
                             root_task_id=root.task.task_id)
        record_submission_unknown(connection, replacement.task.task_id, observed_at=ARCHIVED)
        budget = load_quality_budgets(connection, (parent,))[parent.task.task_id]
        assert (budget.attempted, budget.reserved, budget.available) == (80, 0, 0)


def test_automatic_planning_and_resume_cannot_renew_the_lineage_allowance(researcher):
    from execution.backtests import prepare_backtest_task, fail_backtest_task
    from execution.cycles import plan_automated_cycle
    from persistence.backtests import BacktestMutationRecord, create_backtest_mutation
    root_id = researcher._optimization_parent()
    with open_database(researcher.database_path) as connection:
        parent = get_backtest_task(connection, root_id)
        for index in range(79):
            child = prepare_backtest_task(connection, account_scope=parent.task.account_scope,
                formula=f"rank(ts_rank(close,{index + 100})+ts_zscore(open,66))",
                settings=researcher.settings, created_at="2026-08-30T00:04:00+08:00")
            create_backtest_mutation(connection, BacktestMutationRecord(child.task.task_id, parent.task.task_id,
                "single_window_mutation", "root", parent.task.formula, child.task.formula))
            reserve_quality_task(connection, parent=parent, child_task_id=child.task.task_id, root_task_id=root_id)
            record_submission_accepted(connection, child.task.task_id, remote_id=f"sim-{index}", observed_at=FINISHED)
            if index in (19, 39, 59, 78):
                parent = complete_candidate(connection, child, sharpe=1.5 + (index + 1) / 1000)
            else:
                fail_backtest_task(connection, child.task.task_id, failure_code="request_failed",
                                   failure_message="isolated sample", observed_at=FINISHED)
        assert load_quality_budgets(connection, (parent,))[parent.task.task_id].available == 1
    run_id = researcher._start_run(optimization_only=True, generation_count=6, backtest_count=4,
        max_backtests=4, created_at=ARCHIVED, started_at="2026-09-02T00:01:00+00:00")
    first = plan_automated_cycle(researcher.database_path, run_id=run_id, created_at="2026-09-02T00:02:00+00:00")
    assert len(first.backtests) == 1
    recovered = plan_automated_cycle(researcher.database_path, run_id=run_id, created_at="2026-09-02T00:03:00+00:00")
    assert recovered.recovered and recovered.backtests == first.backtests
    with open_database(researcher.database_path) as connection:
        budget = load_quality_budgets(connection, (parent,))[parent.task.task_id]
        assert (budget.attempted, budget.reserved, budget.available) == (79, 1, 0)
        # Historical optimization links identify quality attempts even without a purpose row.
        connection.execute("DELETE FROM quality_research_tasks WHERE task_id=?", (first.backtests[0].task.task_id,))
        assert load_quality_budgets(connection, (parent,))[parent.task.task_id] == budget
