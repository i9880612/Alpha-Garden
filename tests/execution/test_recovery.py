import math
import json
import pytest
from datetime import date, datetime, timedelta
from dataclasses import replace
from unittest.mock import Mock

from execution.recovery import capture_next_recovery_series, load_recovery_comparisons
from execution.seeds import load_signal_frontiers, synchronize_signal_seeds
from execution.cycles import plan_automated_cycle
from execution.catalog import load_generation_catalog
from execution.runs import record_automated_cycle_settlement
from execution.backtest_batches import AutomatedCandidateBacktest, prepare_automated_candidate_backtest_batch
from generation.candidate import exploration_candidate, CandidateChange
from generation.self_correlation import (
    SELF_CORRELATION_REPAIR_FAMILIES,
    SELF_CORRELATION_LIGHT_FAMILIES,
    SELF_CORRELATION_INTERNAL_FAMILIES,
    SELF_CORRELATION_HALF_FAMILIES,
    SELF_CORRELATION_RESEARCH_FAMILIES,
    iter_self_correlation_leaves, SelfCorrelationLeaf,
)
from generation.parser import parse_formula
from generation.formula import render_formula
from persistence.catalog import FieldCatalogContext
from persistence.submission_checks import SubmissionCheckRecord, save_submission_check
from persistence.backtests import BacktestMutationRecord, create_backtest_mutation, get_backtest_task
from learning.recovery import RecoveryComparison, recovery_comparisons, recovery_task_ids
from learning.self_correlation import SelfCorrelationReference
from persistence.database import open_database
from persistence.pnl import initialize_pnl_schema, list_pnl_series, save_pnl_series
from persistence.submissions import list_platform_submitted_alphas
from learning.seed_correlation import assess_seed_correlation
from worldquant.backtests import BacktestSettings, STANDARD_REGULAR_CHECK_NAMES, WorldQuantProtocolError
from worldquant.client import WorldQuantRequestError
from worldquant.pnl import PnlObservation
from tests.execution import test_cycles as fixtures
from tests.learning.test_recovery import series


@pytest.fixture
def pending_recovery_series(tmp_path, monkeypatch):
    database_path = tmp_path / "recovery.sqlite3"
    comparisons = tuple(
        RecoveryComparison(child, child, "parent-task", "account", child, "parent", "reference")
        for child in ("missing-child", "other-child")
    )
    monkeypatch.setattr("execution.recovery.load_recovery_comparisons", lambda *args, **kwargs: comparisons)
    points = [math.sin(i) for i in range(300)]
    with open_database(database_path) as connection:
        initialize_pnl_schema(connection)
        save_pnl_series(connection, series("parent", points))
        save_pnl_series(connection, series("reference", points))
        save_pnl_series(connection, series("missing-child", points, account="other-account"))
    return database_path, comparisons


def test_capture_does_not_decode_curves_to_choose_missing_evidence(pending_recovery_series, monkeypatch):
    database_path, _ = pending_recovery_series
    def forbid_decode(*args, **kwargs):
        pytest.fail("capture planning should inspect readiness, not decode curves")
    monkeypatch.setattr("persistence.pnl.json.loads", forbid_decode)
    client = Mock()
    client.fetch_pnl.return_value = PnlObservation(None, 0)
    assert capture_next_recovery_series(database_path, client, account_scope="account",
                                        observed_at="2026-09-07T00:00:00+00:00") == 0
    client.fetch_pnl.assert_called_once_with(platform_alpha_id="missing-child")


def test_frontier_curve_decoding_is_bounded_by_comparisons_not_database_history(monkeypatch):
    from tests.execution.test_seeds import SignalSeedExecutionTests
    case = SignalSeedExecutionTests()
    case.setUp()
    try:
        with open_database(case.database_path) as connection:
            snapshot = case._completed(connection, "rank(close)", "1", 1.2, 0.8)
            synchronize_signal_seeds(connection)
            comparison = RecoveryComparison(snapshot.task.task_id, snapshot.task.task_id,
                snapshot.task.task_id, "group-account", snapshot.task.platform_alpha_id, "parent", "reference")
            monkeypatch.setattr("execution.seeds.load_recovery_comparisons", lambda *a, **kw: (comparison,))
            for alpha in ("parent", "reference", snapshot.task.platform_alpha_id):
                points = [math.cos(i) if alpha == snapshot.task.platform_alpha_id else math.sin(i) for i in range(300)]
                save_pnl_series(connection, series(alpha, points, account="group-account"))
            expected = load_signal_frontiers(connection)
            payload = json.dumps([["2020-01-01", 0.0]])
            connection.executemany("INSERT INTO platform_pnl_series VALUES (?, ?, ?, ?, NULL)",
                (("group-account", f"unrelated-{i}", "2026-09-07T00:00:00+00:00", payload) for i in range(10000)))
            connection.execute("INSERT INTO platform_pnl_series VALUES (?, ?, ?, ?, NULL)",
                               ("other-account", snapshot.task.platform_alpha_id, "2026-09-07T00:00:00+00:00", payload))
            loads = json.loads
            decoded_curves = []
            def tracked_loads(value, *args, **kwargs):
                if value.startswith('[["2020-'):
                    decoded_curves.append(value)
                return loads(value, *args, **kwargs)
            monkeypatch.setattr("persistence.pnl.json.loads", tracked_loads)
            assert load_signal_frontiers(connection) == expected
            assert len(decoded_curves) == 3
            assert payload not in decoded_curves
    finally:
        case.doCleanups()


@pytest.mark.parametrize("retry_after, retry_at", (
    (None, "2026-09-07T00:10:00+00:00"),
    (45.0, "2026-09-07T00:10:00+00:00"),
))
def test_missing_recovery_series_defers_only_its_object_and_recovers_after_cooldown(
    pending_recovery_series, retry_after, retry_at, caplog,
):
    caplog.set_level("INFO", logger="execution.progress")
    database_path, comparisons = pending_recovery_series
    child_points = series("child", [math.cos(i) for i in range(300)]).points
    client = Mock()
    client.fetch_pnl.side_effect = [
        WorldQuantRequestError("worldquant_pnl_http_error", status_code=404,
                              retryable=False, outcome_unknown=False, retry_after_seconds=retry_after),
        PnlObservation(child_points),
        PnlObservation(child_points),
    ]
    assert capture_next_recovery_series(database_path, client, account_scope="account",
        observed_at="2026-09-07T00:00:00+00:00") == (retry_after or 0.0)
    assert "正在获取时序数据 missing-child（已获取 2/4）" in caplog.messages[0]
    assert not getattr(caplog.records[0], "transient", False)
    assert "读取失败（HTTP 404），证据待定，600 秒后可重试" in caplog.messages[-1]
    assert "获取完成" not in caplog.text
    with open_database(database_path) as connection:
        records = list_pnl_series(connection)
        missing = next(r for r in records if r.account_scope == "account" and r.platform_alpha_id == "missing-child")
        assert missing.points is None and missing.retry_not_before == retry_at
        assert not recovery_task_ids(comparisons, records)

    # A fresh call/database connection skips the deferred object and captures its peer.
    peer_at = (datetime.fromisoformat("2026-09-07T00:00:00+00:00")
               + timedelta(seconds=(retry_after or 0.0) + 1)).isoformat()
    assert capture_next_recovery_series(database_path, client, account_scope="account",
        observed_at=peer_at) == 0.0
    with open_database(database_path) as connection:
        records = list_pnl_series(connection)
        assert recovery_task_ids(comparisons, records) == frozenset({"other-child"})
    assert capture_next_recovery_series(database_path, client, account_scope="account",
        observed_at="2026-09-07T00:09:59+00:00") is None
    assert client.fetch_pnl.call_count == 2
    assert capture_next_recovery_series(database_path, client, account_scope="account",
        observed_at=retry_at) == 0.0
    assert "获取完成（已获取 4/4）" in caplog.messages[-1]
    with open_database(database_path) as connection:
        records = list_pnl_series(connection)
        assert recovery_task_ids(comparisons, records) == frozenset({"missing-child", "other-child"})
        assert next(r for r in records if r.account_scope == "other-account").points != child_points
    assert [call.kwargs["platform_alpha_id"] for call in client.fetch_pnl.call_args_list] == [
        "missing-child", "other-child", "missing-child",
    ]
    client.submit_backtest.assert_not_called()
    client.submit_formal_alpha.assert_not_called()


@pytest.mark.parametrize("error", (
    *(WorldQuantRequestError("worldquant_pnl_http_error", status_code=status,
                            retryable=status in {429, 503}, outcome_unknown=False)
      for status in (401, 403, 429, 503)),
    WorldQuantRequestError("worldquant_pnl_request_failed", retryable=True, outcome_unknown=False),
    WorldQuantProtocolError("worldquant_pnl_response_invalid"),
    ValueError("program_invariant_broken"),
))
def test_other_recovery_errors_preserve_their_existing_boundary(pending_recovery_series, error):
    database_path, _ = pending_recovery_series
    client = Mock()
    client.fetch_pnl.side_effect = error
    with open_database(database_path) as connection:
        before = list_pnl_series(connection)
    with pytest.raises(type(error)) as raised:
        capture_next_recovery_series(database_path, client, account_scope="account",
                                     observed_at="2026-09-07T00:00:00+00:00")
    assert raised.value is error
    with open_database(database_path) as connection:
        assert list_pnl_series(connection) == before


@pytest.mark.parametrize("repair_action,base_passed,bound_reference", (
    (action, base_passed, bound)
    for action, base_passed in (
        *((action, False) for action in SELF_CORRELATION_REPAIR_FAMILIES),
        (SELF_CORRELATION_INTERNAL_FAMILIES[0], True),
    ) for bound in (False, True)
    # New actions are always created with immutable conflict bindings.
    if bound or action not in SELF_CORRELATION_RESEARCH_FAMILIES
))
def test_measured_repair_gets_existing_research_budget_without_new_seed_or_submit(repair_action, base_passed, bound_reference):
    research = repair_action in SELF_CORRELATION_RESEARCH_FAMILIES
    expected_band = (
        SELF_CORRELATION_RESEARCH_FAMILIES if research else SELF_CORRELATION_LIGHT_FAMILIES if repair_action in SELF_CORRELATION_LIGHT_FAMILIES
        else SELF_CORRELATION_INTERNAL_FAMILIES if repair_action in SELF_CORRELATION_INTERNAL_FAMILIES
        else SELF_CORRELATION_HALF_FAMILIES
    )
    correlation = (
        0.74 if expected_band == SELF_CORRELATION_LIGHT_FAMILIES
        else 0.75 if expected_band == SELF_CORRELATION_INTERNAL_FAMILIES else 0.85
    )
    fixture = fixtures.AutomatedCyclePlanningTests()
    fixture.setUp()
    try:
        fixture._replace_catalog(
            fields=("close", "open", "returns", "volume", "earnings") if research else ("close", "open", "returns", "volume"),
            field_categories={"earnings": "fundamental"}, field_datasets={"earnings": "financials"},
            extra_operators=(
                fixture._operator("days_from_last_change", "Time Series", (("x", "expr"),)),
                fixture._operator("trade_when", "Logical", (("x", "expr"), ("y", "expr"), ("z", "expr"))),
            ) if research else (),
            cross_sectional=("rank", "zscore"),
            time_series=("ts_rank", "ts_zscore", "ts_decay_linear", "ts_delta", "ts_av_diff") if research else ("ts_rank", "ts_zscore", "ts_decay_linear"),
            group_fields=("industry",), group=("group_rank", "group_neutralize"),
            windows=(5, 22, 66, 120, 250),
            pairwise=("vector_neut",),
        )
        batch_count = 6
        run_id = fixture._start_run(
            generation_count=8, backtest_count=batch_count,
            max_cycles=2, max_backtests=2 * batch_count,
        )
        parent_formula = "rank(ts_rank(close,5))" if research else "ts_rank(ts_rank(close,5),22)"
        prepare_automated_candidate_backtest_batch(
            fixture.database_path, run_id=run_id,
            candidates=tuple(AutomatedCandidateBacktest(
                exploration_candidate(parse_formula(formula).expression),
                BacktestSettings.from_platform_dict(fixture.settings),
            ) for formula in (parent_formula, "rank(open)", "rank(volume)",
                              "rank(returns)", "ts_rank(open,22)", "ts_rank(volume,22)")),
            created_at="2026-08-30T00:03:00+08:00",
        )
        first = plan_automated_cycle(
            fixture.database_path, run_id=run_id, created_at="2026-08-30T00:04:00+08:00"
        )
        parent = next(s for s in first.backtests if s.task.formula == parent_formula)
        fixture._complete_plan(
            first,
            observed_at="2026-08-30T00:05:00+08:00",
            qualified_task_id=parent.task.task_id,
            qualified_sharpe=1.5,
        )
        with open_database(fixture.database_path) as connection:
            connection.execute(
                "INSERT INTO backtest_yearly_stats(task_id,stage,year,sharpe) VALUES (?, 'IS', 2023, 1.5)",
                (parent.task.task_id,),
            )
            synchronize_signal_seeds(connection)
            record_automated_cycle_settlement(
                connection,
                run_id,
                cycle_number=1,
                outcome="qualified",
                frontier_advanced=True,
                observed_at="2026-08-30T00:06:00+08:00",
            )
        reference = "ts_rank(close,5)"
        fixture._record_submitted_alpha(
            reference,
            observed_at="2026-08-30T00:06:00+08:00",
            settings=fixture.settings,
            sharpe=1.5,
        )
        with open_database(fixture.database_path) as connection:
            original = get_backtest_task(connection, parent.task.task_id)
        fixture._record_submitted_alpha(
            parent_formula, settings=fixture.settings,
            observed_at="2026-08-30T00:06:00+08:00",
            alpha_id=original.task.platform_alpha_id,
            sharpe=original.result.sharpe,
        )
        fixture._record_formal_attempt(
            task_id=parent.task.task_id,
            run_id=run_id,
            family_root_task_id=parent.task.task_id,
            observed_at="2026-08-30T00:07:00+08:00",
            checks={
                name: "FAIL" if name == "SELF_CORRELATION" else "PASS"
                for name in STANDARD_REGULAR_CHECK_NAMES
            },
            correlation_detail={
                "max": correlation,
                "schema": {"properties": [{"name": "id"}, {"name": "correlation"}]},
                "records": [["submitted-alpha", correlation]],
            },
        )
        with open_database(fixture.database_path) as connection:
            catalog = load_generation_catalog(
                connection, FieldCatalogContext("EQUITY", "USA", "TOP3000", 1),
                account_scope="group-account",
            )
        if repair_action in (*SELF_CORRELATION_LIGHT_FAMILIES, *SELF_CORRELATION_INTERNAL_FAMILIES, *SELF_CORRELATION_RESEARCH_FAMILIES):
            leaf = next(item for item in iter_self_correlation_leaves(
                parse_formula(parent.task.formula).expression,
                parse_formula(reference).expression, catalog, field_candidates=("open", "earnings") if research else ("open",),
                families=(repair_action,), neutralization=fixture.settings["neutralization"],
            ) if item.family == repair_action)
        else:
            # Archived formulas remain research evidence; retired actions are never generated.
            historical = {
                "self_correlation_conflict_reference_residual": f"vector_neut(zscore({parent_formula}),zscore({reference}))",
                "self_correlation_shared_left_replacement": "ts_rank(ts_zscore(close,5),22)",
                "self_correlation_shared_right_replacement": "ts_rank(ts_rank(open,5),22)",
            }[repair_action]
            parsed = parse_formula(historical)
            leaf = SelfCorrelationLeaf(parsed.fingerprint, repair_action, parsed.expression,
                CandidateChange(repair_action, "formula", parent_formula, historical))
        formula = render_formula(leaf.expression)
        child_id = fixture._completed_backtest(
            formula,
            sharpe=1.4 if base_passed else 1.1,
            fitness=1.1 if base_passed else 0.9,
            sharpe_status="PASS" if base_passed else "FAIL",
            fitness_status="PASS" if base_passed else "FAIL",
            time_offset=8,
        )
        with open_database(fixture.database_path) as connection:
            create_backtest_mutation(
                connection,
                BacktestMutationRecord(
                    child_id,
                    parent.task.task_id,
                    repair_action,
                    leaf.change.location,
                    parent.task.formula,
                    formula,
                    "submitted-alpha" if bound_reference else None,
                    reference if bound_reference else None,
                ),
            )
            assert (
                child_id not in load_signal_frontiers(connection).active_branch_task_ids
            )
            assert synchronize_signal_seeds(connection) == ()
            comparisons = load_recovery_comparisons(connection)
            assert len(comparisons) == 1
        if base_passed:
            with open_database(fixture.database_path) as connection:
                connection.execute("INSERT INTO backtest_yearly_stats(task_id,stage,year,sharpe) VALUES (?, 'IS', 2023, 1.4)", (child_id,))
                payload = {"is": {
                    "checks": [{"name": name, "result": "FAIL" if name == "SELF_CORRELATION" else "PASS",
                                **({"value": 0.74, "limit": 0.7} if name == "SELF_CORRELATION" else {})}
                               for name in STANDARD_REGULAR_CHECK_NAMES],
                    "selfCorrelated": {"max": 0.74,
                        "schema": {"properties": [{"name": "id"}, {"name": "correlation"}]},
                        "records": [["submitted-alpha", 0.74]]}}}
                save_submission_check(connection, SubmissionCheckRecord(child_id,
                    "2026-08-30T00:08:30+08:00", json.dumps(payload), None))
        comparison = comparisons[0]
        with open_database(fixture.database_path) as connection:
            original = get_backtest_task(connection, parent.task.task_id)
            child = get_backtest_task(connection, child_id)
        mutation = BacktestMutationRecord(child_id, original.task.task_id, repair_action,
                                         leaf.change.location, original.task.formula, formula)
        references = {child_id: SelfCorrelationReference(original.task.task_id, reference, "submitted-alpha")}
        assert recovery_comparisons((original, child), (mutation,), references, catalog=catalog)
        assert not recovery_comparisons((original, child), (mutation,), {}, catalog=catalog)
        assert not recovery_comparisons((original, child), (replace(mutation, after=original.task.formula),), references, catalog=catalog)
        for altered in (
            replace(child, result=replace(child.result, fitness=0.69)),
            replace(child, task=replace(child.task, account_scope="another-account")),
            replace(child, task=replace(child.task, settings_json="{}")),
        ):
            assert not recovery_comparisons((original, altered), (mutation,), references, catalog=catalog)

        def pnl(*, platform_alpha_id):
            fn = (
                (lambda i: 0.73 * math.sin(i) + math.sqrt(1-0.73**2) * math.cos(i))
                if platform_alpha_id == comparison.child_alpha_id else math.sin
            )
            points = [("2020-01-01", 0.0)]
            for i in range(300):
                points.append(
                    (
                        (date(2020, 1, 2) + timedelta(days=i)).isoformat(),
                        points[-1][1] + fn(i),
                    )
                )
            return PnlObservation(tuple(points))

        client = Mock()
        client.fetch_pnl.return_value = PnlObservation(None, 8.0)
        assert (
            capture_next_recovery_series(
                fixture.database_path,
                client,
                account_scope="group-account",
                observed_at="2026-08-30T00:09:00+08:00",
            )
            == 8.0
        )
        with open_database(fixture.database_path) as connection:
            assert list_pnl_series(connection)[0].points is None
        client.fetch_pnl.side_effect = pnl
        for _ in range(3):
            assert (
                capture_next_recovery_series(
                    fixture.database_path,
                    client,
                    account_scope="group-account",
                    observed_at="2026-08-30T00:20:00+08:00",
                )
                == 0
            )
        assert (
            capture_next_recovery_series(
                fixture.database_path,
                client,
                account_scope="group-account",
                observed_at="2026-08-30T00:09:00+08:00",
            )
            is None
        )
        with open_database(fixture.database_path) as connection:
            assessed = assess_seed_correlation(get_backtest_task(connection, child_id),
                list_platform_submitted_alphas(connection, account_scope="group-account"), list_pnl_series(connection))
            assert assessed.state == "failed" and 0.7 < assessed.maximum < 0.75
            assert set(load_signal_frontiers(connection).active_branch_task_ids) == {
                parent.task.task_id,
                child_id,
            }
            assert (
                connection.execute("SELECT COUNT(*) FROM signal_seeds").fetchone()[0]
                == 1
            )
        plan = plan_automated_cycle(
            fixture.database_path, run_id=run_id, created_at="2026-08-30T00:21:00+08:00"
        )
        allocations = plan.planned_source_allocation.signal_improvements
        parent_allocations = [a for a in allocations if a.parent_task_id == parent.task.task_id]
        if correlation < 0.85 or research:
            assert parent_allocations
            assert parent_allocations[0].candidate_family in expected_band, [a.candidate_family for a in parent_allocations]
        else:
            assert all(a.candidate_family not in SELF_CORRELATION_REPAIR_FAMILIES for a in parent_allocations)
        frozen = plan_automated_cycle(
            fixture.database_path, run_id=run_id, created_at="2026-08-30T00:22:00+08:00"
        )
        assert tuple(s.task.task_id for s in frozen.backtests) == tuple(s.task.task_id for s in plan.backtests)
        assert len(plan.backtests) == batch_count
        assert plan.exploration_backtest_count >= (batch_count * 3 + 9) // 10
        recovery_allocation = next(
            item for item in allocations if item.parent_task_id == child_id
        )
        if base_passed:
            assert recovery_allocation.target.check_name == "SELF_CORRELATION"
            assert recovery_allocation.candidate_family in SELF_CORRELATION_LIGHT_FAMILIES
        else:
            assert recovery_allocation.target.check_name in {"LOW_SHARPE", "LOW_FITNESS"}
        with open_database(fixture.database_path) as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM formal_submission_attempts"
                ).fetchone()[0]
                == 1
            )
            assert (
                connection.execute("SELECT COUNT(*) FROM signal_seeds").fetchone()[0]
                == 1
            )
        # Later checks and catalog refreshes must not rebind a measured repair.
        with open_database(fixture.database_path) as connection:
            payload = {"is": {
                "checks": [{"name": name, "result": "FAIL" if name == "SELF_CORRELATION" else "PASS",
                            **({"value": .8, "limit": .7} if name == "SELF_CORRELATION" else {})}
                           for name in STANDARD_REGULAR_CHECK_NAMES],
                "selfCorrelated": {"max": .8,
                    "schema": {"properties": [{"name": "id"}, {"name": "correlation"}]},
                    "records": [["new-peer", .8]]}}}
            save_submission_check(connection, SubmissionCheckRecord(parent.task.task_id,
                "2026-08-30T00:30:00+08:00", json.dumps(payload), None))
            assert comparison in load_recovery_comparisons(connection)
            # Unbound history with no remaining contemporaneous evidence is unknown.
            connection.execute("DELETE FROM formal_submission_attempts")
            connection.execute("DELETE FROM platform_submitted_alphas")
            assert (comparison in load_recovery_comparisons(connection)) == bound_reference
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize("action", ("internal_field_replacement", "single_window_mutation", "temporal_persistence_reframe"))
def test_sc_bound_ordinary_edits_keep_measured_recovery_in_the_original_lineage(action):
    from generation.internal_edits import iter_internal_edit_candidates
    from generation.polishing import iter_window_mutation_leaves
    from generation.transformations import iter_transformation_leaves
    fixture = fixtures.AutomatedCyclePlanningTests()
    fixture.setUp()
    try:
        fixture._replace_catalog(fields=("close", "open"), cross_sectional=("rank", "zscore"),
            time_series=("ts_rank", "ts_mean", "ts_decay_linear"), windows=(5, 22, 66))
        with open_database(fixture.database_path) as connection:
            fixtures.replace_operator_roles(connection, tuple(fixtures.OperatorRoleRecord(name, role) for name, role in (
                ("rank", "cross_sectional_normalization"), ("zscore", "cross_sectional_normalization"),
                ("ts_rank", "time_series_normalization"), ("ts_mean", "time_series_smoothing"),
                ("ts_decay_linear", "time_series_smoothing"))))
        parent_id = fixture._signal_seed("rank(ts_rank(close,22))", sharpe_status="PASS", fitness_status="PASS")
        with open_database(fixture.database_path) as connection:
            parent = get_backtest_task(connection, parent_id)
            catalog = load_generation_catalog(connection, FieldCatalogContext("EQUITY", "USA", "TOP3000", 1), account_scope="group-account")
        expression = parse_formula(parent.task.formula).expression
        if action == "internal_field_replacement":
            leaf = next(iter_internal_edit_candidates(expression, catalog, parent_task_id=parent_id,
                families=(action,), field_candidates=("open",)))
        elif action == "single_window_mutation":
            leaf = next(iter_window_mutation_leaves(expression, catalog))
        else:
            leaf = next(iter_transformation_leaves(expression, catalog, families=(action,)))
        formula = render_formula(leaf.expression)
        child_id = fixture._completed_backtest(formula, sharpe=1.05, fitness=.75,
            sharpe_status="FAIL", fitness_status="FAIL", time_offset=3)
        mutation = BacktestMutationRecord(child_id, parent_id, action, leaf.change.location,
            leaf.change.before, leaf.change.after, "original-conflict", parent.task.formula)
        with open_database(fixture.database_path) as connection:
            create_backtest_mutation(connection, mutation)
            comparisons = load_recovery_comparisons(connection)
            assert len(comparisons) == 1
            child = get_backtest_task(connection, child_id)
            refs = {child_id: SelfCorrelationReference(parent_id, parent.task.formula, "original-conflict")}
            assert not recovery_comparisons((parent, child), (replace(mutation, before="forged"),), refs, catalog=catalog)
            assert not recovery_comparisons((parent, child), (mutation,), refs, catalog=None)
            for alpha in (parent.task.platform_alpha_id, "original-conflict"):
                save_pnl_series(connection, series(alpha, [math.sin(i) for i in range(300)], "group-account"))
            save_pnl_series(connection, series(child.task.platform_alpha_id, [math.cos(i) for i in range(300)], "group-account"))
            assert child_id in load_signal_frontiers(connection).active_branch_task_ids
            assert synchronize_signal_seeds(connection) == ()
            assert connection.execute("SELECT COUNT(*) FROM signal_seeds").fetchone()[0] == 1
    finally:
        fixture.doCleanups()
