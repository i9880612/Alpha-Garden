from dataclasses import replace
from datetime import datetime

import pytest

from learning.action_effects import ActionRequest, ExplicitSelfCorrelationEvidence, TaskRunEvidence, build_action_strategies
from learning.evidence import LearningEvidenceSet, build_mutation_learning_evidence
from learning.quality import QUALITY_IMPROVEMENT
from learning.quality_budget import quality_budgets
from persistence.backtests import BacktestMutationRecord
from persistence.quality_research import QualityResearchTask
from tests.learning import test_action_effects as effect_fixtures
from tests.learning.test_qualified_evolution import checked, mutation


def quality_record(task, *, sharpe=1.5, fitness=1.2, grade="GOOD", defects=()):
    return replace(effect_fixtures.DefectActionEffectTests._record(task, defects=defects, sharpe=sharpe,
        fitness=fitness, low_sub_actual=.5), grade=grade)


def formal(task, status="PASS", *, other=True):
    return ExplicitSelfCorrelationEvidence(task, status,
        "passed" if status == "PASS" and other else "failed" if status == "FAIL" or other is False else "pending",
        "2026-09-01T00:00:00+00:00", other_checks_passed=other)


@pytest.mark.parametrize("change,status,other,expected", [
    ({"sharpe": 1.51}, "PASS", True, "safe_progress"),
    ({"grade": "EXCELLENT"}, "PASS", True, "safe_progress"),
    ({}, "PASS", True, "no_progress"),
    ({"fitness": 1.3, "sharpe": 1.49}, "PASS", True, "conflict"),
    ({"grade": "AVERAGE", "sharpe": 1.8}, "PASS", True, "conflict"),
    ({"fitness": 1.8, "defects": ("LOW_SUB_UNIVERSE_SHARPE",)}, "PASS", False, "conflict"),
    ({"fitness": 1.8, "defects": ("LOW_TURNOVER",)}, "PASS", False, "conflict"),
    ({"sharpe": 1.8}, "PENDING", None, "unresolved"),
    ({"sharpe": 1.8}, "FAIL", True, "conflict"),
    ({"grade": None}, "PASS", True, "unresolved"),
])
def test_quality_feedback_requires_current_full_checks_and_platform_metrics(change, status, other, expected):
    parent, child = quality_record("parent"), quality_record("child", **change)
    evidence = LearningEvidenceSet((parent, child), ())
    mutations = build_mutation_learning_evidence(evidence, (BacktestMutationRecord("child", "parent", "window", "root", "a", "b"),))
    result = build_action_strategies(evidence, mutations,
        (ActionRequest("parent", QUALITY_IMPROVEMENT, ("window",)),), seed_root_task_ids=("parent",),
        task_runs=(TaskRunEvidence("child", "run"),), explicit_self_correlations=(formal("parent"), formal("child", status, other=other)))
    assert result.observations[0].outcome == expected
    assert result.observations[0].defect_checks == ()
    assert result.records[0].mode == "exploration"  # One observation is not a learned preference.


def test_quality_preferences_need_diverse_lineages_and_only_recent_outcomes():
    roots = tuple(f"root-{index}" for index in range(5))
    records = [quality_record(root) for root in roots]
    checks = [formal(root) for root in roots]
    mutations, runs = [], []
    for action in ("improves", "flat", "request_error"):
        for root in roots:
            for index in range(2):
                child = f"{root}-{action}-{index}"
                records.append(quality_record(child, sharpe=1.6 if action != "flat" else 1.5))
                checks.append(formal(child, "PENDING", other=None) if action == "request_error" else formal(child))
                mutations.append(BacktestMutationRecord(child, root, action, "root", "a", "b"))
                runs.append(TaskRunEvidence(child, f"run-{index}"))
    evidence = LearningEvidenceSet(tuple(records), ())
    def build(start):
        return build_action_strategies(evidence, build_mutation_learning_evidence(evidence, tuple(mutations)),
            (ActionRequest(roots[0], QUALITY_IMPROVEMENT, ("improves", "flat", "request_error")),),
            seed_root_task_ids=roots, task_runs=runs, explicit_self_correlations=checks,
            quality_observation_start=datetime.fromisoformat(start))
    strategy = build("2026-08-01T00:00:00+00:00").records[0]
    assert strategy.preferred_actions == ("improves",)
    assert strategy.deprioritized_actions == ("flat",)
    assert next(item for item in strategy.actions if item.action == "request_error").resolved_count == 0
    assert build("2026-09-01T00:00:00+00:00").records[0].preferred_actions == ()


def test_quality_budget_survives_descendant_changes_and_separates_accounts_settings_and_reservations():
    candidates = tuple(checked(f"parent-{index}", 1.5 + index / 100) for index in range(5))
    mutations = tuple(mutation("root" if index == 0 else f"parent-{index-1}", f"parent-{index}") for index in range(5))
    tasks = tuple(QualityResearchTask(f"trial-{i}", "account", '{"delay":1}', "root", i < 79) for i in range(80))
    other_account = replace(candidates[-1], task=replace(candidates[-1].task, task_id="other-account", account_scope="another"))
    other_settings = replace(candidates[-1], task=replace(candidates[-1].task, task_id="other-settings", settings_json='{"delay":0}'))
    budget = quality_budgets((*candidates, other_account, other_settings),
        (*mutations, mutation("root", "other-account"), mutation("root", "other-settings")), frozenset({"root"}), tasks)
    assert all(budget[item.task.task_id].available == 0 for item in candidates)
    assert (budget["parent-4"].attempted, budget["parent-4"].reserved) == (79, 1)
    assert budget["other-account"].available == budget["other-settings"].available == 80
    assert quality_budgets(candidates, mutations, frozenset({"root"}), tasks[:-1])["parent-4"].available == 1
