"""Derive research feedback from saved tasks and latest checks without writing state."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from evaluation.backtests import evaluate_backtest
from execution.self_correlation import load_action_check_evidence, load_latest_submission_checks
from generation.self_correlation import SELF_CORRELATION_REPAIR_FAMILIES
from learning.action_effects import TaskRunEvidence, build_action_strategies
from learning.evidence import build_learning_evidence, build_mutation_learning_evidence
from learning.quality import QUALITY_GRADES, QUALITY_IMPROVEMENT
from learning.research_progress import ResearchObservation, ResearchProgress, assess_research_progress
from persistence.backtests import get_backtest_task, list_backtest_mutations, list_completed_backtests
from persistence.quality_research import list_quality_research_tasks
from persistence.runs import get_automated_run, list_all_automated_run_backtests, list_automated_run_backtests
from persistence.seeds import list_signal_seeds
from selection.settings import BacktestSettingsPolicy
from submission.formal import local_formal_submission_eligible
from worldquant.backtests import BacktestSettings


@dataclass(frozen=True, slots=True)
class ResearchSourceFeedback:
    source: str
    planned: int
    attempted: int
    qualified: int
    rejected: int
    pending: int
    request_failed: int


@dataclass(frozen=True, slots=True)
class ResearchFeedback:
    run_id: str
    progress: ResearchProgress
    sources: tuple[ResearchSourceFeedback, ...]
    quality_outcomes: tuple[tuple[str, int], ...]


def load_research_feedback(connection, run_id: str, *, observed_at: datetime) -> ResearchFeedback:
    run = get_automated_run(connection, run_id)
    if run is None:
        raise ValueError("automated_run_missing")
    if observed_at.utcoffset() is None:
        raise ValueError("research_feedback_timezone_required")
    snapshots = tuple(get_backtest_task(connection, link.task_id) for link in list_automated_run_backtests(connection, run_id))
    if any(snapshot is None for snapshot in snapshots):
        raise ValueError("automated_run_backtest_task_missing")
    completed = tuple(item for item in list_completed_backtests(connection)
                      if item.task.account_scope == run.account_scope
                      and datetime.fromisoformat(item.task.finished_at) <= observed_at)
    evidence = build_learning_evidence(completed)
    mutations = list_backtest_mutations(connection)
    changes = {item.child_task_id: item for item in mutations}
    quality_ids = {item.task_id for item in list_quality_research_tasks(connection)}
    seeds = list_signal_seeds(connection)
    seed_ids = {item.root_task_id for item in seeds if datetime.fromisoformat(item.promoted_at) <= observed_at}
    action_feedback = build_action_strategies(evidence, build_mutation_learning_evidence(evidence, mutations), (),
        seed_root_task_ids=seed_ids,
        task_runs=tuple(TaskRunEvidence(link.task_id, link.run_id) for link in list_all_automated_run_backtests(connection)),
        explicit_self_correlations=load_action_check_evidence(connection, account_scope=run.account_scope, evidence_cutoff=observed_at))
    recent_tasks = {item.task.task_id for item in completed
                    if datetime.fromisoformat(item.task.finished_at) >= observed_at - timedelta(days=30)}
    quality_outcomes = Counter(item.outcome for item in action_feedback.observations
                               if item.run_id == run_id and item.target_name == QUALITY_IMPROVEMENT
                               and item.child_task_id in recent_tasks)
    significant = {item.child_task_id for item in action_feedback.observations if item.run_id == run_id
                   and item.outcome == "safe_progress" and (item.target_name != QUALITY_IMPROVEMENT
                       or item.parent_grade in QUALITY_GRADES and item.child_grade in QUALITY_GRADES
                       and QUALITY_GRADES.index(item.child_grade) > QUALITY_GRADES.index(item.parent_grade))}
    checks = load_latest_submission_checks(connection, account_scope=run.account_scope, observed_at=observed_at)
    policy = BacktestSettingsPolicy.from_config_dict(json.loads(run.settings_policy_json))
    has_initial_seed = False
    for seed in seeds:
        if datetime.fromisoformat(seed.promoted_at) > datetime.fromisoformat(run.created_at):
            continue
        snapshot = get_backtest_task(connection, seed.root_task_id)
        if snapshot is None:
            raise ValueError("signal_seed_task_missing")
        if snapshot.task.account_scope != run.account_scope:
            continue
        if policy.allows(BacktestSettings.from_platform_dict(json.loads(snapshot.task.settings_json))):
            has_initial_seed = True
            break
    counts = {}
    observations = []
    for snapshot in snapshots:
        task = snapshot.task
        if datetime.fromisoformat(task.created_at) > observed_at:
            continue
        mutation = changes.get(task.task_id)
        source = ("quality" if task.task_id in quality_ids else "exploration" if mutation is None else
                  "sc_repair" if mutation.conflict_reference_alpha_id is not None
                  or mutation.action in SELF_CORRELATION_REPAIR_FAMILIES else "mutation")
        count = counts.setdefault(source, Counter())
        count["planned"] += 1
        if task.submission_started_at is None or datetime.fromisoformat(task.submission_started_at) > observed_at:
            continue
        count["attempted"] += 1
        if task.status == "failed":
            count["request_failed"] += 1
            continue
        state = "pending"
        if task.status == "completed" and datetime.fromisoformat(task.finished_at) <= observed_at:
            latest = checks.get(task.task_id)
            state = latest.assessment.state if latest is not None else evaluate_backtest(snapshot).state
            if state == "passed" and not local_formal_submission_eligible(snapshot):
                state = "pending"
        count[{"passed": "qualified", "failed": "rejected"}.get(state, "pending")] += 1
        new_seed = task.task_id in seed_ids
        observations.append(ResearchObservation(task.task_id, task.finished_at or task.created_at,
            state in {"passed", "failed"}, new_seed or state == "passed" and task.task_id in significant, new_seed))
    progress = assess_research_progress(tuple(observations), has_initial_seed=has_initial_seed,
        max_cycles=run.max_cycles, max_backtests=run.max_backtests,
        focused=run.optimization_only or run.self_correlation_plan_json is not None)
    return ResearchFeedback(run_id, progress, tuple(ResearchSourceFeedback(source,
        *(count[key] for key in ("planned", "attempted", "qualified", "rejected", "pending", "request_failed")))
        for source, count in sorted(counts.items())), tuple(sorted(quality_outcomes.items())))
