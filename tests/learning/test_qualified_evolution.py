from __future__ import annotations

import unittest
import math
from dataclasses import replace

from learning.qualified_evolution import assess_qualified_evolution
from persistence.backtests import BacktestMutationRecord, BacktestResultRecord, BacktestSnapshot, BacktestTaskRecord
from tests.learning.test_seed_correlation import series


def checked(task_id, sharpe):
    return BacktestSnapshot(
        task=BacktestTaskRecord(
            task_id=task_id, account_scope="account", formula="rank(close)",
            formula_fingerprint=task_id, settings_json='{"delay":1}', request_fingerprint=task_id,
            status="completed", remote_id=task_id, platform_alpha_id=task_id,
            created_at="2026-09-01T00:00:00+00:00", submission_started_at="2026-09-01T00:01:00+00:00",
            last_observed_at="2026-09-01T00:02:00+00:00", retry_not_before=None,
            finished_at="2026-09-01T00:02:00+00:00", failure_code=None, failure_message=None,
        ),
        result=BacktestResultRecord(
            task_id, sharpe, 1.2, 0.1, 0.08, 0.04, 0.001, 20_000_000, 100_000,
            100, 100, True, (), "GOOD",
        ),
        yearly_stats=(),
    )


def mutation(parent, child):
    return BacktestMutationRecord(child, parent, "structural", "formula", "rank(close)", "rank(open)")


class QualifiedEvolutionTests(unittest.TestCase):
    def test_better_descendant_replaces_only_its_ancestors_not_siblings_or_lower_children(self):
        snapshots = (checked("parent", 1.5), checked("sibling", 1.4), checked("lower", 1.3),
                     checked("better", 1.7), checked("equal", 1.5))
        mutations = (mutation("root", "parent"), mutation("root", "sibling"),
                     mutation("parent", "better"), mutation("parent", "lower"), mutation("sibling", "equal"))
        decisions = {r.task_id: r for r in assess_qualified_evolution(snapshots, mutations, frozenset())}
        self.assertEqual(decisions["parent"].replacement_task_id, "better")
        self.assertEqual(decisions["sibling"].replacement_task_id, "equal")
        self.assertFalse(decisions["lower"].retired)
        self.assertFalse(decisions["equal"].retired)
        self.assertFalse(decisions["better"].retired)

    def test_equal_result_never_replaces_parent_and_budget_is_not_shared(self):
        mutations = tuple(mutation("parent", f"child-{i}") for i in range(21))
        snapshots = (checked("parent", 1.5), checked("child-0", 1.5))
        for count in (0, 19, 20, 21):
            with self.subTest(count=count):
                decisions = {r.task_id: r for r in assess_qualified_evolution(
                    snapshots, mutations, frozenset(f"child-{i}" for i in range(count)),
                )}
                self.assertEqual(decisions["parent"].remaining_attempts, max(0, 20-count))
                self.assertIsNone(decisions["parent"].replacement_task_id)
                self.assertEqual(decisions["parent"].retired, count >= 20)
                self.assertEqual(decisions["child-0"].remaining_attempts, 20)

    def test_account_and_settings_boundaries_prevent_replacement(self):
        parent, child = checked("parent", 1.5), checked("child", 1.8)
        for change in ({"account_scope": "other"}, {"settings_json": '{"delay":0}'}):
            with self.subTest(change=change):
                candidate = replace(child, task=replace(child.task, **change))
                decisions = assess_qualified_evolution((parent, candidate), (mutation("parent", "child"),), frozenset())
                self.assertTrue(all(not r.retired for r in decisions))

    def test_invalid_ancestry_fails_instead_of_hanging(self):
        with self.assertRaisesRegex(ValueError, "qualified_evolution_cycle_detected"):
            assess_qualified_evolution((checked("parent", 1.5),),
                (mutation("parent", "child"), mutation("child", "parent")), frozenset())

    def test_measured_duplicate_children_do_not_restart_an_older_parents_budget(self):
        parent = checked("parent", 1.5)
        children = tuple(replace(checked(name, 1.5), task=replace(checked(name, 1.5).task,
            finished_at="2026-09-02T00:02:00+00:00")) for name in ("identical", "near"))
        changes = tuple(mutation("parent", f"attempt-{i}") for i in range(20)) + tuple(
            mutation("parent", child.task.task_id) for child in children)
        increments = [math.sin(i) for i in range(300)]
        curves = (series("parent", increments), series("identical", increments),
                  series("near", [value + 0.001 * math.cos(i) for i, value in enumerate(increments)]))
        for count in (10, 20):
            decisions = {r.task_id: r for r in assess_qualified_evolution(
                (parent, *children), changes, frozenset(f"attempt-{i}" for i in range(count)), curves)}
            self.assertEqual(decisions["parent"].remaining_attempts, 20 - count)
            self.assertEqual(decisions["parent"].retired, count == 20)
            for child in children:
                decision = decisions[child.task.task_id]
                self.assertEqual(decision.remaining_attempts, 20)
                self.assertEqual(decision.replacement_task_id, "parent")
                self.assertTrue(decision.retired)

    def test_distinct_signals_and_unknown_correlation_keep_individual_opportunities(self):
        first, second = checked("first", 1.5), checked("second", 1.5)
        changes = (mutation("root", "first"), mutation("root", "second"))
        increments = [math.sin(i) for i in range(300)]
        for other in (None, [math.cos(i) for i in range(300)], [-v for v in increments],
                      [0.0] * 300, increments[:100]):
            with self.subTest(other=other is None or len(other)):
                curves = (series("first", increments),)
                if other is not None:
                    curves += (series("second", other),)
                decisions = assess_qualified_evolution((first, second), changes, frozenset(), curves)
                self.assertTrue(all(not decision.retired for decision in decisions))

    def test_duplicate_comparison_respects_family_settings_account_and_unknown_grade(self):
        first, second = checked("first", 1.5), checked("second", 1.5)
        increments = [math.sin(i) for i in range(300)]
        changes = (mutation("root", "first"), mutation("root", "second"))
        cases = (
            (second, (mutation("root", "first"), mutation("other-root", "second"))),
            (replace(second, task=replace(second.task, settings_json='{"delay":0}')), changes),
            (replace(second, task=replace(second.task, account_scope="other")), changes),
            (replace(second, result=replace(second.result, grade=None)), changes),
        )
        for candidate, lineage in cases:
            curves = (series("first", increments),
                      series("second", increments, account=candidate.task.account_scope))
            decisions = assess_qualified_evolution((first, candidate), lineage, frozenset(), curves)
            self.assertTrue(all(not decision.retired for decision in decisions))

    def test_duplicate_representative_prefers_grade_then_sharpe_then_fitness(self):
        first, second = checked("first", 1.5), checked("second", 1.5)
        changes = (mutation("root", "first"), mutation("root", "second"))
        curves = tuple(series(name, [math.sin(i) for i in range(300)]) for name in ("first", "second"))
        for result in (replace(second.result, grade="EXCELLENT", sharpe=1.4),
                       replace(second.result, sharpe=1.6), replace(second.result, fitness=1.3)):
            candidate = replace(second, result=result)
            decisions = {r.task_id: r for r in assess_qualified_evolution(
                (first, candidate), changes, frozenset(), curves)}
            self.assertEqual(decisions["first"].replacement_task_id, "second")
            self.assertFalse(decisions["second"].retired)

    def test_lower_grade_duplicate_does_not_replace_its_preferred_parent_with_higher_sharpe(self):
        parent, child = checked("parent", 1.5), checked("child", 1.6)
        parent = replace(parent, result=replace(parent.result, grade="EXCELLENT"))
        curves = tuple(series(name, [math.sin(i) for i in range(300)]) for name in ("parent", "child"))
        decisions = {r.task_id: r for r in assess_qualified_evolution(
            (parent, child), (mutation("parent", "child"),), frozenset(), curves)}
        self.assertFalse(decisions["parent"].retired)
        self.assertEqual(decisions["child"].replacement_task_id, "parent")

    def test_better_descendant_does_not_revive_a_duplicate_of_the_retired_parent(self):
        parent, duplicate, better = (checked("parent", 1.5), checked("duplicate", 1.5), checked("better", 1.7))
        duplicate = replace(duplicate, task=replace(duplicate.task, finished_at="2026-09-02T00:00:00+00:00"))
        curves = tuple(series(s.task.task_id, [math.sin(i + phase) for i in range(300)])
                       for s, phase in ((parent, 0), (duplicate, 0), (better, 1)))
        decisions = {r.task_id: r for r in assess_qualified_evolution((parent, duplicate, better),
            (mutation("parent", "duplicate"), mutation("parent", "better")), frozenset(), curves)}
        self.assertEqual(decisions["parent"].replacement_task_id, "better")
        self.assertEqual(decisions["duplicate"].replacement_task_id, "parent")
        self.assertFalse(decisions["better"].retired)

    def test_similarity_is_measured_against_representatives_not_transitive_neighbors(self):
        snapshots = tuple(checked(name, 1.5) for name in ("a", "b", "c"))
        changes = tuple(mutation("root", s.task.task_id) for s in snapshots)
        curves = tuple(series(name, [math.sin(i + phase) for i in range(300)])
                       for name, phase in (("a", 0), ("b", 0.01), ("c", 0.02)))
        decisions = {r.task_id: r for r in assess_qualified_evolution(snapshots, changes, frozenset(), curves)}
        self.assertFalse(decisions["a"].retired)
        self.assertEqual(decisions["b"].replacement_task_id, "a")
        self.assertFalse(decisions["c"].retired)
