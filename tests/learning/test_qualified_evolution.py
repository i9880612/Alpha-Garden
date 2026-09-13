from __future__ import annotations

import unittest
from dataclasses import replace

from learning.qualified_evolution import assess_qualified_evolution
from persistence.backtests import BacktestMutationRecord, BacktestResultRecord, BacktestSnapshot, BacktestTaskRecord


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
