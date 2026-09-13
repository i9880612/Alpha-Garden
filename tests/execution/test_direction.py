from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from execution.catalog import load_generation_catalog
from execution.direction import build_direction_candidates
from generation.parser import parse_formula
from persistence.backtests import (
    BacktestMutationRecord,
    BACKTEST_ACTIVE_STATUSES,
    BACKTEST_TERMINAL_STATUSES,
)
from persistence.catalog import FieldCatalogContext
from selection.settings import BacktestSettingsPolicy, CategoryNeutralization
from tests.execution.catalog_fixture import initialize_test_generation_catalog
from tests.learning import test_seeds as seed_tests
from worldquant.backtests import BacktestSettings


class DirectionCandidateTests(unittest.TestCase):
    def setUp(self):
        connection = sqlite3.connect(":memory:")
        self.addCleanup(connection.close)
        connection.row_factory = sqlite3.Row
        initialize_test_generation_catalog(connection)
        self.catalog = load_generation_catalog(
            connection,
            FieldCatalogContext("EQUITY", "USA", "TOP3000", 1),
            account_scope="group-account",
        )
        self.policy = BacktestSettingsPolicy(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
            decay=4,
            default_neutralization="SECTOR",
            root_group_neutralization="NONE",
            category_neutralizations=(CategoryNeutralization("sample", "SECTOR"),),
            default_truncation=0.08,
            tail_risk_truncation=0.05,
            pasteurization="ON",
            unit_handling="VERIFY",
            nan_handling="OFF",
            language="FASTEXPR",
            visualization=False,
            max_trade="OFF",
            max_position="OFF",
        )
        self.settings = BacktestSettings(
            "EQUITY",
            "USA",
            "TOP3000",
            1,
            4,
            "SECTOR",
            0.08,
            "ON",
            "VERIFY",
            "OFF",
            "FASTEXPR",
            False,
            "OFF",
            "OFF",
        )
        checks = seed_tests.SignalSeedAssessmentTests._positive_signal_checks()
        checks.update(LOW_SHARPE="FAIL", LOW_FITNESS="FAIL", SELF_CORRELATION="PENDING")
        snapshot = seed_tests.SignalSeedAssessmentTests._snapshot(
            sharpe=-1.4,
            fitness=-1.1,
            checks=checks,
        )
        self.source = replace(
            snapshot,
            task=replace(
                snapshot.task,
                settings_json=json.dumps(
                    self.settings.as_platform_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                formula_fingerprint=parse_formula(snapshot.task.formula).fingerprint,
            ),
        )

    def _build(self, completed=None, **changes):
        arguments = dict(
            completed=(self.source,) if completed is None else completed,
            mutations=(),
            reserved_tasks=(self.source.task,),
            excluded_formula_fingerprints=frozenset(),
            account_scope="group-account",
            evidence_cutoff=datetime.fromisoformat("2026-08-31T00:03:00+00:00"),
            max_candidates=1,
        )
        arguments.update(changes)
        return build_direction_candidates(self.catalog, self.policy, **arguments)

    def test_discovery_preserves_full_settings_and_does_not_change_source(self):
        original = self.source
        candidates = self._build()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].settings, self.settings)
        self.assertEqual(candidates[0].candidate.formula, "-rank(close)")
        self.assertEqual(candidates[0].candidate.parent_task_id, original.task.task_id)
        self.assertEqual(self.source, original)
        self.assertEqual(self._build(max_candidates=0), ())

    def test_unrelated_late_or_incompatible_history_is_skipped(self):
        for changes in (
            {"account_scope": "other"},
            {"finished_at": "2026-08-31T00:04:00+00:00"},
            {
                "settings_json": json.dumps(
                    replace(self.settings, universe="TOP1000").as_platform_dict()
                )
            },
            {
                "settings_json": json.dumps(
                    replace(self.settings, decay=8).as_platform_dict()
                )
            },
            {"formula": "rank(unknown_field)"},
            {"formula": "unknown_operator(close)"},
            {"formula": "rank("},
        ):
            with self.subTest(changes=changes):
                unrelated = replace(
                    self.source,
                    task=replace(
                        self.source.task,
                        task_id="0-earlier-id",
                        **changes,
                    ),
                    result=replace(self.source.result, task_id="0-earlier-id"),
                )
                self.assertEqual(self._build((unrelated, self.source)), self._build())

    def test_any_existing_request_status_blocks_repetition(self):
        child = replace(
            self.source.task,
            task_id="child",
            formula="-rank(close)",
            formula_fingerprint=parse_formula("-rank(close)").fingerprint,
        )
        for status in BACKTEST_ACTIVE_STATUSES | BACKTEST_TERMINAL_STATUSES:
            with self.subTest(status=status):
                self.assertEqual(
                    self._build(reserved_tasks=(replace(child, status=status),)), ()
                )
        self.assertEqual(
            len(self._build(reserved_tasks=(replace(child, account_scope="other"),))), 1
        )

    def test_submitted_formula_and_mutation_children_are_excluded(self):
        self.assertEqual(
            self._build(
                excluded_formula_fingerprints=frozenset(
                    {
                        parse_formula("-rank(close)").fingerprint,
                    }
                )
            ),
            (),
        )
        for action in ("direction_reversal", "single_window_mutation"):
            with self.subTest(action=action):
                lineage = BacktestMutationRecord(
                    self.source.task.task_id,
                    "ancestor",
                    action,
                    "formula",
                    "before",
                    "after",
                )
                self.assertEqual(self._build(mutations=(lineage,)), ())

    def test_cancelled_unsent_reversal_is_available_but_sent_failure_is_not(self):
        cancelled = replace(self.source.task, task_id="cancelled", formula="-rank(close)",
            formula_fingerprint=parse_formula("-rank(close)").fingerprint, status="failed",
            failure_code="automated_run_stopped_before_submission", submission_started_at=None,
            remote_id=None, platform_alpha_id=None)
        self.assertEqual(len(self._build(reserved_tasks=(cancelled,))), 1)
        attempted = replace(cancelled, submission_started_at="2026-08-31T00:02:00+00:00")
        self.assertEqual(self._build(reserved_tasks=(cancelled, attempted)), ())

    def test_duplicate_direction_requests_are_not_generated_twice(self):
        duplicate = replace(
            self.source,
            task=replace(self.source.task, task_id="duplicate"),
            result=replace(self.source.result, task_id="duplicate"),
        )
        self.assertEqual(
            len(self._build((self.source, duplicate), max_candidates=2)), 1
        )
