from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.backtest_batches import (
    AutomatedCandidateBacktest,
    prepare_automated_candidate_backtest_batch,
)
from execution.catalog import load_generation_catalog
from execution.idea_planning import field_profiles_path, load_idea_pool
from execution.progress import LOGGER_NAME
from execution.runs import (
    AutomatedRunLimits,
    prepare_automated_run,
    start_automated_run,
)
from generation.candidate import exploration_candidate
from generation.parser import parse_formula
from persistence.catalog import (
    FieldCatalogContext,
    FieldCatalogRecord,
    OperatorCatalogRecord,
    OperatorOutputRecord,
    OperatorRoleRecord,
    PlatformCatalogSyncRecord,
    WindowCatalogRecord,
    replace_operator_outputs,
    replace_operator_roles,
    replace_platform_catalog,
    replace_window_catalog,
)
from persistence.database import open_database
from persistence.formula_ideas import get_run_idea_percent, list_formula_ideas
from persistence.schema import initialize_database_schema
from worldquant.backtests import BacktestSettings

ACCOUNT = "lab-account"
CONTEXT = FieldCatalogContext("EQUITY", "USA", "TOP3000", 1)
SYNCED_AT = "2026-10-19T00:00:00+00:00"
_X = {"name": "x", "kind": "expr"}
_D = {"name": "d", "kind": "window"}
_OPERATORS = {
    "rank": (_X, {"name": "rate", "kind": "float", "optional": True}),
    "abs": (_X,), "vec_avg": (_X,),
    "ts_backfill": (_X, {"name": "lookback", "kind": "window", "optional": True},
                    {"name": "k", "kind": "int", "optional": True}),
    "ts_corr": (_X, {"name": "y", "kind": "expr"}, _D),
    "ts_delay": (_X, _D), "ts_delta": (_X, _D), "ts_mean": (_X, _D), "ts_std_dev": (_X, _D),
}
_FIELDS = ("close", "vwap", "volume", "returns")


def install_catalog(connection, fields=_FIELDS):
    records = tuple(
        FieldCatalogRecord(
            context=CONTEXT, field_id=field_id, dataset_id="pv1", category="Price Volume",
            subcategory=None, field_type="MATRIX", coverage=1.0, description=f"Daily {field_id}",
            dataset_name="pv", category_id="pv", subcategory_id=None,
            raw_payload={"id": field_id}, synced_at=SYNCED_AT,
        )
        for field_id in fields
    )
    operators = tuple(
        OperatorCatalogRecord(
            operator_name=name, category=None, definition=name, description=name,
            documentation=None, level="ALL", scope=("REGULAR",), parameters=parameters,
            raw_payload={"name": name}, synced_at=SYNCED_AT,
        )
        for name, parameters in _OPERATORS.items()
    )
    replace_operator_roles(connection, (OperatorRoleRecord("rank", "cross_sectional_normalization"),))
    replace_operator_outputs(connection, tuple(OperatorOutputRecord(name, "signal") for name in _OPERATORS))
    replace_window_catalog(connection, (WindowCatalogRecord(22, "month"),))
    replace_platform_catalog(
        connection,
        PlatformCatalogSyncRecord(account_scope=ACCOUNT, context=CONTEXT, field_count=len(records),
                                  operator_count=len(operators), synced_at=SYNCED_AT),
        records, operators,
    )


def profile_entry(field_id, *, tier="reviewed", kind="price", scale="price"):
    return {"field_id": field_id, "dataset": "pv1", "type": "MATRIX", "tier": tier,
            "kind": kind, "cadence": "daily", "scale": scale, "direction": "unknown"}


def write_profiles(database_path, entries):
    field_profiles_path(database_path).write_text(
        json.dumps({"version": 1, "fields": entries}), encoding="utf-8")


_ALL_REVIEWED = [
    profile_entry("close"), profile_entry("vwap"), profile_entry("returns", scale="ratio"),
    profile_entry("volume", kind="volume", scale="count"),
]


class IdeaPoolLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database_path = Path(directory.name) / "lab.sqlite3"
        with open_database(self.database_path) as connection:
            initialize_database_schema(connection)
            install_catalog(connection)
            connection.commit()

    def _load(self):
        with open_database(self.database_path) as connection:
            catalog = load_generation_catalog(connection, CONTEXT, account_scope=ACCOUNT)
            return load_idea_pool(connection, self.database_path, catalog, CONTEXT, cycle_number=3)

    def test_missing_or_broken_profile_file_hands_the_share_back_without_failing(self):
        with self.assertLogs(LOGGER_NAME, logging.INFO) as logs:
            self.assertIsNone(self._load())
        self.assertIn("随机探索补齐", logs.output[0])
        field_profiles_path(self.database_path).write_text("{not json", encoding="utf-8")
        with self.assertLogs(LOGGER_NAME, logging.INFO):
            self.assertIsNone(self._load())

    def test_reviewed_fields_build_price_templates(self):
        write_profiles(self.database_path, _ALL_REVIEWED)
        pool = self._load()
        self.assertEqual(set(pool.by_template()),
                         {"reversal", "mean_deviation", "momentum", "price_volume", "low_vol",
                          "abnormal_volume"})

    def test_catalog_changes_are_reconciled_and_rule_fields_extend_the_pool(self):
        write_profiles(self.database_path, [
            profile_entry("close"), profile_entry("returns", scale="ratio"),
            profile_entry("volume", tier="rule", kind="volume", scale="count"),
            profile_entry("gone"), profile_entry("odd", tier="unclassified"),
        ])  # vwap missing from the file, gone missing from the catalog
        with self.assertLogs(LOGGER_NAME, logging.INFO) as logs:
            pool = self._load()
        self.assertIn("停用 2 个已消失字段，1 个新字段已按规则画像", logs.output[0])
        used = {field for variant in pool.variants for field in variant.field_ids}
        self.assertEqual(used, {"close", "returns", "vwap", "volume"})
        reviewed = {v.formula for v in pool.variants if v.reviewed}
        self.assertTrue(reviewed)
        self.assertTrue(all({"close", "returns"} >= set(v.field_ids)
                            for v in pool.variants if v.formula in reviewed))


class IdeaRunAndBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.database_path = root / "lab.sqlite3"
        self.settings_path = root / "backtest.json"
        self.settings_path.write_text(json.dumps({
            "catalogContext": {"instrumentType": "EQUITY", "region": "USA",
                               "universe": "TOP3000", "delay": 1},
            "decay": 4,
            "neutralization": {"default": "SECTOR", "rootGroupNeutralize": "NONE",
                               "byFieldCategory": {"Price Volume": "SECTOR"}},
            "truncation": {"default": 0.08, "tailRisk": 0.05},
            "pasteurization": "ON", "unitHandling": "VERIFY", "nanHandling": "OFF",
            "language": "FASTEXPR", "visualization": False, "maxTrade": "OFF",
            "maxPosition": "OFF",
        }), encoding="utf-8")
        with open_database(self.database_path) as connection:
            initialize_database_schema(connection)
            install_catalog(connection)
            connection.commit()
        self.settings = BacktestSettings(
            instrument_type="EQUITY", region="USA", universe="TOP3000", delay=1, decay=4,
            neutralization="SECTOR", truncation=0.08, pasteurization="ON",
            unit_handling="VERIFY", nan_handling="OFF", language="FASTEXPR",
            visualization=False, max_trade="OFF", max_position="OFF",
        )

    def _prepare_run(self, idea_percent):
        return prepare_automated_run(
            self.database_path, self.settings_path, account_scope=ACCOUNT,
            limits=AutomatedRunLimits(
                exploration_percent=100, self_correlation_percent=0, mutation_percent=0,
                direction_validation_percent=0, generation_count=10, backtest_count=4,
                max_cycles=1, max_backtests=4, max_pending_seconds=3600,
                max_consecutive_failures=2, max_request_failures=3,
                max_in_flight_backtests=3, exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=True, idea_percent=idea_percent,
            ),
            created_at="2026-10-19T00:01:00+00:00",
        )

    def _pool(self):
        with open_database(self.database_path) as connection:
            catalog = load_generation_catalog(connection, CONTEXT, account_scope=ACCOUNT)
            return load_idea_pool(connection, self.database_path, catalog, CONTEXT, cycle_number=1)

    def test_missing_profile_file_does_not_block_a_run(self):
        # Each cycle then hands the template share back to random exploration.
        prepared = self._prepare_run(90)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_run_idea_percent(connection, prepared.run_id), 90)

    def test_run_freezes_its_share(self):
        write_profiles(self.database_path, _ALL_REVIEWED)
        prepared = self._prepare_run(50)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_run_idea_percent(connection, prepared.run_id), 50)

    def test_idea_is_recorded_with_its_task_and_mismatches_roll_back(self):
        write_profiles(self.database_path, _ALL_REVIEWED)
        prepared = self._prepare_run(50)
        start_automated_run(self.database_path, prepared.run_id,
                            started_at="2026-10-19T00:02:00+00:00")
        variant = self._pool().by_template()["reversal"][0]
        candidate = exploration_candidate(variant.expression, strategy_label="idea:reversal")
        other = exploration_candidate(parse_formula("rank(close)").expression)

        with self.assertRaisesRegex(ValueError, "backtest_automated_candidate_idea_invalid"):
            prepare_automated_candidate_backtest_batch(
                self.database_path, run_id=prepared.run_id,
                candidates=(AutomatedCandidateBacktest(candidate, self.settings),
                            AutomatedCandidateBacktest(other, self.settings, idea=variant)),
                created_at="2026-10-19T00:03:00+00:00",
            )
        with open_database(self.database_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM backtest_tasks").fetchone()[0], 0)

        batch = prepare_automated_candidate_backtest_batch(
            self.database_path, run_id=prepared.run_id,
            candidates=(AutomatedCandidateBacktest(candidate, self.settings, idea=variant),
                        AutomatedCandidateBacktest(other, self.settings)),
            created_at="2026-10-19T00:03:00+00:00",
        )
        with open_database(self.database_path) as connection:
            ideas = list_formula_ideas(connection)
            formulas = dict(connection.execute("SELECT task_id, formula FROM backtest_tasks"))
        self.assertEqual(len(batch), 2)
        self.assertEqual(len(ideas), 1)
        self.assertEqual(formulas[ideas[0].task_id], variant.formula)
        self.assertEqual((ideas[0].template_id, ideas[0].family, ideas[0].field_ids),
                         ("reversal", "price_action", variant.field_ids))


if __name__ == "__main__":
    unittest.main()
