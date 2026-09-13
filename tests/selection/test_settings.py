from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generation.catalog import (
    CatalogContext,
    FieldDefinition,
    GenerationCatalog,
    OperatorDefinition,
    OperatorParameter,
)
from generation.parser import parse_formula
from selection.settings import (
    BacktestSettingsPolicy,
    backtest_settings_fingerprint,
    choose_backtest_settings,
    load_backtest_settings_policy,
)


class BacktestSettingsPolicyTests(unittest.TestCase):
    def test_loads_and_canonicalizes_the_frozen_project_policy(self) -> None:
        policy = self._policy()

        self.assertEqual(
            (
                policy.instrument_type,
                policy.region,
                policy.universe,
                policy.delay,
                policy.decay,
            ),
            ("EQUITY", "USA", "TOP3000", 1, 4),
        )
        self.assertEqual(policy.default_neutralization, "SUBINDUSTRY")
        self.assertEqual(policy.root_group_neutralization, "NONE")
        self.assertEqual(policy.default_truncation, 0.08)
        self.assertEqual(policy.tail_risk_truncation, 0.05)
        self.assertEqual((policy.max_trade, policy.max_position), ("OFF", "OFF"))
        self.assertEqual(
            BacktestSettingsPolicy.from_config_dict(policy.as_config_dict()),
            policy,
        )
        self.assertEqual(len(policy.fingerprint), 64)

    def test_neutralization_is_derived_from_formula_field_categories(self) -> None:
        catalog = self._catalog()
        policy = self._policy()

        industry = self._choose("fundamental", catalog, policy)
        sector = self._choose("option", catalog, policy)
        most_granular = self._choose("fundamental+model", catalog, policy)
        fallback = self._choose("unmapped", catalog, policy)

        self.assertEqual(industry.neutralization, "INDUSTRY")
        self.assertEqual(sector.neutralization, "SECTOR")
        self.assertEqual(most_granular.neutralization, "SUBINDUSTRY")
        self.assertEqual(fallback.neutralization, "SUBINDUSTRY")

    def test_root_formula_neutralization_is_not_duplicated_by_platform_setting(
        self,
    ) -> None:
        settings = self._choose(
            "group_neutralize(fundamental,sector)",
            self._catalog(),
            self._policy(),
        )

        self.assertEqual(settings.neutralization, "NONE")

    def test_tail_structure_uses_the_stricter_truncation(self) -> None:
        catalog = self._catalog()
        policy = self._policy()

        ordinary = self._choose("rank(model)", catalog, policy)
        tail = self._choose("rank(divide(model,fundamental))", catalog, policy)

        self.assertEqual(ordinary.truncation, 0.08)
        self.assertEqual(tail.truncation, 0.05)
        self.assertTrue(policy.allows(ordinary))
        self.assertTrue(policy.allows(tail))
        self.assertFalse(policy.allows(replace(ordinary, decay=5)))

    def test_complete_settings_have_a_stable_distinct_identity(self) -> None:
        settings = self._choose("fundamental", self._catalog(), self._policy())

        self.assertEqual(
            backtest_settings_fingerprint(settings),
            backtest_settings_fingerprint(settings),
        )
        self.assertNotEqual(
            backtest_settings_fingerprint(settings),
            backtest_settings_fingerprint(
                replace(settings, neutralization="SUBINDUSTRY")
            ),
        )
        self.assertNotEqual(
            backtest_settings_fingerprint(settings),
            backtest_settings_fingerprint(replace(settings, max_trade="ON")),
        )
        self.assertNotEqual(
            backtest_settings_fingerprint(settings),
            backtest_settings_fingerprint(replace(settings, max_position="ON")),
        )
        self.assertFalse(self._policy().allows(replace(settings, max_trade="ON")))
        self.assertFalse(
            self._policy().allows(replace(settings, max_position="ON"))
        )

    def test_policy_rejects_a_different_catalog_context(self) -> None:
        catalog = self._catalog(delay=0)

        with self.assertRaisesRegex(
            ValueError,
            "backtest_settings_catalog_context_not_allowed",
        ):
            self._choose("fundamental", catalog, self._policy())

    def test_policy_file_requires_one_explicit_catalog_delay(self) -> None:
        payload = self._policy().as_config_dict()
        context = dict(payload["catalogContext"])
        context["delays"] = [context.pop("delay")]
        payload["catalogContext"] = context
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backtest.json"
            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "backtest_settings_policy_file_invalid",
            ):
                load_backtest_settings_policy(path)

    @staticmethod
    def _choose(
        formula: str,
        catalog: GenerationCatalog,
        policy: BacktestSettingsPolicy,
    ):
        return choose_backtest_settings(
            parse_formula(formula).expression,
            catalog,
            policy,
        )

    @staticmethod
    def _policy() -> BacktestSettingsPolicy:
        project_root = Path(__file__).resolve().parents[2]
        return load_backtest_settings_policy(
            project_root / "config" / "backtest.default.json"
        )

    @staticmethod
    def _catalog(delay: int = 1) -> GenerationCatalog:
        return GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", delay),
            fields=(
                FieldDefinition(
                    "fundamental",
                    "fundamental-data",
                    "Fundamental",
                    None,
                    "MATRIX",
                    1.0,
                ),
                FieldDefinition(
                    "model",
                    "model-data",
                    "Model",
                    None,
                    "MATRIX",
                    1.0,
                ),
                FieldDefinition(
                    "option",
                    "option-data",
                    "Option",
                    None,
                    "VECTOR",
                    1.0,
                ),
                FieldDefinition(
                    "sector",
                    "classifications",
                    None,
                    None,
                    "GROUP",
                    1.0,
                ),
                FieldDefinition(
                    "unmapped",
                    "other-data",
                    None,
                    None,
                    "MATRIX",
                    1.0,
                ),
            ),
            operators=(
                OperatorDefinition(
                    name="rank",
                    category="Cross Sectional",
                    scope=("REGULAR",),
                    parameters=(OperatorParameter("x", "expr"),),
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
