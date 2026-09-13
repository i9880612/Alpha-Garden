from __future__ import annotations

import sys
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
from generation.formula import formula_structure_fingerprint
from selection.formulas import (
    formula_polishing_candidate,
    formula_selection_candidate,
)
from selection.settings import backtest_settings_fingerprint
from worldquant.backtests import BacktestSettings


class FormulaSelectionCandidateTests(unittest.TestCase):
    def test_describes_an_open_formula_without_assuming_a_fixed_skeleton(self) -> None:
        expression = parse_formula(
            "if_else(greater(rank(close),0.5),"
            "group_rank(ts_mean(vec_avg(news),22),densify(sector)),rank(open))"
        ).expression

        settings = self._settings()
        candidate = formula_selection_candidate(
            expression,
            self._catalog(),
            settings,
        )
        dimensions = {
            dimension.name: dimension.values for dimension in candidate.diversity
        }

        self.assertEqual(
            dimensions,
            {
                "operator": (
                    "densify",
                    "greater",
                    "group_rank",
                    "if_else",
                    "rank",
                    "ts_mean",
                    "vec_avg",
                ),
                "operator_category": (
                    "Arithmetic",
                    "Cross Sectional",
                    "Group",
                    "Logical",
                    "Time Series",
                    "Vector",
                ),
                "operator_output": ("condition", "group", "signal"),
                "field": ("close", "news", "open", "sector"),
                "field_type": ("GROUP", "MATRIX", "VECTOR"),
                "field_category": ("News", "Price Volume"),
                "field_subcategory": ("Price", "Sentiment"),
                "field_source": ("market", "news"),
                "window": ("22",),
                "formula_structure": (
                    formula_structure_fingerprint(expression, self._catalog()),
                ),
                "depth": ("5",),
                "neutralization": ("SUBINDUSTRY",),
                "decay": ("4",),
                "truncation": ("0.08",),
                "delay": ("1",),
            },
        )
        self.assertTrue(
            candidate.candidate_id.endswith(
                f":{backtest_settings_fingerprint(settings)}"
            )
        )

    def test_complete_settings_change_candidate_but_not_formula_family(self) -> None:
        expression = parse_formula("rank(close)").expression
        first = formula_selection_candidate(
            expression,
            self._catalog(),
            self._settings(),
        )
        second = formula_selection_candidate(
            expression,
            self._catalog(),
            BacktestSettings.from_platform_dict(
                {
                    **self._settings().as_platform_dict(),
                    "neutralization": "INDUSTRY",
                }
            ),
        )

        self.assertNotEqual(first.candidate_id, second.candidate_id)
        self.assertEqual(
            first.family_root_candidate_id,
            second.family_root_candidate_id,
        )

    def test_structure_diversity_groups_window_variants_but_preserves_tree(
        self,
    ) -> None:
        catalog = self._catalog()
        settings = self._settings()
        short = formula_selection_candidate(
            parse_formula("ts_mean(rank(close),5)").expression,
            catalog,
            settings,
        )
        long = formula_selection_candidate(
            parse_formula("ts_mean(rank(close),22)").expression,
            catalog,
            settings,
        )
        different_tree = formula_selection_candidate(
            parse_formula("rank(ts_mean(close,22))").expression,
            catalog,
            settings,
        )

        short_dimensions = {
            item.name: item.values for item in short.diversity
        }
        long_dimensions = {
            item.name: item.values for item in long.diversity
        }
        different_dimensions = {
            item.name: item.values for item in different_tree.diversity
        }

        self.assertNotEqual(short_dimensions["window"], long_dimensions["window"])
        self.assertEqual(
            short_dimensions["formula_structure"],
            long_dimensions["formula_structure"],
        )
        self.assertNotEqual(
            long_dimensions["formula_structure"],
            different_dimensions["formula_structure"],
        )

    def test_polishing_can_inherit_only_proven_parent_tail_structure(self) -> None:
        catalog = self._catalog()
        parent = parse_formula("ts_mean(power(rank(close),2),22)").expression
        child = parse_formula("ts_mean(power(rank(close),2),66)").expression

        accepted = formula_polishing_candidate(
            child,
            parent,
            catalog,
            self._settings(),
            parent_passed_checks=(
                "CONCENTRATED_WEIGHT",
                "LOW_SUB_UNIVERSE_SHARPE",
            ),
        )
        missing_parent_check = formula_polishing_candidate(
            child,
            parent,
            catalog,
            self._settings(),
            parent_passed_checks=("CONCENTRATED_WEIGHT",),
        )

        self.assertEqual(accepted.rejection_reasons, ())
        self.assertEqual(
            missing_parent_check.rejection_reasons,
            ("uncontrolled_tail",),
        )

    def test_polishing_keeps_every_other_child_risk(self) -> None:
        base_catalog = self._catalog()
        catalog = replace(
            base_catalog,
            fields=tuple(
                replace(field, coverage=0.0) if field.field_id == "close" else field
                for field in base_catalog.fields
            ),
        )
        parent = parse_formula("ts_mean(power(rank(close),2),22)").expression
        child = parse_formula("ts_mean(power(rank(close),2),66)").expression

        assessed = formula_polishing_candidate(
            child,
            parent,
            catalog,
            self._settings(),
            parent_passed_checks=(
                "CONCENTRATED_WEIGHT",
                "LOW_SUB_UNIVERSE_SHARPE",
            ),
        )

        self.assertEqual(assessed.rejection_reasons, ("field_coverage_zero",))

    def test_polishing_never_inherits_tail_risk_absent_from_parent(self) -> None:
        catalog = self._catalog()
        parent = parse_formula("ts_mean(rank(close),22)").expression
        child = parse_formula("ts_mean(power(rank(close),2),66)").expression

        assessed = formula_polishing_candidate(
            child,
            parent,
            catalog,
            self._settings(),
            parent_passed_checks=(
                "CONCENTRATED_WEIGHT",
                "LOW_SUB_UNIVERSE_SHARPE",
            ),
        )

        self.assertEqual(assessed.rejection_reasons, ("uncontrolled_tail",))

    def _catalog(self) -> GenerationCatalog:
        return GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            fields=(
                self._field("close", "market", "MATRIX"),
                self._field("open", "market", "MATRIX"),
                self._field("news", "news", "VECTOR"),
                self._field("sector", "classifications", "GROUP"),
            ),
            operators=(
                self._operator("densify", "Arithmetic", "group", (("x", "expr"),)),
                self._operator(
                    "greater",
                    "Logical",
                    "condition",
                    (("input1", "expr"), ("input2", "expr")),
                ),
                self._operator(
                    "group_rank",
                    "Group",
                    "signal",
                    (("x", "expr"), ("group", "group")),
                ),
                self._operator(
                    "if_else",
                    "Logical",
                    "signal",
                    (
                        ("input1", "expr"),
                        ("input2", "expr"),
                        ("input3", "expr"),
                    ),
                ),
                self._operator("rank", "Cross Sectional", "signal", (("x", "expr"),)),
                self._operator(
                    "power",
                    "Arithmetic",
                    "signal",
                    (("x", "expr"), ("y", "float")),
                ),
                self._operator(
                    "ts_mean",
                    "Time Series",
                    "signal",
                    (("x", "expr"), ("d", "window")),
                ),
                self._operator("vec_avg", "Vector", "signal", (("x", "expr"),)),
            ),
        )

    @staticmethod
    def _settings() -> BacktestSettings:
        return BacktestSettings(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
            decay=4,
            neutralization="SUBINDUSTRY",
            truncation=0.08,
            pasteurization="ON",
            unit_handling="VERIFY",
            nan_handling="OFF",
            language="FASTEXPR",
            visualization=False,
            max_trade="OFF",
            max_position="OFF",
        )

    @staticmethod
    def _field(
        field_id: str,
        dataset_id: str,
        field_type: str,
    ) -> FieldDefinition:
        category, subcategory = {
            "classifications": ("Classification", "Industry"),
            "market": ("Price Volume", "Price"),
            "news": ("News", "Sentiment"),
        }[dataset_id]
        return FieldDefinition(
            field_id=field_id,
            dataset_id=dataset_id,
            category=category,
            subcategory=subcategory,
            field_type=field_type,
            coverage=1.0,
        )

    @staticmethod
    def _operator(
        name: str,
        category: str,
        output_kind: str,
        parameters: tuple[tuple[str, str], ...],
    ) -> OperatorDefinition:
        return OperatorDefinition(
            name=name,
            category=category,
            scope=("REGULAR",),
            parameters=tuple(
                OperatorParameter(parameter_name, kind)
                for parameter_name, kind in parameters
            ),
            output_kind=output_kind,
        )


if __name__ == "__main__":
    unittest.main()
