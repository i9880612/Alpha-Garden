import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from generation.catalog import (
    CatalogContext,
    FieldDefinition,
    GenerationCatalog,
    OperatorDefinition,
    OperatorParameter,
    WindowDefinition,
)
from generation.parser import parse_formula
from generation.formula import analyze_formula, render_formula
from generation.self_correlation import (
    SELF_CORRELATION_FIELD_REPLACEMENT,
    SELF_CORRELATION_FIELD_OPERATOR_REPLACEMENT,
    SELF_CORRELATION_INTERNAL_FAMILIES,
    SELF_CORRELATION_HALF_FAMILIES,
    SELF_CORRELATION_LIGHT_FAMILIES,
    iter_self_correlation_leaves,
    matches_self_correlation_repair,
    shared_field_replacements,
)


class SelfCorrelationGenerationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            fields=tuple(
                FieldDefinition(name, "sample", "sample", "sample", "MATRIX", 1.0)
                for name in ("close", "open")
            ),
            operators=tuple(
                OperatorDefinition(
                    name=name,
                    category="sample",
                    scope=("REGULAR",),
                    parameters=tuple(OperatorParameter(p, "expr") for p in parameters),
                    roles=("cross_sectional_normalization",)
                    if name in {"rank", "zscore"}
                    else (),
                    output_kind="signal",
                )
                for name, parameters in (
                    ("rank", ("x",)),
                    ("zscore", ("x",)),
                    ("vector_neut", ("x", "y")),
                )
            ),
            windows=(),
        )

    def leaves(self, source="rank(close)", reference="rank(open)", *, neutralization="NONE"):
        return tuple(
            iter_self_correlation_leaves(
                parse_formula(source).expression,
                parse_formula(reference).expression,
                self.catalog, families=(*SELF_CORRELATION_LIGHT_FAMILIES, *SELF_CORRELATION_INTERNAL_FAMILIES),
                neutralization=neutralization,
            )
        )

    def test_retired_actions_are_only_available_for_historical_verification(self):
        source, reference = (parse_formula(f"rank({field})").expression for field in ("close", "open"))
        residual = parse_formula("vector_neut(zscore(rank(close)),zscore(rank(open)))").expression
        self.assertTrue(matches_self_correlation_repair(source, reference, residual,
            action="self_correlation_conflict_reference_residual", location="formula"))
        self.assertEqual(self.leaves(), ())
        with self.assertRaises(ValueError):
            tuple(iter_self_correlation_leaves(source, reference, self.catalog,
                families=("self_correlation_conflict_reference_residual",), neutralization="NONE"))

    def test_light_edits_validate_shape_catalog_depth_and_recovery_identity(self):
        self.catalog = replace(self.catalog,
            fields=self.catalog.fields + (
                FieldDefinition("industry", "groups", "group", "group", "GROUP", 1.0),),
            operators=self.catalog.operators + (
                OperatorDefinition("group_rank", "Group", ("REGULAR",), (
                    OperatorParameter("x", "expr"), OperatorParameter("group", "group")), (), "signal"),
                OperatorDefinition("ts_decay_linear", "Time Series", ("REGULAR",), (
                    OperatorParameter("x", "expr"), OperatorParameter("d", "window")), (), "signal"),),
            windows=(WindowDefinition(5, "week"), WindowDefinition(22, "month")))
        source, reference = parse_formula("rank(close)").expression, parse_formula("rank(open)").expression
        leaves = [leaf for leaf in self.leaves() if leaf.family in SELF_CORRELATION_LIGHT_FAMILIES]
        self.assertEqual({render_formula(leaf.expression) for leaf in leaves},
                         {"group_rank(rank(close),industry)", "ts_decay_linear(rank(close),5)"})
        for leaf in leaves:
            self.assertTrue(matches_self_correlation_repair(source, reference, leaf.expression,
                action=leaf.family, location="formula", catalog=self.catalog))
            self.assertFalse(matches_self_correlation_repair(source, reference, leaf.expression,
                action=leaf.family, location="formula.arguments[0]", catalog=self.catalog))
        own_leaves = [leaf for leaf in self.leaves(reference="rank(close)")
                      if leaf.family in SELF_CORRELATION_LIGHT_FAMILIES]
        self.assertEqual(len(own_leaves), 2)
        self.assertTrue(all(matches_self_correlation_repair(source, source, leaf.expression,
            action=leaf.family, location="formula", catalog=self.catalog) for leaf in own_leaves))
        self.assertFalse(matches_self_correlation_repair(source, reference,
            parse_formula("ts_decay_linear(rank(close),22)").expression,
            action=SELF_CORRELATION_LIGHT_FAMILIES[1], location="formula", catalog=self.catalog))
        self.assertFalse(any(leaf.family in SELF_CORRELATION_LIGHT_FAMILIES
            for leaf in self.leaves(source="rank(" * 10 + "close" + ")" * 10)))
        self.catalog = replace(self.catalog, fields=tuple(f for f in self.catalog.fields if f.field_id != "industry"))
        self.assertEqual({leaf.family for leaf in self.leaves() if leaf.family in SELF_CORRELATION_LIGHT_FAMILIES},
                         {SELF_CORRELATION_LIGHT_FAMILIES[1]})

    def test_industry_neutralization_respects_existing_settings_and_recovery_shape(self):
        self.catalog = replace(self.catalog,
            fields=self.catalog.fields + tuple(FieldDefinition(g, "groups", "group", "group", "GROUP", 1.0)
                                               for g in ("industry", "subindustry")),
            operators=self.catalog.operators + (OperatorDefinition("group_neutralize", "Group", ("REGULAR",), (
                OperatorParameter("x", "expr"), OperatorParameter("group", "group")), (), "signal"),))
        family = "self_correlation_industry_neutralization"
        source, reference = (parse_formula(f"rank({f})").expression for f in ("close", "open"))
        for setting in ("NONE", "MARKET", "SECTOR"):
            leaf, = [x for x in self.leaves(neutralization=setting) if x.family == family]
            self.assertEqual(render_formula(leaf.expression), "group_neutralize(rank(close),industry)")
            self.assertTrue(matches_self_correlation_repair(source, reference, leaf.expression,
                action=family, location="formula", catalog=self.catalog))
        for setting in ("INDUSTRY", "SUBINDUSTRY"):
            self.assertFalse(any(x.family == family for x in self.leaves(neutralization=setting)))
        for group in ("industry", "subindustry"):
            self.assertFalse(any(x.family == family for x in self.leaves(source=f"group_neutralize(rank(close),{group})")))
        self.assertFalse(matches_self_correlation_repair(source, reference,
            parse_formula("group_neutralize(rank(close),subindustry)").expression,
            action=family, location="formula", catalog=self.catalog))

    def test_self_projection_unknown_field_missing_operator_and_depth_are_rejected(
        self,
    ):
        self.assertFalse(any(leaf.family == "self_correlation_conflict_reference_residual"
                             for leaf in self.leaves(reference="rank(close)")))
        self.assertEqual(self.leaves(reference="rank(missing)"), ())
        self.assertEqual(self.leaves(source="rank(" * 10 + "close" + ")" * 10), ())
        self.catalog = replace(self.catalog, operators=self.catalog.operators[:-1])
        self.assertEqual(self.leaves(), ())

    def internal_leaves(self, source="rank(ts_mean(close,5))", reference="ts_mean(close,22)"):
        catalog = replace(
            self.catalog,
            operators=self.catalog.operators + tuple(
                OperatorDefinition(name, "test", ("REGULAR",), (
                    OperatorParameter("x", "expr"), OperatorParameter("d", "window"),
                ), (), "signal") for name in ("ts_mean", "ts_delta")
            ),
            windows=(WindowDefinition(5, "week"), WindowDefinition(22, "month")),
        )
        return tuple(iter_self_correlation_leaves(
            parse_formula(source).expression, parse_formula(reference).expression,
            catalog, field_candidates=("open",), families=SELF_CORRELATION_INTERNAL_FAMILIES, neutralization="NONE",
        ))

    def test_internal_edits_preserve_wrappers_windows_and_verified_repair_shape(self):
        source, reference = "rank(ts_mean(close,5))", "ts_mean(close,22)"
        leaves = {render_formula(item.expression): item for item in self.internal_leaves()
                  if item.family in SELF_CORRELATION_INTERNAL_FAMILIES}
        for text, family in (
            ("rank(ts_mean(open,5))", SELF_CORRELATION_FIELD_REPLACEMENT),
            ("rank(ts_delta(open,5))", SELF_CORRELATION_FIELD_OPERATOR_REPLACEMENT),
        ):
            leaf = leaves[text]
            self.assertEqual(leaf.family, family)
            self.assertEqual(leaf.change.before, source)
            self.assertEqual(leaf.change.after, text)
            self.assertTrue(matches_self_correlation_repair(
                parse_formula(source).expression, parse_formula(reference).expression,
                leaf.expression, action=family, location=leaf.change.location,
            ))
        internal = [r for r in leaves.values() if r.family in SELF_CORRELATION_INTERNAL_FAMILIES]
        self.assertEqual(len(internal), 2)

    def test_self_reference_can_verify_internal_actions_without_retired_generation(self):
        source = "rank(ts_mean(close,5)+ts_mean(close,22))"
        leaves = self.internal_leaves(source, source)
        families = {leaf.family for leaf in leaves}
        self.assertTrue(set(SELF_CORRELATION_INTERNAL_FAMILIES) <= families)
        self.assertFalse(set(SELF_CORRELATION_HALF_FAMILIES) & families)
        self.assertNotIn("self_correlation_conflict_reference_residual", families)
        expression = parse_formula(source).expression
        for leaf in leaves:
            self.assertNotEqual(expression, leaf.expression)
            self.assertLessEqual(analyze_formula(leaf.expression).depth, 10)
            if leaf.family in SELF_CORRELATION_INTERNAL_FAMILIES:
                self.assertTrue(matches_self_correlation_repair(expression, expression, leaf.expression,
                    action=leaf.family, location=leaf.change.location))
        self.assertFalse(matches_self_correlation_repair(expression, expression,
            parse_formula(f"vector_neut(zscore({source}),zscore({source}))").expression,
            action="self_correlation_conflict_reference_residual", location="formula"))

    def test_internal_edits_survive_depth_limit_and_keep_repeated_positions_separate(self):
        source = "ts_mean(" * 9 + "close" + ",5)" * 9
        leaves = self.internal_leaves(source, "ts_mean(close,22)")
        self.assertTrue(set(SELF_CORRELATION_INTERNAL_FAMILIES) <= {r.family for r in leaves})
        self.assertTrue({r.family for r in leaves} <= set((*SELF_CORRELATION_INTERNAL_FAMILIES, *SELF_CORRELATION_HALF_FAMILIES)))
        self.assertTrue(all(analyze_formula(r.expression).depth == 10 for r in leaves))
        repeated = self.internal_leaves("rank(ts_mean(close,5)+ts_mean(close,22))")
        fields = {render_formula(r.expression) for r in repeated
                  if r.family == SELF_CORRELATION_FIELD_REPLACEMENT}
        self.assertEqual(fields, {
            "rank(ts_mean(open,5)+ts_mean(close,22))",
            "rank(ts_mean(close,5)+ts_mean(open,22))",
        })
        for leaf in repeated:
            if leaf.family in SELF_CORRELATION_INTERNAL_FAMILIES:
                self.assertEqual(render_formula(leaf.expression).count("open"), 1)

    def test_field_scope_uses_shared_catalog_dataset_and_type_not_name_similarity(self):
        self.catalog = replace(self.catalog, fields=self.catalog.fields + (
            FieldDefinition("close_other", "unrelated", "sample", "sample", "MATRIX", 1.0),
            FieldDefinition("vector", "sample", "sample", "sample", "VECTOR", 1.0),
        ))
        selected = shared_field_replacements(
            parse_formula("rank(close)").expression,
            parse_formula("close").expression, self.catalog,
            ("open", "close_other", "vector"),
        )
        self.assertEqual(selected, ("open",))
        unrelated = self.internal_leaves(reference="ts_mean(open,22)")
        self.assertFalse(any(r.family in SELF_CORRELATION_INTERNAL_FAMILIES for r in unrelated))

    def test_forged_repair_labels_extra_edits_and_wrong_location_are_rejected(self):
        source = parse_formula("rank(ts_mean(close,5)+ts_mean(close,22))").expression
        reference = parse_formula("ts_mean(close,22)").expression
        location = "formula.arguments[0].left.arguments[0]"
        for text, action, position in (
            ("rank(ts_mean(open,5)+ts_mean(open,22))", SELF_CORRELATION_FIELD_REPLACEMENT, location),
            ("rank(ts_mean(open,22)+ts_mean(close,22))", SELF_CORRELATION_FIELD_REPLACEMENT, location),
            ("zscore(ts_delta(open,5)+ts_mean(close,22))", SELF_CORRELATION_FIELD_OPERATOR_REPLACEMENT, location),
            ("rank(ts_mean(open,5)+ts_delta(close,22))", SELF_CORRELATION_FIELD_OPERATOR_REPLACEMENT, location),
            ("rank(ts_mean(open,5)+ts_mean(close,22))", SELF_CORRELATION_FIELD_REPLACEMENT, "formula"),
        ):
            with self.subTest(formula=text):
                self.assertFalse(matches_self_correlation_repair(
                    source, reference, parse_formula(text).expression,
                    action=action, location=position,
                ))
