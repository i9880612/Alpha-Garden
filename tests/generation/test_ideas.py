from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generation.catalog import (
    CatalogContext,
    FieldDefinition,
    GenerationCatalog,
    OperatorDefinition,
    OperatorParameter,
    WindowDefinition,
)
from generation.field_profiles import FieldProfile, FieldProfileSet
from generation.formula import Call, analyze_formula
from generation.ideas import (
    FAMILIES,
    FORBIDDEN_OPERATORS,
    TEMPLATES,
    build_combo_variant,
    build_idea_variant_pool,
    plan_idea_batch,
)
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import validate_formula

_NORMALIZERS = frozenset({"rank", "zscore", "normalize", "quantile", "scale", "winsorize"})


def _operator(name, *parameters, roles=()):
    return OperatorDefinition(
        name, None, ("REGULAR",),
        tuple(OperatorParameter(key, kind, optional=optional)
              for key, kind, optional in parameters),
        roles=roles, output_kind="signal")


def _catalog(fields=None, drop_operators=()):
    x, d = ("x", "expr", False), ("d", "window", False)
    operators = (
        _operator("rank", x, ("rate", "float", True), roles=("cross_sectional_normalization",)),
        _operator("abs", x), _operator("vec_avg", x),
        _operator("ts_backfill", x, ("lookback", "window", True), ("k", "int", True)),
        _operator("ts_corr", x, ("y", "expr", False), d),
        *(_operator(name, x, d) for name in ("ts_delay", "ts_delta", "ts_mean", "ts_std_dev")),
    )
    operators = tuple(op for op in operators if op.name not in drop_operators)
    fields = fields if fields is not None else tuple(
        FieldDefinition(profile.field_id, profile.dataset, None, None,
                        profile.field_type, 0.9)
        for profile in _PROFILES)
    return GenerationCatalog(
        CatalogContext("EQUITY", "USA", "TOP3000", 1), fields, operators,
        tuple(WindowDefinition(v, "test") for v in (5, 22, 66, 120, 250)))


def _p(field_id, kind, cadence, scale, direction="unknown", role=None, metric=None,
       dataset="ds", field_type="MATRIX", tier="reviewed"):
    return FieldProfile(field_id, dataset, field_type, tier, kind, cadence, scale,
                        direction, role, metric or field_id)


_PROFILES = (
    _p("close", "price", "daily", "price", dataset="pv1"),
    _p("vwap", "price", "daily", "price", dataset="pv1"),
    _p("returns", "price", "daily", "ratio", dataset="pv1"),
    _p("volume", "volume", "daily", "count", dataset="pv1"),
    _p("cap", "price", "daily", "amount", "-", dataset="pv1"),
    _p("sharesout", "fundamental", "daily", "count", "-", dataset="pv1"),
    _p("assets", "fundamental", "quarterly", "amount"),
    _p("cashflow_op", "fundamental", "quarterly", "amount", "+"),
    _p("income", "fundamental", "quarterly", "amount", "+"),
    _p("debt", "fundamental", "quarterly", "amount", "-"),
    _p("eps", "fundamental", "quarterly", "per_share", "+"),
    _p("est_ebit", "analyst_estimate", "event", "amount", "+", "mean", "ebit", "analyst4"),
    _p("anl4_ebit_high", "analyst_estimate", "event", "amount", "+", "high", "ebit", "analyst4"),
    _p("anl4_ebit_low", "analyst_estimate", "event", "amount", "+", "low", "ebit", "analyst4"),
    _p("anl4_ebit_value", "analyst_actual", "quarterly", "amount", "+", "actual", "ebit",
       "analyst4"),
    _p("iv30", "option_vol", "daily", "ratio", "-"),
    _p("pcr30", "option_flow", "daily", "ratio", "-"),
    _p("snt", "sentiment", "daily", "score"),
    _p("nws", "news", "event", "score", "+", field_type="VECTOR"),
)
_ALL = frozenset(p.field_id for p in _PROFILES)


def _pool(profiles=_PROFILES, enabled=_ALL, catalog=None):
    return build_idea_variant_pool(
        catalog or _catalog(), FieldProfileSet.from_profiles(profiles), enabled_field_ids=enabled)


def test_every_template_has_variants_and_all_pass_local_checks():
    catalog = _catalog()
    pool = _pool(catalog=catalog)
    assert set(pool.by_template()) == {t.template_id for t in TEMPLATES}
    for variant in pool.variants:
        assert validate_formula(variant.expression, catalog).is_valid, variant.formula
        assert not find_coarse_unit_issues(variant.expression, catalog), variant.formula
        assert not FORBIDDEN_OPERATORS & set(analyze_formula(variant.expression).operator_names)
        assert variant.formula.startswith("rank(")
        assert variant.family in FAMILIES


def test_no_template_stacks_cross_sectional_normalizers():
    def nested(node, inside=False):
        if isinstance(node, Call):
            here = node.operator in _NORMALIZERS
            if here and inside:
                return True
            return any(nested(a.value, inside or here) for a in node.arguments)
        return any(nested(getattr(node, k), inside) for k in ("left", "right", "operand")
                   if hasattr(node, k))
    assert not any(nested(v.expression) for v in _pool().variants)


def test_only_enabled_profiled_catalog_fields_are_used():
    unclassified = FieldProfile("mystery", "ds", "MATRIX", "unclassified")
    excluded = replace(_p("zip", "price", "daily", "price", tier="rule"), excluded="编码")
    profiles = _PROFILES + (unclassified, excluded, _p("not_in_catalog", "sentiment", "daily",
                                                       "score"))
    enabled = (_ALL - {"snt"}) | {"mystery", "zip", "not_in_catalog"}
    used = {f for v in _pool(profiles, enabled).variants for f in v.field_ids}
    assert not used & {"snt", "mystery", "zip", "not_in_catalog"}


def test_directions_shape_the_sign_and_unknown_is_generated_positive():
    formulas = {v.formula for v in _pool().variants}
    assert "rank(-iv30)" in formulas
    assert "rank(-pcr30)" in formulas
    assert "rank(snt-ts_mean(snt,22))" in formulas
    assert "rank(ts_backfill(eps,lookback=66)/close)" in formulas
    assert "rank(ts_backfill(income,lookback=66)/cap)" in formulas
    assert not any("debt,lookback=66)/cap" in f for f in formulas)  # value skips known-bad items


def test_vector_fields_are_reduced_and_sparse_fields_backfilled():
    sentiment = [v.formula for v in _pool().by_template()["sentiment_shift"] if "nws" in v.formula]
    assert sentiment and all("ts_backfill(vec_avg(nws),lookback=22)" in f for f in sentiment)


def test_paired_templates_only_pair_the_same_item():
    pool = _pool().by_template()
    assert [v.field_ids for v in pool["dispersion"]] == [("anl4_ebit_high", "anl4_ebit_low")]
    assert [v.field_ids for v in pool["earnings_surprise"]] == [("anl4_ebit_value", "est_ebit")]


def test_missing_operator_only_removes_templates_that_need_it():
    full = _pool().by_template()
    reduced = _pool(catalog=_catalog(drop_operators=("ts_corr",)))
    by_template = reduced.by_template()
    assert "price_volume" not in by_template
    assert {k: len(v) for k, v in by_template.items()} == {
        k: len(v) for k, v in full.items() if k != "price_volume"}
    assert any(code.startswith("price_volume:invalid") for code, _ in reduced.exclusions)


def test_batch_is_deterministic_and_spreads_evenly_over_families():
    pool = _pool()
    first = plan_idea_batch(pool, target_count=12, seed=3)
    again = plan_idea_batch(pool, target_count=12, seed=3)
    assert [c.candidate.formula for c in first.candidates] == [
        c.candidate.formula for c in again.candidates]
    counts = Counter(c.variant.family for c in first.candidates)
    assert set(counts) == set(FAMILIES) and set(counts.values()) == {2}
    other = plan_idea_batch(pool, target_count=12, seed=4)
    assert [c.candidate.formula for c in other.candidates] != [
        c.candidate.formula for c in first.candidates]
    assert all(c.candidate.strategy_label == "idea:" + c.variant.template_id
               for c in first.candidates)


def test_exhausted_families_hand_their_share_to_the_rest_then_report_shortfall():
    pool = _pool()
    total = len(pool.variants)
    batch = plan_idea_batch(pool, target_count=total + 5, seed=1)
    assert len(batch.candidates) == total
    assert batch.missing_count == 5 and batch.saturated
    assert batch.remaining_by_template == ()
    assert len({c.candidate.formula for c in batch.candidates}) == total


def test_already_tested_variants_are_skipped():
    pool = _pool()
    tested = frozenset(v.fingerprint for v in pool.variants[:10])
    batch = plan_idea_batch(pool, target_count=len(pool.variants), seed=2,
                            excluded_fingerprints=tested)
    assert len(batch.candidates) == len(pool.variants) - 10
    assert not tested & {c.variant.fingerprint for c in batch.candidates}


def test_reviewed_variants_are_drawn_before_rule_tier_ones():
    profiles = tuple(replace(p, tier="rule", direction="unknown")
                     if p.field_id in {"snt", "iv30", "pcr30"} else p for p in _PROFILES)
    pool = _pool(profiles)
    rule_count = sum(not v.reviewed for v in pool.variants)
    assert rule_count > 0
    batch = plan_idea_batch(pool, target_count=len(pool.variants), seed=5)
    order = {}
    for item in batch.candidates:
        order.setdefault(item.variant.template_id, []).append(item.variant.reviewed)
    for flags in order.values():
        assert flags == sorted(flags, reverse=True)  # all reviewed before any rule-tier


def test_batch_rejects_bad_requests_and_allows_zero():
    pool = _pool()
    assert plan_idea_batch(pool, target_count=0, seed=1).candidates == ()
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError):
            plan_idea_batch(pool, target_count=bad, seed=1)


def test_combo_needs_two_families_and_stays_valid():
    catalog = _catalog()
    by_template = _pool(catalog=catalog).by_template()
    reversal, momentum = by_template["reversal"][0], by_template["momentum"][0]
    value = by_template["value"][0]
    assert build_combo_variant(reversal, momentum, catalog) is None
    combo = build_combo_variant(reversal, value, catalog)
    assert combo is not None and combo.family == "combo"
    assert combo.formula.count("rank(") == 2 and "+" in combo.formula
    assert validate_formula(combo.expression, catalog).is_valid
    assert build_combo_variant(value, reversal, catalog).formula == combo.formula
    assert build_combo_variant(combo, value, catalog) is None
