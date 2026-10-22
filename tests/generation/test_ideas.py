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
    COMBO_MAX_USES_PER_PARENT,
    FAMILIES,
    FORBIDDEN_OPERATORS,
    TEMPLATES,
    ComboParent,
    build_combo_variant,
    build_idea_variant_pool,
    plan_combo_variants,
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
        *(_operator(name, x, d) for name in ("ts_delay", "ts_delta", "ts_mean", "ts_std_dev", "ts_sum",
                                             "ts_rank")),
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
    _p("open", "price", "daily", "price", dataset="pv1"),
    _p("adv20", "volume", "daily", "count", dataset="pv1"),
    _p("dividend", "fundamental", "event", "per_share", "+", dataset="pv1"),
    _p("rel_ret_comp", "relationship", "daily", "ratio", "+", dataset="pv13"),
    _p("high", "price", "daily", "price", dataset="pv1"),
    _p("low", "price", "daily", "price", dataset="pv1"),
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
    assert "rank(-ts_backfill(iv30,lookback=5))" in formulas
    assert "rank(-ts_backfill(pcr30,lookback=5))" in formulas
    assert "rank(snt-ts_mean(snt,22))" in formulas
    # Fundamentals are used as the platform carries them; no ts_backfill layer.
    assert "rank(eps/close)" in formulas
    assert "rank(income/cap)" in formulas
    assert not any("debt,lookback=66)/cap" in f for f in formulas)  # value skips known-bad items


def test_no_template_adds_a_bare_constant_to_a_field_value():
    # The platform's unit check rejects e.g. high-low+0.0001 ("Incompatible unit for add").
    for variant in _pool().variants:
        assert "+0.0001" not in variant.formula and "(1+" not in variant.formula, variant.formula


def test_per_share_surprise_only_pairs_fields_from_the_same_source():
    profiles = _PROFILES + (
        _p("est_eps", "analyst_estimate", "event", "per_share", "+", "mean", "eps", "analyst4"),
        _p("anl4_eps_value", "analyst_actual", "quarterly", "per_share", "+", "actual", "eps",
           "analyst4"),
        _p("anl4_eps_median", "analyst_estimate", "event", "per_share", "+", "median", "eps",
           "analyst4"),
    )
    catalog = _catalog(fields=tuple(
        FieldDefinition(p.field_id, p.dataset, None, None, p.field_type, 0.9) for p in profiles))
    pairs = [v.field_ids for v in _pool(profiles, _ALL | {p.field_id for p in profiles},
                                        catalog).by_template()["earnings_surprise"]]
    assert ("anl4_eps_value", "anl4_eps_median") in pairs
    assert ("anl4_eps_value", "est_eps") not in pairs
    assert ("anl4_ebit_value", "est_ebit") in pairs  # amounts carry no unit on the platform


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
    first = plan_idea_batch(pool, target_count=2 * len(FAMILIES), seed=3)
    again = plan_idea_batch(pool, target_count=2 * len(FAMILIES), seed=3)
    assert [c.candidate.formula for c in first.candidates] == [
        c.candidate.formula for c in again.candidates]
    counts = Counter(c.variant.family for c in first.candidates)
    assert set(counts) == set(FAMILIES) and set(counts.values()) == {2}
    other = plan_idea_batch(pool, target_count=2 * len(FAMILIES), seed=4)
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


def _parent(variant, sharpe, wave, noise=0.0):
    import math
    days = [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(300)]
    return ComboParent(variant, sharpe, {d: math.sin(i * wave) + noise * math.cos(i * 7.3)
                                         for i, d in enumerate(days)})


def test_combo_flips_negative_parents_and_keeps_formula_valid():
    catalog = _catalog()
    by_template = _pool(catalog=catalog).by_template()
    combo = build_combo_variant(by_template["reversal"][0], by_template["value"][0], catalog,
                                signs=(-1, 1))
    assert validate_formula(combo.expression, catalog).is_valid
    assert "+-" not in combo.formula and combo.formula.count("rank(") == 2
    assert combo.formula.count("-rank(") == 1
    with pytest.raises(ValueError):
        build_combo_variant(by_template["reversal"][0], by_template["value"][0], catalog,
                            signs=(2, 1))


def test_combo_planning_pairs_independent_strong_parents_across_families():
    catalog = _catalog()
    t = _pool(catalog=catalog).by_template()
    reversal, momentum, value, quality = (t["reversal"][0], t["momentum"][0], t["value"][0],
                                          t["quality"][0])
    parents = [
        _parent(reversal, 1.4, 0.11),
        _parent(momentum, 1.3, 0.11),           # same family as reversal: never paired with it
        _parent(value, -1.1, 0.53),             # negative: enters flipped
        _parent(quality, 1.2, 0.11, noise=0.1),  # moves with reversal: too correlated to pair
        _parent(t["low_vol"][0], 0.3, 0.97),    # too weak to be a parent
    ]
    combos = plan_combo_variants(parents, catalog)
    pairs = {frozenset(v.hypothesis.split("：")[1].split(" + ")) for v in combos}
    assert frozenset({"reversal", "momentum"}) not in pairs
    assert frozenset({"reversal", "quality"}) not in pairs
    assert all("low_vol" not in pair for pair in pairs)
    assert frozenset({"reversal", "value"}) in pairs
    flipped = [v for v in combos if "value" in v.hypothesis]
    assert flipped and all(v.formula.startswith("-rank(") or "-rank(" in v.formula for v in flipped)
    assert all(v.family == "combo" and v.reviewed for v in combos)
    assert plan_combo_variants(parents, catalog) == combos


def test_combo_planning_spreads_parents_and_needs_enough_shared_days():
    catalog = _catalog()
    t = _pool(catalog=catalog).by_template()
    star = _parent(t["reversal"][0], 2.0, 0.11)
    others = [_parent(v, 1.0, 0.37 + k * 0.29) for k, v in
              enumerate(t["value"][:1] + t["forward_value"][:1] + t["quality"][:1])]
    combos = plan_combo_variants([star, *others], catalog)
    assert sum("reversal" in v.hypothesis for v in combos) == COMBO_MAX_USES_PER_PARENT
    short = ComboParent(t["momentum"][0], 1.5, dict(list(star.daily_pnl.items())[:100]))
    assert plan_combo_variants([short, others[0]], catalog) == ()


def test_near_duplicate_parents_share_one_idea_budget():
    catalog = _catalog()
    t = _pool(catalog=catalog).by_template()
    # Three reversal-like parents that move together (vwap reversal written three ways).
    twins = [_parent(v, 1.5 - 0.1 * k, 0.11, noise=0.05 * k)
             for k, v in enumerate(t["reversal"][:2] + t["mean_deviation"][:1])]
    others = [_parent(v, 1.0, 0.37 + k * 0.29) for k, v in
              enumerate(t["value"][:1] + t["forward_value"][:1] + t["quality"][:1])]
    combos = plan_combo_variants([*twins, *others], catalog)
    price_side = [v for v in combos if "reversal" in v.hypothesis or "mean_deviation" in v.hypothesis]
    assert len(price_side) == COMBO_MAX_USES_PER_PARENT  # one budget for the three twins
    partners = [v.hypothesis.split("：")[1] for v in price_side]
    assert len(set(partners)) == len(partners)


def test_calmer_parent_is_weighted_up_to_balance_the_combo():
    import math
    catalog = _catalog()
    t = _pool(catalog=catalog).by_template()
    days = [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(300)]
    loud = ComboParent(t["reversal"][0], 1.5, {d: 3 * math.sin(i * 0.11) for i, d in enumerate(days)})
    calm = ComboParent(t["value"][0], 1.2, {d: math.sin(i * 0.53) for i, d in enumerate(days)})
    (combo,) = plan_combo_variants([loud, calm], catalog)
    assert f"3*{t['value'][0].formula}" in combo.formula
    assert validate_formula(combo.expression, catalog).is_valid
    assert not find_coarse_unit_issues(combo.expression, catalog)
    with pytest.raises(ValueError):
        build_combo_variant(t["reversal"][0], t["value"][0], catalog, weights=(1.0, 9.0))


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


def test_new_price_and_linked_templates_are_generated():
    t = _pool().by_template()
    assert "rank(-((open-ts_delay(close,1))/ts_delay(close,1)))" in {v.formula for v in t["overnight_gap"]}
    assert {v.window for v in t["reversal"]} == {1, 2, 3, 5, 10, 15, 22}
    assert {f for v in t["reversal"] for f in v.field_ids} == {"close", "vwap", "open"}
    assert {v.formula for v in t["linked_momentum"]} >= {"rank(ts_mean(rel_ret_comp,5))"}
    assert all(v.family == "linked" for v in t["linked_momentum"] + t["linked_gap"])
    # `returns` carries a price unit on the platform and rel_ret_* none: never subtract them.
    assert t["linked_gap"]
    assert all("returns" not in v.formula for v in t["linked_gap"])
    assert "rank(ts_mean(rel_ret_comp-(close/ts_delay(close,1)-1),22))" in {
        v.formula for v in t["linked_gap"]}


def test_fundamentals_drop_backfill_but_remember_their_older_form():
    pool = _pool()
    by_formula = {v.formula: v for v in pool.variants}
    variant = by_formula["rank(income/cap)"]
    assert variant.legacy_fingerprint is not None
    # Analyst fields keep the layer and have no older form.
    analyst = [v for v in pool.variants if "est_ebit" in v.field_ids]
    assert analyst and all("ts_backfill(est_ebit" in v.formula and v.legacy_fingerprint is None for v in analyst)
    # A variant tested in its older form is not offered again.
    batch = plan_idea_batch(pool, target_count=len(pool.variants), seed=1,
                            excluded_fingerprints=frozenset({variant.legacy_fingerprint}))
    assert "rank(income/cap)" not in {c.variant.formula for c in batch.candidates}


def test_option_fields_fill_missing_days():
    # A missing option day leaves a few stocks holding the whole book (CONCENTRATED_WEIGHT).
    option = [v.formula for v in _pool().by_template()["option_signal"]]
    assert option and all("ts_backfill(" in f for f in option)
    assert not any("iv30" in f and "ts_backfill(iv30,lookback=5)" not in f for f in option)
