from __future__ import annotations

import pytest

from generation.catalog import (
    CatalogContext,
    FieldDefinition,
    GenerationCatalog,
    OperatorDefinition,
    OperatorParameter,
    WindowDefinition,
)
from generation.formula import analyze_formula
from generation.internal_edits import (
    INTERNAL_EDIT_FAMILIES,
    INTERNAL_FIELD_REPLACEMENT,
    INTERNAL_LAYER_REMOVAL,
    INTERNAL_OPERATOR_REPLACEMENT,
    iter_internal_edit_candidates,
)
from generation.logic import prepare_formula_logic
from generation.parser import parse_formula
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import validate_formula
from selection.risks import assess_candidate_risk


@pytest.fixture
def catalog():
    def field(name, kind="MATRIX", coverage=1.0):
        return FieldDefinition(name, "dataset", "Price Volume", "Price", kind, coverage)

    def operator(name, *parameters, roles=(), output="signal", scope=("REGULAR",)):
        return OperatorDefinition(name, "test", scope, parameters, roles, output)

    x, d = OperatorParameter("x", "expr"), OperatorParameter("d", "window")
    return GenerationCatalog(
        CatalogContext("EQUITY", "USA", "TOP3000", 1),
        (
            field("close"),
            field("open"),
            field("sparse", coverage=0),
            field("vector", "VECTOR"),
            field("vector_other", "VECTOR"),
            field("sector", "GROUP"),
        ),
        (
            operator("rank", x, roles=("cross_sectional_normalization",)),
            operator("zscore", x, roles=("cross_sectional_normalization",)),
            operator("ts_mean", x, d),
            operator("ts_delta", x, d),
            operator("ts_scale", x, d, OperatorParameter("constant", "float", optional=True)),
            operator("ts_backfill", x, OperatorParameter("lookback", "window", optional=True),
                     OperatorParameter("k", "int", optional=True)),
            operator(
                "ts_decay_linear",
                x,
                d,
                OperatorParameter("dense", "bool", optional=True),
            ),
            operator("other_window", x, OperatorParameter("lookback", "window")),
            operator("integer_input", x, OperatorParameter("n", "int")),
            operator("bool_output", x, d, output="condition"),
            operator("unavailable", x, d, scope=("SUPER",)),
            operator("needs_input", x, d, OperatorParameter("y", "expr")),
            operator("vec_avg", x),
            operator("vec_sum", x),
            operator("vector_neut", x, OperatorParameter("y", "expr")),
            operator("group_rank", x, OperatorParameter("group", "group")),
            operator("option", x, OperatorParameter("mode", "string")),
            operator("log", x),
            operator("inverse", x),
        ),
        (WindowDefinition(22, "month"), WindowDefinition(66, "quarter")),
    )


def candidates(formula, catalog, fields=("open",)):
    return tuple(
        iter_internal_edit_candidates(
            parse_formula(formula).expression,
            catalog,
            parent_task_id="parent-task",
            field_candidates=fields,
        )
    )


def test_three_edit_kinds_preserve_parent_identity_and_one_local_change(catalog):
    formula = "rank(ts_mean(close,66))"
    result = candidates(formula, catalog)
    by_formula = {c.formula: c for c in result}
    for text, action in (
        ("rank(ts_delta(close,66))", INTERNAL_OPERATOR_REPLACEMENT),
        ("rank(ts_mean(open,66))", INTERNAL_FIELD_REPLACEMENT),
        ("rank(close)", INTERNAL_LAYER_REMOVAL),
    ):
        child = by_formula[text]
        assert child.change.action == action
        assert child.parent_task_id == "parent-task"
        assert child.parent_formula_fingerprint == parse_formula(formula).fingerprint
        assert child.generation_action == "mutation"
    assert by_formula["rank(close)"].change.before == "ts_mean(close,66)"
    assert by_formula["rank(close)"].change.after == "close"


def test_depth_ten_still_has_equal_depth_and_lower_depth_candidates(catalog):
    formula = "close"
    for _ in range(9):
        formula = f"ts_mean({formula},66)"
    parent = parse_formula(formula).expression
    assert analyze_formula(parent).depth == 10
    result = candidates(formula, catalog)
    assert set(c.change.action for c in result) == set(INTERNAL_EDIT_FAMILIES)
    assert any(analyze_formula(c.expression).depth == 10 for c in result)
    assert any(analyze_formula(c.expression).depth == 9 for c in result)
    for child in result:
        assert analyze_formula(child.expression).depth <= 10
        assert (
            analyze_formula(child.expression).complexity
            <= analyze_formula(parent).complexity
        )
        assert validate_formula(child.expression, catalog).is_valid
        assert not find_coarse_unit_issues(child.expression, catalog)
        assert prepare_formula_logic(child.expression).expression == child.expression


def test_repeated_field_occurrences_are_changed_separately(catalog):
    formula = "rank(ts_mean(close,66)+ts_delta(close,22))"
    edits = [
        c
        for c in candidates(formula, catalog)
        if c.change.action == INTERNAL_FIELD_REPLACEMENT
    ]
    assert {c.formula for c in edits} == {
        "rank(ts_mean(open,66)+ts_delta(close,22))",
        "rank(ts_mean(close,66)+ts_delta(open,22))",
    }


def test_equal_arity_is_not_enough_and_missing_arguments_are_rejected(catalog):
    result = candidates("rank(ts_mean(close,66))", catalog)
    assert not any(
        name in c.formula
        for c in result
        for name in ("integer_input", "bool_output", "unavailable", "needs_input")
    )


def test_named_windows_are_rebound_without_changing_values(catalog):
    result = candidates("rank(ts_mean(close,d=66))", catalog)
    assert "rank(other_window(close,66))" in {c.formula for c in result}
    assert "rank(ts_backfill(close,lookback=66))" in {c.formula for c in result}


@pytest.mark.parametrize("formula", [
    "rank(ts_backfill(close,lookback=66))",
    "rank(ts_backfill(lookback=66,x=close))",
])
def test_backfill_replacements_use_positional_required_windows(catalog, formula):
    result = candidates(formula, catalog)
    formulas = {c.formula for c in result}
    for operator in ("ts_mean", "ts_delta", "ts_scale", "ts_decay_linear"):
        assert f"rank({operator}(close,66))" in formulas
        assert not any(f"{operator}(close,d=" in text for text in formulas)
    assert all(validate_formula(c.expression, catalog).is_valid for c in result)


def test_omitted_optional_argument_does_not_block_operator_replacement(catalog):
    result = candidates("rank(ts_decay_linear(close,66))", catalog)
    assert "rank(ts_delta(close,66))" in {c.formula for c in result}
    explicit = candidates("rank(ts_decay_linear(close,66,dense=true))", catalog)
    assert "rank(ts_delta(close,66))" not in {c.formula for c in explicit}


def test_vector_inputs_stay_vectors_and_cannot_escape_reduction(catalog):
    result = candidates(
        "rank(vec_avg(vector))", catalog, ("open", "vector_other", "sector")
    )
    fields = [c for c in result if c.change.action == INTERNAL_FIELD_REPLACEMENT]
    assert {c.formula for c in fields} == {"rank(vec_avg(vector_other))"}
    assert "rank(vector)" not in {c.formula for c in result}
    assert "rank(vec_sum(vector))" in {c.formula for c in result}


def test_layer_removal_does_not_break_required_dimensionless_input(catalog):
    result = candidates("vector_neut(zscore(close),zscore(open))", catalog)
    assert "vector_neut(close,zscore(open))" not in {c.formula for c in result}
    assert "vector_neut(zscore(close),open)" not in {c.formula for c in result}
    assert all(not find_coarse_unit_issues(c.expression, catalog) for c in result)


def test_string_options_and_group_fields_are_not_replaced(catalog):
    result = candidates("rank(option(close,mode=open))", catalog, ("close", "open"))
    assert all("mode=open" in c.formula for c in result if "option" in c.formula)
    groups = candidates("group_rank(close,sector)", catalog, ("open", "sector"))
    assert all(
        "group_rank(open,sector)" == c.formula
        for c in groups
        if c.change.action == INTERNAL_FIELD_REPLACEMENT
    )


def test_existing_selection_risk_checks_are_still_required(catalog):
    result = candidates("rank(close)", catalog, ("sparse",))
    sparse = next(c for c in result if c.formula == "rank(sparse)")
    assert not assess_candidate_risk(sparse.expression, catalog).eligible
    removal = next(
        c
        for c in candidates("rank(inverse(rank(close)))", catalog)
        if c.formula == "inverse(rank(close))"
    )
    assert not assess_candidate_risk(removal.expression, catalog).eligible


def test_noop_duplicates_and_implicit_second_edits_are_excluded(catalog):
    result = candidates("rank(ts_mean(close,66))", catalog, ("open", "open", "close"))
    assert len({c.fingerprint for c in result}) == len(result)
    assert result == candidates("rank(ts_mean(close,66))", catalog, ("close", "open"))
    # Replacing the left input by the right would collapse the subtraction.
    cancelling = candidates("rank(close-open)", catalog)
    assert "rank(open-open)" not in {c.formula for c in cancelling}


@pytest.mark.parametrize(
    "formula", ["unknown(close)", "rank(vector)", "rank(close-close)"]
)
def test_invalid_parents_are_rejected(catalog, formula):
    with pytest.raises(ValueError, match="internal_edit_parent_invalid"):
        candidates(formula, catalog)


def test_unknown_field_candidates_are_rejected(catalog):
    with pytest.raises(ValueError, match="internal_edit_field_candidates_invalid"):
        candidates("rank(close)", catalog, ("absent",))


def test_prefix_and_binary_siblings_remain_unchanged(catalog):
    result = candidates("rank(-ts_mean(close,66)+open)", catalog)
    child = next(c for c in result if c.formula == "rank(-ts_delta(close,66)+open)")
    assert child.change.location == "formula.arguments[0].left.operand"
    assert child.change.before == "ts_mean(close,66)"
    assert child.change.after == "ts_delta(close,66)"


def test_cannot_relax_depth_limit_to_accept_an_over_limit_parent(catalog):
    formula = "close"
    for _ in range(10):
        formula = f"ts_mean({formula},66)"
    with pytest.raises(ValueError, match="internal_edit_parent_invalid"):
        candidates(formula, catalog)
