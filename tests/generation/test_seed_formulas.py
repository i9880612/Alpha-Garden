from dataclasses import replace
from unittest.mock import patch

from generation.catalog import (
    CatalogContext, FieldDefinition, GenerationCatalog, OperatorDefinition,
    OperatorParameter, WindowDefinition,
)
from generation.seed_formulas import SEED_FORMULAS, seed_mother_candidates
from generation.validation import validate_formula


def catalog():
    def operator(name, parameters):
        return OperatorDefinition(name, None, ("REGULAR",), tuple(
            OperatorParameter(key, kind) for key, kind in parameters), output_kind="signal")
    return GenerationCatalog(
        CatalogContext("EQUITY", "USA", "TOP3000", 1),
        tuple(FieldDefinition(name, "price", "Price Volume", None, "MATRIX", 1.0)
              for name in ("close", "open", "high", "low", "returns", "volume")),
        tuple(operator(name, (("x", "expr"),)) for name in ("rank", "sign", "log"))
        + tuple(operator(name, (("x", "expr"), ("d", "window"))) for name in (
            "ts_delta", "ts_rank", "ts_std_dev", "ts_zscore", "ts_mean", "ts_delay"))
        + (operator("ts_corr", (("x", "expr"), ("y", "expr"), ("d", "window"))),),
        tuple(WindowDefinition(value, "test") for value in (1, 5, 10, 20, 40)),
    )


def test_mothers_require_supported_fields_operators_types_and_coverage():
    complete = catalog()
    valid = seed_mother_candidates(complete)
    assert len(valid) == len(SEED_FORMULAS)
    assert all(validate_formula(c.expression, complete).is_valid for c in valid)
    for changed in (
        replace(complete, operators=tuple(op for op in complete.operators if op.name != "ts_delay")),
        replace(complete, fields=tuple(f for f in complete.fields if f.field_id != "volume")),
        replace(complete, fields=tuple(replace(f, field_type="VECTOR") if f.field_id == "close" else f
                                      for f in complete.fields)),
        replace(complete, fields=tuple(replace(f, coverage=0) if f.field_id == "volume" else f
                                      for f in complete.fields)),
    ):
        remaining = seed_mother_candidates(changed)
        assert 0 < len(remaining) < len(valid)
        assert all(validate_formula(c.expression, changed).is_valid for c in remaining)


def test_invalid_mother_does_not_block_independent_valid_mothers():
    formulas = (
        "rank(ts_delta(close,20))", "rank(ts_delta(close,20))",
        "rank(unknown(close))", "rank(missing)", "rank(close/close)",
        "rank(log(close))", "rank(",
    )
    with patch("generation.seed_formulas.SEED_FORMULAS", formulas):
        selected = seed_mother_candidates(catalog())
    assert len(selected) == 1
    assert selected[0].formula == "rank(ts_delta(close,20))"


def test_mothers_do_not_hide_unexpected_program_errors():
    import pytest
    with patch("generation.seed_formulas.validate_formula", side_effect=RuntimeError("bug")):
        with pytest.raises(RuntimeError, match="bug"):
            seed_mother_candidates(catalog())
