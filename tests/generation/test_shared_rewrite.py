from __future__ import annotations

import pytest

from generation.catalog import (
    CatalogContext, FieldDefinition, GenerationCatalog, OperatorDefinition,
    OperatorParameter, WindowDefinition,
)
from generation.parser import parse_formula
from generation.shared_rewrite import matches_shared_half


@pytest.fixture
def catalog():
    def field(name, dataset="dataset", kind="MATRIX"):
        return FieldDefinition(name, dataset, "Model", "Model", kind, 1.0)

    x, y, d = (OperatorParameter("x", "expr"), OperatorParameter("y", "expr"),
               OperatorParameter("d", "window"))

    def op(name, *parameters):
        roles = ("cross_sectional_normalization",) if name in {"rank", "zscore", "normalize", "quantile"} else ()
        return OperatorDefinition(name, "test", ("REGULAR",), parameters, roles, "signal")

    return GenerationCatalog(
        CatalogContext("EQUITY", "USA", "TOP3000", 1),
        (field("close"), field("open"), field("other", "unrelated"),
         field("opt6_10dorhv", "option6"), field("opt6_slopestd1y", "option6"),
         field("editorial_commentary_sentiment_2", "news18", "VECTOR"),
         field("nws18_qep", "news18", "VECTOR"),
         field("vector", "dataset", "VECTOR"), field("sector", "dataset", "GROUP")),
        (op("rank", x), op("zscore", x), op("normalize", x), op("quantile", x),
         op("abs", x), op("ts_mean", x, d), op("ts_delta", x, d),
         op("ts_count_nans", x, d), op("ts_decay_linear", x, d),
         op("ts_step", OperatorParameter("step", "int")), op("vec_avg", x),
         op("vec_sum", x), op("vector_neut", x, y), op("group_rank", x, OperatorParameter("group", "group")),
         op("option", x, OperatorParameter("mode", "string"))),
        tuple(WindowDefinition(w, "test") for w in (5, 22, 66, 120)),
    )


def expression(text):
    return parse_formula(text).expression


@pytest.mark.parametrize("parent,reference,child,side", (
    ("rank(ts_mean(close,5))", "rank(ts_mean(close,22))", "zscore(ts_delta(close,5))", "left"),
    ("rank(ts_mean(close,5))", "rank(ts_mean(close,22))", "rank(ts_mean(open,5))", "right"),
    ("rank(ts_mean(close,5))", "rank(ts_mean(close,5))", "rank(ts_mean(open,5))", "right"),
))
def test_archived_half_edits_remain_verifiable_with_original_windows_and_dataset(catalog, parent, reference, child, side):
    assert matches_shared_half(expression(parent), expression(reference), expression(child), catalog, side)
    for altered in (child.replace(",5)", ",22)"), child.replace("open", "other"), child.replace("open", "vector")):
        if altered != child:
            assert not matches_shared_half(expression(parent), expression(reference), expression(altered), catalog, side)
    assert not matches_shared_half(expression(parent), expression(reference), expression(child), catalog,
                                   "right" if side == "left" else "left")


@pytest.mark.parametrize("child", ("rank(ts_mean(close,5))", "rank(ts_mean(open,22))", "zscore(ts_mean(open,5))",
                                   "rank(ts_mean(missing,5))", "rank(ts_mean(open,5)+ts_mean(open,5))"))
def test_forged_historical_half_does_not_grant_research(catalog, child):
    assert not matches_shared_half(expression("rank(ts_mean(close,5))"), expression("rank(ts_mean(close,22))"),
                                   expression(child), catalog, "right")
