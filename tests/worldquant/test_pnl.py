import pytest

from worldquant.pnl import parse_pnl


def payload(rows):
    return {
        "schema": {"properties": [{"name": "pnl"}, {"name": "date"}]},
        "records": rows,
    }


def test_pnl_uses_column_names_and_preserves_empty_observation():
    assert parse_pnl(payload([[10, "2020-01-01"], [12, "2020-01-02"]])) == (
        ("2020-01-01", 10.0),
        ("2020-01-02", 12.0),
    )
    assert parse_pnl(payload([])) == ()


@pytest.mark.parametrize(
    "rows",
    [
        [[True, "2020-01-01"]],
        [[float("nan"), "2020-01-01"]],
        [[None, "2020-01-01"]],
        [[1, "2020-02-30"]],
        [[1, "2020-01-01"], [2, "2020-01-01"]],
        [[1, "2020-01-02"], [2, "2020-01-01"]],
        [[1]],
    ],
)
def test_bad_pnl_is_not_silently_filled(rows):
    with pytest.raises(ValueError):
        parse_pnl(payload(rows))
