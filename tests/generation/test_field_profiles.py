from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generation.field_profiles import (
    CatalogFieldInfo,
    FieldProfile,
    FieldProfileSet,
    load_field_profiles,
    profile_field_by_rules,
    promote_to_rule_tier,
    reconcile_field_profiles,
)


def _write(payload) -> Path:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return Path(handle.name)


def _entry(field_id, **values):
    entry = {"field_id": field_id, "dataset": "pv1", "type": "MATRIX", "tier": "rule",
             "kind": "price", "cadence": "daily", "scale": "price", "direction": "unknown"}
    entry.update(values)
    return entry


def _info(field_id, dataset, description, field_type="MATRIX", coverage=0.9):
    return CatalogFieldInfo(field_id, dataset, field_type, coverage, description)


def test_load_reads_entries_and_ignores_extra_keys():
    path = _write({"version": 1, "fields": [
        _entry("close", tier="reviewed", direction="-", note="说明", coverage=1.0,
               description="Daily close price"),
        _entry("zip", excluded="编码"),
    ]})
    profiles = load_field_profiles(path)
    assert profiles.get("close").direction == "-"
    assert profiles.get("close").note == "说明"
    assert [p.field_id for p in profiles.template_fields()] == ["close"]


@pytest.mark.parametrize("payload, code", [
    ({"version": 2, "fields": []}, "field_profile_file_version_unsupported"),
    ({"version": 1, "fields": [_entry("close", direction="+")]},
     "field_profile_unreviewed_direction:close"),
    ({"version": 1, "fields": [_entry("close"), _entry("close")]},
     "field_profile_duplicate:close"),
    ({"version": 1, "fields": [_entry("close", kind="weird")]}, "field_profile_kind_invalid:close"),
    ({"version": 1, "fields": [_entry("close", scale=None)]},
     "field_profile_labels_missing:close"),
])
def test_load_rejects_malformed_files(payload, code):
    with pytest.raises(ValueError, match=code):
        load_field_profiles(_write(payload))


def test_reconcile_drops_vanished_fields_and_profiles_new_ones_without_failing():
    profiles = FieldProfileSet.from_profiles([
        FieldProfile("close", "pv1", "MATRIX", "reviewed", "price", "daily", "price"),
        FieldProfile("gone", "pv1", "MATRIX", "rule", "price", "daily", "price"),
    ])
    result = reconcile_field_profiles(profiles, [
        _info("close", "pv1", "Daily close price"),
        _info("assets", "fundamental6", "Assets - Total"),
        _info("sector", "pv1", "grouping fields", field_type="GROUP"),
    ])
    assert result.missing_field_ids == ("gone",)
    assert result.added_field_ids == ("assets",)
    assert result.profiles.get("close").tier == "reviewed"
    added = result.profiles.get("assets")
    assert (added.tier, added.kind, added.cadence, added.scale, added.direction) == (
        "rule", "fundamental", "quarterly", "amount", "unknown")
    assert result.profiles.get("sector") is None


@pytest.mark.parametrize("info, expected", [
    (_info("anl4_x_flag", "analyst4", "Net profit - forecast type (revision/new/...)"), "excluded"),
    (_info("spy_close", "news12", "Closing price of SPY ETF for the session"), "excluded"),
    (_info("fnd6_ein", "fundamental6", "Employer Identification Number code"), "excluded"),
    (_info("split", "pv1", "Stock split ratio"), "excluded"),
    (_info("x", "fundamental6", "Assets - Total", coverage=0.0), "excluded"),
    (_info("beta_60", "model51", "The rolling beta value relative to SPY"), "risk"),
    (_info("acc", "fundamental2", "Carrying value as of the balance sheet date of payables"),
     "fundamental"),
])
def test_rules_exclude_codes_and_market_level_values_only(info, expected):
    profile = profile_field_by_rules(info)
    assert (profile.kind if profile.excluded is None else "excluded") == expected
    assert profile.direction == "unknown"


def test_rules_label_analyst_roles_and_scales():
    cases = {
        "Earnings per share - median of estimations": ("analyst_estimate", "median", "per_share"),
        "Net profit - The highest estimation": ("analyst_estimate", "high", "amount"),
        "Net profit - number of estimations": ("analyst_count", "count", "count"),
        "Net profit- announced financial value": ("analyst_actual", "actual", "amount"),
        "Minimum guidance value for Free Cash Flow": ("analyst_guidance", "guidance_min", "amount"),
        "Free Cash Flow - median (50th percentile) estimate": ("analyst_estimate", "median",
                                                               "amount"),
    }
    for description, expected in cases.items():
        profile = profile_field_by_rules(_info("f", "analyst4", description))
        assert (profile.kind, profile.role, profile.scale) == expected, description


def test_rules_leave_intraday_news_timings_to_random_exploration():
    timing = profile_field_by_rules(_info(
        "n1", "news12", "Number of minutes that elapsed before price went up 4 percentage points",
        field_type="VECTOR"))
    reaction = profile_field_by_rules(_info(
        "n2", "news12", "Percent change in price during the first minute following the news"))
    assert timing.tier == "unclassified"
    assert reaction.tier == "rule" and reaction.scale == "ratio"


def test_promote_moves_only_labelled_unclassified_fields():
    profiles = FieldProfileSet.from_profiles([
        FieldProfile("n1", "news12", "VECTOR", "unclassified", "news", "event", "count"),
        FieldProfile("odd", None, "MATRIX", "unclassified"),
        FieldProfile("close", "pv1", "MATRIX", "reviewed", "price", "daily", "price"),
    ])
    promoted = promote_to_rule_tier(profiles, "n1")
    assert promoted.get("n1").tier == "rule"
    assert promote_to_rule_tier(profiles, "odd") is profiles
    assert promote_to_rule_tier(profiles, "close") is profiles
    assert promote_to_rule_tier(profiles, "absent") is profiles
