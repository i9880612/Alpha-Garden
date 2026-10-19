"""Field profiles: what each platform field means, for idea-driven generation.

The profile file (``data/field-profiles.json``) is a research asset kept out of Git.
Reviewed entries carry hand-checked labels; ``rule`` entries come from
``profile_field_by_rules``; ``unclassified`` entries are left to random exploration.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

FIELD_PROFILE_FILE_VERSION = 1

KINDS = frozenset({
    "price", "volume", "fundamental", "analyst_estimate", "analyst_guidance",
    "analyst_actual", "analyst_count", "sentiment", "news", "option_vol",
    "option_flow", "option_price", "risk", "score", "relationship",
})
CADENCES = frozenset({"daily", "event", "quarterly"})
SCALES = frozenset({"price", "amount", "per_share", "ratio", "count", "score"})
DIRECTIONS = frozenset({"+", "-", "unknown"})
TIERS = frozenset({"reviewed", "rule", "unclassified"})
TEMPLATE_TIERS = frozenset({"reviewed", "rule"})


@dataclass(frozen=True, slots=True)
class FieldProfile:
    field_id: str
    dataset: str | None
    field_type: str | None
    tier: str
    kind: str | None = None
    cadence: str | None = None
    scale: str | None = None
    direction: str = "unknown"
    role: str | None = None
    metric: str | None = None
    synonym_group: str | None = None
    excluded: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.field_id, str) or not self.field_id.strip():
            raise ValueError("field_profile_id_invalid")
        if self.tier not in TIERS:
            raise ValueError(f"field_profile_tier_invalid:{self.field_id}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"field_profile_direction_invalid:{self.field_id}")
        if self.tier != "reviewed" and self.direction != "unknown":
            raise ValueError(f"field_profile_unreviewed_direction:{self.field_id}")
        for value, allowed, code in (
            (self.kind, KINDS, "kind"),
            (self.cadence, CADENCES, "cadence"),
            (self.scale, SCALES, "scale"),
        ):
            if value is not None and value not in allowed:
                raise ValueError(f"field_profile_{code}_invalid:{self.field_id}")
        if self.usable_by_templates and (
            self.kind is None or self.cadence is None or self.scale is None
        ):
            raise ValueError(f"field_profile_labels_missing:{self.field_id}")

    @property
    def usable_by_templates(self) -> bool:
        return self.excluded is None and self.tier in TEMPLATE_TIERS

    @property
    def sign(self) -> int:
        return -1 if self.direction == "-" else 1


@dataclass(frozen=True, slots=True)
class FieldProfileSet:
    profiles: Mapping[str, FieldProfile]

    def __post_init__(self) -> None:
        if any(key != value.field_id for key, value in self.profiles.items()):
            raise ValueError("field_profile_set_key_mismatch")
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))

    @classmethod
    def from_profiles(cls, profiles: Iterable[FieldProfile]) -> FieldProfileSet:
        by_id: dict[str, FieldProfile] = {}
        for profile in profiles:
            if profile.field_id in by_id:
                raise ValueError(f"field_profile_duplicate:{profile.field_id}")
            by_id[profile.field_id] = profile
        return cls(by_id)

    def get(self, field_id: str) -> FieldProfile | None:
        return self.profiles.get(field_id)

    def template_fields(self) -> tuple[FieldProfile, ...]:
        return tuple(
            profile for _, profile in sorted(self.profiles.items())
            if profile.usable_by_templates
        )


@dataclass(frozen=True, slots=True)
class CatalogFieldInfo:
    """The catalog facts rule profiling needs (description is not in GenerationCatalog)."""
    field_id: str
    dataset_id: str | None
    field_type: str | None
    coverage: float | None
    description: str | None


@dataclass(frozen=True, slots=True)
class ReconciledFieldProfiles:
    profiles: FieldProfileSet
    missing_field_ids: tuple[str, ...]
    added_field_ids: tuple[str, ...]


def load_field_profiles(path: Path) -> FieldProfileSet:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != FIELD_PROFILE_FILE_VERSION:
        raise ValueError("field_profile_file_version_unsupported")
    entries = payload.get("fields")
    if not isinstance(entries, list):
        raise ValueError("field_profile_file_fields_invalid")
    return FieldProfileSet.from_profiles(_profile_from_entry(entry) for entry in entries)


def reconcile_field_profiles(
    profiles: FieldProfileSet,
    catalog_fields: Iterable[CatalogFieldInfo],
) -> ReconciledFieldProfiles:
    """Align the profile file with the live catalog without stopping a run.

    Fields gone from the catalog are dropped for this run and reported; fields new to
    the catalog are profiled by rules immediately (rule tier, direction unknown).
    """
    signal_fields = {
        info.field_id: info for info in catalog_fields
        if info.field_type in {"MATRIX", "VECTOR"}
    }
    kept: list[FieldProfile] = []
    added: list[str] = []
    for field_id, info in sorted(signal_fields.items()):
        profile = profiles.get(field_id)
        if profile is None:
            kept.append(profile_field_by_rules(info))
            added.append(field_id)
        else:
            kept.append(profile)
    missing = tuple(sorted(set(profiles.profiles) - set(signal_fields)))
    return ReconciledFieldProfiles(
        profiles=FieldProfileSet.from_profiles(kept),
        missing_field_ids=missing,
        added_field_ids=tuple(added),
    )


def promote_to_rule_tier(profiles: FieldProfileSet, field_id: str) -> FieldProfileSet:
    """Make an unclassified field available to templates after random exploration found signal."""
    profile = profiles.get(field_id)
    if profile is None or profile.tier != "unclassified" or profile.excluded is not None:
        return profiles
    if profile.kind is None or profile.cadence is None or profile.scale is None:
        return profiles
    updated = dict(profiles.profiles)
    updated[field_id] = replace(profile, tier="rule")
    return FieldProfileSet(updated)


def _profile_from_entry(entry: object) -> FieldProfile:
    if not isinstance(entry, dict):
        raise ValueError("field_profile_entry_invalid")
    try:
        return FieldProfile(
            field_id=entry["field_id"],
            dataset=entry.get("dataset"),
            field_type=entry.get("type"),
            tier=entry["tier"],
            kind=entry.get("kind"),
            cadence=entry.get("cadence"),
            scale=entry.get("scale"),
            direction=entry.get("direction", "unknown"),
            role=entry.get("role"),
            metric=entry.get("metric"),
            synonym_group=entry.get("synonym_group"),
            excluded=entry.get("excluded"),
            note=entry.get("note"),
        )
    except KeyError as exc:
        raise ValueError(f"field_profile_entry_key_missing:{exc.args[0]}") from exc


# --- Rule profiling -------------------------------------------------------------

# Identifier / code / metadata fields: numeric but not a magnitude.
_CODE_PATTERNS = (
    r"forecast type", r"estimation type", r"broker (?:name|id)", r"^financial item$",
    r"currency code", r"pricing currency", r"identification number", r"home currency",
    r"reporting currency", r"currency in which", r"currency of", r"type ind",
    r"instrument type", r"city code", r"country code", r"zip code", r"mapping data",
    r"segment type", r"level of the sector", r"terminal sector \(",
    (r"^(?:the )?(?:utc |event |announcement |report |filing |record |ex-?dividend |"
     r"payment |fiscal |period )?(?:date|timestamp)\b"),
    r"\bdate of (?:the )?(?:report|announcement|filing|event|release)\b",
    r"reporting period (?:start|end)", r"marking the (?:beginning|end)",
    r"\bsession on which\b", r"index of session", r"timestamp", r"\butc\b",
    r"value of specified index", r"industry code", r"\bsic\b", r"\bnaics\b",
    r"\bticker\b", r"\bcusip\b", r"identifier",
)
# One value for the whole market: ranking across stocks is meaningless.
_MARKET_LEVEL_PATTERNS = (
    r"spy etf", r"(?:price|close|last)[a-z ]* (?:of )?(?:the )?spy\b",
    r"\bspy (?:price|close|last)", r"global equities",
)
_MECHANICAL_FIELDS = frozenset({"adjfactor", "split"})

_PER_SHARE = re.compile(
    r"per share|\beps\b|\bbvps\b|\bcfps\b|\bdps\b|\bfcfps\b|earnings per|"
    r"dividends? per|book value per", re.IGNORECASE)
_RATIO = re.compile(
    r"\(%\)|\bpercent\b|percentage|\bratio\b|\brate\b|\byield\b|margin|"
    r"\breturn on\b|z-score|\bpct\b", re.IGNORECASE)
_COUNT = re.compile(
    r"\bnumber of\b|\bcount\b|employees|shares outstanding|\bshares used\b", re.IGNORECASE)

_ANALYST_ROLES = (
    ("count", (r"number of estimat|number of forecasts|numest|estimates count|"
               r"count of|number of .* estimates|\bnumber\b$")),
    ("up", r"number of upper|upper estimations|up(?:ward)? revisions"),
    ("down", r"number of lower|lower estimations|down(?:ward)? revisions"),
    ("std", r"standard deviation|\bstd\b"),
    ("high", r"highest|\bhigh\b|\bmax(?:imum)? estimat"),
    ("low", r"lowest|\blow\b|\bmin(?:imum)? estimat"),
    ("median", r"median|50th percentile"),
    ("mean", r"\bmean\b|average of estimat|average of analyst|consensus"),
    ("prev", r"previous estimation"),
    ("guidance_max", r"max(?:imum)? guidance|maxguidance|guidance.*maximum"),
    ("guidance_min", r"min(?:imum)? guidance|minguidance|guidance.*minimum"),
    ("actual", r"actual|announced|reported value|reported financial|\bvalue\b$"),
)

_EXCLUDED_CODE = "编码/标识/日期类字段，数值没有大小含义"
_EXCLUDED_MARKET = "全市场同一个值，横截面排序无意义"
_EXCLUDED_MECHANICAL = "拆股/复权等机械调整因子，不是信号"
_EXCLUDED_NO_COVERAGE = "覆盖率为0"
_EXCLUDED_RELATIONSHIP_CODE = "关系图谱中的结构/编码字段"


def profile_field_by_rules(info: CatalogFieldInfo) -> FieldProfile:
    """First-pass profile from dataset, id and description. Never guesses direction."""
    field_id = info.field_id
    dataset = info.dataset_id or ""
    description = (info.description or "").strip()
    lowered = description.lower()

    def excluded(reason: str) -> FieldProfile:
        return FieldProfile(field_id, info.dataset_id, info.field_type, "rule", excluded=reason)

    if field_id in _MECHANICAL_FIELDS:
        return excluded(_EXCLUDED_MECHANICAL)
    if any(re.search(pattern, lowered) for pattern in _CODE_PATTERNS):
        return excluded(_EXCLUDED_CODE)
    if any(re.search(pattern, lowered) for pattern in _MARKET_LEVEL_PATTERNS):
        return excluded(_EXCLUDED_MARKET)
    if not info.coverage or info.coverage <= 0:
        return excluded(_EXCLUDED_NO_COVERAGE)

    tier = "rule"
    role = None
    metric = _metric_text(description)
    if dataset == "pv1":
        kind = "volume" if field_id in {"volume", "adv20", "sharesout"} else "price"
        scale = {"returns": "ratio", "cap": "amount", "volume": "count", "adv20": "count",
                 "sharesout": "count", "dividend": "per_share"}.get(field_id, "price")
        cadence, metric = "daily", field_id
    elif dataset.startswith("fundamental"):
        kind = "fundamental"
        cadence = "event" if "daily" in lowered else "quarterly"
        scale = _scale_from_text(description)
    elif dataset == "analyst4":
        role = next((name for name, pattern in _ANALYST_ROLES if re.search(pattern, lowered)),
                    "estimate")
        if role.startswith("guidance"):
            kind = "analyst_guidance"
        elif role == "actual":
            kind = "analyst_actual"
        elif role in {"count", "up", "down"}:
            kind = "analyst_count"
        elif "rating" in lowered or "score on the given" in lowered:
            kind, role = "score", None
        else:
            kind = "analyst_estimate"
        scale = "count" if kind == "analyst_count" else _scale_from_text(
            description + " " + field_id)
        if scale == "count" and kind != "analyst_count":
            scale = "amount"
        cadence = "quarterly" if kind == "analyst_actual" else "event"
    elif dataset == "option8":
        kind, cadence, scale = "option_vol", "daily", "ratio"
    elif dataset == "option9":
        cadence = "daily"
        if field_id.startswith("pcr_"):
            kind, scale = "option_flow", "ratio"
        elif "breakeven" in field_id or "forward_price" in field_id or "price" in lowered:
            kind, scale = "option_price", "price"
        else:
            kind, scale = "option_flow", _scale_from_text(description, "ratio")
    elif dataset == "model16":
        kind, scale = "score", "score"
        cadence = "daily" if "derivative" in field_id else "event"
    elif dataset == "model51":
        kind, cadence, scale = "risk", "daily", "ratio"
    elif dataset == "pv13":
        cadence, kind = "daily", "relationship"
        if field_id.startswith("rel_ret") or "retsig" in field_id:
            scale = "ratio"
        elif field_id.startswith("rel_num") or "total" in field_id or "count" in field_id:
            scale = "count"
        elif "rank" in field_id or "rk_" in field_id:
            scale = "score"
        elif "cap" in field_id:
            kind, scale = "volume", "amount"
        else:
            return excluded(_EXCLUDED_RELATIONSHIP_CODE)
    elif dataset.startswith("socialmedia"):
        kind, cadence = "sentiment", "daily"
        scale = "count" if "volume" in lowered or "buzz" in field_id else "score"
    elif dataset == "news18":
        kind, cadence, scale = "news", "event", "score"
    elif dataset == "news12":
        kind, cadence = "news", "event"
        if re.search(r"minutes|elapsed", lowered):
            scale = "count"
        elif re.search(r"percent|pct|ratio|return|%", lowered):
            scale = "ratio"
        elif re.search(r"price|close|open|high|low|vwap", lowered):
            scale = "price"
        elif "volume" in lowered or "cap" in lowered:
            scale = "amount"
        else:
            scale = _scale_from_text(description, "ratio")
        if scale in {"count", "price", "amount", "per_share"}:
            # Intraday reaction timings and raw price levels: too unclear for templates.
            tier = "unclassified"
    else:
        return FieldProfile(field_id, info.dataset_id, info.field_type, "unclassified")

    return FieldProfile(
        field_id=field_id, dataset=info.dataset_id, field_type=info.field_type,
        tier=tier, kind=kind, cadence=cadence, scale=scale, role=role, metric=metric,
        synonym_group=synonym_group(kind, metric, role),
    )


def synonym_group(kind: str, metric: str | None, role: str | None) -> str:
    group = f"{kind}:{metric}"
    if kind == "analyst_estimate" and role:
        group += ":dispersion" if role == "std" else ":level"
    return group


def _metric_text(description: str) -> str:
    """Normalize the financial item a field refers to, for synonym grouping."""
    text = description.lower()
    text = re.sub(r"\s+-\s+(?:quarterly|annual|daily|ttm|ytd|12mm)\b.*$", "", text)
    text = re.sub(r"\s+-\s+(?:mean|median|number|the highest|the lowest|highest|lowest|"
                  r"standard deviation|std|announced|estimation|forecast|average|"
                  r"actual)\b.*$", "", text)
    text = re.sub(r"\((?:annual|quarterly|daily)\)", "", text)
    text = re.sub(r"[^a-z0-9&/ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _scale_from_text(text: str, default: str = "amount") -> str:
    if _PER_SHARE.search(text):
        return "per_share"
    if _RATIO.search(text):
        return "ratio"
    if _COUNT.search(text):
        return "count"
    return default
