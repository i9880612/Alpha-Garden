from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from generation.catalog import GenerationCatalog
from generation.formula import Call, Expression, analyze_formula
from selection.risks import has_tail_risk_structure
from worldquant.backtests import BacktestSettings


_SUPPORTED_NEUTRALIZATIONS = frozenset(
    {"NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"}
)
_NEUTRALIZATION_GRANULARITY = {
    "NONE": 0,
    "MARKET": 1,
    "SECTOR": 2,
    "INDUSTRY": 3,
    "SUBINDUSTRY": 4,
}
_SIGNAL_FIELD_TYPES = frozenset({"MATRIX", "VECTOR"})


@dataclass(frozen=True, slots=True)
class CategoryNeutralization:
    category: str
    neutralization: str


@dataclass(frozen=True, slots=True)
class BacktestSettingsPolicy:
    instrument_type: str
    region: str
    universe: str
    delay: int
    decay: int
    default_neutralization: str
    root_group_neutralization: str
    category_neutralizations: tuple[CategoryNeutralization, ...]
    default_truncation: float
    tail_risk_truncation: float
    pasteurization: str
    unit_handling: str
    nan_handling: str
    language: str
    visualization: bool
    max_trade: str
    max_position: str

    def __post_init__(self) -> None:
        for value in (
            self.instrument_type,
            self.region,
            self.universe,
            self.pasteurization,
            self.unit_handling,
            self.nan_handling,
            self.language,
            self.max_trade,
            self.max_position,
        ):
            _require_text(value, "backtest_settings_policy_text_invalid")
        if (
            isinstance(self.delay, bool)
            or not isinstance(self.delay, int)
            or self.delay < 0
        ):
            raise ValueError("backtest_settings_policy_delay_invalid")
        if (
            isinstance(self.decay, bool)
            or not isinstance(self.decay, int)
            or self.decay < 0
        ):
            raise ValueError("backtest_settings_policy_decay_invalid")
        for value in (
            self.default_neutralization,
            self.root_group_neutralization,
        ):
            if value not in _SUPPORTED_NEUTRALIZATIONS:
                raise ValueError(
                    "backtest_settings_policy_neutralization_invalid"
                )
        if (
            not isinstance(self.category_neutralizations, tuple)
            or not self.category_neutralizations
        ):
            raise ValueError(
                "backtest_settings_policy_category_neutralizations_invalid"
            )
        categories: set[str] = set()
        for rule in self.category_neutralizations:
            if not isinstance(rule, CategoryNeutralization):
                raise ValueError(
                    "backtest_settings_policy_category_neutralization_invalid"
                )
            _require_text(
                rule.category,
                "backtest_settings_policy_category_neutralization_invalid",
            )
            if (
                rule.category in categories
                or rule.neutralization not in _SUPPORTED_NEUTRALIZATIONS
            ):
                raise ValueError(
                    "backtest_settings_policy_category_neutralization_invalid"
                )
            categories.add(rule.category)
        for value in (self.default_truncation, self.tail_risk_truncation):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("backtest_settings_policy_truncation_invalid")
        if self.tail_risk_truncation > self.default_truncation:
            raise ValueError("backtest_settings_policy_truncation_invalid")
        if not isinstance(self.visualization, bool):
            raise ValueError("backtest_settings_policy_visualization_invalid")

    def as_config_dict(self) -> dict[str, object]:
        return {
            "catalogContext": {
                "instrumentType": self.instrument_type,
                "region": self.region,
                "universe": self.universe,
                "delay": self.delay,
            },
            "decay": self.decay,
            "neutralization": {
                "default": self.default_neutralization,
                "rootGroupNeutralize": self.root_group_neutralization,
                "byFieldCategory": {
                    rule.category: rule.neutralization
                    for rule in sorted(
                        self.category_neutralizations,
                        key=lambda item: item.category,
                    )
                },
            },
            "truncation": {
                "default": self.default_truncation,
                "tailRisk": self.tail_risk_truncation,
            },
            "pasteurization": self.pasteurization,
            "unitHandling": self.unit_handling,
            "nanHandling": self.nan_handling,
            "language": self.language,
            "visualization": self.visualization,
            "maxTrade": self.max_trade,
            "maxPosition": self.max_position,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.as_config_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def fingerprint(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def allows(self, settings: BacktestSettings) -> bool:
        if not isinstance(settings, BacktestSettings):
            return False
        return (
            settings.instrument_type == self.instrument_type
            and settings.region == self.region
            and settings.universe == self.universe
            and settings.delay == self.delay
            and settings.decay == self.decay
            and settings.neutralization in self.allowed_neutralizations
            and settings.truncation
            in {self.default_truncation, self.tail_risk_truncation}
            and settings.pasteurization == self.pasteurization
            and settings.unit_handling == self.unit_handling
            and settings.nan_handling == self.nan_handling
            and settings.language == self.language
            and settings.visualization == self.visualization
            and settings.max_trade == self.max_trade
            and settings.max_position == self.max_position
        )

    @property
    def allowed_neutralizations(self) -> frozenset[str]:
        return frozenset(
            {
                self.default_neutralization,
                self.root_group_neutralization,
                *(rule.neutralization for rule in self.category_neutralizations),
            }
        )

    @classmethod
    def from_config_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "BacktestSettingsPolicy":
        required = {
            "catalogContext",
            "decay",
            "neutralization",
            "truncation",
            "pasteurization",
            "unitHandling",
            "nanHandling",
            "language",
            "visualization",
            "maxTrade",
            "maxPosition",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("backtest_settings_policy_file_invalid")
        context = payload["catalogContext"]
        neutralization = payload["neutralization"]
        truncation = payload["truncation"]
        if (
            not isinstance(context, Mapping)
            or set(context) != {
                "instrumentType",
                "region",
                "universe",
                "delay",
            }
            or not isinstance(neutralization, Mapping)
            or set(neutralization)
            != {"default", "rootGroupNeutralize", "byFieldCategory"}
            or not isinstance(neutralization["byFieldCategory"], Mapping)
            or not isinstance(truncation, Mapping)
            or set(truncation) != {"default", "tailRisk"}
        ):
            raise ValueError("backtest_settings_policy_file_invalid")
        category_rules = neutralization["byFieldCategory"]
        return cls(
            instrument_type=context["instrumentType"],
            region=context["region"],
            universe=context["universe"],
            delay=context["delay"],
            decay=payload["decay"],
            default_neutralization=neutralization["default"],
            root_group_neutralization=neutralization["rootGroupNeutralize"],
            category_neutralizations=tuple(
                CategoryNeutralization(category=category, neutralization=value)
                for category, value in sorted(category_rules.items())
            ),
            default_truncation=truncation["default"],
            tail_risk_truncation=truncation["tailRisk"],
            pasteurization=payload["pasteurization"],
            unit_handling=payload["unitHandling"],
            nan_handling=payload["nanHandling"],
            language=payload["language"],
            visualization=payload["visualization"],
            max_trade=payload["maxTrade"],
            max_position=payload["maxPosition"],
        )


def load_backtest_settings_policy(path: str | Path) -> BacktestSettingsPolicy:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("backtest_settings_policy_file_invalid") from exc
    try:
        return BacktestSettingsPolicy.from_config_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("backtest_settings_policy_file_invalid") from exc


def choose_backtest_settings(
    expression: Expression,
    catalog: GenerationCatalog,
    policy: BacktestSettingsPolicy,
) -> BacktestSettings:
    if not isinstance(catalog, GenerationCatalog):
        raise ValueError("backtest_settings_catalog_invalid")
    if not isinstance(policy, BacktestSettingsPolicy):
        raise ValueError("backtest_settings_policy_invalid")
    context = catalog.context
    if (
        context.instrument_type != policy.instrument_type
        or context.region != policy.region
        or context.universe != policy.universe
        or context.delay != policy.delay
    ):
        raise ValueError("backtest_settings_catalog_context_not_allowed")

    settings = BacktestSettings(
        instrument_type=context.instrument_type,
        region=context.region,
        universe=context.universe,
        delay=context.delay,
        decay=policy.decay,
        neutralization=_choose_neutralization(expression, catalog, policy),
        truncation=(
            policy.tail_risk_truncation
            if has_tail_risk_structure(expression)
            else policy.default_truncation
        ),
        pasteurization=policy.pasteurization,
        unit_handling=policy.unit_handling,
        nan_handling=policy.nan_handling,
        language=policy.language,
        visualization=policy.visualization,
        max_trade=policy.max_trade,
        max_position=policy.max_position,
    )
    if not policy.allows(settings):
        raise ValueError("backtest_settings_not_allowed")
    return settings


def backtest_settings_fingerprint(settings: BacktestSettings) -> str:
    if not isinstance(settings, BacktestSettings):
        raise ValueError("backtest_settings_invalid")
    payload = json.dumps(
        settings.as_platform_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _choose_neutralization(
    expression: Expression,
    catalog: GenerationCatalog,
    policy: BacktestSettingsPolicy,
) -> str:
    if isinstance(expression, Call) and expression.operator == "group_neutralize":
        return policy.root_group_neutralization
    by_category = {
        rule.category: rule.neutralization
        for rule in policy.category_neutralizations
    }
    facts = analyze_formula(expression)
    choices = {
        by_category.get(field.category, policy.default_neutralization)
        for name in facts.referenced_names
        if (field := catalog.field(name)) is not None
        and field.field_type in _SIGNAL_FIELD_TYPES
    }
    return max(
        choices or {policy.default_neutralization},
        key=lambda value: _NEUTRALIZATION_GRANULARITY[value],
    )


def _require_text(value: object, error: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)
