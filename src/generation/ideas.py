"""Idea-driven exploration: economic hypotheses turned into formula variants.

Each template states a hypothesis, the field profiles it accepts, and a fixed formula
shape. Every (template, fields, window) combination is one variant; variants are
validated with the same checks as random exploration and then sampled evenly by
family, then by template. Nothing here touches the database or the platform.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from random import Random

from generation.candidate import FormulaCandidate, exploration_candidate
from generation.catalog import GenerationCatalog
from generation.field_profiles import FieldProfile, FieldProfileSet
from generation.formula import (
    Expression,
    analyze_formula,
    formula_fingerprint,
    render_formula,
)
from generation.logic import prepare_formula_logic
from generation.parser import FormulaSyntaxError, parse_formula
from generation.structure_limits import fits_formula_structure_limits
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import validate_formula

FAMILIES = ("price_action", "valuation", "quality", "growth_financing", "analyst", "alternative", "linked")
FAMILY_LABELS = {
    "price_action": "价格行为", "valuation": "估值", "quality": "质量",
    "growth_financing": "成长与融资", "analyst": "分析师", "alternative": "另类数据",
    "linked": "关联公司",
}
# Operators templates must never produce (data-availability signals, not economic ones).
FORBIDDEN_OPERATORS = frozenset({"ts_count_nans", "days_from_last_change"})
STRATEGY_LABEL_PREFIX = "idea:"
_BACKFILL_LOOKBACK = {"quarterly": 66, "event": 22}
# The platform already carries fundamental values forward between filings: without
# ts_backfill three fundamental alphas backtested the same (daily PnL correlation
# 0.996-0.997), while an analyst-estimate alpha lost 4% of its Sharpe (2026-10-21). Only
# fundamentals drop the layer.
_UNFILLED_KINDS = frozenset({"fundamental"})


@dataclass(frozen=True, slots=True)
class IdeaTemplate:
    template_id: str
    family: str
    hypothesis: str
    expand: Callable[[_FieldPool], Iterable[tuple[tuple[str, ...], int | None, str]]]


@dataclass(frozen=True, slots=True)
class IdeaVariant:
    template_id: str
    family: str
    hypothesis: str
    field_ids: tuple[str, ...]
    window: int | None
    expression: Expression
    # Every field hand-reviewed; such variants are drawn before rule-profiled ones.
    reviewed: bool = True
    # The same variant as written before fundamentals dropped ts_backfill, if different:
    # a variant tested in that form is not tested again.
    legacy_fingerprint: str | None = None

    @property
    def formula(self) -> str:
        return render_formula(self.expression)

    @property
    def fingerprint(self) -> str:
        return formula_fingerprint(self.expression)


@dataclass(frozen=True, slots=True)
class IdeaVariantPool:
    variants: tuple[IdeaVariant, ...]
    exclusions: tuple[tuple[str, int], ...]

    def by_template(self) -> dict[str, tuple[IdeaVariant, ...]]:
        grouped: dict[str, list[IdeaVariant]] = defaultdict(list)
        for variant in self.variants:
            grouped[variant.template_id].append(variant)
        return {key: tuple(value) for key, value in grouped.items()}


@dataclass(frozen=True, slots=True)
class IdeaCandidate:
    candidate: FormulaCandidate
    variant: IdeaVariant


@dataclass(frozen=True, slots=True)
class IdeaBatch:
    candidates: tuple[IdeaCandidate, ...]
    missing_count: int
    remaining_by_template: tuple[tuple[str, int], ...]

    @property
    def saturated(self) -> bool:
        return self.missing_count > 0


# --- Field selection --------------------------------------------------------------

class _FieldPool:
    def __init__(self, catalog: GenerationCatalog, profiles: tuple[FieldProfile, ...], *,
                 backfill_fundamentals: bool = False) -> None:
        self.catalog = catalog
        self.profiles = profiles
        self.by_id = {profile.field_id: profile for profile in profiles}
        self.backfill_fundamentals = backfill_fundamentals

    def select(self, predicate: Callable[[FieldProfile], bool]) -> tuple[FieldProfile, ...]:
        return tuple(profile for profile in self.profiles if predicate(profile))

    def has(self, field_id: str) -> bool:
        return field_id in self.by_id

    def atom(self, profile: FieldProfile) -> str:
        """Field reference with vector reduction and gap filling for sparse data."""
        text = profile.field_id
        if profile.field_type == "VECTOR":
            text = f"vec_avg({text})"
        lookback = _BACKFILL_LOOKBACK.get(profile.cadence or "")
        if lookback is not None and (self.backfill_fundamentals or profile.kind not in _UNFILLED_KINDS):
            text = f"ts_backfill({text},lookback={lookback})"
        return text

    def atom_by_id(self, field_id: str) -> str:
        profile = self.by_id.get(field_id)
        return self.atom(profile) if profile is not None else field_id

    def deflated(self, profile: FieldProfile) -> str | None:
        """Make a level comparable across stocks according to its scale."""
        atom = self.atom(profile)
        if profile.scale == "amount":
            return f"{atom}/cap"
        if profile.scale in {"per_share", "price"}:
            return f"{atom}/close"
        if profile.scale in {"ratio", "score"}:
            return atom
        return None


def _signed(profile: FieldProfile, body: str) -> str:
    return f"-({body})" if profile.sign < 0 else body


def _not_negative(profile: FieldProfile) -> bool:
    """Known-good or unknown: unknown is generated as + and left to direction validation."""
    return profile.direction in {"+", "unknown"}


def _metric_pairs(
    left: Iterable[FieldProfile], right: Iterable[FieldProfile],
) -> list[tuple[FieldProfile, FieldProfile]]:
    """Pair fields that describe the same item (same dataset and metric text)."""
    index: dict[tuple[str | None, str | None], list[FieldProfile]] = defaultdict(list)
    for profile in right:
        index[(profile.dataset, profile.metric)].append(profile)
    pairs = []
    for profile in left:
        for partner in index.get((profile.dataset, profile.metric), ()):
            if partner.field_id != profile.field_id:
                pairs.append((profile, partner))
    return pairs


# --- Templates ----------------------------------------------------------------------

# Price-volume templates were the only ones that kept producing seeds, and with two
# price fields and two or three windows each they ran out after 25 variants. They now
# span open as well and a fuller window ladder (2026-10-21).
def _price_fields(pool: _FieldPool) -> tuple[str, ...]:
    return tuple(name for name in ("close", "vwap", "open") if pool.has(name))


def _reversal(pool):
    for name in _price_fields(pool):
        for w in (1, 2, 3, 5, 10, 15, 22):
            yield (name,), w, f"-({name}/ts_delay({name},{w})-1)"


def _mean_deviation(pool):
    for name in _price_fields(pool):
        for w in (5, 10, 22, 44, 66):
            yield (name,), w, f"-({name}-ts_mean({name},{w}))/ts_std_dev({name},{w})"


def _momentum(pool):
    if pool.has("close"):
        for skip in (5, 20):
            for w in (60, 120, 230):
                yield ("close",), w, f"ts_delay(close,{skip})/ts_delay(close,{w + skip})-1"


def _price_volume(pool):
    if not pool.has("volume"):
        return
    for name in _price_fields(pool):
        for w in (5, 10, 22, 44):
            yield (name, "volume"), w, f"-ts_corr({name},volume,{w})"
    if pool.has("returns") and pool.has("adv20"):
        for w in (5, 10, 22, 44):
            yield ("returns", "volume", "adv20"), w, f"-ts_corr(returns,volume/adv20,{w})"


def _low_vol(pool):
    if pool.has("returns"):
        for w in (22, 66, 120, 250):
            yield ("returns",), w, f"-ts_std_dev(returns,{w})"
    for profile in pool.select(lambda p: p.kind in {"option_vol", "risk"} and p.direction == "-"
                               and "skew" not in (p.metric or "")):
        yield (profile.field_id,), None, f"-{pool.atom(profile)}"


def _intraday_position(pool):
    if not all(pool.has(name) for name in ("high", "low", "close")):
        return
    # high==low forces close==high, so the ratio is 0/0 (NaN), never x/0.
    position = "(high-close)/(high-low)"
    yield ("high", "low", "close"), None, position
    for w in (3, 5, 10, 15, 22):
        yield ("high", "low", "close"), w, f"ts_mean({position},{w})"


def _abnormal_volume(pool):
    if not pool.has("volume"):
        return
    for w in (5, 10, 20, 40, 60):
        yield ("volume",), w, f"volume/ts_mean(volume,{w})"
    if pool.has("adv20"):
        yield ("volume", "adv20"), None, "volume/adv20"


def _smoothed(fields, body, windows, *, sign=""):
    """The raw daily value, then its average over each window."""
    yield fields, None, f"{sign}({body})"
    for w in windows:
        yield fields, w, f"{sign}ts_mean({body},{w})"


def _overnight_gap(pool):
    if pool.has("open") and pool.has("close"):
        # Written as a difference over the price, not a ratio minus one: a constant next
        # to a ratio of two fields fails the local unit check.
        yield from _smoothed(("open", "close"), "(open-ts_delay(close,1))/ts_delay(close,1)",
                             (3, 5, 10, 22), sign="-")


def _intraday_return(pool):
    if pool.has("open") and pool.has("close"):
        yield from _smoothed(("open", "close"), "(close-open)/open", (3, 5, 10, 22), sign="-")


def _close_vs_vwap(pool):
    if pool.has("close") and pool.has("vwap"):
        yield from _smoothed(("close", "vwap"), "(close-vwap)/vwap", (3, 5, 10), sign="-")


def _daily_range(pool):
    if all(pool.has(name) for name in ("high", "low", "close")):
        for w in (5, 10, 22, 66):
            yield ("high", "low", "close"), w, f"-ts_mean((high-low)/close,{w})"


def _share_turnover(pool):
    if pool.has("volume") and pool.has("sharesout"):
        for w in (5, 22, 66, 120):
            yield ("volume", "sharesout"), w, f"-ts_mean(volume/sharesout,{w})"


def _year_high(pool):
    # Where today's price sits within its recent range: near the top tends to keep going.
    if pool.has("close"):
        for w in (120, 250):
            yield ("close",), w, f"ts_rank(close,{w})"


def _dividend_yield(pool):
    if pool.has("dividend") and pool.has("close"):
        yield ("dividend", "close"), 250, "ts_sum(dividend,250)/close"


def _linked_fields(pool):
    return pool.select(lambda p: p.kind == "relationship" and p.scale == "ratio")


def _linked_momentum(pool):
    # News about customers, suppliers and competitors reaches a stock's price late.
    for profile in _linked_fields(pool):
        atom = pool.atom(profile)
        for w in (5, 10, 22, 66):
            yield (profile.field_id,), w, _signed(profile, f"ts_mean({atom},{w})")


def _linked_gap(pool):
    if not pool.has("returns"):
        return
    # Only the averaged returns of linked companies are on the same scale as returns;
    # a sign field minus a return means nothing.
    for profile in _linked_fields(pool):
        if not profile.field_id.startswith("rel_ret_"):
            continue
        atom = pool.atom(profile)
        for w in (5, 10, 22, 66):
            yield (profile.field_id, "returns"), w, _signed(profile, f"ts_mean({atom}-returns,{w})")


def _value(pool):
    for profile in pool.select(lambda p: p.kind == "fundamental" and _not_negative(p)
                               and p.scale in {"amount", "per_share"}):
        body = pool.deflated(profile)
        if body is not None:
            yield (profile.field_id,), None, body


def _forward_value(pool):
    for profile in pool.select(lambda p: p.kind == "analyst_estimate" and _not_negative(p)
                               and p.role in {"mean", "median"}
                               and p.scale in {"amount", "per_share"}):
        body = pool.deflated(profile)
        if body is not None:
            yield (profile.field_id,), None, body


def _quality(pool):
    if not pool.has("assets"):
        return
    assets = pool.atom_by_id("assets")
    for profile in pool.select(lambda p: p.kind == "fundamental" and _not_negative(p)
                               and p.field_id != "assets"):
        if profile.scale == "amount":
            yield (profile.field_id, "assets"), None, f"{pool.atom(profile)}/{assets}"
        elif profile.scale == "ratio":
            yield (profile.field_id,), None, pool.atom(profile)


_ACCRUAL_EARNINGS = ("income", "income_beforeextra", "fnd6_ibmii", "operating_income")


def _accruals(pool):
    if not (pool.has("assets") and pool.has("cashflow_op")):
        return
    assets, cfo = pool.atom_by_id("assets"), pool.atom_by_id("cashflow_op")
    for name in _ACCRUAL_EARNINGS:
        if pool.has(name):
            yield (name, "cashflow_op", "assets"), None, \
                f"-({pool.atom_by_id(name)}-{cfo})/{assets}"


def _fundamental_growth(pool):
    assets = pool.atom_by_id("assets") if pool.has("assets") else None
    for profile in pool.select(lambda p: p.kind == "fundamental"
                               and p.field_id != "assets" and p.scale in {"amount", "ratio"}):
        for w in (66, 125, 250):
            change = f"ts_delta({pool.atom(profile)},{w})"
            if profile.scale == "amount":
                if assets is None:
                    continue
                yield (profile.field_id, "assets"), w, _signed(profile, f"{change}/{assets}")
            else:
                yield (profile.field_id,), w, _signed(profile, change)


_SHARE_COUNT_FIELDS = ("sharesout", "fnd6_mfma1_csho")


def _issuance(pool):
    for name in _SHARE_COUNT_FIELDS:
        if pool.has(name):
            atom = pool.atom_by_id(name)
            for w in (125, 250):
                yield (name,), w, f"-({atom}/ts_delay({atom},{w})-1)"


def _revision(pool):
    for profile in pool.select(lambda p: p.kind == "analyst_estimate"
                               and p.role in {"mean", "median"}):
        atom = pool.atom(profile)
        for w in (22, 66, 120):
            # mean(|x|) over w+1 days covers both ends of the change, so it is
            # zero only when the change is too: no epsilon, which the platform's
            # unit check rejects next to fields that carry units.
            body = f"ts_delta({atom},{w})/ts_mean(abs({atom}),{w + 1})"
            yield (profile.field_id,), w, _signed(profile, body)


def _dispersion(pool):
    estimates = pool.select(lambda p: p.kind == "analyst_estimate")
    highs = [p for p in estimates if p.role == "high"]
    lows = [p for p in estimates if p.role == "low"]
    for high, low in _metric_pairs(highs, lows):
        if _same_stem(high.field_id, low.field_id, "high", "low"):
            h, l = pool.atom(high), pool.atom(low)
            yield (high.field_id, low.field_id), None, \
                f"-({h}-{l})/(abs({h})+abs({l}))"


def _earnings_surprise(pool):
    actuals = pool.select(lambda p: p.kind == "analyst_actual" and p.direction == "+")
    estimates = pool.select(lambda p: p.kind == "analyst_estimate" and p.direction == "+"
                            and p.role in {"mean", "median"})
    for actual, estimate in _metric_pairs(actuals, estimates):
        if actual.scale != estimate.scale:
            continue
        if actual.scale == "per_share" and _name_source(actual) != _name_source(estimate):
            # The platform gives est_* per-share estimates a per-share unit but
            # anl4_* actuals none, so their difference fails unit verification.
            continue
        a, prior = pool.atom(actual), f"ts_delay({pool.atom(estimate)},66)"
        yield (actual.field_id, estimate.field_id), None, \
            f"({a}-{prior})/(abs({a})+abs({prior}))"


def _sentiment_shift(pool):
    for profile in pool.select(lambda p: p.kind in {"sentiment", "news"}
                               and p.scale in {"score", "ratio", "count"}):
        atom = pool.atom(profile)
        for w in (5, 22, 66):
            if profile.scale == "count":
                body = f"{atom}/ts_mean({atom},{w})-1"
            else:
                body = f"{atom}-ts_mean({atom},{w})"
            yield (profile.field_id,), w, _signed(profile, body)


def _option_signal(pool):
    for profile in pool.select(lambda p: p.kind in {"option_flow", "option_vol", "option_price"}):
        atom = pool.atom(profile)
        if profile.kind == "option_price":
            # rank ignores the constant in x/close-1; dropping it keeps the ratio's
            # unit question out of an addition.
            yield (profile.field_id,), None, _signed(profile, f"{atom}/close")
            continue
        yield (profile.field_id,), None, _signed(profile, atom)
        for w in (22, 66):
            yield (profile.field_id,), w, _signed(profile, f"{atom}-ts_mean({atom},{w})")


def _name_source(profile: FieldProfile) -> str:
    return profile.field_id.split("_", 1)[0]


def _same_stem(left: str, right: str, left_role: str, right_role: str) -> bool:
    return left.replace(left_role, "\x00") == right.replace(right_role, "\x00")


TEMPLATES: tuple[IdeaTemplate, ...] = (
    IdeaTemplate("reversal", "price_action", "短期涨多的股票会回落", _reversal),
    IdeaTemplate("mean_deviation", "price_action", "价格偏离近期均值过多会回归", _mean_deviation),
    IdeaTemplate("momentum", "price_action", "中期趋势会延续（跳过最近一个月）", _momentum),
    IdeaTemplate("price_volume", "price_action", "量价同向的上涨不可持续", _price_volume),
    IdeaTemplate("low_vol", "price_action", "低波动、低贝塔的股票长期表现更好", _low_vol),
    IdeaTemplate("intraday_position", "price_action", "收盘离当日高点越远，之后越容易反弹", _intraday_position),
    IdeaTemplate("abnormal_volume", "price_action", "成交量异常放大的股票之后表现更好", _abnormal_volume),
    IdeaTemplate("overnight_gap", "price_action", "开盘相对前收跳得太多，之后会回落", _overnight_gap),
    IdeaTemplate("intraday_return", "price_action", "开盘到收盘的涨跌会部分回吐", _intraday_return),
    IdeaTemplate("close_vs_vwap", "price_action", "收盘偏离当日均价，之后会回归", _close_vs_vwap),
    IdeaTemplate("daily_range", "price_action", "日内振幅小的股票之后表现更好", _daily_range),
    IdeaTemplate("share_turnover", "price_action", "换手率低的股票有流动性溢价", _share_turnover),
    IdeaTemplate("year_high", "price_action", "价格接近一段时间的高点，之后更容易继续涨", _year_high),
    IdeaTemplate("value", "valuation", "同样的基本面，价格越低越好", _value),
    IdeaTemplate("forward_value", "valuation", "预期盈利相对价格越高越好", _forward_value),
    IdeaTemplate("dividend_yield", "valuation", "股息率高的股票长期表现更好", _dividend_yield),
    IdeaTemplate("quality", "quality", "资产回报率高的公司更好", _quality),
    IdeaTemplate("accruals", "quality", "利润主要来自账面而非现金的公司更差", _accruals),
    IdeaTemplate("fundamental_growth", "growth_financing", "基本面改善的公司更好", _fundamental_growth),
    IdeaTemplate("issuance", "growth_financing", "增发越多越差，回购越多越好", _issuance),
    IdeaTemplate("revision", "analyst", "分析师上调预期的股票会涨", _revision),
    IdeaTemplate("dispersion", "analyst", "分析师分歧大意味着风险高", _dispersion),
    IdeaTemplate("earnings_surprise", "analyst", "业绩超预期的股票之后继续上涨", _earnings_surprise),
    IdeaTemplate("sentiment_shift", "alternative", "情绪相对自身常态的变化有信息", _sentiment_shift),
    IdeaTemplate("option_signal", "alternative", "期权市场的看法领先股价", _option_signal),
    IdeaTemplate("linked_momentum", "linked", "客户、供应商、竞争对手的涨跌会延迟传到股价", _linked_momentum),
    IdeaTemplate("linked_gap", "linked", "关联公司涨了而自身没跟上，之后会补涨", _linked_gap),
)
COMBO_TEMPLATE_ID = "combo"
COMBO_HYPOTHESIS = "两个独立想法叠加更稳"


# --- Variant pool and batch planning -----------------------------------------------------

def build_idea_variant_pool(
    catalog: GenerationCatalog,
    profiles: FieldProfileSet,
    *,
    enabled_field_ids: frozenset[str],
    templates: tuple[IdeaTemplate, ...] = TEMPLATES,
) -> IdeaVariantPool:
    """Enumerate every legal variant over the enabled fields, in a stable order.

    A field is usable when it is enabled for this run, profiled for templates, and present
    in the catalog with coverage. A variant that fails a check is skipped and counted;
    it never blocks other variants.
    """
    usable = tuple(
        profile for profile in profiles.template_fields()
        if profile.field_id in enabled_field_ids and _in_catalog(catalog, profile.field_id)
    )
    pool = _FieldPool(catalog, usable)
    legacy_pool = _FieldPool(catalog, usable, backfill_fundamentals=True)
    exclusions: Counter[str] = Counter()
    seen: set[str] = set()
    variants: list[IdeaVariant] = []
    for template in templates:
        expanded = tuple(template.expand(pool))
        legacy = tuple(template.expand(legacy_pool))
        if len(legacy) != len(expanded):
            legacy = (None,) * len(expanded)
        for (field_ids, window, body), legacy_item in zip(expanded, legacy):
            outcome = _prepare_expression(f"rank({body})", catalog)
            if isinstance(outcome, str):
                exclusions[f"{template.template_id}:{outcome}"] += 1
                continue
            fingerprint = formula_fingerprint(outcome)
            if fingerprint in seen:
                exclusions[f"{template.template_id}:duplicate"] += 1
                continue
            seen.add(fingerprint)
            legacy_fingerprint = None
            if legacy_item is not None and legacy_item[2] != body:
                legacy_outcome = _prepare_expression(f"rank({legacy_item[2]})", catalog)
                if not isinstance(legacy_outcome, str):
                    legacy_fingerprint = formula_fingerprint(legacy_outcome)
            variants.append(IdeaVariant(
                template_id=template.template_id, family=template.family,
                hypothesis=template.hypothesis, field_ids=field_ids, window=window,
                expression=outcome,
                reviewed=all(_is_reviewed(profiles, field_id) for field_id in field_ids),
                legacy_fingerprint=legacy_fingerprint,
            ))
    return IdeaVariantPool(tuple(variants), tuple(sorted(exclusions.items())))


def plan_idea_batch(
    pool: IdeaVariantPool,
    *,
    target_count: int,
    seed: int,
    excluded_fingerprints: frozenset[str] = frozenset(),
) -> IdeaBatch:
    """Pick untested variants: families in turn, templates in turn within a family.

    Families or templates with nothing left drop out of the rotation, so their share flows
    to the rest. If everything runs out the batch is short and reports how many are missing.
    """
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 0:
        raise ValueError("idea_batch_target_count_invalid")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("idea_batch_seed_invalid")
    random = Random(seed)
    remaining: dict[str, dict[str, list[IdeaVariant]]] = defaultdict(dict)
    for template_id, variants in sorted(pool.by_template().items()):
        open_variants = [v for v in variants if v.fingerprint not in excluded_fingerprints
                         and v.legacy_fingerprint not in excluded_fingerprints]
        random.shuffle(open_variants)
        # pop() takes from the end: reviewed variants go last so they are used up first.
        open_variants.sort(key=lambda variant: variant.reviewed)
        if open_variants:
            remaining[variants[0].family][template_id] = open_variants
    family_order = [family for family in FAMILIES if family in remaining]
    random.shuffle(family_order)
    template_turn = {family: 0 for family in family_order}

    chosen: list[IdeaCandidate] = []
    while len(chosen) < target_count and family_order:
        for family in tuple(family_order):
            if len(chosen) == target_count:
                break
            templates = remaining[family]
            template_ids = sorted(templates)
            template_id = template_ids[template_turn[family] % len(template_ids)]
            template_turn[family] += 1
            variant = templates[template_id].pop()
            if not templates[template_id]:
                del templates[template_id]
            if not templates:
                family_order.remove(family)
            chosen.append(IdeaCandidate(
                candidate=exploration_candidate(
                    variant.expression,
                    strategy_label=STRATEGY_LABEL_PREFIX + variant.template_id),
                variant=variant,
            ))
    left = Counter()
    for templates in remaining.values():
        for template_id, variants in templates.items():
            left[template_id] = len(variants)
    return IdeaBatch(
        candidates=tuple(chosen),
        missing_count=target_count - len(chosen),
        remaining_by_template=tuple(sorted(left.items())),
    )


def build_combo_variant(
    first: IdeaVariant,
    second: IdeaVariant,
    catalog: GenerationCatalog,
    *,
    signs: tuple[int, int] = (1, 1),
    weights: tuple[float, float] = (1.0, 1.0),
) -> IdeaVariant | None:
    """Add two single-template signals from different families; None if not combinable.

    A sign of -1 flips a parent whose backtest showed the opposite direction. Weights let
    the calmer signal count for more, so the louder one does not drown it out.
    """
    if first.family == second.family or COMBO_TEMPLATE_ID in {first.template_id, second.template_id}:
        return None
    if any(sign not in (1, -1) for sign in signs):
        raise ValueError("idea_combo_sign_invalid")
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not 0 < w <= COMBO_MAX_WEIGHT
           for w in weights):
        raise ValueError("idea_combo_weight_invalid")
    left, right = sorted(zip((first, second), signs, weights),
                         key=lambda item: item[0].fingerprint)
    (left_variant, left_sign, left_weight), (right_variant, right_sign, right_weight) = left, right
    formula = (("-" if left_sign < 0 else "") + _weight_prefix(left_weight) + left_variant.formula
               + ("-" if right_sign < 0 else "+") + _weight_prefix(right_weight)
               + right_variant.formula)
    outcome = _prepare_expression(formula, catalog)
    if isinstance(outcome, str):
        return None
    return IdeaVariant(
        template_id=COMBO_TEMPLATE_ID, family="combo",
        hypothesis=f"{COMBO_HYPOTHESIS}：{left[0].template_id} + {right[0].template_id}",
        field_ids=tuple(dict.fromkeys(left[0].field_ids + right[0].field_ids)), window=None,
        expression=outcome,
    )


def _weight_prefix(weight: float) -> str:
    return "" if weight == 1 else f"{weight:g}*"


@dataclass(frozen=True, slots=True)
class ComboParent:
    """A backtested single-template formula and its daily PnL changes."""
    variant: IdeaVariant
    sharpe: float
    daily_pnl: Mapping[str, float]


COMBO_MIN_ABS_SHARPE = 0.7
COMBO_MAX_CORRELATION = 0.5
COMBO_MAX_PARENTS = 60
COMBO_MAX_USES_PER_PARENT = 3
# Parents this correlated are one idea written differently (the platform's own
# self-correlation line): they share one budget of pairs and pair at most once each.
COMBO_SAME_IDEA_CORRELATION = 0.7
COMBO_MAX_WEIGHT = 4.0
_COMBO_MIN_OVERLAP_DAYS = 250


def plan_combo_variants(
    parents: Iterable[ComboParent],
    catalog: GenerationCatalog,
) -> tuple[IdeaVariant, ...]:
    """Pair strong parents from different families whose PnL moves independently.

    Parents with a negative Sharpe enter flipped. Near-duplicate parents (PnL correlation
    above COMBO_SAME_IDEA_CORRELATION) count as one idea, and each idea joins at most a few
    pairs so one strong idea does not crowd out the rest. The calmer parent is weighted up
    to the louder one's daily PnL swing, which is what the expected Sharpe used for ordering
    assumes. All limits are fixed, so the work per cycle stays bounded however much history
    accumulates.
    """
    strong = sorted(
        (parent for parent in parents
         if abs(parent.sharpe) >= COMBO_MIN_ABS_SHARPE
         and parent.variant.template_id != COMBO_TEMPLATE_ID),
        key=lambda parent: (-abs(parent.sharpe), parent.variant.fingerprint),
    )[:COMBO_MAX_PARENTS]
    ideas = _idea_clusters(strong)
    scored: list[tuple[float, int, int]] = []
    for i, first in enumerate(strong):
        for j in range(i + 1, len(strong)):
            second = strong[j]
            if first.variant.family == second.variant.family or ideas[i] == ideas[j]:
                continue
            correlation = _signed_correlation(first, second)
            if correlation is None or correlation > COMBO_MAX_CORRELATION:
                continue
            expected = (abs(first.sharpe) + abs(second.sharpe)) / (2.0 + 2.0 * correlation) ** 0.5
            scored.append((expected, i, j))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    uses: Counter[int] = Counter()
    paired: set[frozenset[int]] = set()
    combos: list[IdeaVariant] = []
    for _, i, j in scored:
        idea_pair = frozenset((ideas[i], ideas[j]))
        if (uses[ideas[i]] >= COMBO_MAX_USES_PER_PARENT
                or uses[ideas[j]] >= COMBO_MAX_USES_PER_PARENT or idea_pair in paired):
            continue
        first, second = strong[i], strong[j]
        combo = build_combo_variant(
            first.variant, second.variant, catalog,
            signs=(1 if first.sharpe > 0 else -1, 1 if second.sharpe > 0 else -1),
            weights=_balancing_weights(first, second),
        )
        if combo is None:
            continue
        uses[ideas[i]] += 1
        uses[ideas[j]] += 1
        paired.add(idea_pair)
        combos.append(combo)
    return tuple(combos)


def _idea_clusters(parents: list[ComboParent]) -> list[int]:
    """Label each parent with the strongest earlier parent it nearly duplicates."""
    labels: list[int] = []
    leaders: list[int] = []
    for index, parent in enumerate(parents):
        for label, leader in enumerate(leaders):
            correlation = _signed_correlation(parents[leader], parent)
            if correlation is not None and correlation > COMBO_SAME_IDEA_CORRELATION:
                labels.append(label)
                break
        else:
            labels.append(len(leaders))
            leaders.append(index)
    return labels


def _balancing_weights(first: ComboParent, second: ComboParent) -> tuple[float, float]:
    """Weight the calmer parent up so both move the combined PnL about equally."""
    swings = [_daily_swing(first.daily_pnl), _daily_swing(second.daily_pnl)]
    if min(swings) <= 0:
        return (1.0, 1.0)
    ratio = min(round(max(swings) / min(swings), 1), COMBO_MAX_WEIGHT)
    if ratio < 1.2:
        return (1.0, 1.0)
    return (ratio, 1.0) if swings[0] < swings[1] else (1.0, ratio)


def _daily_swing(daily_pnl: Mapping[str, float]) -> float:
    values = list(daily_pnl.values())
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _signed_correlation(first: ComboParent, second: ComboParent) -> float | None:
    days = sorted(set(first.daily_pnl) & set(second.daily_pnl))
    if len(days) < _COMBO_MIN_OVERLAP_DAYS:
        return None
    x = [first.daily_pnl[day] for day in days]
    y = [second.daily_pnl[day] for day in days]
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((value - mean_x) ** 2 for value in x)
    syy = sum((value - mean_y) ** 2 for value in y)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    sign = (1 if first.sharpe > 0 else -1) * (1 if second.sharpe > 0 else -1)
    return sign * sxy / (sxx * syy) ** 0.5


def _is_reviewed(profiles: FieldProfileSet, field_id: str) -> bool:
    profile = profiles.get(field_id)
    return profile is not None and profile.tier == "reviewed"


def _in_catalog(catalog: GenerationCatalog, field_id: str) -> bool:
    field = catalog.field(field_id)
    return (field is not None and field.field_type in {"MATRIX", "VECTOR"}
            and field.coverage is not None and field.coverage > 0)


def _prepare_expression(formula: str, catalog: GenerationCatalog) -> Expression | str:
    """Run the same local checks as random exploration; return a reason code on failure."""
    try:
        expression = parse_formula(formula).expression
    except FormulaSyntaxError as exc:
        return f"syntax:{exc.code}"
    if FORBIDDEN_OPERATORS.intersection(analyze_formula(expression).operator_names):
        return "forbidden_operator"
    if not fits_formula_structure_limits(expression):
        return "structure_budget_exceeded"
    validation = validate_formula(expression, catalog)
    if not validation.is_valid:
        return "invalid:" + validation.issues[0].code
    if find_coarse_unit_issues(expression, catalog):
        return "unit_incompatible"
    logic = prepare_formula_logic(expression)
    if logic.issue is not None:
        return "logic:" + logic.issue.code
    return parse_formula(render_formula(logic.expression)).expression
