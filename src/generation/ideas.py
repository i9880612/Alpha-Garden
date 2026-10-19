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

FAMILIES = ("price_action", "valuation", "quality", "growth_financing", "analyst", "alternative")
FAMILY_LABELS = {
    "price_action": "价格行为", "valuation": "估值", "quality": "质量",
    "growth_financing": "成长与融资", "analyst": "分析师", "alternative": "另类数据",
}
# Operators templates must never produce (data-availability signals, not economic ones).
FORBIDDEN_OPERATORS = frozenset({"ts_count_nans", "days_from_last_change"})
STRATEGY_LABEL_PREFIX = "idea:"
_EPSILON = "0.0001"
_BACKFILL_LOOKBACK = {"quarterly": 66, "event": 22}


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
    def __init__(self, catalog: GenerationCatalog, profiles: tuple[FieldProfile, ...]) -> None:
        self.catalog = catalog
        self.profiles = profiles
        self.by_id = {profile.field_id: profile for profile in profiles}

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
        if lookback is not None:
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

def _price_fields(pool: _FieldPool) -> tuple[str, ...]:
    return tuple(name for name in ("close", "vwap") if pool.has(name))


def _reversal(pool):
    for name in _price_fields(pool):
        for w in (3, 5, 10):
            yield (name,), w, f"-({name}/ts_delay({name},{w})-1)"


def _mean_deviation(pool):
    for name in _price_fields(pool):
        for w in (10, 22, 66):
            yield (name,), w, f"-({name}-ts_mean({name},{w}))/ts_std_dev({name},{w})"


def _momentum(pool):
    if pool.has("close"):
        for w in (60, 120, 230):
            yield ("close",), w, f"ts_delay(close,20)/ts_delay(close,{w + 20})-1"


def _price_volume(pool):
    if not pool.has("volume"):
        return
    for name in _price_fields(pool):
        for w in (10, 22):
            yield (name, "volume"), w, f"-ts_corr({name},volume,{w})"


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
    position = f"(high-close)/(high-low+{_EPSILON})"
    yield ("high", "low", "close"), None, position
    for w in (5, 10):
        yield ("high", "low", "close"), w, f"ts_mean({position},{w})"


def _abnormal_volume(pool):
    if not pool.has("volume"):
        return
    for w in (10, 20, 60):
        yield ("volume",), w, f"volume/(ts_mean(volume,{w})+{_EPSILON})"


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
            body = f"ts_delta({atom},{w})/(abs(ts_mean({atom},{w}))+{_EPSILON})"
            yield (profile.field_id,), w, _signed(profile, body)


def _dispersion(pool):
    estimates = pool.select(lambda p: p.kind == "analyst_estimate")
    highs = [p for p in estimates if p.role == "high"]
    lows = [p for p in estimates if p.role == "low"]
    for high, low in _metric_pairs(highs, lows):
        if _same_stem(high.field_id, low.field_id, "high", "low"):
            h, l = pool.atom(high), pool.atom(low)
            yield (high.field_id, low.field_id), None, \
                f"-({h}-{l})/(abs({h})+abs({l})+{_EPSILON})"


def _earnings_surprise(pool):
    actuals = pool.select(lambda p: p.kind == "analyst_actual" and p.direction == "+")
    estimates = pool.select(lambda p: p.kind == "analyst_estimate" and p.direction == "+"
                            and p.role in {"mean", "median"})
    for actual, estimate in _metric_pairs(actuals, estimates):
        if actual.scale != estimate.scale:
            continue
        prior = f"ts_delay({pool.atom(estimate)},66)"
        yield (actual.field_id, estimate.field_id), None, \
            f"({pool.atom(actual)}-{prior})/(abs({prior})+{_EPSILON})"


def _sentiment_shift(pool):
    for profile in pool.select(lambda p: p.kind in {"sentiment", "news"}
                               and p.scale in {"score", "ratio", "count"}):
        atom = pool.atom(profile)
        for w in (5, 22, 66):
            if profile.scale == "count":
                body = f"{atom}/(ts_mean({atom},{w})+{_EPSILON})-1"
            else:
                body = f"{atom}-ts_mean({atom},{w})"
            yield (profile.field_id,), w, _signed(profile, body)


def _option_signal(pool):
    for profile in pool.select(lambda p: p.kind in {"option_flow", "option_vol", "option_price"}):
        atom = pool.atom(profile)
        if profile.kind == "option_price":
            yield (profile.field_id,), None, _signed(profile, f"{atom}/close-1")
            continue
        yield (profile.field_id,), None, _signed(profile, atom)
        for w in (22, 66):
            yield (profile.field_id,), w, _signed(profile, f"{atom}-ts_mean({atom},{w})")


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
    IdeaTemplate("value", "valuation", "同样的基本面，价格越低越好", _value),
    IdeaTemplate("forward_value", "valuation", "预期盈利相对价格越高越好", _forward_value),
    IdeaTemplate("quality", "quality", "资产回报率高的公司更好", _quality),
    IdeaTemplate("accruals", "quality", "利润主要来自账面而非现金的公司更差", _accruals),
    IdeaTemplate("fundamental_growth", "growth_financing", "基本面改善的公司更好", _fundamental_growth),
    IdeaTemplate("issuance", "growth_financing", "增发越多越差，回购越多越好", _issuance),
    IdeaTemplate("revision", "analyst", "分析师上调预期的股票会涨", _revision),
    IdeaTemplate("dispersion", "analyst", "分析师分歧大意味着风险高", _dispersion),
    IdeaTemplate("earnings_surprise", "analyst", "业绩超预期的股票之后继续上涨", _earnings_surprise),
    IdeaTemplate("sentiment_shift", "alternative", "情绪相对自身常态的变化有信息", _sentiment_shift),
    IdeaTemplate("option_signal", "alternative", "期权市场的看法领先股价", _option_signal),
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
    exclusions: Counter[str] = Counter()
    seen: set[str] = set()
    variants: list[IdeaVariant] = []
    for template in templates:
        for field_ids, window, body in template.expand(pool):
            outcome = _prepare_expression(f"rank({body})", catalog)
            if isinstance(outcome, str):
                exclusions[f"{template.template_id}:{outcome}"] += 1
                continue
            fingerprint = formula_fingerprint(outcome)
            if fingerprint in seen:
                exclusions[f"{template.template_id}:duplicate"] += 1
                continue
            seen.add(fingerprint)
            variants.append(IdeaVariant(
                template_id=template.template_id, family=template.family,
                hypothesis=template.hypothesis, field_ids=field_ids, window=window,
                expression=outcome,
                reviewed=all(_is_reviewed(profiles, field_id) for field_id in field_ids),
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
        open_variants = [v for v in variants if v.fingerprint not in excluded_fingerprints]
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
) -> IdeaVariant | None:
    """Add two single-template signals from different families; None if not combinable.

    A sign of -1 flips a parent whose backtest showed the opposite direction.
    """
    if first.family == second.family or COMBO_TEMPLATE_ID in {first.template_id, second.template_id}:
        return None
    if any(sign not in (1, -1) for sign in signs):
        raise ValueError("idea_combo_sign_invalid")
    left, right = sorted(zip((first, second), signs), key=lambda item: item[0].fingerprint)
    (left_variant, left_sign), (right_variant, right_sign) = left, right
    formula = (("-" if left_sign < 0 else "") + left_variant.formula
               + ("-" if right_sign < 0 else "+") + right_variant.formula)
    outcome = _prepare_expression(formula, catalog)
    if isinstance(outcome, str):
        return None
    return IdeaVariant(
        template_id=COMBO_TEMPLATE_ID, family="combo",
        hypothesis=f"{COMBO_HYPOTHESIS}：{left[0].template_id} + {right[0].template_id}",
        field_ids=tuple(dict.fromkeys(left[0].field_ids + right[0].field_ids)), window=None,
        expression=outcome,
    )


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
_COMBO_MIN_OVERLAP_DAYS = 250


def plan_combo_variants(
    parents: Iterable[ComboParent],
    catalog: GenerationCatalog,
) -> tuple[IdeaVariant, ...]:
    """Pair strong parents from different families whose PnL moves independently.

    Parents with a negative Sharpe enter flipped. Pairs are ordered by the Sharpe two
    equally weighted signals would reach given their correlation; each parent joins at
    most a few pairs so one strong idea does not crowd out the rest. All limits are fixed,
    so the work per cycle stays bounded however much history accumulates.
    """
    strong = sorted(
        (parent for parent in parents
         if abs(parent.sharpe) >= COMBO_MIN_ABS_SHARPE
         and parent.variant.template_id != COMBO_TEMPLATE_ID),
        key=lambda parent: (-abs(parent.sharpe), parent.variant.fingerprint),
    )[:COMBO_MAX_PARENTS]
    scored: list[tuple[float, int, int]] = []
    for i, first in enumerate(strong):
        for j in range(i + 1, len(strong)):
            second = strong[j]
            if first.variant.family == second.variant.family:
                continue
            correlation = _signed_correlation(first, second)
            if correlation is None or correlation > COMBO_MAX_CORRELATION:
                continue
            expected = (abs(first.sharpe) + abs(second.sharpe)) / (2.0 + 2.0 * correlation) ** 0.5
            scored.append((expected, i, j))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    uses: Counter[int] = Counter()
    combos: list[IdeaVariant] = []
    for _, i, j in scored:
        if uses[i] >= COMBO_MAX_USES_PER_PARENT or uses[j] >= COMBO_MAX_USES_PER_PARENT:
            continue
        first, second = strong[i], strong[j]
        combo = build_combo_variant(
            first.variant, second.variant, catalog,
            signs=(1 if first.sharpe > 0 else -1, 1 if second.sharpe > 0 else -1),
        )
        if combo is None:
            continue
        uses[i] += 1
        uses[j] += 1
        combos.append(combo)
    return tuple(combos)


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
