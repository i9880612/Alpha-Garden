from __future__ import annotations

from generation.candidate import FormulaCandidate, exploration_candidate
from generation.catalog import GenerationCatalog
from generation.formula import analyze_formula
from generation.logic import prepare_formula_logic
from generation.parser import FormulaSyntaxError, parse_formula
from generation.structure_limits import fits_formula_structure_limits
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import validate_formula

# Curated cold-start seed formulas ("mothers").
#
# Each encodes a distinct, economically motivated signal structure adapted from
# public Alpha101-style patterns. They are starting points only: the normal
# backtest -> checks -> seed admission pipeline decides which ones become
# seeds. Unknown or invalid formulas are skipped before task creation, and the
# seed admission rules (including cross-seed decorrelation) apply unchanged.
#
# The first ordinary batch of an account without local backtest or submitted
# history may use these mothers. Later batches use normal exploration/mutation;
# recovery reuses the frozen tasks instead of generating another mother batch.
SEED_FORMULAS: tuple[str, ...] = (
    # Momentum: 20-day price change persists.
    "rank(ts_delta(close, 20))",
    # Short-term reversal: 5-day move mean-reverts.
    "rank(-ts_delta(close, 5))",
    # Dual horizon: short-rank minus long-rank spread.
    "rank(ts_rank(close, 10) - ts_rank(close, 40))",
    # Volatility-adjusted momentum.
    "rank(ts_delta(close, 10) / (1 + ts_std_dev(returns, 20)))",
    # Low-volatility anomaly.
    "rank(-ts_std_dev(returns, 20))",
    # Volume momentum.
    "rank(ts_delta(volume, 10))",
    # Price-volume correlation.
    "rank(ts_corr(close, volume, 20))",
    # Breakout proximity: intraday position within the daily range.
    "rank((high - close) / (high - low))",
    # Z-score reversal.
    "rank(-ts_zscore(close, 20))",
    # Bollinger-style deviation from the mean.
    "rank((close - ts_mean(close, 20)) / ts_std_dev(close, 20))",
    # Abnormal volume.
    "rank(volume / ts_mean(volume, 20))",
    # Intraday range as volatility proxy.
    "rank((high - low) / close)",
    # Overnight gap return.
    "rank((open - ts_delay(close, 1)) / ts_delay(close, 1))",
    # Trend consistency: share of up days.
    "rank(ts_mean(sign(ts_delta(close, 1)), 10))",
    # Lagged momentum.
    "rank(ts_delta(ts_delay(close, 5), 20))",
)


def seed_mother_candidates(catalog: GenerationCatalog) -> tuple[FormulaCandidate, ...]:
    """Return locally legal mothers; no task writes or platform requests."""
    candidates = []
    seen = set()
    for formula in SEED_FORMULAS:
        try:
            expression = parse_formula(formula).expression
        except FormulaSyntaxError:
            continue
        logic = prepare_formula_logic(expression)
        if logic.issue is not None:
            continue
        expression = logic.expression
        if (not fits_formula_structure_limits(expression)
                or not validate_formula(expression, catalog).is_valid
                or find_coarse_unit_issues(expression, catalog)):
            continue
        fields = tuple(catalog.field(name) for name in analyze_formula(expression).referenced_names)
        if not fields or any(field is None or field.coverage is None or field.coverage <= 0
                             for field in fields):
            continue
        candidate = exploration_candidate(expression, strategy_label="cold_start_mother")
        if candidate.fingerprint not in seen:
            candidates.append(candidate)
            seen.add(candidate.fingerprint)
    return tuple(candidates)
