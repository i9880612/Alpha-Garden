from __future__ import annotations

# Curated cold-start seed formulas ("mothers").
#
# Each encodes a distinct, economically motivated signal structure adapted from
# public Alpha101-style patterns. They are starting points only: the normal
# backtest -> checks -> seed admission pipeline decides which ones become
# seeds. Unknown or invalid formulas are skipped at task creation, and the
# seed admission rules (including cross-seed decorrelation) apply unchanged.
#
# The list is static by design. After cold start the research loop evolves the
# population on its own via exploration and mutation; mothers are not reused.
# To refresh, edit this list and re-run the cold-start helper.
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
    # Breakout proximity: closeness to 20-day high.
    "rank(close / ts_max(high, 20))",
    # Z-score reversal.
    "rank(-ts_zscore(close, 20))",
    # Bollinger-style deviation from the mean.
    "rank((close - ts_mean(close, 20)) / ts_std_dev(close, 20))",
    # Abnormal volume.
    "rank(volume / ts_mean(volume, 20))",
    # Intraday range as volatility proxy.
    "rank((high - low) / close)",
    # Overnight gap return.
    "rank((open - delay(close, 1)) / delay(close, 1))",
    # Trend consistency: share of up days.
    "rank(ts_mean(sign(ts_delta(close, 1)), 10))",
    # Lagged momentum.
    "rank(ts_delta(delay(close, 5), 20))",
)
