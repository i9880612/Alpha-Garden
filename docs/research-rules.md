# Research rules

**English** · [简体中文](research-rules.zh-CN.md) · [Back to README](../README.md)

This page describes the research rules Alpha Garden **currently implements**: where formulas come from, how they improve, how self-correlation (SC) is controlled, and the order of submission. Commands and database operations are in the [Operations guide](operations.md); module boundaries are in [Architecture](architecture.md).

The target is the platform's **Spectacular** grade. Optimization tries to improve a formula but cannot guarantee a higher grade; platform backtests and checks are always the final word.

## 1. Two run modes

- **Normal research**: explore new formulas, mutate existing ones and handle SC, split by the percentages in the run configuration.
- **Focused optimization** (`run -opt`): only continues active parents that fully passed checks, are not yet top-graded and still have attempts left.

Both modes share the same attempt budgets; switching modes resets nothing.

## 2. Exploration: idea templates

- By default 90% of exploration slots (`ideaPercent`) come from **idea templates**; random exploration fills the rest.
- There are 27 templates in 7 families: price action, valuation, quality, growth & financing, analyst, alternative data and **linked companies**.
  - Price action covers reversal, momentum, deviation from moving averages, price–volume divergence, overnight gaps, intraday returns, close vs. VWAP, daily range, share turnover, recent highs and more.
  - Linked companies covers the momentum and gaps of related companies.
- Templates are expanded into concrete formulas using the field profile `data/field-profiles.json` (not stored in the database). If the profile is missing or broken, the cycle falls back to random exploration without stopping.
- Fundamental fields are referenced directly without `ts_backfill` (the platform already carries them forward; experiments showed nearly identical backtests). Analyst and event fields still use `ts_backfill`.
- Option fields use `ts_backfill(field, 5)` to fill occasional missing days: on such a day only a few stocks have a value and the book piles into them (CONCENTRATED_WEIGHT failure).
- Template variants already tested in their old form are not retested in the new form.
- **Combination templates** take at most 30% of template slots: they add strong parents from different families whose daily returns correlate ≤0.5. Parents correlated >0.7 count as the same idea and share one combination budget. Formulas that failed CONCENTRATED_WEIGHT are never combination parents.

## 3. Seeds and lineages

- An independent formula that passes checks becomes a **seed**; its descendants are researched along the same lineage.
- Before admission, its daily returns are compared with existing seeds, current research parents and submitted alphas. Missing evidence means no independent seed status.
- Qualified formulas outside any seed lineage (for example combinations) get their own optimization lineage, strongest Sharpe first. They are skipped if they correlate ≥0.7 with a formula being optimized in another lineage, or are blocked by a submitted alpha.
- **Budgets**:
  - up to **20** mutation attempts per formula;
  - up to **80** quality-research backtests per root lineage (same account and settings);
  - at most 2 qualified-evolution backtests per lineage per cycle.
- A better qualified descendant can **take over** from its parent with its own remaining attempts; the replaced formula and its history are kept.

## 4. Mutations and weak-spot diagnosis

- Both normal mutations and focused optimization first diagnose the parent's weak spot and try the matching mutations first:
  - **High turnover**: turnover >30%, or Fitness is the weak spot and turnover >12.5%. Above 30% turnover a decay ladder `ts_decay_linear(parent, n)` with n = 3, 6, 10, 15 is also available (around 20% turnover it almost always lowered Fitness).
  - **Low turnover**, **low returns** (Fitness is the weak spot and turnover ≤12.5%), **low Sharpe**.
- Each cycle recomputes "weak spot → mutation family" from the last 30 days: a mutation counts as effective only if the child's Fitness beats the parent's, the weak metric moves the right way and the child does not fail SC.
- Action preferences learned for a parent come first; about one in five slots ignores the diagnosis to keep exploring.
- Field swaps stay within the same field class of the profile; operator swaps stay within the same role.
- No-op mutations that don't change the signal are skipped at generation time: a monotone transform inside `group_rank`, a repeated `rank`, or a child with the same signal as its parent or a sibling (at the root, `normalize(x)` and `zscore(x)` give the same positions, so only one is tested).
- Parents with turnover below 5% get no smoothing mutations: the signal barely moves, so smoothing returns the parent.
- `winsorize` is not wrapped around a weighted sum of up to 3 rank terms: such a sum never reaches the 4-standard-deviation clip, so nothing changes.
- A parent that is already a weighted sum of up to 3 bounded terms (rank, quantile, ts_rank, ts_corr and similar) gets no distribution stabilization: it has no tails, so another zscore or rank barely moves it.
- With neutralization in the settings, a child that only adds or removes a root `zscore`/`normalize`/`scale` is a no-op: the platform already removes each group's mean and rescales the book, so positions stay the same.
- Group fields with a single group are not used for group mutations: `market` is one group everywhere, and in a single-country region (such as USA) so are `country` and `currency`.
- Fields the profile marks as non-signals (codes, market-wide constants, adjustment factors like `adjfactor`) are not used for random exploration or mutations.

## 5. Combo enhancement

Qualified parents (in focused optimization and the qualified stage of normal research) first try adding an already-backtested companion signal:

```text
zscore(parent) + w * zscore(companion)
```

- A side that is a single `rank(...)` drops the `zscore` and its weight is multiplied by √12 (both forms backtest the same, with one less nesting level).
- Companion requirements: |Sharpe| ≥0.7, daily-return correlation with the parent ≤0.5, no CONCENTRATED_WEIGHT failure.
- The weight is the Sharpe-optimal mix of the two daily return curves. Children whose estimated Fitness gain is under 5% are skipped.
- Near-identical companions count as one idea, and each companion serves at most 3 parents, so lineages don't converge.
- **Cross-lineage veto**: before backtesting, the child's returns are predicted as "parent curve + weight × companion curve" and compared with other lineages' qualified formulas, submitted alphas and children already chosen this cycle. At ≥0.7 the child is skipped and the next candidate takes the slot.
- The child stays in the parent's lineage and shares its budget; it counts as effective only if its Fitness beats the parent's.
- Estimates use in-sample curves and lean optimistic; real backtests decide.

## 6. Self-correlation (SC) control

**Local SC rule** (same as the platform):
- Uses daily PnL changes from the **last four calendar years**, with at least **252 shared intervals**.
- Correlation **≥0.7** with a submitted alpha fails unless Sharpe is at least **10%** higher.
- Missing curves, too little overlap or missing Sharpe leave the result **pending**, never passed.

**PnL collection**:
- Each saved result gets its daily PnL curve collected for the comparisons above.
- At least 2 seconds pass between collections; on HTTP 429 the whole run backs off, waiting at least 60 seconds when there is no `Retry-After`.

**Near-duplicate convergence**:
- Before generating mutations in a normal batch, parents whose daily returns correlate **≥0.9999** keep a single representative (grade → Sharpe → Fitness, earliest on ties).
- Focused optimization converges only within a lineage.
- High correlation at 0.7 or 0.9 alone is not a reason to drop a parent; representatives keep their own budgets, which are never merged.

**SC risk ranking**: normal batches rank parents by the platform SC results of the parent and its direct children (at least 5 clear results needed). This lowers priority; it never removes a parent.

**SC-directed repair** (parents that fail only SC with complete blocking evidence) compares three directions within the existing budget, two attempts each:
1. switch the signal source (only catalog-confirmed fundamental, analyst or news-sentiment fields, with type, coverage and unit checks);
2. turn a level into a change;
3. update positions on changes in the raw data.

A lower local correlation is not a successful repair; only platform checks decide.

## 7. Queue checks and submission order

**Queue checks run locally**:
- Other checks are read straight from the backtest result; SC is computed from local curves with the rule above, without calling the platform check API.
- Missing curves leave the check pending and it is retried automatically without blocking the batch.
- SC-directed research runs still use platform checks.
- Formulas that pass and are graded Spectacular enter the submission queue. Other qualified formulas keep being researched and move to the qualified archive once their budget is used up or a better descendant takes over.

**Submission order (quality first)**:
- Sorted by grade → Fitness → Sharpe, best first; how many other candidates a formula blocks only matters on ties.
- A weaker formula (including one still being optimized) never blocks a stronger one from going first.
- **Only the peak of each lineage is submitted**: of highly correlated 1.50 → 1.60 → 1.70, only 1.70 is submitted, while weaker levels wait in the archive, so weak submissions don't drag down the score or use up SC room.
- The order is recomputed after every submission.

**"All grades, best first"** (the default on the submissions page):
- Has a minimum grade, Good by default.
- Candidates come from the Spectacular queue, the qualified archive, and formulas still in focused optimization graded **Excellent or above**.

**Every formal submission is re-checked on the platform first**; local checks only filter. Backtests never submit unless that is explicitly enabled.

## 8. Quality feedback and stagnation

- Qualified parents with an insufficient grade get a separate quality-improvement target. A child must still pass all checks, and either raise the grade or keep it while improving Sharpe or Fitness without the other getting worse.
- Each action needs at least 10 clear results from 5 independent root lineages (same account and settings) before it is learned from; until then it keeps exploring.
- Stagnation: in non-optimization runs with a budget of at least 200, every 100 clear results form a window; two windows with no new seed, defect fix or grade gain count as stagnation. This is **display only** for now; it never stops a run or adds budget.

## Still being validated

These rules are implemented, but their effect still needs several rounds of real backtests: the weak-spot thresholds, the combo-enhancement gain estimate, and the new price-action and linked-company templates.
