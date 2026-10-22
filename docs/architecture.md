# Architecture

**English** · [简体中文](architecture.zh-CN.md) · [Back to README](../README.md)

This page records the current module responsibilities, main flow and data ownership. Research rules are in [Research rules](research-rules.md); operations are in the [Operations guide](operations.md).

## Dependency direction

```text
alpha_garden            CLI and web entry points
    ↓
execution               use-case orchestration, transactions, scheduling, recovery
    ↓
generation / selection / evaluation / learning / submission
    ↓
persistence / worldquant
```

Code is organized flat by business capability, without type layers, a global service locator or a second orchestration framework. `tests/` mirrors `src/`.

## Module responsibilities

| Module | Owns | Does not own |
| --- | --- | --- |
| `alpha_garden` | Command input, web API, presentation | Quality judgements, transactions, platform protocol |
| `execution` | Initialization, run plans, cycle scheduling, transactions, ordering of external calls, local queue checks, recovery and stopping | Copies of generation, selection, learning or submission algorithms |
| `generation` | Formula trees, parsing, normalization, type and unit checks, idea templates, mutations, combo enhancement | Claims that a mutation will improve returns |
| `selection` | Risk filtering, exploration/improvement slot allocation, diversity | Generating formulas, predicting platform scores |
| `evaluation` | Interpreting complete platform results and individual checks, correlation | Changing platform verdicts or picking the next formulas |
| `learning` | Recomputing seed frontiers, targets, parent budgets, weak-spot diagnosis and action preferences from real results | Storing drifting model scores or active pointers |
| `submission` | Local eligibility, quality ordering, check interpretation, verifying success details | Network I/O and state transitions |
| `persistence` | Fact storage, integrity checks, transactions, explicit upgrades | Quality algorithms and run decisions |
| `worldquant` | Authentication, catalog, requests, rate limits, response parsing | Judging formulas or whether research continues |

## Main flow

```text
Freeze account, settings, run limits and submission authorization
→ finish queue checks left over from earlier runs
→ learning recomputes seeds, active parents, targets and action evidence
→ generation produces valid formulas (templates, random, mutations, combo enhancement); selection allocates and filters
→ atomically save final tasks, full settings, lineage and cycle membership
→ create, poll and collect platform results and PnL within the in-flight limit
→ once every task in the cycle has finished, run local queue checks; Spectacular enters the queue, other qualified formulas keep researching or are archived
→ handle or skip formal submission per the frozen authorization (platform re-check first)
→ settle the cycle, then continue or stop
```

Key boundaries:
- A task never changes cycle; resuming reads frozen tasks without regenerating or reselecting.
- Before any platform request, the unique right to send it is claimed atomically; uncertain outcomes are never resent.
- Details, PnL and yearly data are collected in separate steps, each saved atomically; partial facts never pose as complete results.
- Automatic and manual formal submissions share one state machine; only confirmed platform details write a submitted object.

## Data ownership

SQLite is the only mutable source of business facts; docs, logs, caches and memory never overwrite it.

| Fact | Tables |
| --- | --- |
| Platform fields, operators and sync identity | `platform_fields`, `platform_operators`, `platform_catalog_syncs` |
| Standard windows and project generation semantics | `generation_windows`, `generation_operator_roles`, `generation_operator_outputs` |
| Formula, actual settings, remote identity and task state | `backtest_tasks` |
| Platform results and individual checks | `backtest_results`, `backtest_checks` |
| Yearly statistics | `backtest_yearly_stats_captures`, `backtest_yearly_stats` |
| Daily PnL curves | `platform_pnl_series` |
| Parent–child links, seed roots | `backtest_mutations`, `signal_seeds` |
| Qualified archive candidates | `qualified_alpha_archive` |
| Run limits, authorization and state | `automated_runs`, `automated_run_allocations` |
| Task-to-run and cycle membership, cycle settlement | `automated_run_backtests`, `automated_cycle_settlements` |
| Queue check records (local or platform) | `submission_checks` |
| Submission queue | `submission_queue` |
| Formal submission and reconciliation state | `formal_submission_attempts` |
| Confirmed submitted platform objects | `platform_submitted_alphas` |

Generation pools, unselected formulas, active branches, targets, remaining parent budgets and action preferences are all recomputed from these facts and never stored as copies.

`submitted-formulas.md` is a display file of `platform_submitted_alphas`, rebuilt with `export-submitted`; it takes no part in decisions and stays out of Git.

## Identity and evidence

- A task's identity is the account scope, the normalized formula and the full backtest settings; the same identity only resumes the original task and is never backtested twice.
- Only parent and child results with the same account and settings are compared directly. Configuration changes never rewrite existing runs or tasks.
- The platform catalog is kept separate from project semantics: new operators without local semantics are stored as platform facts but not used for generation.
- Learning uses only trusted real results and never predicts Sharpe, Fitness or success probability for unseen formulas.
