# Operations guide

**English** · [简体中文](operations.zh-CN.md) · [Back to README](../README.md)

This page collects CLI operations, catalog sync, runs and recovery, SC-directed research, database upgrades, rate limits and storage. The research rules themselves are in [Research rules](research-rules.md).

All commands default to `data/alpha_garden.sqlite3`, `config/` and `.env`; override them with `--database`, `--settings` and `--env`.

## Command overview

| Action | Command | Calls the platform? |
| --- | --- | --- |
| Initialize database and catalog | `alpha-garden init` | Only when the catalog needs syncing |
| Refresh operators only | `alpha-garden sync-operators` | Yes |
| Sync submitted alphas and PnL | `alpha-garden sync-submitted` | Yes |
| Run N cycles (`-1` runs continuously) | `alpha-garden run [N]` | Yes |
| Focused optimization | `alpha-garden run -opt [N]` | Yes |
| Resume a run | `alpha-garden resume <run_id>` | Yes |
| Preview SC-directed research | `alpha-garden sc-plan <parent_task_id>` | No |
| Run SC-directed research | `alpha-garden run 3 --sc-parent <parent_task_id> --sc-plan-key <key>` | Yes |
| Submit | `alpha-garden submit [N]` / `alpha-garden submit <grade> [N]` | Yes |
| Rebuild the submitted-formula list | `alpha-garden export-submitted` | No |
| Research feedback | `alpha-garden research-status <run_id>` | No |
| Storage usage / retention preview | `alpha-garden storage inspect` / `storage prune --dry-run` | No |
| Web console | `alpha-garden web [--read-only]` | Depends on the action |

## Initialization and catalog sync

- `init` creates or verifies the database. If no complete, matching catalog exists, it authenticates and syncs fields and operators. It never starts backtests or submits.
- Fields are saved to a staging area page by page. If interrupted, rerun `init` with the same arguments to continue. Checkpoints are separated by account, platform URL, instrument type, region, universe and delay.
- The live catalog is replaced atomically only after everything validates; a partial download never overwrites the old catalog. If totals change or paging looks wrong, staging is cleared and the next run starts from page one.
- The first time an older database uses paged saving, run `alpha-garden init --upgrade-catalog-storage`.
- When operator permissions or the platform operator list change, `sync-operators` refreshes only operators. It refuses while a run is active or a submission is unresolved.

## Submitted-alpha baseline

- `sync-submitted` pages through every submitted alpha on the account (including hidden and decommissioned ones) and saves details and PnL. It only records submitted facts; it does not import seeds or submit anything.
- **An account without a complete sync cannot start or resume research**; a new account confirmed to have zero submissions can cold-start.
- If some curves are temporarily unavailable, completed work is kept and rerunning continues.
- Inconsistent lists, missing pages or identity conflicts fail explicitly.
- Keep `WQB_ACCOUNT_SCOPE` mapped one-to-one to the real account; never reuse a scope when switching accounts.

## Runs, pausing and resuming

- `run` always creates a new batch and wraps up stopped older ones; use `resume` to continue an existing batch with its original mode and frozen settings.
- **Formal submission is off by default** and only enabled with `--auto-submit`; runs started from the web never submit automatically.
- Pause in the web console stops the engine at a safe boundary. Closing the browser does not stop a run.
- Resuming reads the frozen tasks without regenerating them; requests with an uncertain outcome are never resent.
- New settings apply only to later runs.

## Submitting

- `submit 2`: submit 2 successfully from the Spectacular queue.
- `submit good 2`: pick Good-graded formulas from the qualified archive that currently pass the correlation rule, and submit 2 successfully. `inferior`, `average`, `excellent` and `spectacular` also work.
- The count is **successful submissions**: candidates that fail the re-check are skipped, the command ends early when candidates run out, and uncertain outcomes are confirmed before continuing.
- The web "All grades, best first" mode submits one at a time in quality order; see [Research rules §7](research-rules.md#7-queue-checks-and-submission-order).
- Every formal submission first reads the platform details and runs the latest checks.

## SC-directed research

For parents that fail only SC, this compares three ways to lower correlation (see [Research rules §6](research-rules.md#6-self-correlation-sc-control)):

1. `sc-plan <parent_task_id>` previews the six candidates read-only and prints a plan key.
2. `run 3 --sc-parent <parent_task_id> --sc-plan-key <key>` runs them: two per batch, at most three batches, sharing the parent's 20-attempt budget. It refuses to start if local facts changed after the preview.

Requirements:
- The parent fails only SC, the blocking evidence is complete, at least six attempts remain, and the account baseline sync is complete.
- Formal submission stays off. Operator availability is checked on the platform before the first request; missing operators stop new requests without using attempts.

Check retries:
- Each task gets up to 3 short retries, then up to 3 deferred retries at least 600 seconds apart.
- When exhausted, the result stays pending; it is never turned into a pass or a failure.

## Rate limits and failures

- Authentication failures, or HTTP 429 after bounded retries, stop further requests immediately; wait for the printed `Retry-After` before trying again.
- A 429 on formal checks waits at least 60 seconds; the log shows that the queue check was rate-limited, and backtest results are unaffected.
- After each PnL collection the next request waits at least 2 seconds; on 429 the whole run backs off and the saved cooldown survives restarts. 2 seconds is a conservative local value, not a guarantee against 429s.
- The platform sometimes files a formula that uses an alias field under an existing alpha (whose code is the other spelling); such a task is recorded as `platform_duplicate_alpha` at once instead of being re-read.
- `maxConsecutiveFailures`, `maxRequestFailures` and `maxPendingSeconds` in the run configuration set the failure limits.

## Database upgrades

Run commands **never migrate** the database automatically. With explicit approval, run these `persistence.schema` upgrades on an older database in order, inside one `BEGIN IMMEDIATE` transaction; any error rolls everything back:

1. `add_optimization_submission_source` (if the older submission-source constraint is still in place)
2. `add_mutation_reference_storage`
3. `add_sc_research_plan_storage`
4. `add_submitted_sync_storage` (or leave the transaction and let `sync-submitted` create it)
5. `add_quality_research_storage`

These only add tables or columns; historical tasks, budgets and results are untouched. A new empty database just needs `init`.

## Research feedback

`research-status <run_id>` reports, read-only, how many tasks were frozen, sent, qualified, rejected, pending or failed by request error, split into exploration, normal mutation, SC repair and quality research, plus quality-action results from the last 30 days. It counts frozen tasks only, not formulas generated but never sent.

## Storage and retention

```bash
alpha-garden storage inspect --daily-curves 1000
alpha-garden storage prune --dry-run --retention-days 90
```

- Both are read-only: no account file, no platform calls, no migrations.
- `inspect` breaks down the database file, WAL, free pages and PnL size, and estimates growth from the number of new curves per day.
- `prune` is **preview only**: retention defaults to 90 days (minimum 30) and it lists only old curves with an explicit non-SC check failure and no protected dependency. Submitted references, lineages, seeds, unfinished runs and submission records are all protected.
- Actual deletion, prune markers and a `VACUUM` maintenance command are not implemented yet.

## Logs

The web service keeps up to 500 log lines per operation; they are cleared on restart and never affect saved research results. PnL collection shows up as separate `[PnL采集]` lines.
