# Alpha Garden web console

A local React console connected to the existing Python research engine. Dashboard charts, run history, formula lists, grades, checks, budgets and submission history come from the configured SQLite database. No sample data is used as a fallback.

## Start

Complete the project setup first: install the Python package, configure `.env`, and initialize the research database using the existing project instructions. The web service never initializes or migrates a database during a read.

From the repository root (Node.js 24 and pnpm 11):

```powershell
pnpm --dir webui install --frozen-lockfile
pnpm --dir webui build
alpha-garden web
```

Open [the local console](http://127.0.0.1:8787/). The Python service serves both the built frontend and its API. Starting it does not trigger platform authentication, backtests or submissions.

Unauthenticated visits open the Pro login page. The local console account is `admin` with password `123123`; both are prefilled in the inputs and are separate from WorldQuant credentials. After login, the browser returns to the requested console page. The account menu provides Sign out. Login lasts up to 12 hours and is cleared by logout or a backend restart. It does not start, pause or resume research.

For observation only:

```powershell
alpha-garden web --read-only
```

This disables write endpoints as well as the UI controls. `--port`, `--database`, `--settings`, `--run-config`, `--env` and `--assets` select the local paths and port; defaults match the CLI and `webui/dist`.

## Research workspace

The Formulas page requests `execution=started`: only tasks with a recorded backtest submission start are included, including pending, uncertain and failed requests. Unsent candidates and cancellations before submission are excluded before search, grade filtering and pagination, so totals match the list. Run details keep their separate finished, in-progress and not-started lists; no task or research plan is changed by this display filter.

The sidebar groups twelve pages into Overview, Research, Results and System. Formulas is a single searchable collection; Optimization formulas (专项优化公式), Qualified archive and Submitted formulas are independent pages. Submissions continues to manage candidate submission and attempt history.

- **Seed pool:** admitted roots come from persisted engine membership. Active research branches come from `load_signal_frontiers`, with the engine's remaining budgets and root links. Merely visiting the pool never admits a new seed, changes eligibility or starts research. Local PnL correlation is part of seed admission, not a repeated per-descendant condition of the active-branch list. Optimization formulas use the separate checked-candidate decision and have up to 20 mutation attempts to explore possible grade improvements. Unused attempts alone do not prohibit manual submission, but current correlation evidence and protection of other improvement opportunities determine whether Submit is enabled.
- **Data catalog and Operators:** search, category filters, dataset filters (fields), pagination and details use the locally synchronized catalog. Context and sync time are shown. A catalog belonging to another account is unavailable; viewing does not trigger platform synchronization.
- **Formula details:** click a formula row in collections, run details, admitted seeds, active branches, submission candidates/history or direct descendants to inspect its highlighted expression, settings, recorded metrics, backtest checks, formal-check summary, annual results, saved PnL and descendants. IDs are plain text; clickable rows also open with Enter or Space, and selecting text does not open them. Detail drawers stay within the current menu and preserve the list's filters, tab and pagination when closed. Active branches have a separate View root seed action that opens the root without triggering the branch row. Mutation descriptions follow the selected Chinese or English language and explain the changed operator, field or window directly, without internal argument paths. The main expression and before/after expressions stay on one line, scroll horizontally when needed and copy the original text. Direct-parent links and descendant rows open another formula drawer above the current one. Each layer retains its scroll position, selected curve and descendant page; closing the top layer with its mask, close button or Escape returns to the previous detail without replacing it or changing the menu. Catalog and analysis links connect their respective views. Missing series or results remain unavailable. Descendants use an independent ten-row paginated read (`GET /api/formula/children?task_id=...&page=...&page_size=10`). Changing its page updates only the descendant table, keeps the detail content mounted and does not refetch the formula or platform curves. Loading and errors stay within this table; pagination does not scroll the drawer back to the top. Submitted Alphas without a local task keep their platform summary without a fabricated detail record and their rows do not open a drawer.
- **Platform curves:** opening formula details automatically requests PnL. Sharpe and Turnover are requested only when their respective tabs are selected, through separate endpoints that each read one recordset using the existing authenticated Python client. Loaded tabs retain their data until the drawer closes or the formula changes. Tabs switch between 420-pixel charts, with date labels 28 pixels below the axis (20 pixels more spacing). PnL uses K; Turnover ratios display as percentages; null Sharpe observations remain gaps. Dark formula cards and the summary grid use the black container background. The service serializes chart reads and caches each Alpha/metric separately. Ready/error responses are reused for 30 seconds or a longer Retry-After; pending responses expire after their retry delay (at least one second). The active tab retries pending data up to three total attempts, with waits of at most 30 seconds; switching tabs or closing details cancels waiting. Account errors stop further platform access during the cooldown. A failed series stays separate from successful ones; saved PnL remains visible if its refresh fails. These requests do not start research or write the database, and are disabled in read-only mode.
- **Backtest analysis:** `/analysis` provides Quality diagnosis: Sharpe/Fitness groups, turnover distribution (including all values at or above 80%), failed-check reasons, platform self-correlation and a separate local return-correlation reference. Time, research mode, run and source filters apply to sent tasks for the current account. Each view uses one exact backtest configuration, initially the largest group, with an explicit selector for other groups. Date filtering uses completion time or submission time for unfinished tasks, not the strategy's backtest years. Unsent candidates are excluded.

Analysis reads the latest saved check evidence without reviving an older pass when a newer observation is pending, incomplete or erroneous. Failed-check rates divide failures by explicit passes plus failures; missing and pending evidence is excluded. Shared Sharpe/Fitness reference lines require complete, uniform saved limits. Local correlation loads independently and reuses the engine's saved daily-return comparison: at least 252 matching intervals, correlation below 0.7 or at least 10% higher Sharpe against every correlated submitted reference. Missing evidence and no references remain distinct. This display does not authorize submission or bypass existing opportunity protection.

All scatter groups use colored circles. Select a scatter legend to hide or restore its group; hidden legends turn gray with a strikethrough, and Reset restores all groups. Legend toggles affect only the scatter plot, leaving statistics and record lists unchanged. Select a chart point to open formula details; turnover bins, check rows and correlation legends reveal the corresponding records. These lightweight scoped records paginate locally, preserving scroll and avoiding another aggregate read. Formula drawers opened from analysis, including their parents and descendants, use the same on-demand platform curves as the formula library: PnL on opening, and Sharpe or Turnover when selected. Actual service read-only mode still disables platform curve requests. Analysis statistics themselves continue to use only saved local data. Analysis endpoints share in-flight reads and retain one filter scope per endpoint until the database, account/configuration or calendar revision changes. No research state is written.


Local reads use the current account and existing read-only database transaction. They share the existing single topic-based SSE connection. Platform-curve requests use protected POST endpoints and are triggered by opening details or selecting a curve tab, never by SSE notifications. Native formula expressions, parameters and locally saved curves stay in the local console.

## Operations

- **Run settings:** edit Backtest settings or Research & runs in compact forms and save each tab independently. The forms write the configured policy and run-default files used by the CLI and new web runs. Validation reuses the engine's policy and run-limit checks, including allocation totaling 100%, backtests not exceeding generated candidates, direction validation within mutation allocation, and tail-risk truncation not exceeding the default. Existing plans retain their saved settings on resume; cycle count, research mode and submission authorization remain start-time choices. Changing data scope requires a matching synchronized catalog before starting research. Category-specific neutralization rules expand within the form. Switching tabs keeps drafts; Discard changes restores the most recently loaded/saved values. Live refreshes preserve unsaved edits. Revision checks reject overwriting another editor's changes. Saves use an atomic replacement of one configuration file and are blocked in read-only mode or during web/CLI research or submission operations. Saving does not write research data or call the platform.

- **Runs:** choose normal research or focused optimization and a positive cycle count. New web runs keep automatic submission disabled. Review the mode and budget before starting. Each history row always shows link-style Details, Logs and Resume actions. Logs is disabled when no eligible operation log belongs to that run; Resume is disabled when the run cannot be recovered or operations are unavailable. Terminal runs with no recoverable work are rejected before a worker is created. After a start or resume is accepted, its log drawer opens and further starts remain disabled until the operation appears in the refreshed job list. Results are loaded only while that run's details drawer is open, with 10 formulas per page.
- **Pause and resume:** pause a web-owned operation after its current request completes. Completed work and remote identities stay in the database. Resume by run ID using the original configuration and stop conditions, including its original automatic-submission setting.
- **Submissions:** Spectacular uses the submission queue. Other grades use fully checked, unsubmitted archive candidates whose research has ended through exhaustion or replacement. Archive membership alone does not authorize submission: the page and server use the same current correlation and improvement-step selection. The count measures confirmed new submissions, excludes already-active Alphas, skips rejected candidates and stops early when candidates run out. An empty candidate list still allows starting submission so an already-claimed, unfinished item can resume; with no pending work the operation ends immediately.
- **Submit an optimization formula:** the Action column offers Submit for a specific formula without waiting for its remaining mutation budget to reach zero. Confirmation identifies that Alpha and explains the existing interrupted-research settlement behavior. The protected command is `POST /api/jobs` with `{"kind":"submit","task_id":"<local task ID>"}`. It rechecks and submits only that formula using the formal submission workflow; failed checks do not cause another formula to be submitted. Missing, consumed or unavailable candidates are rejected before an operation starts. Requests with the same idempotency key reuse the same operation. History records the Optimization formulas source and offers Continue for an unfinished item, checking uncertain platform outcomes before any retry. A different unfinished submission must be resolved first.

Optimization formulas can be filtered by Alpha ID, grade and source (Exploration, SC repair, Mutation or Reversal). Source filtering happens before pagination on the server; changing or clearing it returns the list to page 1. The page explanation has a theme-colored question icon. Its action buttons use the same text height as the other formula cells, so adding Submit does not increase row height.

The archive Action column also supports exact selection using `{"kind":"submit","task_id":"<local task ID>","source":"qualified_archive"}`. Continue retains this source. Both manual entry points plan across the account before filtering by source, grade or selected ID; an unavailable selected formula is rejected without substituting another one. Local screening uses saved daily PnL increments (at least 252 matched intervals), correlation below 0.7 or at least 10% higher Sharpe against every correlated submitted reference, with unrounded values. Missing evidence stays pending. Fully checked active optimization opportunities are protected; already-blocked opportunities do not veto other submissions. Among retired candidates, prefer a step that blocks fewer remaining candidates, then grade, Sharpe, Fitness and earlier completion. This deterministic ordering does not promise a globally optimal portfolio.

The final POST guard repeats the local plan within the claiming transaction. If eligibility changed, the latest check evidence is retained and the unposted reservation is released back to its source, allowing independent work to continue. A newer pending or failed check blocks an older ready observation; releasing the reservation never overwrites newer evidence or its cooldown. It never releases a claimed or uncertain POST. Formula results and budgets remain intact, and a deferred formula can become eligible after evidence or references change. Each confirmed submission changes the next selection; an operation remains within its original authorized candidate set. Missing PnL for qualified comparisons is collected at existing research boundaries with the existing bounded retry policy, never by list reads or service startup. Historical candidates lacking that evidence remain unavailable until a separately authorized research operation collects it. No new database migration is needed for these eligibility changes.

Existing databases with only `queue` and `qualified_archive` submission sources require the explicit `persistence.schema.add_optimization_submission_source` migration before enabling manual optimization submission. Run it in a transaction under the exclusive database process lock after stopping the service. It changes only the submission-source constraint, preserves attempt rows and indexes, rejects unrelated schema changes, and rolls back on failure. It never runs from a read endpoint or automatically during service startup. Restart the service after an approved migration. Source filtering and exact archive selection need only the updated service and do not require migration. On an unmigrated database, selected optimization submissions are rejected before creating a worker, settling research or accessing the platform.
- **History:** database run status is historical evidence, not a process heartbeat. CLI-owned processes retain their original terminal logs. The web service cannot pause a separate CLI process; the shared process lock prevents competing writers.
- **Logs:** the last 20 web operations retain up to 500 log lines each in memory. The run page shows logs for the current running or stopping web operation and retains the most recent failed or needs-attention research operation, using its exact operation ID. The live connection, worker busy state and operation status determine active availability; a historical database status of running is insufficient. An operation without a run record has a Startup logs link in the configuration card. Successful completion closes the run-log drawer; failure keeps its error and logs visible until a later operation replaces it. Known errors have readable messages alongside their original codes. Submission logs remain scoped to submission operations. Restarting the service clears temporary logs; research results remain in SQLite.
- **Settings:** displays the actual configuration rules, including category-dependent neutralization and truncation. Editing still uses the local configuration files.

Seed and recovery correlation logs identify the Alpha before fetching its time series, then report completion with the collected/required count, or a pending/error result with its retry cooldown. These messages are retained in web operation logs and refreshed after a job-version notification; collection and retry rules are unchanged. Terminal platform failures retain the platform's message when supplied and include it after the error code in the log. Missing messages remain missing, and an unknown grade is not an error classification. The result-column legend displays its check-status explanation on a separate line. Backtest result rows use fixed-width status, grade, cycle, sequence, source and ID columns with right-aligned metric values, so mixed Chinese and English text stays aligned. They remain single-line text when selected and copied.

Main list tables have borders, left-aligned columns, 16-pixel horizontal cell padding and no surrounding cards. Detail and analysis sections group related data in cards. All lists and run details default to 10 records per page and allow choosing 10, 20, 50 or 100. Changing the page size returns to page 1 and updates the server query; resizing the window does not change the selected size or page. Submission candidates and history keep independent pagination. Run history smoothly adjusts row spacing and control padding to the desktop window height, keeping the default 10 records and pagination on one screen from 1280 by 720 without compressing medium-height windows unnecessarily. Larger page sizes can scroll. The run controls have no title bar. Its ID and action columns keep fixed widths instead of stretching with the table. Other lists retain their roomy spacing. Run-detail tables inherit the drawer background; formula-detail sections use the standard container background. Submission logs open from the configuration panel's Logs link in a drawer.

Table headers, cells and tags stay on one line in both languages, including tables inside drawers. The English Check results column has extra room; longer failure tags use an ellipsis and retain their full check details on hover. Multiple failure tags stay in one row.

The light theme uses its original light-gray page background throughout the console and dashboard, including the initial loading screen. The dark page background stays unchanged. Pagination buttons and the page-size selector have transparent backgrounds so they match the surrounding page or drawer; the active page retains its blue outline. Component pagination labels follow the selected language.

Formula details display the latest saved full-check observation from research or formal submission, with its observation time and actual values/limits. Without a full-check observation, the table is explicitly labeled Backtest snapshot and retains its original statuses, including pending self-correlation. A newer pending, missing-detail or error observation never borrows an older pass; historical backtest records are not overwritten, and viewing checks never requests the platform.

The Check results column distinguishes basic threshold failures, specific backtest check failures and formal checks. Basic Sharpe/Fitness/Turnover failures show “未达标”; otherwise failed sub-universe, weight concentration and correlation checks show their specific reasons. Hover retains the source, observation time, original codes and available actual/threshold values on one line. Backtest checks never imply that a formal check passed; request errors and pending observations remain distinct. Turnover uses its English name in both locales. Grade and status tags use Ant Design outlined colors: red for Spectacular, green for Excellent, lime for Good, blue for Average and neutral gray for Inferior.

Run details default to finished backtests (completed and failed). Requests awaiting acceptance or results appear separately under In progress, with Awaiting acceptance or Awaiting backtest result instead of Awaiting checks. Created tasks that have never been submitted appear under Not started with Awaiting backtest. All three views use server-side pagination and update through the existing SSE connection; receiving a terminal result moves its record out of In progress. Filtering never changes tasks or cancels plans. Formula rows open a nested detail drawer within Runs, including when following parent or descendant formulas. Closing its mask, close button or Escape closes only the formula drawer; the run drawer retains its selected view, pagination and scroll position. Run status uses outlined tags in both the list and details: paused is gray, failed is red, and completed or running is green.

Operation logs use a black terminal background in both themes, with colored check statuses, grades, sources, phase messages and retry warnings. Timestamps are muted; text remains selectable with its existing spacing and line breaks. They automatically scroll to the latest line when opened, updated or resized, including updates that replace older entries at the 500-line limit. Horizontal scroll position is preserved. Operation logs display the most recent transient progress separately, with its observation time. This exposes submitted/in-flight/waiting progress before the first backtest result arrives, including resumed runs. Only the latest transient message is retained; repeated identical messages do not trigger updates or fill the 500-line result log. A regular log or operation completion clears the previous transient message. The progress is display-only and never controls scheduling. Backend changes require a service restart; do not interrupt an active operation to deploy them without user authorization.

Research mode uses outlined tags in the run list, details and resume confirmation: blue for normal research and volcano for focused optimization. The mode selector remains a radio control.

The resume confirmation identifies the selected run by research mode and start time (creation time if it has not started), and lists finished/planned counts and automatic submission status. The long internal run ID stays in the detail view; the resume request still uses that exact ID.

Cooperative pause saves the current response before stopping at an execution boundary. While holding the exclusive run lock, the launch/resume owner records `stop_reason=user_paused` on an unfinished plan without changing its tasks, budgets or terminal results. History displays that observation as Paused even after the web service restarts; while the current web operation is stopping, its live status displays Pausing. Resume clears the pause reason under the same lock and continues the original identities and limits. Starting a new run or manually submitting still uses the existing plan-retirement rules.

Closing the browser does not stop a run. Stop the service with Ctrl+C to request a cooperative pause. After an unexpected shutdown, use the existing run or submission recovery flow; do not treat an uncertain platform result as a failed submission.

## Data boundaries

The service binds only to `127.0.0.1`, validates the request host and browser origin, and checks the local account on the server. Login issues a random, expiring HttpOnly/SameSite=Strict cookie; private reads, SSE and writes require that session, and writes additionally require the existing request token. Logout invalidates the session and its open streams. Credentials and session IDs are never stored in frontend localStorage. This remains a single-account local console, with no public hosting or multi-user account management.

Read endpoints use SQLite read-only connections. They do not reconcile state, create schema, query WorldQuant BRAIN, or expose credentials. The recent-results endpoint includes expressions for the current account's four most recent completed backtests, truncated in the page with the full expression available on hover or keyboard focus. Other list endpoints omit expressions. Research expressions stay local and must not be included in public screenshots or repository assets.

The grade chart uses platform grades from the current account's locally recorded submitted Alphas. Each submitted Alpha is counted once; missing grades are shown as unknown, so the full distribution matches the submitted total. The research overview shows available optimization parents, their remaining attempts, manually submittable archived formulas, and seven-day backtest and submission totals. Optimization selection does not rebuild historical PnL recovery correlations, which do not affect its checked lineage membership. Loading is shown separately from a failed read; missing data or failed reads are never displayed as zero or success.

Formula categories and submission decisions reuse the engine's existing selectors. Attempt counts include sent children and exclude candidates cancelled before sending. A displayed remaining budget does not by itself mean a formula is an eligible active parent.

## Live dashboard updates

The dashboard uses five independent reads: activity and totals, submitted grades, research eligibility, batch progress, and recent results. Each section loads and fails independently. Grade and recent-result reads do not reconstruct optimization eligibility. Completed research snapshots are loaded in batches instead of querying each historical task separately.

The dashboard backtest trend covers the last 15 local calendar days, including today, with zero counts for days without completed backtests. Research summary totals and KPI sparklines retain their seven-day window. The trend reserves space for both endpoint date labels and skips intermediate labels when space is limited.

Dashboard charts play ECharts entrance animations when their first data arrives. Live updates and grade legend filters reuse the existing series for transitions instead of restarting the entrance. Resize notifications only resize charts when their dimensions actually change. Reduced-motion preferences disable these animations.

Each open console tab keeps one stable `/api/events` SSE connection across route changes, drawers, filters and pagination. It sends only `versions` events for the database, jobs and session/configuration topics. Opening it never executes business queries or transmits business rows. Its long duration in browser network tools is the connection lifetime, not a list-query duration.

After the initial version frame, mounted sections perform one HTTP read per exact resource path. Later changes refresh only mounted resources for the changed topic, with their current filters and page size. Unchanged versions trigger no reads. Unmounted sections abort requests and release data; development replay initializes only surviving subscriptions. There is no periodic browser polling.

Each resource allows one in-flight read. Notifications during slow reads coalesce into one follow-up, while other sections continue independently. Abandoned responses cannot populate a new page; failures stay local to their section. Reconnection refreshes mounted sections even when a service restart resets revision counters.

One observer checks SQLite commit versions, configuration timestamps, job revisions and the calendar day every second, using an autocommit read-only connection. It does not rebuild business data while idle. CLI commits are detected too, at database rather than row level. Heartbeat comments every ten seconds do not refresh data. Connection failures are shown explicitly.

All formulas and run-detail queries filter and paginate in SQLite before loading metrics, checks and attempt information. Parent budgets include submitted children outside the page or run. Run history aggregates task counts without loading full formula records. Other categories retain the engine's eligibility selectors. This changes reads only, without schema changes, result caching or writes to research state.

Research counts reuse the optimization engine. It reads formal checks in bulk and loads complete backtests only for candidates whose formal checks passed, then loads their ancestor chains to verify seed membership. Archive budgets read only archived parents and their direct children. Submission consumption reads account/formula identities without parsing full historical responses. Each read still applies current checks, replacement decisions, consumed identities and the 20-attempt budget.

Submission eligibility reads saved PnL only for the current account's checked candidates and submitted references. Within one decision, each curve's daily increments are prepared once and shared across near-duplicate, submitted-reference and opportunity-protection comparisons; repeated pairs reuse the same result. This data is discarded after the decision, so a later read or final submission guard uses fresh facts. An empty submission queue does not run another eligibility calculation. These reductions preserve the matching-date, correlation and 10% Sharpe-improvement rules without introducing a persistent eligibility cache.

The optimization list also uses the engine's measured near-duplicate retirement: same account, lineage and settings, at least 252 matching daily PnL intervals, and correlation >= 0.9999. It retains the best grade, then Sharpe and Fitness, then earliest completion, with the representative's existing budget. It reads only saved PnL for checked candidates and never fetches platform curves from this list. Missing evidence does not hide candidates; retired duplicates remain in formula history and enter the qualified archive through the normal research settlement process.

The activity and research HTTP endpoints share concurrent reads and retain their display statistics while the database revision, account/configuration and calendar day are unchanged. Every request synchronously checks SQLite's commit version before reusing a result, including WAL commits; it does not wait for an SSE tick or a time-to-live expiry. A commit during computation prevents that older result from populating the current cache. Failures are not cached. Formula lists and final submission guards continue to calculate eligibility from current facts; the dashboard cache never authorizes a submission. Opportunity selection checks whether a candidate would block protected formulas before doing its full submitted-reference comparison, and only evaluates the protected formulas whose eligibility can affect that decision.

Idle streams send a heartbeat comment every ten seconds; comments do not refresh the page. Network disconnects show a reconnecting status. A terminal stream error is shown as unavailable with a reload instruction instead of claiming that reconnection is still running. Starting, stopping and submitting still use the existing protected HTTP commands; version notifications refresh operation status and bounded logs through their read endpoint.

## Frontend development

The saved theme and brand loading screen are applied by `index.html` before the application modules load. Route loading uses the same screen; table and settings reads use a compact version with the same centered layout. The logo stays still while a thin highlight sweeps beneath it. Loading ends when the relevant content is ready, without a fixed animation delay, and the highlight remains static for reduced-motion preferences.

In separate terminals from the repository root:

```powershell
alpha-garden web --read-only
```

```powershell
pnpm --dir webui dev
```

The [development page](http://127.0.0.1:5173/) proxies `/api` to port 8787. `pnpm --dir webui preview` uses port 4173 with the same proxy. Both bind locally and refuse an occupied port. If changing the API port, update the local Vite proxy target accordingly.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/auth` | Local console login status; no research data |
| POST | `/api/login` | Validate the local account and issue its session cookie |
| POST | `/api/logout` | Revoke the current local console session |
| GET | `/api/session` | Local readiness, read-only mode and write token |
| GET | `/api/events` | SSE topic revisions and heartbeat comments |
| GET | `/api/dashboard/activity` | Daily activity, totals and seven-day trends |
| GET | `/api/dashboard/submitted-grades` | Platform grades of locally recorded submitted Alphas |
| GET | `/api/dashboard/research` | Eligible optimization parents, attempts and manual candidates |
| GET | `/api/dashboard/progress` | Latest batch and its progress by formula source |
| GET | `/api/dashboard/recent` | Four latest completed results with expressions |
| GET | `/api/analysis/quality` | Account-scoped quality diagnostics and lightweight drilldown records; days, mode, run, source and exact configuration filters |
| GET | `/api/analysis/correlation` | Independent saved-PnL reference comparison for the same diagnostic scope |
| GET | `/api/runs` | Paginated run history |
| GET | `/api/catalog/fields`, `/api/catalog/operators` | Paginated local catalog, search, filters and details |
| GET | `/api/seeds` | Admitted roots or active normal-research branches; search and pagination |
| GET | `/api/formula` | Scoped formula detail by `task_id`, with `child_page` |
| POST | `/api/formula/series/pnl` | Read only the PnL recordset for `task_id`; no database writes |
| POST | `/api/formula/series/sharpe` | Read only the Sharpe recordset for `task_id`; no database writes |
| POST | `/api/formula/series/turnover` | Read only the Turnover recordset for `task_id`; no database writes |
| GET | `/api/formulas` | Search, grade, category, run and execution (`all`, `finished`, `inflight`, `planned`) filters |
| GET | `/api/submissions` | Paginated submission attempts |
| GET | `/api/settings` | Validated current configuration |
| POST | `/api/settings` | Save one defaults section (`backtest` or `run`) with its current revision |
| GET | `/api/jobs` | Web-owned operations and bounded logs |
| POST | `/api/jobs` | Start normal/optimization run, resume, or submit |
| POST | `/api/jobs/stop` | Request cooperative pause |

Write requests use JSON and `X-Console-Token`. Starting a job also requires an `Idempotency-Key`; repeating the same request and key returns the existing job. One operation may execute at a time. Paths, credentials and arbitrary commands cannot be supplied through the API.

## Verification

```powershell
pnpm --dir webui lint
pnpm --dir webui test
pnpm --dir webui build
python -m unittest tests.alpha_garden.test_web tests.alpha_garden.test_web_events tests.alpha_garden.test_cli tests.execution.test_console tests.execution.test_console_jobs tests.persistence.test_console_research
```

Tests use synthetic databases and simulated platform clients. They cover read-only behavior, account scoping, budget handling, input and origin validation, idempotency, the existing process lock, pause/resume and counted submissions that consume only selected candidates. SSE tests cover read-free revision notifications, committed changes, configuration changes, missing-database recovery and reconnects. Frontend transport tests cover single initialization, stable connections across filters and pagination, abandoned response cleanup, topic-specific refreshes, coalesced slow reads and independent errors. Browser checks verify visible updates and an unchanged long-lived connection across drawer and page changes.

The dashboard fits desktop viewports from 1280 × 720; smaller screens retain natural scrolling. The login page uses the original dot-matrix shader and form entrance animation from `Alpha-Garden-Pro/webui/src/components/ui/sign-in-flow.tsx`, with matching versions of Three.js, React Three Fiber and Framer Motion. Its animation parameters are preserved; the mock login marker and fixed loading delay are replaced with actual server authentication. Login tests cover valid and invalid credentials, unauthenticated requests, logout, session expiry, safe return routes and revoked stream handling. The build currently reports a large JavaScript bundle warning. Dependencies, build output, local databases and logs remain excluded from Git.
