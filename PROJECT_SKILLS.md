# OpenAlgo Project Skills - Digest

A condensed study guide to the nine project skills in `.claude/skills/`. Each
`SKILL.md` (and its `reference/` or `references/` folder) remains the source of
truth; this file summarises them and can drift if they change. Snapshot taken
2026-11-04.

| Skill | Use it when | The one rule |
| --- | --- | --- |
| `broker-integration` | Adding or modifying a broker plugin | Translate broker shapes into the common symbol / API / WebSocket contracts; copy the closest family template |
| `chart-indicator` | Writing a JavaScript indicator for the `/trading` chart | Never write into `strategies/indicators/` directly; validate, then `--install` |
| `openscript` | Writing a `.oscript` study or strategy | Never write into `strategies/openscript/` by hand; source and program are written together from one compile |
| `flow-builder` | Building or debugging a `/flow` workflow JSON | Run `validate.py` before handing anything over; unknown `data` keys are silently ignored |
| `fd-audit` | After any change touching DB, sockets, threads, subprocesses, files, caches, registries | Audit the change, check every exit path, measure when a leak is suspected, report before fixing |
| `verify` | Before claiming a control holds, a bug is fixed, or a test guards a fix | Execute, do not reason; break the code to prove the test |
| `security-audit` | Monthly review, before a release, after a dependency bump | Run `audit.py`; work FAIL, ERROR, REVIEW, WARN in that order |
| `version-bump` | Platform release or bumping the `openalgo` SDK pin | Two unrelated versions in different files; release notes and contributors are part of the bump |
| `grokbot-setup` | Running OpenAlgo inside Grok Bot's Linux box | Knowledge file first, never guess, uv only, never echo secrets, Sandbox by default |

Three unrelated things are called "indicator" in this repo:

- **`chart-indicator`**: plain JavaScript descriptor on `openalgo-charts`, drawing only, loaded at runtime in the browser.
- **`openscript`**: a compiled language (`.oscript`) that can draw *and* place orders.
- **`openalgo.ta`**: the Python library used from the strategy host, scanners and backtests (no skill).

---

## 1. broker-integration

**Stack position.** `restx_api/<endpoint>.py` (thin, validated, rate-limited) ->
`services/<x>_service.py` (auth, analyze-mode branch, events) ->
`broker/<name>/api/*` (the only place broker specifics live).

- Broker modules **never emit events** and **never see analyze mode** (the service routes it to `sandbox_service` first). "Works in analyzer" proves nothing.
- A broker never adds anything to `restx_api/`. A new capability goes service layer first, then `restx_api/`, for every broker.

**Step 1: pick the family template.**

| Family | Members | Copy |
| --- | --- | --- |
| OAuth2 / checksum redirect | zerodha, upstox, fyers, dhan, arrow, groww, paytm, aliceblue, definedge | `broker/zerodha/` |
| Noren / Finvasia | shoonya, flattrade, tradesmart, zebu, ibulls, wisdom | `broker/shoonya/` or `broker/flattrade/` |
| Symphony XTS (dual credentials) | fivepaisaxts, jainamxts, compositedge, rmoney, iiflcapital | `broker/fivepaisaxts/` |
| Direct login + TOTP | angel, mstock, motilal, kotak, firstock, samco, nubra, tradejini, fivepaisa | `broker/angel/` |

Special cases: `broker/fyers/` for depth above 5 levels; `broker/deltaexchange/`
is the only crypto plugin. White-labels give themselves away by their endpoint paths (`/NorenWClientTP/`, `/interactive/user/session`).

**Step 2: read the contracts, not a reference broker's code.**
`docs/prompt/symbol-format.md`, `order-constants.md`, `LotSize.md`,
`websockets-format.md`, `services_documentation.md`, `docs/api/**`,
`docs/broker-integration-guide.md`, `references/cross-broker-reference.md`.

- **Inspect a working broker's live `symtoken` table before overwriting it.** It is the ground truth for NCO symbols, MCX_INDEX names, expiry formatting and the index set.
- **Download the broker's official SDK from PyPI and read it.** It gives the real field names, enum codes, binary offsets and routes.

**Step 3: module contract.** Services import these exact names:

| Module | Functions |
| --- | --- |
| `api.auth_api` | `authenticate_broker(...)` -> `(auth_token, error)` |
| `api.order_api` | `place_order_api`, `place_smartorder_api` -> `(response, data, orderid)`; `modify_order`, `cancel_order`, `close_all_positions` -> `(dict, status)`; `cancel_all_orders_api` -> `(canceled, failed)`; `get_order_book`, `get_trade_book`, `get_positions`, `get_holdings`, `get_open_position` (net qty as str) |
| `mapping.order_data` | `map_order_data`, `calculate_order_statistics`, `transform_order_data`, `map_trade_data`, `transform_tradebook_data`, `map_position_data`, `transform_positions_data`, `map_portfolio_data`, `calculate_portfolio_statistics`, `transform_holdings_data` |
| `api.funds` | `get_margin_data(auth)` |
| `api.data` | `BrokerData(auth[, feed_token])`: `get_quotes`, `get_depth` (5 levels), `get_history` (DataFrame), `get_multiquotes` (optional), `timeframe_map` |
| `database.master_contract_db` | `master_contract_download()` |
| `api.margin_api` | `calculate_margin_api(positions, auth)` (optional) |
| `api.gtt_api` | GTT functions (optional; only Dhan and Zerodha; absent -> 501) |

Nuances:
- The arity of `BrokerData.__init__` is introspected: declare a second `feed_token` argument only if the broker has a separate feed credential.
- If `get_multiquotes` is missing, the service falls back to one `get_quotes` call per symbol, which is very slow for options tools (180+ symbols). Implement it whenever the broker has a batch endpoint.
- Margin: copy `broker/dhan/api/margin_api.py`. One leg goes to the single-order calculator, two or more go to the basket calculator, so hedges net off. Keep its JSON-decode guard (502) and `_normalise_success_response` (an error sent with HTTP 200 becomes a 400).
- The optional order-update stream lives in `services/order_update_service.py` (17 of 36 brokers have one).

**Symbol helpers** (from `database.token_db`): `get_token`, `get_symbol`,
`get_br_symbol`, `get_oa_symbol`, `get_brexchange`, `get_symbol_info`,
`get_tokens_bulk`. Never hand-roll lookups.

**Layout:** `plugin.json`, `api/{auth_api,order_api,data,funds,margin_api,gtt_api,baseurl}.py`,
`mapping/{transform_data,order_data,exchange}.py`, `database/master_contract_db.py`,
`streaming/{<name>_adapter,<name>_websocket,<name>_mapping}.py`.
`plugin.json` keys that matter: `supported_exchanges`, `broker_type` (`IN_stock` / `crypto`), `leverage_config`.

**Shared utils:** `utils.logging.get_logger` (its redaction filter stops tokens reaching the logs), `utils.httpx_client.get_httpx_client()` with an explicit `timeout=`, `utils.mpp_slab` (MARKET/SL-M emulation), `utils.config` getters, `utils.constants` exchange codes.

**Build order:** research -> `plugin.json` + auth -> platform wiring
(`VALID_BROKERS`, install scripts, `BrokerSelect.tsx`, `websocket_proxy/__init__.py`,
the `brlogin.py` branch) -> master contract -> quotes, depth and history -> orders and books -> funds and margin
-> streaming -> live-hardening pass -> `fd-audit`.

**What "done" does not mean:** passing in analyze mode, a clean code review (all six Arrow bugs survived review), or NSE working.
Test every exchange in `plugin.json`.

---

## 2. chart-indicator

**Rule:** draft to scratch, validate, then install. Never write into `strategies/indicators/` directly:
a short column or a mismatched plot key draws nothing and raises nothing.

```bash
node .claude/skills/chart-indicator/validate.mjs <scratch>/x.js
node .claude/skills/chart-indicator/validate.mjs <scratch>/x.js --install
```

- Never `npm install` for this. The validator finds `openalgo-charts` itself (`frontend/node_modules`, then `.cache/`, then a fetch of the pinned version).
- If the fetch fails, say so and offer the choice; do not silently skip validation.
- After installing, reopen the indicator picker; no reload is needed.

**File shape:** plain JS (no TypeScript, JSX or imports) with one default-exported function that receives the API:

```js
export default function ({ registerIndicator, sma, nulls }) {
  registerIndicator({ id, name, category, placement: 'onchart'|'pane',
    inputs: [...], plots: [...], calc(bars, settings, store, ctx) { return {...} } })
}
```

A bar is `{time (UTC seconds), open, high, low, close, volume}`, with `oi` optional.

**Workflow:** read `reference/cookbook.md` first (reuse built-ins through
`getIndicator('macd').calc(bars, settings, {})`), then `contract.md`, `api.md` and
`pitfalls.md`, pick the closest of the twelve examples, then draft, validate and install.

**Library features to use instead of hand-rolling:** `draws(ctx)`, `background(ctx)`,
`barColors(ctx)`, `levels(ctx)`, `alerts[]`, `plot.overlay`, `plot.ohlc`,
`plot.offset`, `securitySeries(bars, tf, opts)` for higher timeframes,
`ctx.requestBars` for another instrument, `ctx.tickSize`, `parseSessionSpec` /
`inSessionAt`, `crossesAbove` / `crossesBelow` / `rising` / `falling`,
`IndicatorInputError` for errors the user can fix, `withTimeframe(descriptor)`.

**Top four pitfalls:**
1. Every column must be exactly `bars.length`.
2. Warmup values are `null`, never `0`.
3. `5 > null` is true in JS, so guard comparisons with `x != null`.
4. Markers using `aboveBar` / `belowBar` anchor to the indicator's own plot. Use `atPrice` or `markerAnchor: 'price'` to place relative to the candle.

**Do not:** add colour or width inputs for plots (the chart generates them), add precision or
tick-size inputs, reuse a built-in id (custom indicators register last and override, and there are 112 built-ins), or assume the
browser's local time zone (use `zonedDayIndex` with `DEFAULT_TIMEZONE`).

`calc` runs once per animation frame and must be pure in `(bars, settings)`.
Porting from another chart language follows a construct-by-construct table: `overlay` -> `placement`,
`plotshape` -> `markers()`, `hline` -> `levels`, `request.security` -> `securitySeries`, `na` ->
`null`, `var` -> a variable outside the loop.

Bumping `openalgo-charts` means regenerating the skill in the same change: `generate-api-index.mjs`, `coverage.mjs`, the
drawing and chrome-icon generators, and the prose.

---

## 3. openscript

**Rule:** the runner reads only `<name>.oscript.program.json`, never the source.
A source with no program cannot run; a program built from different text runs a different strategy from the one shown.
Always:

```bash
node .claude/skills/openscript/validate.mjs <scratch>/s.oscript
node .claude/skills/openscript/validate.mjs <scratch>/s.oscript --install [--as name.oscript]
```

`--install` writes both files only when there are zero errors and the program's source hash matches.

- **A pass only means it compiled.** No bar has run. Runtime codes such as `OS4008` (a cell outside its grid) can still appear; `OS6025` is a platform data problem, so report the instrument and timeframe.
- **Read before writing:** `reference/library.md` (350 names with warmups), `pitfalls.md` (with diagnostic codes), `strategies.md`, `porting-from-javascript.md`, and six compiling examples. Look names up; do not recall them.

**The three costliest traps:**
1. **95 of the 350 names are planned, not implemented** (`OS2020`). Examples: `order.qtyForRisk`, all of `leg.*` and all of `book.*`.
2. **`buy` / `sell` move a position by an amount; they do not set a side.** Selling one lot while long one lot leaves you flat, not short. See `examples/stop-and-reverse.oscript`.
3. **Absent is not zero.** Do not substitute values during warmup with `orElse`.

**Porting a `//@version=5/6` script:** use `importScript` from `openalgo-script`. Its findings are `OS9001`-`OS9012`. The output is a draft: read every finding, then validate it.

This version of the skill describes `openalgo-script` 0.8.1. A bump means running `generate-reference.mjs`,
`coverage.mjs` (must print COVERAGE COMPLETE) and `check-pitfalls.mjs` (must print PITFALLS VERIFIED), then updating the prose.

---

## 4. flow-builder

**Rule:** `uv run python .claude/skills/flow-builder/validate.py <workflow.json>`.
It wraps the importer's own validator and adds a check for unread `data` keys. Treat
warnings as errors.

**Steps:**
1. Read `reference/nodes.md` (61 node types).
2. Pick exactly one trigger: `start`, `webhookTrigger`, `priceAlert` or `orderUpdateTrigger`.
3. Wire the edges. Edges leaving a condition node need a `sourceHandle`.
4. Validate.
5. Import at `/flow` or `POST /flow/api/workflows/import`.

**Shape:** `{name, nodes:[{id,type,position,data}], edges:[{id,source,target,type,sourceHandle?}]}`.

**Webhook values:** `{{webhook.key}}`, `{{webhook.legs[0].qty}}`. Every field on an order node
accepts one. A field that holds exactly one token keeps its type. Enumerated fields are case-insensitive. **An
unresolved reference fails the node**; it never falls back to a default.

**Traps:**
- `strikeOffset` instead of `offset` is silently ignored, so the order goes ATM.
- `source` means the data source (`api` / `db`); the price field is `sourceField`.
- `expiryType` takes a relative value or a `DDMMMYY` date. `expiryDate` wins if both are set.
- A `webhookTrigger` carries no instrument; symbol and exchange come from the request.
- A failed read in `priceCondition`, `positionCheck` or `fundCheck` takes neither branch.
- Gates wait for `inputCount`, not for the wires you drew. A condition evaluates once per run.
- `timeWindow` can cross midnight.
- `unsubscribe` needs a symbol unless `streamType` is `all`. Otherwise it clears every subscription, including the Sandbox engine's.
- `waitUntil` is capped at 30 minutes.
- A `start` schedule is clock-aligned (HH:MM:02). **`marketHoursOnly` defaults to false in imported JSON**, so set it explicitly.
- For 1-minute strategies, set `FLOW_HISTORY_CACHE_TTL` to 2-3 seconds (default 30).
- Use `{{ind.previous.*}}` for the last closed bar; `latest` is the bar still forming and repaints.
- Tuple indicators return `out0` / `out1`. Supertrend direction is inverted: **-1 is the uptrend**.

**Test:** `curl -X POST http://127.0.0.1:5000/flow/webhook/<token> -d '{"secret":...}'`,
then read `flow_workflow_executions.logs`. Check `settings.analyze_mode` first: 1 means Sandbox, 0 means Live.

---

## 5. fd-audit

Production is a single Gunicorn worker that never restarts, so any per-request leak keeps growing. There are two failure modes:
file descriptors ("Too many open files", refused DB connections, dropped sockets) and memory (rising RSS, OOM).

1. **Scope.** If the change touches no resource or memory holder, say so and stop.
2. **Descriptor conventions:**
   - SQLite only through `engine_factory.create_db_engine()` (NullPool, never StaticPool).
   - DuckDB (`historify_db.py`) bypasses that factory, so close each connection explicitly.
   - Every `scoped_session` is registered in teardown or used as `with db_session()`.
   - HTTP only through the shared httpx client, with a timeout.
   - Close WebSocket adapters before reconnecting.
   - ZMQ: no context per call; `cleanup_zmq()` in `disconnect()`.
   - Subprocesses log to a file, not `PIPE`, and are `.wait()`-reaped (watch the kaleido renderer).
   - Executors and threads are module-level singletons. Files open in `with` blocks.
3. **Memory conventions:**
   - Every cache is bounded with both maxsize and TTL. The root CLAUDE.md requires `LockedTTLCache` from `utils/thread_safe_cache.py`, not a bare `TTLCache`.
   - Every registry add has a matching remove, and the remove also runs on the error path.
   - Ask what the key space is: per user is bounded; per symbol, strategy or request is not.
   - Do not keep DataFrames past the request.
   - Known examples to model on: `_STRIKES_CACHE`, `RUNNING_STRATEGIES`, `_POOLED_ADAPTERS`.
4. **Exit paths.** Check release on success, on exception (`finally` or a context manager) and in retry loops. A `continue` in a retry branch that skips the close is the most common real leak.
5. **Measure.** `ls /proc/$PID/fd | wc -l`, `lsof` grouped by type, `ps -o rss=` and `tracemalloc` over 100+ iterations. A flat count is the proof; a plateau is a cache filling up; linear growth is a leak.
6. **Report.** For each leak: file and line, the exit path that misses the release, descriptor or memory, what bounds it, and the production cost. **Ask for approval before fixing.**

---

## 6. verify

1. **Execute the pattern; never just read it.** For example, `redaction_check.py "Access Token obtained: {t}" <secret>` prints LEAKS or redact. Whether a log line leaks depends on the real parameter names: `?token=` is redacted, `?jKey=` is not.
2. **Break the code to validate the test.** Revert the fix and confirm the test goes red. Watch for tautological assertions and vacuous passes. Target the exact guard line, not the first match.
3. **Grep for the sink, not the variable name.** Secrets also leak through URLs, request and response bodies, headers dicts and exception messages (`HTTPStatusError` includes the URL).
4. **Take a baseline before and after.** Ruff counts on HEAD versus the change. `F401` is ignored in this repo, so check for unused imports by grep. On a Windows checkout, Biome reports CRLF errors on every file, so establish that baseline first.
5. **Distinguish "already safe" from "fixed".** Triage scanner false positives: a docstring match, an inline guard the scanner misses, synthetic `0o666` permissions on Windows.
6. **An unrun check is not a pass.** A timeout or a missing tool means the area is unverified.

When you are wrong, correct it in one plain sentence with the evidence.

---

## 7. security-audit

```bash
uv run --with openpyxl python .claude/skills/security-audit/audit.py
```

Writes `tmp/security-audits/<YYYY-MM>/security_audit_<date>_<time>.xlsx`. The run is read-only, and the report contains no
secrets (only `sha256:` fingerprints). Exit code 1 means a CRITICAL or HIGH check failed.

**Coverage:**
- **Secrets:** `APP_KEY` and pepper entropy, the known leaked pre-2.0.0.6 values, credentials tracked in git, detect-secrets.
- **Runtime posture:** debug, CSRF, CSP, CORS, HTTPS, ngrok, rate limits.
- **Routes:** 461 routes checked against `PUBLIC_ROUTE_ALLOWLIST`.
- **Database:** permissions, Fernet encryption, peppered keys, NullPool.
- **Caches:** bounded TTL; the ZMQ publisher must connect, never bind.
- **Backend patterns:** eval, `shell=True`, pickle, SQL f-strings, SSRF, `verify=False`, missing timeouts.
- **Frontend patterns:** `dangerouslySetInnerHTML`, credentials in browser storage, source maps.
- **Dependencies and static analysis:** pip-audit plus npm audit, and triaged bandit output (about 930 raw findings, about 5 that matter).

**Work order:** FAIL, then ERROR (a check that could not run is not a pass), then REVIEW (where the real findings are), then WARN.

**Recurring REVIEW items:**
- SQL f-strings that interpolate table names from constants are safe; trace any new one to its source.
- Direct `create_engine` calls in broker master-contract modules are an FD issue, not a breach.
- The Telegram credential cache has a 30-minute TTL; make sure nothing outlives the 3 AM token rollover.
- Add legitimately public routes to the allowlist with a reason.
- Commit a `.secrets.baseline` so later audits show only new candidates.

**Manual checks:**
- The broker static-IP whitelist still lists only this server.
- Revoke stale API keys.
- Ports 5000, 8765 and 5555 are not reachable from the internet (ZMQ 5555 carries the tick feed with no authentication).
- Webhook URLs are bearer credentials.
- Rotate the pepper only with `upgrade/rotate_pepper.py`.

The threat model is single-user and self-hosted: filesystem, secret and network exposure matter more than IDOR.

---

## 8. version-bump

**Two independent versions:**

| | Files |
| --- | --- |
| **Platform** (e.g. 2.0.3.1) | `utils/version.py` `VERSION`, `pyproject.toml` line 4, `uv sync`. Never the requirements files. |
| **SDK pin** (`openalgo==X.Y.Z`) | `pyproject.toml` dependencies, `requirements.txt`, `requirements-nginx.txt` (the one most often forgotten), `uv sync` |

**Platform release order:**
1. Edit both version files.
2. Run `uv sync`.
3. Collect the commit range (`git log v<prev>..HEAD`) and resolve every contributor's GitHub handle (`gh pr view`, `gh api .../commits/<sha>`). Exclude `github-actions[bot]` and Claude co-author trailers.
4. Write `docs/releases/version-<v>-released.md`: title and date, a bold summary, an overview with the commit count, Highlights citing SHAs and issues, Dependencies, Contributors, Links.
5. Add a `docs/CHANGELOG.md` stanza: `## [v] - YYYY-MM-DD`, then Highlights, the fix sections, Dependencies and Contributors.
6. Verify with `get_version()`.
7. Commit all five paths together.
8. Only if asked: tag, then `gh release create <descriptive-tag> --title "v<v>"`. Tags are descriptive names, not version numbers.

**Contributors go in all three places** (release notes, changelog and GitHub release), formatted
`**@handle (Real Name)** - what, PR #`, maintainer first.

Commit messages: `chore(release): bump platform to X` or `chore(deps): bump openalgo SDK to X`. No emojis, and no commit or push unless asked.

---

## 9. grokbot-setup

For running OpenAlgo on Grok Bot's shared Linux box. Start every task with
`references/openalgo-knowledge.md`, then the specific reference:
`install-and-run.md`, `brokers.md`, `login-routines.md` or `weekly-upgrade.md`.

**Rules:**
1. Knowledge file first.
2. Never guess.
3. When docs conflict, test the installed package.
4. uv only.
5. Never echo secrets; collect them through secure forms and store them `chmod 600`.
6. Sandbox by default; check `analyzerstatus()`.
7. Warn that the box has no static IP.
8. Use trader language.
9. One instance; check ports 5000 and 8765 first.

**Install:** clone into `/workspace/repos`, `uv sync --locked`, `cp .sample.env .env && chmod 600`.
Set `APP_KEY`, `API_KEY_PEPPER` and `FERNET_SALT` before the account exists. **The pepper and the salt must never change afterwards**, or the user is locked out.

**Start headless:** `nohup setsid script -qfec "uv run app.py" app.log >/dev/null 2>&1 &`.
A plain `nohup` start dies because flask_socketio refuses to run without a TTY. Check with `curl /auth/app-info`.

**Setup:**
1. Create the account at `/setup`.
2. Connect the broker at `/broker`.
3. Read the API key at `/apikey` and save it to `/workspace/pyenv/.env`.

**Broker login:**
- **Indian brokers expire at about 3 AM IST.** Log in manually, or set up an automated 8:00 AM IST routine that generates TOTP codes with `pyotp` from the base32 secret.
- **SMS or email OTP brokers** (Nubra, Definedge) cannot be automated.
- **Delta Exchange (crypto)**: set `DISABLE_SESSION_EXPIRY='true'`, connect once, no daily routine.

**Weekly upgrade (Sunday 6 AM IST):**
1. Stop the app.
2. Back up `.env` and all six databases.
3. `git pull --ff-only`, then `uv sync --locked`.
4. `migrate_all.py`.
5. Merge new `.sample.env` keys without changing existing values.
6. Restart, re-login and verify.
7. On any failure, roll back.

Do not use `install/update.sh` on the box.

**Limits:**
- No static IP (rotating Cloudflare egress), so Live orders are rejected under the SEBI mandate. Use a static-IP VPS for Live.
- Everything else works in Sandbox.
- IIFL Capital, Tradejini and TradeSmart may whitelist all API calls, so even login or data may fail from the box.

There is a 12-item verification checklist (permissions, placeholders replaced, ping, a quote, routines, the static-IP warning given).

---

## Cross-cutting rules that recur in every skill

- **Validate with the real tool before handing anything over.** `validate.mjs` (chart and OpenScript), `validate.py` (Flow), `audit.py`, `redaction_check.py`.
- **Generated references must stay in sync with the pinned packages.** Each skill has a generator plus a coverage check that CI runs.
- **Shared helpers are the contract:** `engine_factory`, `get_httpx_client`, `get_logger`, `db_session`, `utils/real_threading`, `LockedTTLCache`.
- **Code must be correct under eventlet (the default), gthread and the dev server.** No asyncio in request code.
- **Trader-facing text:** say Sandbox / analyzer, never "paper"; never "arm" or "armed"; no status codes; no emojis anywhere.
- **Sandbox first.** Analyze mode never exercises broker code, so passing there proves nothing about a broker integration.
