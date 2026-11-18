# Liquidity sweep backtester (NSE basket)

Research only, not financial advice. This folder is a self-contained research tool.
It places no orders and does not change anything else in OpenAlgo.

## What it does

The strategy assumes price hunts the stop losses that cluster beyond obvious levels.
When a support or resistance level has been touched three or more times, stops
collect just beyond it. Price wicks through the level (the "sweep"), takes those
stops, closes back inside and reverses. The strategy trades that reversal, and only
in the direction of the higher-timeframe trend.

The backtester runs this rule bar by bar, separately for each of 17 NSE stocks
(ICICIBANK, HDFCBANK, BSE, BAJFINANCE, M&M, RELIANCE, SBIN, CUMMINSIND, BHARTIARTL,
PIIND, LENSKART, GROWW, TCS, ASHOKLEY, LT, TONBO, MARUTI). It holds one position at
a time per stock and pools the results across the basket. LENSKART, GROWW and TONBO
are recent listings with short histories, so they are flagged and also reported
separately. No parameter is optimised.

The original specification was written for BTC/USDT. The NSE changes are listed
under "NSE adaptations" below.

## Setup (uv)

From the repository root:

```sh
uv venv liquidity_sweep/.venv
uv pip install --python liquidity_sweep/.venv -r liquidity_sweep/requirements.txt
```

`requirements.txt` pins pandas, numpy, matplotlib, requests, duckdb and pytest.
The crypto specification used ccxt to pull data from Binance, Bybit and OKX. Those
exchanges carry no NSE data, so ccxt is not installed or used here.

## Providing data

The backtester needs 12 full months of 5-minute bars ending at the last complete
month. The range is computed from today's date: run on 2026-11-18, it is 2025-11-01
to 2026-10-31. It tries two sources in order for each stock:

1. **Historify** (`db/historify.duckdb` at the repo root, opened read-only). In
   OpenAlgo, open Historify, add the stocks to the watchlist with exchange NSE, and
   download 1 minute data for the range above. Historify stores 1-minute bars, and
   the backtester builds 5-minute bars from them. If OpenAlgo is running and the
   file is busy, stop OpenAlgo for a moment and run again.
2. **The OpenAlgo API** (`POST /api/v1/history` on your own running OpenAlgo). Start
   OpenAlgo and log in to your broker. Copy your key from the API Key page and set
   it in the shell you run from:

   ```sh
   export OPENALGO_API_KEY=your_key_here
   export OPENALGO_HOST=http://127.0.0.1:5000   # optional, this is the default
   ```

   The key is read only from that environment variable. It is never printed,
   logged or written to any file. The backtester asks for interval 5m and falls
   back to 1m bars aggregated to 5m. It requests 60 days at a time for 5m and 25
   days at a time for 1m, which is inside the broker's per-request limits (Angel
   allows 100 and 30), and pauses 0.6 s between requests.

A stock with no data from either source is skipped with a message, and the run
continues with the rest. If no stock has data, both scripts print how to provide
it and exit with status 1. No data is ever simulated for a report.

Each stock is cached as `liquidity_sweep/data/<SYMBOL>_5m.csv` (M&M is saved as
`M_and_M_5m.csv`), indexed in Asia/Kolkata time, sorted, deduplicated, with float
columns. The cache keeps only bars starting from 09:15 up to (not including) 15:30
IST, Monday to Friday. `data/source.txt` records each stock's source, the requested
range, its first and last bar, and the bar count. When the API returns data with
some requests failed, the source column says so. `--refresh` downloads again. A
cache made for a different range (for example, last month's run) is downloaded
again automatically.

## Running

Run these from inside `liquidity_sweep/`:

```sh
uv run --no-project --python .venv python run.py                  # 5m, atr_mult 0
uv run --no-project --python .venv python run.py --timeframe 15m --atr-mult 0.5
uv run --no-project --python .venv python run.py --symbols SBIN,TCS --refresh
uv run --no-project --python .venv python experiments.py
uv run --no-project --python .venv python -m pytest
```

`run.py` runs three variants with no tuning: (a) the baseline, (b) the same without
the bias filter, and (c) the same with min_touches=1. It prints the tables and writes:

- `output/report.md`: in-sample, out-of-sample and full statistics for each variant;
  established stocks against recent listings; a per-stock table (trades, win rate,
  average net R, total R) with recent listings flagged; the data source of each
  stock; and the first entry per variant, which shows the EMA warm-up.
- `output/trades.csv`: variant, symbol, side, entry time and price, stop, target,
  exit time and price, exit reason, r_gross, r_net, level, touches.
- `output/equity.png`: cumulative net R per variant across the basket, with a dashed
  line at the in-sample/out-of-sample split.

`experiments.py` runs 9 configurations: timeframe {5m, 15m, 1h} x atr_mult {0, 0.5,
1.0}, with the bias on and min_touches=3. It writes one table to
`output/experiments.md` and `output/experiments_equity.png`, which has one panel per
timeframe and the split marked. Nine configurations were tested, so the best one is
likely flattered by chance. The out-of-sample results are what matter, and rows with
few trades mean nothing.

The statistics reported for each sample are: trades, win rate (a win is a trade with
net R above 0), average net R, average gross R, total net R, profit factor, maximum
drawdown in R, long and short counts, and a count for each exit reason. The basket
drawdown is measured on all trades ordered by exit time.

## Rules

Every parameter is in `config.py` (one frozen dataclass). The defaults are
pivot_n=5, level_tol=0.0015, min_touches=3, bias EMA 50, stop_buffer=0.0005,
atr_mult=0, atr_period=14, rr=2, max_bars=48, fee=0.0004, slippage=0.0002,
low_fee=0.0002, low_slippage=0 and is_frac=0.7. The session settings are an
09:15 open, a 15:30 close, last entry 15:00 and square-off 15:15. Nothing looks ahead.

1. **Pivots.** A pivot high at bar i is where high[i] is the strict maximum of the
   highs in [i-n, i+n]. A pivot low is the mirror. A pivot is confirmed at bar i+n
   and is used only from bar i+n+1.
2. **Levels.** Live pivot highs (resistance) and pivot lows (support) are clustered
   within level_tol of each other. A level's price is the mean of its pivots, and its
   touches are the number of pivots. Only pivots from the last `level_lookback` bars
   count. A level dies when a bar closes beyond it by more than level_tol, or once it
   has been traded. Its pivots are removed, so new pivots can later form a fresh level.
3. **Signal at bar t.** Short when a live resistance L with at least min_touches
   touches has high[t] > L and close[t] < L. Long is the mirror at support. If several
   levels qualify, the one with the most touches wins, then the most extreme price.
   A bar that sweeps both sides is skipped. On each bar the signal is checked first,
   then the invalidations are applied.
4. **Bias.** The EMA of bias-timeframe closes. Shorts only when the last completed
   bias close is below the EMA, longs only when it is above. No trades until the EMA
   exists.
5. **Entry and exits.** Entry is at the next bar's open, skipped if that open is
   already beyond the stop. The short stop is max(sweep_high x (1 + stop_buffer),
   sweep_high + atr_mult x ATR). The long stop mirrors it with the sweep low. ATR is
   Wilder's ATR(14), seeded with the mean of the first 14 true ranges and computed up
   to and including the sweep bar, so atr_mult=0 gives exactly the plain buffer. The
   target is rr x risk. A trade is closed at the bar's close after max_bars bars. When
   stop and target fall in the same bar, the stop is taken. A bar that gaps past the
   stop fills at its open. A trade still open when the data ends exits as
   `end_of_data`. A new signal may come from the exit bar itself.
6. **Costs.** R_gross = (exit - entry) / risk for longs, with the sign flipped for
   shorts. cost_R = (fee + slippage) x (entry + exit) / risk, and R_net = R_gross -
   cost_R. The low-cost scenario prices the same trades again with low_fee and
   low_slippage.

Trades are assigned to in-sample or out-of-sample by entry time.

## NSE adaptations

- **Session-aware bars.** 15m and 1h bars are built from 5m bars within each session,
  anchored at 09:15 (label left, closed left, first/max/min/last/sum). No bar spans
  two sessions. The 1h bars are 09:15, 10:15, 11:15, 12:15, 13:15, 14:15 and a short
  15:15-15:30 bar. The short bar is a complete session bar and is kept.
- **Partial bars.** A 15m or 1h bar is kept only if it has every 5m bar its span
  needs. That drops an incomplete bar at the start or end of the data, and any bar
  with an internal hole. The same rule builds 5m bars from 1m data, so a 5-minute
  window with a missing minute is dropped. A daily bar is kept when its session has
  both its first and its last 5m bar, so its open and close are the true session open
  and close. Internal holes do not drop a daily bar.
- **Lookback in sessions.** The crypto lookback was 3 days of round-the-clock bars.
  Here it is 3 sessions: 225 bars on 5m, 75 on 15m and 21 on 1h.
- **Bias timeframe.** 5m and 15m use the session-anchored 1h bars. A 1h bar is usable
  from its close (start + 1h, or 15:30 for the short bar). On 1h, a 4h bar does not
  fit a 6h15m session, so the bias is the daily bar (one per session) with an EMA(50)
  of daily closes. A daily bar is usable only after its session has closed, so from
  the next session. The bias bars are built from the same 5m data. No extra warm-up
  data is loaded, so the 1h runs need 50 daily closes first and their trades begin
  around mid-January 2026 for the 2025-11-01 start. On 5m and 15m, EMA(50) of 1h bars
  needs about 8 sessions.
- **Intraday only (MIS-style).** Entry is at the next bar's open only if that bar is
  in the same session, so a signal on a session's last bar is dropped. There is no
  entry on a bar opening at or after 15:00. An open trade is closed at the close of
  the last bar that ends at or before 15:15 IST, with exit reason `session_end`. That
  check comes after stop and target on the same bar, with the stop taken first.
  Pivots and levels persist across sessions. The opening gap is a bar like any other
  for the gap-past-stop rule.
- **Basket.** The engine runs per stock and all tables are pooled across the basket.
  The in-sample/out-of-sample split is one timestamp for every stock and every
  timeframe: 70% of the way through the common 12-month calendar range. For
  2025-11-01 to 2026-10-31 that is 2026-07-14 12:00 IST.
- **Costs.** The defaults are kept: fee 0.0004 + slippage 0.0002 per side, about
  0.12% round trip, which is in line with Indian intraday equity costs at moderate
  size. The low-cost scenario is 0.0002 + 0 per side, about 0.04% round trip, roughly
  the statutory charges alone.

## Choices the specification leaves open

These are stated so they can be checked, not hidden:

- **Clustering.** Pivots are sorted by price and grouped greedily. A pivot joins the
  current cluster while it is within level_tol of the cluster's lowest member, so
  every pair inside a cluster is within level_tol.
- **Lookback.** A pivot at bar i counts at bar t while t - i <= level_lookback.
- **Invalidation.** Breaks apply to every cluster, including those below min_touches.
  A level counts as traded when the entry is actually filled. If the entry is skipped
  because the open is already beyond the stop, the level stays alive.
- **Sweeping both sides.** A bar is skipped when it meets the full sweep condition on
  a qualifying resistance and on a qualifying support. This is checked before the
  bias filter.
- **Bias.** A bias bar counts for a trading bar when it became usable at or before
  that bar's start. The EMA is seeded with the simple mean of its first 50 closes, so
  it exists once 50 closes are in.
- **ATR.** The first true range is high - low. True ranges include the overnight gap.
  With atr_mult above 0, a signal is skipped while ATR does not exist yet (the first
  13 bars of a stock's data; ATR exists from the 14th bar).
- **Exits.** The entry bar counts as bar 1 of max_bars. If the time exit and the
  session square-off fall on the same bar, the reason is `time`; the price is the
  same close. A target is filled at the target price even when a bar opens beyond it;
  only the stop has a gap rule. With the 15:15 square-off, max_bars=48 can bind only
  on 5m (a 15m session has 25 bars and a 1h session has 7).
- **Missing square-off bar.** If the bar ending at 15:15 is missing from the data,
  the trade is closed at the open of the next bar of that session. If the rest of the
  session is missing, it is closed at the close of the session's last bar. Either way
  the reason is `session_end`.
- **Times in trades.csv** are bar start labels. A close-price exit happens at the end
  of the labelled bar.
- **Special sessions.** Weekend sessions (such as a Budget-day Sunday) and anything
  outside 09:15-15:30 are excluded by the session filter.
- **Historify coverage.** If Historify has any rows for a stock, it is used even when
  its coverage is shorter than the range. `data/source.txt` and the data table in the
  report show each stock's first and last bar.
- **Exit status.** With no data at all, the scripts exit with status 1, with a
  message and no traceback.

## Tests

`tests/` uses small synthetic data only, with no network and no real data. It covers
pivots, levels, signals, bias, resampling, ATR, exits, the intraday rules, costs,
prefix invariance, a saved regression fixture, data loading from a temporary
Historify-shaped duckdb file and from a mocked API, and the scripts themselves. To
regenerate the regression fixture after an intended rule change:

```sh
uv run --no-project --python .venv python tests/regression_fixture.py --write
```

Research only, not financial advice.
