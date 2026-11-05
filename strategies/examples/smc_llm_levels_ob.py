#!/usr/bin/env python
"""
SMC Liquidity Levels and Order Blocks, planned by an LLM

The level and order-block logic is a Python port of "Shaakuni - Liquidity Levels
& Order Blocks" v1.4.0 (Pine Script, (c) Shaakuni), which is licensed under the
Mozilla Public License 2.0 (https://mozilla.org/MPL/2.0/). This file is a
derivative of that work and is distributed under the same licence.

How it trades
  1. Code computes, on closed 5 minute bars:
       - previous day high/low (PDH/PDL) and its 50% (D-EQ)
       - previous week high/low (PWH/PWL) and its 50% (W-EQ)
       - order blocks with the indicator's swing-break + ATR-impulse rule,
         confirmed on bar close (no repaint)
  2. Trigger: the last closed bar is the FIRST return of price into an
     unmitigated order block (bull OB -> long candidate, bear OB -> short).
  3. The LLM gets the levels, zones, recent candles, the 15m and 1h trend, the
     index and how its recent decisions turned out, and answers with a JSON
     plan: TAKE or SKIP, entry (MARKET or LIMIT), stop loss, target, reason.
  4. Code validates the plan (side, stop side, stop size in ATR, reward:risk,
     confidence) and sizes the quantity from a fixed rupee risk.
  5. Code places the entry, a stop-loss order at the broker, watches the
     target, and squares off before the close.
  6. While a trade is open the LLM reviews it every few bars and may answer
     HOLD, TIGHTEN_STOP or EXIT. It can only reduce risk: code refuses a stop
     that moves away from price, and the target and quantity never change.

Every trigger is also tracked as a "shadow" trade with a fixed mechanical plan
(stop beyond the block, target at MIN_RR), whether the LLM took it or not. Its
result shows whether the LLM's skips are saving you money or costing you
winners. Print the comparison with:

    uv run python strategies/examples/smc_llm_levels_ob.py --report

Every LLM prompt, reply, trade and shadow result is written to a journal (JSON
lines) in smc_llm_data/, so you can review what the model decided and why.

Setup
  - Put your LLM key in OpenAlgo's .env (strategies inherit it), one per line
    and nothing after the closing quote:
      ANTHROPIC_API_KEY = 'sk-ant-...'
    The default model is Claude Sonnet. For another provider set the matching
    key and SMC_LLM_MODEL, for example gemini/gemini-flash-latest.
  - Edit the CONFIG block below, or override any value with an SMC_* entry in .env.
    For a bank stock set SMC_INDEX_SYMBOL = 'BANKNIFTY'.
  - Upload at /python, exchange NSE, schedule 09:15 - 15:20, Mon-Fri.
  - Run it in analyzer (sandbox) mode first. The log says which mode is active.
"""

import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openalgo import api

# litellm otherwise downloads its model price map from GitHub on import
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _env(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    value = value.strip()
    if isinstance(default, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return value


# =============================================================================
# CONFIG
# =============================================================================
STRATEGY_NAME = _env("SMC_STRATEGY_NAME", "SMC LLM Levels OB")
SYMBOL = _env("SMC_SYMBOL", "HDFCBANK")
EXCHANGE = os.getenv("OPENALGO_STRATEGY_EXCHANGE") or _env("SMC_EXCHANGE", "NSE")
PRODUCT = _env("SMC_PRODUCT", "MIS")
INTERVAL = "5m"
BAR_MINUTES = 5

# Risk
RISK_PER_TRADE_RS = _env("SMC_RISK_PER_TRADE", 500.0)  # rupees lost if the stop hits
MAX_QTY = _env("SMC_MAX_QTY", 100)
MAX_TRADES_PER_DAY = _env("SMC_MAX_TRADES_PER_DAY", 2)
MAX_DAILY_LOSS_RS = _env("SMC_MAX_DAILY_LOSS", 1000.0)
MIN_RR = _env("SMC_MIN_RR", 1.5)
MIN_STOP_ATR = _env("SMC_MIN_STOP_ATR", 0.3)  # stop distance bounds, in ATR(14) of 5m bars
MAX_STOP_ATR = _env("SMC_MAX_STOP_ATR", 3.0)
MAX_LIMIT_DISTANCE_ATR = _env("SMC_MAX_LIMIT_DISTANCE_ATR", 1.0)
LIMIT_ENTRY_EXPIRY_MIN = _env("SMC_LIMIT_EXPIRY_MIN", 15)
MIN_CONFIDENCE = _env("SMC_MIN_CONFIDENCE", 60)

# Session (IST)
ENTRY_START = _env("SMC_ENTRY_START", "09:30")
LAST_ENTRY = _env("SMC_LAST_ENTRY", "14:30")
SQUARE_OFF = _env("SMC_SQUARE_OFF", "15:10")

# Order block detector (indicator defaults)
ATR_PERIOD = 14
OB_SWING_LEN = _env("SMC_OB_SWING_LEN", 10)
OB_IMPULSE_ATR = _env("SMC_OB_IMPULSE_ATR", 0.7)
OB_MIN_HOFF = 3
ZONE_LOOKBACK_BARS = 1500  # the indicator's "Auto" lookback for a 5m chart

# LLM
LLM_MODEL = _env("SMC_LLM_MODEL", "anthropic/claude-sonnet-5-5")
LLM_TIMEOUT_SEC = _env("SMC_LLM_TIMEOUT", 60)
MAX_LLM_CALLS_PER_DAY = _env("SMC_MAX_LLM_CALLS", 20)
CANDLES_FOR_LLM = 40
RECENT_RESULTS_FOR_LLM = 5

# Market context given to the LLM
INDEX_SYMBOL = _env("SMC_INDEX_SYMBOL", "NIFTY")  # BANKNIFTY for a bank stock
INDEX_EXCHANGE = _env("SMC_INDEX_EXCHANGE", "NSE_INDEX")

# Open-trade review: the LLM may hold, tighten the stop, or exit early
REVIEW_ENABLED = _env("SMC_REVIEW_ENABLED", True)
REVIEW_EVERY_BARS = _env("SMC_REVIEW_EVERY_BARS", 3)
MAX_REVIEW_CALLS_PER_DAY = _env("SMC_MAX_REVIEW_CALLS", 30)
REVIEW_MIN_STOP_GAP_ATR = 0.2  # a tightened stop stays at least this far from ltp

# Shadow tracking: the mechanical plan every trigger is scored against
SHADOW_STOP_BUFFER_ATR = 0.1
SHADOW_RR = MIN_RR
HISTORY_KEEP = 50

# Backtest (/python Backtest button). The LLM is not called by default, since a
# backtest asks it about every trigger in the period: each valid trigger is
# taken with the mechanical plan instead. Set SMC_BACKTEST_LLM = 'true' to pay
# for the real planner and reviewer.
BACKTEST = os.getenv("OPENALGO_BACKTEST") == "1"
BACKTEST_LLM = _env("SMC_BACKTEST_LLM", False)

# Plumbing
POLL_SECONDS = 10
TELEGRAM_USER = _env("SMC_TELEGRAM_USER", "")  # your OpenAlgo login name, optional

API_KEY = os.getenv("OPENALGO_API_KEY")
HOST = os.getenv("HOST_SERVER") or os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL") or (
    f"ws://{os.getenv('WEBSOCKET_HOST', '127.0.0.1')}:{os.getenv('WEBSOCKET_PORT', '8765')}"
)

IST = timezone(timedelta(hours=5, minutes=30))
DATA_DIR = Path(__file__).resolve().parent / "smc_llm_data"


def now_ist():
    return datetime.now(IST).replace(tzinfo=None)


def hhmm(text):
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def minutes_of(dt):
    return dt.hour * 60 + dt.minute


def log(msg):
    print(f"[{now_ist():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# =============================================================================
# INDICATOR PORT
# =============================================================================
def atr_rma(high, low, close, length=ATR_PERIOD):
    """ta.atr: RMA of true range, seeded with the SMA of the first `length` values."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    out = np.full(n, np.nan)
    if n >= length:
        out[length - 1] = tr[:length].mean()
        for i in range(length, n):
            out[i] = (out[i - 1] * (length - 1) + tr[i]) / length
    return out


@dataclass
class Zone:
    kind: str  # "bull" (demand) or "bear" (supply)
    top: float
    bottom: float
    ob_idx: int  # bar of the order-block candle
    created_idx: int  # bar on which it was confirmed
    mitigated_idx: int = -1  # first bar that traded back into it, -1 while fresh


def detect_order_blocks(o, h, lo, c, atr, swing_len=OB_SWING_LEN, imp_mult=OB_IMPULSE_ATR):
    """Bar-by-bar port of the indicator's ORDER BLOCKS / POI section.

    Index arithmetic mirrors Pine: x[k] on bar t is x[t - k].
    """
    zones = []
    last_bull = last_bear = -1
    for t in range(len(c)):
        if not np.isnan(atr[t]) and t > swing_len + 2:
            a = atr[t]

            # Bullish OB: close breaks the highest high of the last swing_len bars
            hh, h_off = h[t - 1], 1
            for i in range(2, swing_len + 1):
                if h[t - i] > hh:
                    hh, h_off = h[t - i], i
            if c[t] > hh and h_off >= OB_MIN_HOFF:
                mn, mn_off = lo[t - 1], 1
                for i in range(2, h_off):
                    if lo[t - i] < mn:
                        mn, mn_off = lo[t - i], i
                d_off = -1
                for k in range(mn_off):
                    off = mn_off - 1 - k
                    if c[t - off] > o[t - off] and (c[t - off] - o[t - off]) >= a * imp_mult:
                        d_off = off
                        break
                if d_off >= 0:
                    ob_off = mn_off
                    for j in range(d_off + 1, mn_off + 1):
                        if c[t - j] < o[t - j]:
                            ob_off = j
                            break
                    fresh = True
                    for k in range(0, ob_off - 1):
                        if lo[t - k] <= h[t - ob_off]:
                            fresh = False
                            break
                    ob_bar = t - ob_off
                    if fresh and ob_bar != last_bull:
                        last_bull = ob_bar
                        zones.append(Zone("bull", h[ob_bar], lo[ob_bar], ob_bar, t))

            # Bearish OB: close breaks the lowest low of the last swing_len bars
            ll, l_off = lo[t - 1], 1
            for i in range(2, swing_len + 1):
                if lo[t - i] < ll:
                    ll, l_off = lo[t - i], i
            if c[t] < ll and l_off >= OB_MIN_HOFF:
                mx, mx_off = h[t - 1], 1
                for i in range(2, l_off):
                    if h[t - i] > mx:
                        mx, mx_off = h[t - i], i
                d_off = -1
                for k in range(mx_off):
                    off = mx_off - 1 - k
                    if c[t - off] < o[t - off] and (o[t - off] - c[t - off]) >= a * imp_mult:
                        d_off = off
                        break
                if d_off >= 0:
                    ob_off = mx_off
                    for j in range(d_off + 1, mx_off + 1):
                        if c[t - j] > o[t - j]:
                            ob_off = j
                            break
                    fresh = True
                    for k in range(0, ob_off - 1):
                        if h[t - k] >= lo[t - ob_off]:
                            fresh = False
                            break
                    ob_bar = t - ob_off
                    if fresh and ob_bar != last_bear:
                        last_bear = ob_bar
                        zones.append(Zone("bear", h[ob_bar], lo[ob_bar], ob_bar, t))

        # Mitigation: first bar after confirmation that trades into the zone
        for z in zones:
            if t > z.created_idx and z.mitigated_idx < 0 and h[t] >= z.bottom and lo[t] <= z.top:
                z.mitigated_idx = t

        # Drop zones older than the lookback
        cutoff = t - ZONE_LOOKBACK_BARS
        zones = [z for z in zones if z.created_idx >= cutoff]
    return zones


def compute_levels(daily, today_bars, ltp, today):
    """PDH/PDL/D-EQ and PWH/PWL/W-EQ, plus sweep/break status against today's range.

    Status follows the indicator's panel: "broken" while price is beyond the
    level, "swept" if today traded beyond it and came back, else "intact".
    """
    lv = {}
    past = daily[daily.index.date < today]
    if not past.empty:
        lv["PDH"] = float(past["high"].iloc[-1])
        lv["PDL"] = float(past["low"].iloc[-1])
        lv["D_EQ"] = (lv["PDH"] + lv["PDL"]) / 2

    iso = daily.index.isocalendar()
    week_key = list(zip(iso["year"], iso["week"], strict=True))
    this_week = tuple(pd.Timestamp(today).isocalendar())[:2]
    weeks = sorted({k for k in week_key if k < this_week})
    if weeks:
        mask = [k == weeks[-1] for k in week_key]
        wk = daily[mask]
        lv["PWH"] = float(wk["high"].max())
        lv["PWL"] = float(wk["low"].min())
        lv["W_EQ"] = (lv["PWH"] + lv["PWL"]) / 2

    if not today_bars.empty:
        day_high = max(float(today_bars["high"].max()), ltp)
        day_low = min(float(today_bars["low"].min()), ltp)
    else:
        day_high = day_low = ltp
    lv["today_high"], lv["today_low"] = day_high, day_low

    status = {}
    for name in ("PDH", "PWH"):
        if name in lv:
            status[name] = (
                "broken" if ltp > lv[name] else "swept" if day_high > lv[name] else "intact"
            )
    for name in ("PDL", "PWL"):
        if name in lv:
            status[name] = (
                "broken" if ltp < lv[name] else "swept" if day_low < lv[name] else "intact"
            )
    lv["status"] = status
    return lv


def session_phase(dt):
    m = minutes_of(dt)
    if m < hhmm("10:00"):
        return "opening hour"
    if m < hhmm("13:30"):
        return "mid session"
    return "last two hours"


def trend_view(bars, rule):
    """EMA 20/50 trend of 5m bars resampled to `rule` ("15min", "60min")."""
    # offset aligns the hourly bars to the 09:15 open
    closes = bars["close"].resample(rule, offset="15min").last().dropna()
    if len(closes) < 50:
        return None
    ema20 = closes.ewm(span=20, adjust=False).mean()
    ema50 = closes.ewm(span=50, adjust=False).mean()
    close, fast, slow = float(closes.iloc[-1]), float(ema20.iloc[-1]), float(ema50.iloc[-1])
    if close > fast > slow:
        label = "up"
    elif close < fast < slow:
        label = "down"
    else:
        label = "mixed"
    return {
        "trend": label,
        "close": round(close, 2),
        "ema20": round(fast, 2),
        "ema50": round(slow, 2),
        "ema20_slope_5_bars": round(fast - float(ema20.iloc[-6]), 2),
    }


# =============================================================================
# LLM PLANNER
# =============================================================================
SYSTEM_PROMPT = """You are the trade planner for an intraday Smart Money Concepts strategy on Indian markets (NSE, 5 minute bars, positions closed by {square_off} IST).

A rule-based detector has just found that the last closed bar is the FIRST return of price into an unmitigated order block. Decide whether to take that trade, and if so plan the entry, stop loss and target using only the data given.

What the data means
- PDH / PDL: previous day's high and low; D_EQ is their 50% level. PWH / PWL / W_EQ are the same for the previous week.
- status: "swept" = price traded beyond the level today and came back (a liquidity grab), "broken" = price is beyond it now, "intact" = not reached.
- Bull order block = demand zone: the last bearish candle before an ATR-sized impulsive up move that broke the recent swing high. Bear order block = the supply mirror.
- Premium / discount: above / below the D_EQ and W_EQ.
- trend_15m / trend_1h: EMA 20/50 trend of this stock on those timeframes. index: the same for the market index, plus its change today.
- gap_pct: today's open against the previous close.
- recent_results: the latest triggers, what you decided, and the R multiple a fixed mechanical plan made on each (stop beyond the block, target at {min_rr}R). Use it to notice when the market is punishing or rewarding this setup today, not to chase the last result.

How to judge
- Favour longs from bull blocks in discount, especially after a PDL or PWL sweep; shorts from bear blocks in premium, especially after a PDH or PWH sweep.
- Put the stop beyond the far edge of the order block with a small buffer, never inside the block.
- Put the target at the next opposing liquidity (PDH/PDL, PWH/PWL, D_EQ/W_EQ, today's high/low, an opposing fresh block) that still gives reward:risk >= {min_rr}.
- Skip when the trigger bar closed beyond the block's far edge, when opposing liquidity is too close for {min_rr}R, when the move is against both day and week context, when it fights both the 1h trend and the index, or when too little session is left to reach the target. SKIP is a normal answer; most triggers should be skipped.

Hard limits (a plan breaking any of them is rejected)
- side must be one of allowed_sides.
- BUY: stop_loss < entry < target. SELL: target < entry < stop_loss.
- Stop distance from entry between {min_stop} and {max_stop} x ATR.
- reward:risk >= {min_rr}.
- LIMIT entry: BUY at or below ltp, SELL at or above ltp, within {max_limit} x ATR of ltp. A LIMIT not filled within {expiry} minutes is cancelled. For MARKET, entry is ltp.

Reply with one JSON object and nothing else:
{{"decision": "TAKE" or "SKIP", "side": "BUY" or "SELL", "entry_type": "MARKET" or "LIMIT", "entry": number, "stop_loss": number, "target": number, "confidence": 0-100, "reason": "one or two sentences"}}
For SKIP, the price fields may be 0."""


def build_system_prompt():
    return SYSTEM_PROMPT.format(
        square_off=SQUARE_OFF,
        min_rr=MIN_RR,
        min_stop=MIN_STOP_ATR,
        max_stop=MAX_STOP_ATR,
        max_limit=MAX_LIMIT_DISTANCE_ATR,
        expiry=LIMIT_ENTRY_EXPIRY_MIN,
    )


REVIEW_PROMPT = """You are managing an open intraday trade from a Smart Money Concepts order-block strategy on Indian markets (NSE, 5 minute bars). Positions are closed automatically at {square_off} IST.

You get the trade (side, entry, current and initial stop, target, the reason it was taken, open R multiple), the market now (ltp, ATR, levels, candles, trend, index) and the minutes left. Decide one of:
- HOLD: the reason for the trade still stands. This is the normal answer; do not micro-manage noise.
- TIGHTEN_STOP: lock in progress or cut risk, for example to breakeven once the trade is about 1R in profit, or behind a fresh higher low (BUY) / lower high (SELL).
- EXIT: close now at market, only when the reason for the trade is clearly invalidated (a strong close back through the order block against the trade, a sweep and rejection at the target side, the index turning hard against it), or when the target is no longer reachable in the time left.

Hard limits (code enforces them)
- You can only reduce risk. BUY: new_stop above the current stop. SELL: new_stop below the current stop.
- new_stop at least {gap} x ATR away from ltp, on the losing side of ltp.
- The target and quantity never change.

Reply with one JSON object and nothing else:
{{"action": "HOLD" or "TIGHTEN_STOP" or "EXIT", "new_stop": number, "reason": "one sentence"}}
For HOLD and EXIT, new_stop may be 0."""


def build_review_prompt():
    return REVIEW_PROMPT.format(square_off=SQUARE_OFF, gap=REVIEW_MIN_STOP_GAP_ATR)


def _complete(system, context):
    """Return (parsed dict or None, raw text)."""
    import litellm

    extra = {}
    if LLM_MODEL.startswith("gemini/"):
        extra["response_format"] = {"type": "json_object"}
    # Thinking models spend output tokens on reasoning before the JSON, so the
    # cap is well above the size of the reply itself.
    response = litellm.completion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
        ],
        max_tokens=8000,
        timeout=LLM_TIMEOUT_SEC,
        **extra,
    )
    text = response.choices[0].message.content or ""
    return parse_json_object(text), text


def ask_llm(context):
    return _complete(build_system_prompt(), context)


def ask_reviewer(context):
    return _complete(build_review_prompt(), context)


def mechanical_planner(context):
    """Backtest planner without an LLM: TAKE the first trigger the mechanical plan accepts."""
    ltp, atr = float(context["ltp"]), float(context["atr14_5m"])
    for view in context.get("trigger", []):
        kind = "bull" if view["type"].startswith("bull") else "bear"
        base = baseline_plan(Zone(kind, view["top"], view["bottom"], -1, -1), ltp, atr)
        if base is None:
            continue
        raw = {
            "decision": "TAKE",
            "side": base["side"],
            "entry_type": "MARKET",
            "entry": ltp,
            "stop_loss": base["stop"],
            "target": base["target"],
            "confidence": 100,
            "reason": "mechanical (backtest)",
        }
        return raw, json.dumps(raw)
    raw = {"decision": "SKIP", "reason": "mechanical (backtest): price closed beyond the block"}
    return raw, json.dumps(raw)


def mechanical_reviewer(context):
    """Backtest reviewer without an LLM: always HOLD."""
    raw = {"action": "HOLD", "new_stop": 0, "reason": "mechanical (backtest)"}
    return raw, json.dumps(raw)


def parse_json_object(text):
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# =============================================================================
# PLAN VALIDATION
# =============================================================================
@dataclass
class Plan:
    side: str
    entry_type: str
    entry: float
    stop: float
    target: float
    qty: int
    rr: float
    confidence: float
    reason: str


def round_tick(price, tick):
    return round(round(price / tick) * tick, 4)


def validate_plan(raw, allowed_sides, ltp, atr, tick, lot):
    """Return (Plan, "") or (None, why it was rejected)."""
    if not isinstance(raw, dict):
        return None, "reply was not a JSON object"
    if str(raw.get("decision", "")).upper() != "TAKE":
        return None, "model skipped: " + str(raw.get("reason", ""))[:300]

    side = str(raw.get("side", "")).upper()
    entry_type = str(raw.get("entry_type", "MARKET")).upper()
    if side not in allowed_sides:
        return None, f"side {side!r} not in {allowed_sides}"
    if entry_type not in ("MARKET", "LIMIT"):
        return None, f"entry_type {entry_type!r} is not MARKET or LIMIT"
    try:
        entry = float(raw.get("entry") or 0)
        stop = float(raw["stop_loss"])
        target = float(raw["target"])
        confidence = float(raw.get("confidence") or 0)
    except (KeyError, TypeError, ValueError):
        return None, "missing or non-numeric price fields"
    if not all(math.isfinite(x) and x > 0 for x in (stop, target)):
        return None, "stop or target is not a positive number"

    if entry_type == "MARKET":
        entry = ltp
    else:
        if not (math.isfinite(entry) and entry > 0):
            return None, "LIMIT entry is not a positive number"
        if side == "BUY" and entry > ltp:
            return None, "BUY LIMIT above ltp"
        if side == "SELL" and entry < ltp:
            return None, "SELL LIMIT below ltp"
        if abs(entry - ltp) > MAX_LIMIT_DISTANCE_ATR * atr:
            return None, "LIMIT entry too far from ltp"

    entry, stop, target = round_tick(entry, tick), round_tick(stop, tick), round_tick(target, tick)
    if side == "BUY" and not (stop < entry < target):
        return None, f"BUY needs stop < entry < target, got {stop} / {entry} / {target}"
    if side == "SELL" and not (target < entry < stop):
        return None, f"SELL needs target < entry < stop, got {target} / {entry} / {stop}"

    risk, reward = abs(entry - stop), abs(target - entry)
    if risk < MIN_STOP_ATR * atr:
        return None, f"stop {risk:.2f} tighter than {MIN_STOP_ATR} ATR ({MIN_STOP_ATR * atr:.2f})"
    if risk > MAX_STOP_ATR * atr:
        return None, f"stop {risk:.2f} wider than {MAX_STOP_ATR} ATR ({MAX_STOP_ATR * atr:.2f})"
    rr = reward / risk
    if rr < MIN_RR:
        return None, f"reward:risk {rr:.2f} below {MIN_RR}"
    if confidence < MIN_CONFIDENCE:
        return None, f"confidence {confidence:.0f} below {MIN_CONFIDENCE}"

    qty = min(int(RISK_PER_TRADE_RS // risk), MAX_QTY)
    qty = (qty // lot) * lot
    if qty <= 0:
        return (
            None,
            f"risk per unit {risk:.2f} too large for Rs {RISK_PER_TRADE_RS:.0f} at lot size {lot}",
        )
    return Plan(
        side,
        entry_type,
        entry,
        stop,
        target,
        qty,
        round(rr, 2),
        confidence,
        str(raw.get("reason", ""))[:500],
    ), ""


def validate_review(raw, trade, ltp, atr, tick):
    """Return (action, new_stop, why). action is HOLD, TIGHTEN_STOP or EXIT.

    Anything malformed or risk-increasing becomes HOLD, so a bad reply leaves
    the trade exactly as it was.
    """
    if not isinstance(raw, dict):
        return "HOLD", None, "reply was not a JSON object"
    action = str(raw.get("action", "")).upper()
    reason = str(raw.get("reason", ""))[:300]
    if action == "EXIT":
        return "EXIT", None, reason
    if action != "TIGHTEN_STOP":
        return "HOLD", None, reason
    try:
        new_stop = round_tick(float(raw["new_stop"]), tick)
    except (KeyError, TypeError, ValueError):
        return "HOLD", None, "TIGHTEN_STOP without a numeric new_stop"
    if not (math.isfinite(new_stop) and new_stop > 0):
        return "HOLD", None, "new_stop is not a positive number"
    gap = REVIEW_MIN_STOP_GAP_ATR * atr
    if trade["side"] == "BUY":
        if new_stop <= trade["stop"]:
            return "HOLD", None, f"new stop {new_stop} does not tighten {trade['stop']}"
        if new_stop > ltp - gap:
            return "HOLD", None, f"new stop {new_stop} too close to ltp {ltp}"
    else:
        if new_stop >= trade["stop"]:
            return "HOLD", None, f"new stop {new_stop} does not tighten {trade['stop']}"
        if new_stop < ltp + gap:
            return "HOLD", None, f"new stop {new_stop} too close to ltp {ltp}"
    return "TIGHTEN_STOP", new_stop, reason


def baseline_plan(zone, ltp, atr):
    """The fixed mechanical plan a trigger is scored with, LLM or not."""
    buffer = SHADOW_STOP_BUFFER_ATR * atr
    if zone.kind == "bull":
        stop = zone.bottom - buffer
        risk = ltp - stop
        side, target = "BUY", ltp + SHADOW_RR * risk
    else:
        stop = zone.top + buffer
        risk = stop - ltp
        side, target = "SELL", ltp - SHADOW_RR * risk
    if not (math.isfinite(risk) and risk > 0):
        return None  # price already closed beyond the block's far edge
    return {"side": side, "entry": ltp, "stop": stop, "target": target, "risk": risk}


def score_bar(shadow, bar_high, bar_low):
    """R multiple if this bar settles the shadow, else None. A bar touching
    both stop and target counts as the stop, the conservative reading."""
    if shadow["side"] == "BUY":
        if bar_low <= shadow["stop"]:
            return -1.0
        if bar_high >= shadow["target"]:
            return SHADOW_RR
    else:
        if bar_high >= shadow["stop"]:
            return -1.0
        if bar_low <= shadow["target"]:
            return SHADOW_RR
    return None


# =============================================================================
# STRATEGY
# =============================================================================
class SmcLlmStrategy:
    def __init__(
        self, client, planner=ask_llm, reviewer=ask_reviewer, clock=now_ist, data_dir=DATA_DIR
    ):
        self.client = client
        self.planner = planner
        self.reviewer = reviewer
        self.clock = clock
        data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = data_dir / f"state_{SYMBOL}.json"
        self.journal_path = data_dir / f"journal_{SYMBOL}.jsonl"
        self.tick, self.lot = 0.05, 1
        self.daily = None
        self.daily_date = None
        self.last_bar_done = None
        self.state = self._load_state()

    # ---------- persistence ----------
    def _fresh_day(self, day, previous=None):
        previous = previous or {}
        return {
            "date": day,
            "trades": 0,
            "pnl": 0.0,
            "llm_calls": 0,
            "review_calls": 0,
            "open": previous.get("open"),  # verified against the broker in reconcile()
            "pending": None,
            "shadows": [],  # unsettled shadows end with the day, as the trade would have
            "history": previous.get("history", []),
        }

    def _load_state(self):
        day = self.clock().date().isoformat()
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return self._fresh_day(day)
        if state.get("date") != day:
            return self._fresh_day(day, state)
        for key, value in self._fresh_day(day).items():
            state.setdefault(key, value)  # a state file from before a key existed
        return state

    def _save_state(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(self.state_path)

    def journal(self, event, **data):
        row = {
            "ts": self.clock().isoformat(timespec="seconds"),
            "event": event,
            "symbol": SYMBOL,
            **data,
        }
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def notify(self, text):
        log(text)
        if TELEGRAM_USER:
            self.call(
                "telegram", username=TELEGRAM_USER, message=f"{STRATEGY_NAME} {SYMBOL}: {text}"
            )

    # ---------- broker helpers ----------
    def call(self, method, **kwargs):
        try:
            resp = getattr(self.client, method)(**kwargs)
        except Exception as e:
            log(f"{method} failed: {e}")
            return None
        if isinstance(resp, dict) and resp.get("status") == "error":
            log(f"{method} error: {resp.get('message')}")
            return None
        return resp

    def ltp(self):
        resp = self.call("quotes", symbol=SYMBOL, exchange=EXCHANGE)
        try:
            return float(resp["data"]["ltp"])
        except (TypeError, KeyError, ValueError):
            return None

    def position_qty(self):
        resp = self.call(
            "openposition",
            strategy=STRATEGY_NAME,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            product=PRODUCT,
        )
        try:
            return int(float(resp["quantity"]))
        except (TypeError, KeyError, ValueError):
            return None

    def order_status(self, order_id):
        resp = self.call("orderstatus", order_id=order_id, strategy=STRATEGY_NAME)
        try:
            data = resp["data"]
            return str(data.get("order_status", "")).lower(), float(data.get("average_price") or 0)
        except (TypeError, KeyError, ValueError):
            return None, 0.0

    def place(self, action, qty, price_type="MARKET", price=0, trigger_price=0):
        resp = self.call(
            "placeorder",
            strategy=STRATEGY_NAME,
            symbol=SYMBOL,
            action=action,
            exchange=EXCHANGE,
            price_type=price_type,
            product=PRODUCT,
            quantity=qty,
            price=price,
            trigger_price=trigger_price,
        )
        return (resp or {}).get("orderid")

    def wait_final(self, order_id, timeout=30):
        deadline = time.monotonic() + timeout
        status, avg = None, 0.0
        while time.monotonic() < deadline:
            status, avg = self.order_status(order_id)
            if status in ("complete", "rejected", "cancelled"):
                break
            time.sleep(2)
        return status, avg

    # ---------- startup ----------
    def setup(self):
        info = self.call("symbol", symbol=SYMBOL, exchange=EXCHANGE)
        if info and isinstance(info.get("data"), dict):
            self.tick = float(info["data"].get("tick_size") or 0.05)
            self.lot = int(float(info["data"].get("lotsize") or 1))
        mode = self.call("analyzerstatus")
        mode_name = ((mode or {}).get("data") or {}).get("mode", "unknown")
        log(
            f"{STRATEGY_NAME} on {SYMBOL} {EXCHANGE} {PRODUCT}, tick {self.tick}, lot {self.lot}, model {LLM_MODEL}"
        )
        log(f"OpenAlgo mode: {mode_name}  (analyze = sandbox orders, live = real orders)")
        self.reconcile()

    def reconcile(self):
        """Make the saved state agree with the broker's position."""
        qty = self.position_qty()
        if qty is None:
            return
        trade = self.state.get("open")
        if trade and qty == 0:
            self.notify(
                "saved trade is no longer open at the broker; clearing it (P&L not recorded)"
            )
            self.state["open"] = None
        elif trade:
            trade["qty"] = abs(qty)
            log(f"resuming management of open {trade['side']} {abs(qty)} @ {trade['entry_price']}")
        elif qty != 0 and not self.state.get("pending"):
            log(
                f"a position of {qty} exists that this strategy did not open; no new entries until it is flat"
            )
        self._save_state()

    # ---------- market data ----------
    def load_bars(self, now):
        if self.daily_date != now.date():
            start = (now - timedelta(days=45)).strftime("%Y-%m-%d")
            daily = self.call(
                "history",
                symbol=SYMBOL,
                exchange=EXCHANGE,
                interval="D",
                start_date=start,
                end_date=now.strftime("%Y-%m-%d"),
            )
            if isinstance(daily, pd.DataFrame) and not daily.empty:
                self.daily = self._naive(daily)
                self.daily_date = now.date()
        start = (now - timedelta(days=12)).strftime("%Y-%m-%d")
        bars = self.call(
            "history",
            symbol=SYMBOL,
            exchange=EXCHANGE,
            interval=INTERVAL,
            start_date=start,
            end_date=now.strftime("%Y-%m-%d"),
        )
        if not isinstance(bars, pd.DataFrame) or bars.empty or self.daily is None:
            return None
        bars = self._naive(bars)
        forming_start = now.replace(second=0, microsecond=0) - timedelta(
            minutes=now.minute % BAR_MINUTES
        )
        return bars[bars.index < forming_start]

    @staticmethod
    def _naive(df):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        return df.sort_index()[["open", "high", "low", "close", "volume"]].astype(float)

    # ---------- main cycle ----------
    def step(self):
        now = self.clock()
        if self.state["date"] != now.date().isoformat():
            self.state = self._fresh_day(now.date().isoformat(), self.state)
        if self.state.get("pending"):
            self.manage_pending(now)
        if self.state.get("open"):
            self.manage_open(now)

        closed_bar = now.replace(second=0, microsecond=0) - timedelta(
            minutes=now.minute % BAR_MINUTES + BAR_MINUTES
        )
        if closed_bar != self.last_bar_done and now >= closed_bar + timedelta(
            minutes=BAR_MINUTES, seconds=5
        ):
            if self.on_bar(now, closed_bar):
                self.last_bar_done = closed_bar

    def can_enter(self, now):
        s = self.state
        if s.get("open") or s.get("pending"):
            return False, "a trade is already open or pending"
        if not hhmm(ENTRY_START) <= minutes_of(now) <= hhmm(LAST_ENTRY):
            return False, "outside the entry window"
        if s["trades"] >= MAX_TRADES_PER_DAY:
            return False, "max trades for the day reached"
        if s["pnl"] <= -MAX_DAILY_LOSS_RS:
            return False, "daily loss limit reached"
        if s["llm_calls"] >= MAX_LLM_CALLS_PER_DAY:
            return False, "LLM call budget for the day used"
        return True, ""

    def on_bar(self, now, closed_bar):
        """Process one closed bar. Returns False to retry on the next poll."""
        bars = self.load_bars(now)
        if bars is None or len(bars) < OB_SWING_LEN + ATR_PERIOD + 5:
            log("no usable history yet (broker not logged in, or market not open); retrying")
            return False
        if bars.index[-1] < closed_bar and now < closed_bar + timedelta(
            minutes=BAR_MINUTES, seconds=60
        ):
            return False  # the broker has not published the bar yet

        o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
        atr = atr_rma(h, lo, c)
        zones = detect_order_blocks(o, h, lo, c, atr)
        last = len(c) - 1
        ltp = self.ltp() or float(c[last])
        today_bars = bars[bars.index.date == now.date()]
        levels = compute_levels(self.daily, today_bars, ltp, now.date())

        fresh = [z for z in zones if z.mitigated_idx < 0]
        triggered = [z for z in zones if z.mitigated_idx == last]
        log(
            f"bar {bars.index[-1]:%H:%M} close {c[last]:.2f} ltp {ltp:.2f} ATR {atr[last]:.2f} | "
            f"fresh OB {sum(z.kind == 'bull' for z in fresh)} bull / {sum(z.kind == 'bear' for z in fresh)} bear | "
            f"PDH {levels.get('PDH', 0):.2f} {levels['status'].get('PDH', '-')} PDL {levels.get('PDL', 0):.2f} {levels['status'].get('PDL', '-')}"
        )
        self.settle_shadows(bars)
        if self.state.get("open"):
            self.maybe_review(now, bars, float(atr[last]), ltp, levels)
        if not triggered:
            return True

        ok, why = self.can_enter(now)
        if not ok:
            log(f"order block touched, not evaluated: {why}")
            return True
        foreign = self.position_qty()
        if foreign:
            log(f"order block touched, but a position of {foreign} is already open in {SYMBOL}")
            return True

        allowed = sorted({"BUY" if z.kind == "bull" else "SELL" for z in triggered})
        context = self.build_context(now, bars, atr[last], ltp, levels, zones, triggered, allowed)
        self.state["llm_calls"] += 1
        self._save_state()
        try:
            raw, text = self.planner(context)
        except Exception as e:
            log(f"LLM request failed: {e}")
            self.journal("llm_error", error=str(e), context=context)
            return True

        plan, why = validate_plan(raw, allowed, ltp, float(atr[last]), self.tick, self.lot)
        self.journal(
            "plan",
            context=context,
            reply=text,
            accepted=plan is not None,
            rejected_because=why,
            plan=asdict(plan) if plan else None,
        )
        model_took = isinstance(raw, dict) and str(raw.get("decision", "")).upper() == "TAKE"
        for z in triggered:
            self.add_shadow(z, ltp, float(atr[last]), bars.index[-1], plan, model_took, why)
        self._save_state()
        if plan is None:
            log(f"no trade: {why}")
            return True
        log(
            f"plan accepted: {plan.side} {plan.qty} {plan.entry_type} @ {plan.entry} SL {plan.stop} TG {plan.target} RR {plan.rr} - {plan.reason}"
        )
        self.enter(plan, now)
        return True

    def build_context(self, now, bars, atr, ltp, levels, zones, triggered, allowed):
        last = len(bars) - 1

        def zone_view(z):
            return {
                "type": "bull OB (demand)" if z.kind == "bull" else "bear OB (supply)",
                "top": round(z.top, 2),
                "bottom": round(z.bottom, 2),
                "formed_at": f"{bars.index[z.ob_idx]:%d-%b %H:%M}",
                "bars_since_formed": last - z.ob_idx,
            }

        fresh = [z for z in zones if z.mitigated_idx < 0]
        above = sorted((z for z in fresh if z.bottom > ltp), key=lambda z: z.bottom)[:3]
        below = sorted((z for z in fresh if z.top < ltp), key=lambda z: -z.top)[:3]
        recent = bars.iloc[-CANDLES_FOR_LLM:]
        candles = [
            [
                f"{ts:%d %H:%M}",
                round(r.open, 2),
                round(r.high, 2),
                round(r.low, 2),
                round(r.close, 2),
                int(r.volume),
            ]
            for ts, r in recent.iterrows()
        ]
        return {
            "symbol": SYMBOL,
            "exchange": EXCHANGE,
            "time_ist": f"{now:%Y-%m-%d %H:%M}",
            "session_phase": session_phase(now),
            "minutes_to_square_off": hhmm(SQUARE_OFF) - minutes_of(now),
            "ltp": round(ltp, 2),
            "atr14_5m": round(float(atr), 2),
            "tick_size": self.tick,
            "allowed_sides": allowed,
            "trigger": [zone_view(z) for z in triggered],
            "levels": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in levels.items()},
            "fresh_order_blocks_above": [zone_view(z) for z in above],
            "fresh_order_blocks_below": [zone_view(z) for z in below],
            "candles_5m": {
                "columns": ["time", "open", "high", "low", "close", "volume"],
                "rows": candles,
            },
            **self.market_context(now, bars),
            "recent_results": self.state["history"][-RECENT_RESULTS_FOR_LLM:],
            "trades_today": self.state["trades"],
            "realized_pnl_today": round(self.state["pnl"], 2),
        }

    def market_context(self, now, bars):
        """Higher timeframe trend, the index and the opening gap."""
        ctx = {"trend_15m": trend_view(bars, "15min"), "trend_1h": trend_view(bars, "60min")}
        past = self.daily[self.daily.index.date < now.date()]
        today = bars[bars.index.date == now.date()]
        if not past.empty and not today.empty:
            prev_close = float(past["close"].iloc[-1])
            ctx["gap_pct"] = round((float(today["open"].iloc[0]) / prev_close - 1) * 100, 2)
        ctx["index"] = self.index_view(now)
        return ctx

    def index_view(self, now):
        if not INDEX_SYMBOL:
            return None
        start = (now - timedelta(days=12)).strftime("%Y-%m-%d")
        df = self.call(
            "history",
            symbol=INDEX_SYMBOL,
            exchange=INDEX_EXCHANGE,
            interval=INTERVAL,
            start_date=start,
            end_date=now.strftime("%Y-%m-%d"),
        )
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        df = self._naive(df)
        today = df[df.index.date == now.date()]
        before = df[df.index.date < now.date()]
        view = {"symbol": INDEX_SYMBOL, "trend_15m": trend_view(df, "15min")}
        if not today.empty and not before.empty:
            prev_close = float(before["close"].iloc[-1])
            last = float(today["close"].iloc[-1])
            view["last"] = round(last, 2)
            view["change_today_pct"] = round((last / prev_close - 1) * 100, 2)
            view["vs_today_open"] = "above" if last > float(today["open"].iloc[0]) else "below"
        return view

    # ---------- shadow tracking ----------
    def add_shadow(self, zone, ltp, atr, bar_time, plan, model_took, why):
        base = baseline_plan(zone, ltp, atr)
        if base is None:
            return
        if plan is not None and plan.side == base["side"]:
            decision = "taken"
        elif model_took:
            decision = "rejected_by_rules"  # the model said TAKE, validation refused it
        else:
            decision = "skipped"
        self.state["shadows"].append(
            {
                **{k: round(v, 4) for k, v in base.items() if k != "side"},
                "side": base["side"],
                "zone": f"{zone.kind} {zone.bottom:.2f}-{zone.top:.2f}",
                "from_bar": bar_time.isoformat(),
                "decision": decision,
                "why": why[:200],
            }
        )

    def settle_shadows(self, bars):
        """Walk every open shadow forward over the bars closed since it began."""
        open_shadows = []
        for s in self.state["shadows"]:
            r = None
            for ts, bar in bars[bars.index > pd.Timestamp(s["from_bar"])].iterrows():
                if minutes_of(ts) >= hhmm(SQUARE_OFF):
                    sign = 1 if s["side"] == "BUY" else -1
                    r, outcome = sign * (float(bar.open) - s["entry"]) / s["risk"], "square-off"
                    break
                r = score_bar(s, float(bar.high), float(bar.low))
                if r is not None:
                    outcome = "target" if r > 0 else "stop"
                    break
            if r is None:
                open_shadows.append(s)
                continue
            r = round(r, 2)
            self.journal("shadow_result", shadow=s, outcome=outcome, r=r)
            self.state["history"].append(
                {
                    "date": s["from_bar"][:16],
                    "side": s["side"],
                    "your_decision": s["decision"],
                    "mechanical_result_r": r,
                }
            )
            log(
                f"shadow {s['side']} from {s['from_bar'][11:16]} ({s['decision']}): {outcome}, {r}R"
            )
        self.state["history"] = self.state["history"][-HISTORY_KEEP:]
        if len(open_shadows) != len(self.state["shadows"]):
            self.state["shadows"] = open_shadows
            self._save_state()

    # ---------- open-trade review ----------
    def maybe_review(self, now, bars, atr, ltp, levels):
        trade = self.state["open"]
        if not REVIEW_ENABLED or minutes_of(now) >= hhmm(SQUARE_OFF):
            return
        if self.state["review_calls"] >= MAX_REVIEW_CALLS_PER_DAY:
            return
        since = pd.Timestamp(trade.get("last_review") or trade["opened"])
        if int((bars.index >= since).sum()) < REVIEW_EVERY_BARS:
            return

        sign = 1 if trade["side"] == "BUY" else -1
        initial_risk = abs(trade["entry_price"] - trade.get("initial_stop", trade["stop"]))
        recent = bars.iloc[-20:]
        context = {
            "trade": {
                "side": trade["side"],
                "qty": trade["qty"],
                "entry": round(trade["entry_price"], 2),
                "current_stop": trade["stop"],
                "initial_stop": trade.get("initial_stop", trade["stop"]),
                "target": trade["target"],
                "opened": trade["opened"],
                "reason_taken": trade.get("reason", ""),
                "open_r": round(sign * (ltp - trade["entry_price"]) / initial_risk, 2)
                if initial_risk
                else None,
            },
            "time_ist": f"{now:%Y-%m-%d %H:%M}",
            "minutes_to_square_off": hhmm(SQUARE_OFF) - minutes_of(now),
            "ltp": round(ltp, 2),
            "atr14_5m": round(atr, 2),
            "tick_size": self.tick,
            "levels": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in levels.items()},
            "candles_5m": {
                "columns": ["time", "open", "high", "low", "close", "volume"],
                "rows": [
                    [
                        f"{ts:%d %H:%M}",
                        round(r.open, 2),
                        round(r.high, 2),
                        round(r.low, 2),
                        round(r.close, 2),
                        int(r.volume),
                    ]
                    for ts, r in recent.iterrows()
                ],
            },
            **self.market_context(now, bars),
        }
        trade["last_review"] = now.isoformat(timespec="seconds")
        self.state["review_calls"] += 1
        self._save_state()
        try:
            raw, text = self.reviewer(context)
        except Exception as e:
            log(f"trade review request failed, holding: {e}")
            self.journal("review_error", error=str(e))
            return

        action, new_stop, why = validate_review(raw, trade, ltp, atr, self.tick)
        self.journal(
            "review", context=context, reply=text, action=action, new_stop=new_stop, why=why
        )
        if action == "EXIT":
            self.notify(f"review: exit now - {why}")
            self.exit_trade("LLM review: " + why[:80])
        elif action == "TIGHTEN_STOP":
            self.tighten_stop(new_stop, why)
        else:
            log(f"review: hold - {why}")

    def tighten_stop(self, new_stop, why):
        trade = self.state["open"]
        old = trade["stop"]
        trade["stop"] = new_stop
        oid = trade.get("stop_order_id")
        if oid:
            exit_action = "SELL" if trade["side"] == "BUY" else "BUY"
            price_type, price = self.stop_order_prices(
                exit_action, new_stop, trade.get("stop_type", "SL-M")
            )
            resp = self.call(
                "modifyorder",
                order_id=oid,
                strategy=STRATEGY_NAME,
                symbol=SYMBOL,
                action=exit_action,
                exchange=EXCHANGE,
                price_type=price_type,
                product=PRODUCT,
                quantity=trade["qty"],
                price=price,
                trigger_price=new_stop,
            )
            # If the broker refused, its order still sits at the old stop and
            # this script watches the tighter one until the trade closes.
            trade["stop_unsynced"] = resp is None
        self._save_state()
        note = (
            " (broker refused the change; the new stop is watched by this script)"
            if trade.get("stop_unsynced")
            else ""
        )
        self.notify(f"review: stop tightened {old} -> {new_stop}{note} - {why}")

    def stop_order_prices(self, exit_action, stop, price_type):
        """(price_type, limit price) for a stop order at `stop`."""
        if price_type == "SL-M":
            return "SL-M", 0
        buffer = max(self.tick, round_tick(stop * 0.003, self.tick))
        limit = stop - buffer if exit_action == "SELL" else stop + buffer
        return "SL", round_tick(limit, self.tick)

    # ---------- execution ----------
    def enter(self, plan, now):
        action = plan.side
        if plan.entry_type == "LIMIT":
            oid = self.place(action, plan.qty, "LIMIT", price=plan.entry)
            if not oid:
                self.notify("LIMIT entry order was refused; no trade")
                return
            self.state["pending"] = {
                "order_id": oid,
                "plan": asdict(plan),
                "expires": (now + timedelta(minutes=LIMIT_ENTRY_EXPIRY_MIN)).isoformat(),
            }
            self.state["trades"] += 1
            self._save_state()
            self.notify(
                f"LIMIT {action} {plan.qty} @ {plan.entry} placed (SL {plan.stop}, TG {plan.target})"
            )
            return

        oid = self.place(action, plan.qty, "MARKET")
        if not oid:
            self.notify("MARKET entry order was refused; no trade")
            return
        self.state["trades"] += 1
        self._save_state()
        status, avg = self.wait_final(oid)
        qty = self.position_qty()
        if not qty:
            self.notify(f"entry order ended '{status}' with no position; no trade")
            return
        self.on_filled(plan, abs(qty), avg or self.ltp() or plan.entry)

    def manage_pending(self, now):
        p = self.state["pending"]
        plan = Plan(**p["plan"])
        status, avg = self.order_status(p["order_id"])
        expired = now >= datetime.fromisoformat(p["expires"]) or minutes_of(now) >= hhmm(SQUARE_OFF)
        if status == "complete":
            self.state["pending"] = None
            self.on_filled(plan, abs(self.position_qty() or plan.qty), avg or plan.entry)
        elif status in ("rejected", "cancelled"):
            self.state["pending"] = None
            self._save_state()
            self.notify(f"LIMIT entry {status}; no trade")
        elif expired:
            self.call("cancelorder", order_id=p["order_id"], strategy=STRATEGY_NAME)
            status, avg = self.wait_final(p["order_id"], timeout=10)
            if status not in ("complete", "cancelled", "rejected"):
                log(f"LIMIT entry could not be cancelled ({status}); retrying next cycle")
                return
            self.state["pending"] = None
            qty = self.position_qty()  # covers a fill or partial fill that raced the cancel
            if qty:
                self.on_filled(plan, abs(qty), avg or plan.entry)
            else:
                self._save_state()
                self.notify("LIMIT entry not filled in time; cancelled")

    def on_filled(self, plan, qty, fill):
        exit_action = "SELL" if plan.side == "BUY" else "BUY"
        trade = {
            "side": plan.side,
            "qty": qty,
            "entry_price": fill,
            "stop": plan.stop,
            "initial_stop": plan.stop,
            "target": plan.target,
            "reason": plan.reason,
            "stop_order_id": None,
            "stop_type": "SL-M",
            "opened": self.clock().isoformat(timespec="seconds"),
        }
        self.state["open"] = trade
        self._save_state()
        self.journal("entry", trade=trade, plan=asdict(plan))
        self.notify(f"{plan.side} {qty} filled @ {fill:.2f}, SL {plan.stop}, TG {plan.target}")

        gapped = fill <= plan.stop if plan.side == "BUY" else fill >= plan.stop
        if gapped:
            self.exit_trade("filled beyond the stop")
            return

        oid = self.place(exit_action, qty, "SL-M", trigger_price=plan.stop)
        if not oid:
            _, limit = self.stop_order_prices(exit_action, plan.stop, "SL")
            oid = self.place(exit_action, qty, "SL", price=limit, trigger_price=plan.stop)
            trade["stop_type"] = "SL"
        if oid:
            trade["stop_order_id"] = oid
        else:
            self.notify(
                "broker refused the stop-loss order; the stop is now watched by this script only"
            )
        self._save_state()

    def manage_open(self, now):
        trade = self.state["open"]
        if trade.get("stop_order_id"):
            status, avg = self.order_status(trade["stop_order_id"])
            if status == "complete":
                self.close_book(avg or trade["stop"], "stop loss")
                return
            if status in ("rejected", "cancelled"):
                trade["stop_order_id"] = None
                self._save_state()
                self.notify(
                    f"stop-loss order {status} by the broker; the stop is now watched by this script only"
                )

        if minutes_of(now) >= hhmm(SQUARE_OFF):
            self.exit_trade("square-off time")
            return
        ltp = self.ltp()
        if ltp is None:
            return
        long = trade["side"] == "BUY"
        if (ltp >= trade["target"]) if long else (ltp <= trade["target"]):
            self.exit_trade("target")
        elif (not trade.get("stop_order_id") or trade.get("stop_unsynced")) and (
            (ltp <= trade["stop"]) if long else (ltp >= trade["stop"])
        ):
            self.exit_trade("stop loss (script)")

    def exit_trade(self, reason):
        trade = self.state["open"]
        if trade.get("stop_order_id"):
            self.call("cancelorder", order_id=trade["stop_order_id"], strategy=STRATEGY_NAME)
            status, avg = self.wait_final(trade["stop_order_id"], timeout=10)
            if status == "complete":  # the stop filled first; nothing left to close
                self.close_book(avg or trade["stop"], "stop loss")
                return
            if status != "cancelled":
                # Still live at the broker: exiting now as well could reverse the position
                log(
                    f"stop-loss order could not be cancelled ({status}); retrying the {reason} exit next cycle"
                )
                return
            trade["stop_order_id"] = None
            self._save_state()
        qty = self.position_qty()
        if qty == 0:
            self.close_book(self.ltp() or trade["entry_price"], reason + " (already flat)")
            return
        exit_action = "SELL" if trade["side"] == "BUY" else "BUY"
        resp = self.call(
            "placesmartorder",
            strategy=STRATEGY_NAME,
            symbol=SYMBOL,
            action=exit_action,
            exchange=EXCHANGE,
            price_type="MARKET",
            product=PRODUCT,
            quantity=abs(qty) if qty else trade["qty"],
            position_size=0,
        )
        oid = (resp or {}).get("orderid")
        if not oid:
            self.notify(f"EXIT ORDER REFUSED ({reason}); position still open, will retry")
            return
        status, avg = self.wait_final(oid)
        self.close_book(avg or self.ltp() or trade["entry_price"], reason)

    def close_book(self, exit_price, reason):
        trade = self.state["open"]
        sign = 1 if trade["side"] == "BUY" else -1
        pnl = sign * (exit_price - trade["entry_price"]) * trade["qty"]
        self.state["pnl"] += pnl
        self.state["open"] = None
        self._save_state()
        self.journal("exit", trade=trade, exit_price=exit_price, reason=reason, pnl=round(pnl, 2))
        self.notify(
            f"closed {trade['side']} {trade['qty']} @ {exit_price:.2f} ({reason}), P&L {pnl:.2f}, day {self.state['pnl']:.2f}"
        )


def report(journal_path=None):
    """Compare the mechanical result of triggers the LLM took against those it skipped."""
    path = journal_path or DATA_DIR / f"journal_{SYMBOL}.jsonl"
    groups, trades = {}, []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        print(f"no journal yet at {path}")
        return
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a line cut off by a killed run
        if row.get("event") == "shadow_result":
            groups.setdefault(row["shadow"]["decision"], []).append(float(row["r"]))
        elif row.get("event") == "exit":
            trades.append(float(row.get("pnl") or 0))

    print(f"{SYMBOL}: same mechanical plan (target {SHADOW_RR}R) scored on every trigger")
    print(f"{'decision':<20}{'count':>7}{'win %':>8}{'avg R':>8}{'total R':>9}")
    for name in ("taken", "skipped", "rejected_by_rules"):
        rs = groups.get(name, [])
        if rs:
            wins = sum(r > 0 for r in rs) / len(rs) * 100
            print(f"{name:<20}{len(rs):>7}{wins:>7.0f}%{sum(rs) / len(rs):>8.2f}{sum(rs):>9.2f}")
        else:
            print(f"{name:<20}{0:>7}")
    taken, skipped = groups.get("taken", []), groups.get("skipped", [])
    if len(taken) >= 20 and len(skipped) >= 20:
        edge = sum(taken) / len(taken) - sum(skipped) / len(skipped)
        verdict = (
            "the LLM filter is adding value" if edge > 0 else "the LLM filter is NOT adding value"
        )
        print(f"\ntaken minus skipped: {edge:+.2f}R per trigger, {verdict}")
    else:
        print(
            "\nneed at least 20 taken and 20 skipped triggers before this comparison means anything"
        )
    if trades:
        print(f"actual trades closed: {len(trades)}, P&L Rs {sum(trades):.2f}")


def main():
    if "--report" in sys.argv:
        report()
        return
    if not API_KEY:
        log("OPENALGO_API_KEY is not set. Start this strategy from the /python page.")
        raise SystemExit(1)
    planner, reviewer = ask_llm, ask_reviewer
    if BACKTEST and not BACKTEST_LLM:
        planner, reviewer = mechanical_planner, mechanical_reviewer
        log("backtest: the LLM is off, every valid trigger is taken with the mechanical plan")
    strategy = SmcLlmStrategy(
        api(api_key=API_KEY, host=HOST, ws_url=WS_URL), planner=planner, reviewer=reviewer
    )
    strategy.setup()
    while True:
        try:
            strategy.step()
        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"error in cycle: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
