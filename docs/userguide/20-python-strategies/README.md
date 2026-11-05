# 20 - Python Strategies

## Introduction

Python is one of the most powerful ways to build trading strategies with OpenAlgo. Using the official OpenAlgo Python library, you can create sophisticated algorithms, backtest strategies, and execute trades programmatically.

## Getting Started

### Installing the Library

```bash
pip install openalgo
```

### Basic Setup

```python
from openalgo import api

# Initialize client
client = api(
    api_key="YOUR_API_KEY",
    host="http://127.0.0.1:5000"
)

# Test connection
print("Connected to OpenAlgo!")
```

## Core Functions

### Placing Orders

```python
# Market order
response = client.place_order(
    symbol="SBIN",
    exchange="NSE",
    action="BUY",
    quantity=100,
    price_type="MARKET",
    product="MIS",
    strategy="PythonStrategy"
)

print(f"Order ID: {response['orderid']}")
```

### Order Types

```python
# Limit order
client.place_order(
    symbol="SBIN",
    exchange="NSE",
    action="BUY",
    quantity=100,
    price_type="LIMIT",
    price=620.00,
    product="MIS",
    strategy="LimitStrategy"
)

# Stop-loss order
client.place_order(
    symbol="SBIN",
    exchange="NSE",
    action="SELL",
    quantity=100,
    price_type="SL",
    price=614.00,
    trigger_price=615.00,
    product="MIS",
    strategy="SLStrategy"
)

# Stop-loss market order
client.place_order(
    symbol="SBIN",
    exchange="NSE",
    action="SELL",
    quantity=100,
    price_type="SL-M",
    trigger_price=615.00,
    product="MIS",
    strategy="SLMStrategy"
)
```

### Smart Orders

```python
# Position-aware order
response = client.place_smart_order(
    symbol="SBIN",
    exchange="NSE",
    action="BUY",
    quantity=100,
    position_size=100,  # Target position
    price_type="MARKET",
    product="MIS",
    strategy="SmartStrategy"
)
```

### Basket Orders

```python
# Multiple orders at once
basket = [
    {
        "symbol": "SBIN",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 100,
        "price_type": "MARKET",
        "product": "MIS"
    },
    {
        "symbol": "INFY",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 50,
        "price_type": "MARKET",
        "product": "MIS"
    },
    {
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 25,
        "price_type": "MARKET",
        "product": "MIS"
    }
]

response = client.place_basket_order(
    orders=basket,
    strategy="BasketStrategy"
)

print(f"Successful: {response['successful']}")
print(f"Failed: {response['failed']}")
```

### Getting Positions

```python
# Get all positions
positions = client.get_positions()

for pos in positions['data']:
    print(f"{pos['symbol']}: {pos['quantity']} @ {pos['average_price']}")
```

### Getting Holdings

```python
# Get all holdings
holdings = client.get_holdings()

for holding in holdings['data']:
    print(f"{holding['symbol']}: {holding['quantity']} shares")
```

### Getting Order Book

```python
# Get all orders
orders = client.get_orders()

for order in orders['data']:
    print(f"{order['orderid']}: {order['symbol']} {order['action']} {order['status']}")
```

### Closing Positions

```python
# Close specific position
client.close_position(
    symbol="SBIN",
    exchange="NSE",
    product="MIS",
    strategy="CloseStrategy"
)

# Close all positions
client.close_all_positions(strategy="SquareOff")
```

## Strategy Examples

### 1. Simple Moving Average Crossover

```python
import pandas as pd
import numpy as np
from openalgo import api
import time

client = api(api_key="YOUR_KEY", host="http://127.0.0.1:5000")

def get_historical_data(symbol, exchange, interval, days=30):
    """Fetch historical data for analysis"""
    # You would typically use a data provider here
    # This is a placeholder for demonstration
    pass

def calculate_signals(df, fast_period=9, slow_period=21):
    """Calculate moving average crossover signals"""
    df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
    df['slow_ma'] = df['close'].rolling(window=slow_period).mean()

    # Generate signals
    df['signal'] = 0
    df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1
    df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1

    return df

def run_strategy(symbol, exchange, quantity):
    """Main strategy loop"""
    current_position = 0

    while True:
        try:
            # Get latest data
            df = get_historical_data(symbol, exchange, '5min')
            df = calculate_signals(df)

            # Get latest signal
            latest_signal = df['signal'].iloc[-1]

            # Execute trades based on signal
            if latest_signal == 1 and current_position <= 0:
                # Buy signal
                if current_position < 0:
                    # Close short first
                    client.place_order(
                        symbol=symbol,
                        exchange=exchange,
                        action="BUY",
                        quantity=abs(current_position),
                        price_type="MARKET",
                        product="MIS",
                        strategy="MA_Crossover"
                    )

                # Go long
                client.place_order(
                    symbol=symbol,
                    exchange=exchange,
                    action="BUY",
                    quantity=quantity,
                    price_type="MARKET",
                    product="MIS",
                    strategy="MA_Crossover"
                )
                current_position = quantity
                print(f"Bought {quantity} {symbol}")

            elif latest_signal == -1 and current_position >= 0:
                # Sell signal
                if current_position > 0:
                    # Close long first
                    client.place_order(
                        symbol=symbol,
                        exchange=exchange,
                        action="SELL",
                        quantity=current_position,
                        price_type="MARKET",
                        product="MIS",
                        strategy="MA_Crossover"
                    )
                    current_position = 0
                    print(f"Sold {quantity} {symbol}")

            # Wait for next candle
            time.sleep(300)  # 5 minutes

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

# Run the strategy
if __name__ == "__main__":
    run_strategy("SBIN", "NSE", 100)
```

### 2. RSI Mean Reversion

```python
from openalgo import api
import pandas as pd
import numpy as np
import time

client = api(api_key="YOUR_KEY", host="http://127.0.0.1:5000")

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_strategy(symbol, exchange, quantity):
    """RSI mean reversion strategy"""
    position = 0

    while True:
        try:
            # Get data and calculate RSI
            df = get_historical_data(symbol, exchange, '15min')
            df['rsi'] = calculate_rsi(df['close'])

            current_rsi = df['rsi'].iloc[-1]

            # Oversold - Buy signal
            if current_rsi < 30 and position == 0:
                client.place_order(
                    symbol=symbol,
                    exchange=exchange,
                    action="BUY",
                    quantity=quantity,
                    price_type="MARKET",
                    product="MIS",
                    strategy="RSI_Strategy"
                )
                position = quantity
                print(f"RSI {current_rsi:.2f}: Bought {symbol}")

            # Overbought - Sell signal
            elif current_rsi > 70 and position > 0:
                client.place_order(
                    symbol=symbol,
                    exchange=exchange,
                    action="SELL",
                    quantity=position,
                    price_type="MARKET",
                    product="MIS",
                    strategy="RSI_Strategy"
                )
                position = 0
                print(f"RSI {current_rsi:.2f}: Sold {symbol}")

            time.sleep(900)  # 15 minutes

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)
```

### 3. Multi-Symbol Scanner

```python
from openalgo import api
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

client = api(api_key="YOUR_KEY", host="http://127.0.0.1:5000")

# Watchlist
symbols = ["SBIN", "HDFC", "ICICIBANK", "INFY", "TCS", "RELIANCE"]

def analyze_symbol(symbol):
    """Analyze single symbol for trading signal"""
    try:
        df = get_historical_data(symbol, "NSE", "5min")

        # Calculate indicators
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        df['volume_avg'] = df['volume'].rolling(20).mean()

        # Check conditions
        latest = df.iloc[-1]

        # Bullish conditions
        bullish = (
            latest['close'] > latest['sma20'] and
            latest['sma20'] > latest['sma50'] and
            latest['volume'] > latest['volume_avg'] * 1.5
        )

        return {
            'symbol': symbol,
            'bullish': bullish,
            'close': latest['close']
        }

    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}

def scan_and_trade(max_positions=3, quantity=100):
    """Scan all symbols and trade top signals"""

    # Get current positions
    positions = client.get_positions()
    current_symbols = [p['symbol'] for p in positions.get('data', [])]
    open_positions = len(current_symbols)

    # Analyze symbols in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(analyze_symbol, symbols))

    # Filter bullish signals
    bullish_signals = [r for r in results if r.get('bullish') and r['symbol'] not in current_symbols]

    # Trade up to max positions
    available_slots = max_positions - open_positions

    for signal in bullish_signals[:available_slots]:
        client.place_order(
            symbol=signal['symbol'],
            exchange="NSE",
            action="BUY",
            quantity=quantity,
            price_type="MARKET",
            product="MIS",
            strategy="Scanner"
        )
        print(f"Bought {signal['symbol']} at {signal['close']}")

# Run scanner every 5 minutes
while True:
    scan_and_trade()
    time.sleep(300)
```

### 4. Options Strategy

```python
from openalgo import api
import datetime

client = api(api_key="YOUR_KEY", host="http://127.0.0.1:5000")

def get_option_symbol(underlying, expiry, strike, option_type):
    """Construct option symbol"""
    # Format: NIFTY25JAN21500CE
    expiry_str = expiry.strftime("%y%b").upper()
    return f"{underlying}{expiry_str}{strike}{option_type}"

def bull_call_spread(underlying, expiry, lower_strike, upper_strike, lot_size):
    """Execute bull call spread"""

    lower_call = get_option_symbol(underlying, expiry, lower_strike, "CE")
    upper_call = get_option_symbol(underlying, expiry, upper_strike, "CE")

    # Basket order for simultaneous execution
    basket = [
        {
            "symbol": lower_call,
            "exchange": "NFO",
            "action": "BUY",
            "quantity": lot_size,
            "price_type": "MARKET",
            "product": "NRML"
        },
        {
            "symbol": upper_call,
            "exchange": "NFO",
            "action": "SELL",
            "quantity": lot_size,
            "price_type": "MARKET",
            "product": "NRML"
        }
    ]

    response = client.place_basket_order(
        orders=basket,
        strategy="BullCallSpread"
    )

    return response

def iron_condor(underlying, expiry, call_sell, call_buy, put_sell, put_buy, lot_size):
    """Execute iron condor strategy"""

    basket = [
        {
            "symbol": get_option_symbol(underlying, expiry, call_sell, "CE"),
            "exchange": "NFO",
            "action": "SELL",
            "quantity": lot_size,
            "price_type": "MARKET",
            "product": "NRML"
        },
        {
            "symbol": get_option_symbol(underlying, expiry, call_buy, "CE"),
            "exchange": "NFO",
            "action": "BUY",
            "quantity": lot_size,
            "price_type": "MARKET",
            "product": "NRML"
        },
        {
            "symbol": get_option_symbol(underlying, expiry, put_sell, "PE"),
            "exchange": "NFO",
            "action": "SELL",
            "quantity": lot_size,
            "price_type": "MARKET",
            "product": "NRML"
        },
        {
            "symbol": get_option_symbol(underlying, expiry, put_buy, "PE"),
            "exchange": "NFO",
            "action": "BUY",
            "quantity": lot_size,
            "price_type": "MARKET",
            "product": "NRML"
        }
    ]

    response = client.place_basket_order(
        orders=basket,
        strategy="IronCondor"
    )

    return response

# Execute strategies
expiry = datetime.date(2025, 1, 30)  # Next expiry

# Bull call spread
bull_call_spread("NIFTY", expiry, 21500, 21600, 50)

# Iron condor
iron_condor("NIFTY", expiry, 22000, 22100, 21000, 20900, 50)
```

## Error Handling

### Robust Order Placement

```python
import time
from openalgo import api

client = api(api_key="YOUR_KEY", host="http://127.0.0.1:5000")

def place_order_with_retry(order_params, max_retries=3):
    """Place order with automatic retry on failure"""

    for attempt in range(max_retries):
        try:
            response = client.place_order(**order_params)

            if response.get('status') == 'success':
                return response
            else:
                print(f"Order failed: {response.get('message')}")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff

    return {'status': 'error', 'message': 'Max retries exceeded'}

# Usage
order = {
    'symbol': 'SBIN',
    'exchange': 'NSE',
    'action': 'BUY',
    'quantity': 100,
    'price_type': 'MARKET',
    'product': 'MIS',
    'strategy': 'RetryStrategy'
}

result = place_order_with_retry(order)
```

## Scheduling Strategies

### Using Schedule Library

```python
import schedule
import time
from openalgo import api

client = api(api_key="YOUR_KEY", host="http://127.0.0.1:5000")

def morning_scan():
    """Run at market open"""
    print("Running morning scan...")
    # Your scanning logic

def square_off():
    """Run before market close"""
    print("Squaring off positions...")
    client.close_all_positions(strategy="EOD_SquareOff")

def check_positions():
    """Periodic position check"""
    positions = client.get_positions()
    print(f"Open positions: {len(positions.get('data', []))}")

# Schedule tasks
schedule.every().day.at("09:20").do(morning_scan)
schedule.every().day.at("15:15").do(square_off)
schedule.every(5).minutes.do(check_positions)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(1)
```

## Backtesting a Strategy

Every strategy uploaded at `/python` has a **Backtest** button (the flask icon
on its card, and a Backtest button on its edit page). Choose a start date, an
end date (the last 30 days by default, at most 366 days), the capital, the
brokerage per order and the slippage, and press **Run backtest**. The results
page shows the progress while it runs and, when it is done, the net P&L, return,
trade count, win rate, profit factor, maximum drawdown, Sharpe ratio and total
costs, with the equity curve, the drawdown, the daily P&L, every trade and the
last lines your script printed. Past runs of the same strategy are listed
below, so you can compare one setting against another.

### What it does

- **Your script runs unchanged.** It runs as its own process, as it does live,
  from a copy in `strategies/backtests/<strategy>/<run>/`. Any file it writes
  beside itself (a journal, a state file) lands there, never beside the live
  script. A backtest does not stop, start or change the live strategy, its
  schedule or its files, and it can run while the live strategy is running.
- **Time is simulated.** `datetime.now()`, `date.today()`, `time.time()`,
  `time.monotonic()`, `time.sleep()` and `pd.Timestamp.now()` (and `today()`,
  `utcnow()`) follow a clock that starts at the open of the strategy's exchange
  on the first day of the period. A `time.sleep(15)` moves it on 15 seconds.
  Nights, weekends and days with no data are skipped, so a sleep after the close
  wakes the script at the next open. `datetime.now()` returns IST, as it does on
  an Indian server. The script is stopped when the clock passes the close of the
  last day.
- **Two clocks are not simulated:** `numpy.datetime64("now")` and
  `pd.Timestamp("now")` (the text form) still read the real time. Use
  `datetime.now()` or `pd.Timestamp.now()` in a strategy you want to backtest.

  | Exchange of the strategy | Session the clock follows |
  |--------------------------|---------------------------|
  | NSE, BSE, NFO, BFO | 09:15 to 15:30 |
  | CDS, BCD | 09:00 to 17:00 |
  | MCX | 09:00 to 23:30 |
  | NCDEX | 09:00 to 17:00 |

- **Orders stay inside the backtest.** `from openalgo import api` gives the
  script a simulated broker. Orders, positions, the order book, the trade book
  and funds all live inside the backtest, `analyzerstatus()` reports the mode
  `backtest`, and `telegram()` sends nothing.
- **The script cannot reach OpenAlgo itself.** While it runs, every connection
  from the backtest process to the OpenAlgo app is refused (whichever library
  makes it), starting another program is refused, and importing OpenAlgo's own
  code (`services`, `database`, `broker` and the rest) is refused. The script
  sees a stand-in API key, not yours; only the part of the backtest that reads
  history holds the real one. Other sites, such as an LLM provider, stay
  reachable. These protections cover every ordinary way a script makes a
  request, starts a program or imports a module. They are not a sandbox against
  code written on purpose to get around them, which nothing inside Python can
  be: backtest only scripts you would trust to run live.
- **No look-ahead.** `history()` returns the bars up to the simulated moment,
  including the bar still forming, exactly as a broker does live. A forming bar
  holds only what had traded by then. `quotes()` returns the last traded price
  of the moment.

### How orders fill

| Order | Fill |
|-------|------|
| MARKET | At once, at the price of the moment, moved against you by the slippage |
| LIMIT | When a later one-minute bar trades through the limit: at the limit, or at that bar's open if it opened beyond it |
| SL-M | When a bar trades through the trigger: at the trigger, or at the open if the bar gapped past it, plus slippage |
| SL | Triggers like SL-M, then fills as a LIMIT at its price |

`placesmartorder` trades the difference between the open position and
`position_size`, by the same rules as live. MIS positions are squared off at
the square-off times set on the Sandbox page (by default 15:15 for NSE, BSE and
F&O, 16:45 for currency, 23:30 for MCX and 17:00 for NCDEX), and MIS orders
after that time are rejected. Positions still open at the end of the period are
closed at the last price. MIS orders need a fifth of their value free in
capital, other products their full value.

### Costs

Per executed order: brokerage of the amount you enter (20 by default) or 0.03%
of the order value, whichever is lower; STT of 0.025% of the value on sells;
and 0.005% of the value for exchange, regulator and stamp charges. This is a
simple model close to intraday equity; delivery and F&O charges are higher.

### Data

The backtest reads one-minute bars from **Historify** first and asks your
broker for whatever Historify does not have, a few weeks at a time. Fifteen
days before the start date are loaded as well, so indicators have history to
warm up on, and 120 days of daily bars. Download the symbols into Historify
for the fastest and most complete runs; data from the broker needs a broker
login and is limited to what the broker keeps. Finished periods are cached on
disk, so running the same period again starts at once. If there is no data
for the symbol the strategy trades, the run ends with a message saying so.

### Limitations

- WebSocket streaming (`connect`, `subscribe_ltp` and the rest) is not
  available; those calls report that they did not connect. Strategies that
  rely only on streaming cannot be backtested.
- Fills come from one-minute bars, so a stop and a target touched in the same
  minute cannot be ordered, and a fill never improves on the bar's own prices.
  There is no order book depth, partial fill or queue position.
- Option chains, Greeks and other options services are not simulated.
- **Results are unreliable for a strategy that relies on its own threads.**
  Threads see the simulated clock, but the clock only moves when the main loop
  sleeps, so a background thread's timing does not match what would happen live.
- A run may take at most 60 minutes. A strategy that never sleeps between
  cycles is stopped with a message saying so. A strategy that prints on every cycle
  prints thousands of lines a simulated day; the log keeps the first 8 MB and
  the last 2,000 lines.
- The SMC example (`smc_llm_levels_ob.py`) does not call the LLM in a
  backtest. It takes every valid trigger with its mechanical plan instead, so a
  backtest costs nothing in model fees. Set `SMC_BACKTEST_LLM = 'true'` in
  `.env` to backtest with the real model, which calls it for every trigger in
  the period.

## Best Practices

### 1. Always Test in Analyzer Mode

```python
# Use Analyzer Mode for testing
# Enable it in OpenAlgo before running your strategy
```

### 2. Implement Risk Management

```python
def check_risk_limits(symbol, quantity, price):
    """Check if trade is within risk limits"""
    max_position_value = 100000  # ₹1 lakh per position
    max_daily_loss = 5000  # ₹5000 max daily loss

    position_value = quantity * price

    if position_value > max_position_value:
        return False, "Position size exceeds limit"

    # Check daily P&L
    # ... implementation

    return True, "OK"
```

### 3. Log Everything

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading.log'
)

def log_trade(action, symbol, quantity, price):
    logging.info(f"{action} {quantity} {symbol} @ {price}")
```

### 4. Handle Market Hours

```python
from datetime import datetime, time as dt_time

def is_market_open():
    """Check if market is open"""
    now = datetime.now().time()
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)

    weekday = datetime.now().weekday()

    return (
        weekday < 5 and
        market_open <= now <= market_close
    )
```

---

**Previous**: [19 - GoCharting Integration](../19-gocharting-integration/README.md)

**Next**: [21 - Flow Visual Strategy Builder](../21-flow-visual-builder/README.md)
