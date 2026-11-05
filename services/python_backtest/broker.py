"""The simulated broker a backtested strategy trades against.

``BacktestClient`` has the OpenAlgo SDK client's method names, keyword
arguments and response shapes, so a strategy written against ``openalgo.api``
runs unchanged. Orders never leave this process.

Fills, against one-minute bars:

- MARKET fills at once at the last known price (the open of the bar in
  progress), moved against the order by the slippage.
- LIMIT waits for a later bar to trade through the limit and fills at the
  limit, or at that bar's open when the bar opens beyond it.
- SL-M triggers when a bar trades through the trigger and fills at the
  trigger, or at the open when the bar gaps past it, moved by the slippage.
- SL triggers the same way and then behaves as a LIMIT at its price.

A pending order is only checked against bars that start at or after the
moment it was placed, so it can never fill on a price from before it existed.
MIS positions are squared off at the exchange's square-off time, as the
sandbox does.

Costs, per executed order: brokerage of ``brokerage_per_order`` or
``brokerage_pct`` of turnover, whichever is lower; ``stt_sell_pct`` of the
sell-side turnover; ``other_charges_pct`` of all turnover for exchange,
regulator and stamp charges. A deliberately simple model, close to intraday
equity; it is documented on the results page.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from .sessions import day_of, ist_datetime, sec_of_day, session_for

logger = logging.getLogger("python_backtest")

OPEN = "open"
TRIGGER_PENDING = "trigger pending"
COMPLETE = "complete"
CANCELLED = "cancelled"
REJECTED = "rejected"

PRICE_TYPES = ("MARKET", "LIMIT", "SL", "SL-M")
LEVERAGE = {"MIS": 5.0}  # intraday margin; CNC and NRML are taken at full value


@dataclass
class Costs:
    brokerage_per_order: float = 20.0
    brokerage_pct: float = 0.03
    stt_sell_pct: float = 0.025
    other_charges_pct: float = 0.005
    slippage_bps: float = 2.0

    def for_fill(self, action: str, qty: int, price: float) -> float:
        turnover = abs(qty) * price
        brokerage = self.brokerage_per_order
        if self.brokerage_pct > 0:
            brokerage = min(brokerage, turnover * self.brokerage_pct / 100)
        stt = turnover * self.stt_sell_pct / 100 if action == "SELL" else 0.0
        other = turnover * self.other_charges_pct / 100
        return max(0.0, brokerage) + stt + other

    def slip(self, action: str, price: float) -> float:
        factor = self.slippage_bps / 10000
        return price * (1 + factor) if action == "BUY" else price * (1 - factor)


@dataclass
class Order:
    orderid: str
    strategy: str
    symbol: str
    exchange: str
    action: str
    quantity: int
    price_type: str
    product: str
    price: float
    trigger_price: float
    placed_at: float
    status: str
    average_price: float = 0.0
    filled_at: float | None = None
    triggered: bool = False
    reason: str = ""
    tag: str = ""  # why it filled, for the trade list

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "orderid": self.orderid,
            "product": self.product,
            "quantity": str(self.quantity),
            "price": self.price,
            "pricetype": self.price_type,
            "order_status": self.status,
            "trigger_price": self.trigger_price,
            "average_price": round(self.average_price, 4),
            "timestamp": f"{ist_datetime(self.placed_at):%d-%b-%Y %H:%M:%S}",
        }


@dataclass
class Lot:
    qty: int  # signed: positive long, negative short
    price: float
    opened_at: float
    cost_per_unit: float


@dataclass
class Position:
    symbol: str
    exchange: str
    product: str
    lots: deque = field(default_factory=deque)

    @property
    def qty(self) -> int:
        return sum(lot.qty for lot in self.lots)

    @property
    def average_price(self) -> float:
        q = self.qty
        if q == 0:
            return 0.0
        return sum(lot.qty * lot.price for lot in self.lots) / q


class Broker:
    """Orders, fills, positions and costs. Knows nothing about the clock's mechanics."""

    def __init__(self, market, capital: float, costs: Costs, now_fn):
        self.market = market
        self.capital = float(capital)
        self.costs = costs
        self.now = now_fn
        self.orders: dict[str, Order] = {}
        self.pending: list[Order] = []
        self.positions: dict[tuple[str, str, str], Position] = {}
        self.trades: list[dict] = []  # round trips
        self.fills: list[dict] = []  # tradebook
        self.realized = 0.0
        self.total_costs = 0.0
        self._seq = 0

    # ---------- helpers ----------
    def _next_id(self) -> str:
        self._seq += 1
        return f"BT{self._seq:08d}"

    def position(self, symbol, exchange, product) -> Position:
        key = (symbol, exchange, product)
        pos = self.positions.get(key)
        if pos is None:
            pos = Position(symbol, exchange, product)
            self.positions[key] = pos
        return pos

    def net_qty(self, symbol, exchange, product) -> int:
        pos = self.positions.get((symbol, exchange, product))
        return pos.qty if pos else 0

    def mark(self, symbol, exchange, fallback: float) -> float:
        price = self.market.get(symbol, exchange).price_at(self.now())
        return fallback if price is None else price

    def unrealized(self) -> float:
        total = 0.0
        for pos in self.positions.values():
            for lot in pos.lots:
                total += lot.qty * (self.mark(pos.symbol, pos.exchange, lot.price) - lot.price)
        return total

    def equity(self) -> float:
        return self.capital + self.realized + self.unrealized() - self.total_costs

    def exposure(self) -> float:
        total = 0.0
        for pos in self.positions.values():
            q = pos.qty
            if q:
                total += (
                    abs(q)
                    * self.mark(pos.symbol, pos.exchange, pos.average_price)
                    / (LEVERAGE.get(pos.product, 1.0))
                )
        return total

    def has_open_positions(self) -> bool:
        return any(pos.qty for pos in self.positions.values())

    # ---------- placing ----------
    def place(
        self,
        *,
        strategy,
        symbol,
        exchange,
        action,
        quantity,
        price_type="MARKET",
        product="MIS",
        price=0,
        trigger_price=0,
        tag="",
    ) -> dict:
        symbol = str(symbol or "").strip().upper()
        exchange = str(exchange or "").strip().upper()
        action = str(action or "").strip().upper()
        price_type = str(price_type or "MARKET").strip().upper()
        product = str(product or "MIS").strip().upper()
        try:
            qty = int(float(quantity))
            price = float(price or 0)
            trigger = float(trigger_price or 0)
        except (TypeError, ValueError):
            return _error("quantity, price and trigger_price must be numbers")
        if not symbol or not exchange:
            return _error("symbol and exchange are required")
        if action not in ("BUY", "SELL"):
            return _error("action must be BUY or SELL")
        if price_type not in PRICE_TYPES:
            return _error("price_type must be MARKET, LIMIT, SL or SL-M")
        if qty <= 0:
            return _error("quantity must be greater than zero")
        if price_type in ("LIMIT", "SL") and price <= 0:
            return _error(f"A {price_type} order needs a price above zero")
        if price_type in ("SL", "SL-M") and trigger <= 0:
            return _error(f"A {price_type} order needs a trigger_price above zero")

        now = self.now()
        series = self.market.get(symbol, exchange)
        order = Order(
            orderid=self._next_id(),
            strategy=str(strategy or ""),
            symbol=symbol,
            exchange=exchange,
            action=action,
            quantity=qty,
            price_type=price_type,
            product=product,
            price=price if price_type in ("LIMIT", "SL") else 0.0,
            trigger_price=trigger if price_type in ("SL", "SL-M") else 0.0,
            placed_at=now,
            status=OPEN,
            tag=tag,
        )
        self.orders[order.orderid] = order

        ltp = series.price_at(now)
        refusal = self._refusal(order, ltp)
        if refusal:
            order.status = REJECTED
            order.reason = refusal
            logger.info(f"Order {order.orderid} rejected: {refusal}")
            # A broker accepts the request and rejects the order; the caller
            # learns it from orderstatus, as live.
            return {"status": "success", "orderid": order.orderid, "mode": "backtest"}

        if price_type == "MARKET":
            self._fill(order, self.costs.slip(action, ltp), now, tag or "strategy")
        else:
            order.status = TRIGGER_PENDING if price_type in ("SL", "SL-M") else OPEN
            self.pending.append(order)
        return {"status": "success", "orderid": order.orderid, "mode": "backtest"}

    def _refusal(self, order: Order, ltp: float | None) -> str:
        if ltp is None:
            return f"no price for {order.symbol} {order.exchange} at this time"
        session = session_for(order.exchange)
        if (
            order.product == "MIS"
            and session.square_off_sec is not None
            and sec_of_day(order.placed_at) >= session.square_off_sec
            and not order.tag
        ):
            return "MIS orders are not accepted after the square-off time"
        # Margin: reducing a position always passes; adding to it must fit the capital.
        current = self.net_qty(order.symbol, order.exchange, order.product)
        signed = order.quantity if order.action == "BUY" else -order.quantity
        if abs(current + signed) > abs(current):
            needed = order.quantity * ltp / LEVERAGE.get(order.product, 1.0)
            free = self.equity() - self.exposure()
            if needed > free + 1e-6:
                return (
                    f"not enough capital: the order needs about {needed:,.0f} "
                    f"and {max(free, 0):,.0f} is free"
                )
        return ""

    # ---------- smart order ----------
    def place_smart(self, *, position_size, **kwargs) -> dict:
        """Trade the difference between the open position and position_size.

        The same decision table as OpenAlgo's place_smartorder_api, including
        its one surprise: with no position and a target of zero it places the
        order as given.
        """
        try:
            target = int(float(position_size))
            qty = int(float(kwargs.get("quantity", 0)))
        except (TypeError, ValueError):
            return _error("position_size and quantity must be numbers")
        symbol = str(kwargs.get("symbol") or "").strip().upper()
        exchange = str(kwargs.get("exchange") or "").strip().upper()
        product = str(kwargs.get("product") or "MIS").strip().upper()
        current = self.net_qty(symbol, exchange, product)

        action, quantity = None, 0
        if target == 0 and current == 0:
            action, quantity = str(kwargs.get("action", "BUY")).upper(), qty
        elif target == 0:
            action, quantity = ("SELL" if current > 0 else "BUY"), abs(current)
        elif current == 0:
            action, quantity = ("BUY" if target > 0 else "SELL"), abs(target)
        elif target > current:
            action, quantity = "BUY", target - current
        elif target < current:
            action, quantity = "SELL", current - target

        if not action or quantity <= 0:
            return {
                "status": "success",
                "message": "No action needed. Position already matched.",
                "mode": "backtest",
            }
        kwargs = {**kwargs, "action": action, "quantity": quantity}
        return self.place(**kwargs)

    # ---------- managing ----------
    def cancel(self, order_id) -> dict:
        order = self.orders.get(str(order_id))
        if order is None:
            return _error(f"Order {order_id} not found")
        if order.status not in (OPEN, TRIGGER_PENDING):
            return _error(f"Order {order_id} is already {order.status} and cannot be cancelled")
        order.status = CANCELLED
        self.pending = [o for o in self.pending if o is not order]
        return {"status": "success", "orderid": order.orderid, "mode": "backtest"}

    def cancel_all(self) -> dict:
        cancelled = [o.orderid for o in list(self.pending)]
        for order in list(self.pending):
            order.status = CANCELLED
        self.pending = []
        return {
            "status": "success",
            "canceled_orders": cancelled,
            "failed_cancellations": [],
            "message": f"Canceled {len(cancelled)} orders",
            "mode": "backtest",
        }

    def modify(self, order_id, **kwargs) -> dict:
        order = self.orders.get(str(order_id))
        if order is None:
            return _error(f"Order {order_id} not found")
        if order.status not in (OPEN, TRIGGER_PENDING):
            return _error(f"Order {order_id} is already {order.status} and cannot be modified")
        try:
            if kwargs.get("quantity") not in (None, ""):
                order.quantity = max(1, int(float(kwargs["quantity"])))
            price_type = str(kwargs.get("price_type") or order.price_type).upper()
            if price_type not in PRICE_TYPES:
                return _error("price_type must be MARKET, LIMIT, SL or SL-M")
            order.price_type = price_type
            if kwargs.get("price") not in (None, ""):
                order.price = float(kwargs["price"]) if price_type in ("LIMIT", "SL") else 0.0
            if kwargs.get("trigger_price") not in (None, ""):
                trig = float(kwargs["trigger_price"])
                order.trigger_price = trig if price_type in ("SL", "SL-M") else 0.0
        except (TypeError, ValueError):
            return _error("quantity, price and trigger_price must be numbers")
        if price_type == "MARKET":
            self.pending = [o for o in self.pending if o is not order]
            ltp = self.market.get(order.symbol, order.exchange).price_at(self.now())
            if ltp is None:
                order.status = REJECTED
                order.reason = "no price at this time"
            else:
                self._fill(order, self.costs.slip(order.action, ltp), self.now(), "strategy")
        elif not order.triggered:
            order.status = TRIGGER_PENDING if price_type in ("SL", "SL-M") else OPEN
        return {"status": "success", "orderid": order.orderid, "mode": "backtest"}

    def close_all(self, tag="strategy", product=None) -> int:
        """Square off every open position at market. Returns how many were closed."""
        closed = 0
        for pos in list(self.positions.values()):
            if product and pos.product != product:
                continue
            q = pos.qty
            if not q:
                continue
            series = self.market.get(pos.symbol, pos.exchange)
            ltp = series.price_at(self.now())
            if ltp is None:
                ltp = pos.average_price
            action = "SELL" if q > 0 else "BUY"
            order = Order(
                orderid=self._next_id(),
                strategy="",
                symbol=pos.symbol,
                exchange=pos.exchange,
                action=action,
                quantity=abs(q),
                price_type="MARKET",
                product=pos.product,
                price=0.0,
                trigger_price=0.0,
                placed_at=self.now(),
                status=OPEN,
                tag=tag,
            )
            self.orders[order.orderid] = order
            self._fill(order, self.costs.slip(action, ltp), self.now(), tag)
            closed += 1
        return closed

    # ---------- the replay calls these ----------
    def on_bar_closed(self, bar_start: int) -> None:
        """A one-minute bar starting at bar_start has completed: try every pending order on it."""
        if not self.pending:
            return
        still = []
        for order in self.pending:
            if order.status not in (OPEN, TRIGGER_PENDING):
                continue
            if order.placed_at > bar_start:
                still.append(order)
                continue
            bar = self.market.get(order.symbol, order.exchange).bar_at(bar_start)
            if bar is None or not self._try_fill(order, bar, bar_start):
                still.append(order)
        self.pending = still

    def _try_fill(self, order: Order, bar, bar_start: int) -> bool:
        o, h, lo, _c = bar
        buy = order.action == "BUY"
        when = bar_start + 30  # inside the bar; the exact second is not knowable

        if order.price_type in ("SL", "SL-M") and not order.triggered:
            trig = order.trigger_price
            hit = h >= trig if buy else lo <= trig
            if not hit:
                return False
            order.triggered = True
            gap = (o >= trig) if buy else (o <= trig)
            touch = o if gap else trig
            if order.price_type == "SL-M":
                self._fill(order, self.costs.slip(order.action, touch), when, "stop")
                return True
            order.status = OPEN  # now a limit at order.price
            limit = order.price
            if (buy and touch <= limit) or (not buy and touch >= limit):
                self._fill(order, touch, when, "stop")
                return True
            return False

        if order.price_type in ("LIMIT", "SL"):
            limit = order.price
            if buy and lo <= limit:
                self._fill(order, min(limit, o), when, "stop" if order.triggered else "target")
                return True
            if not buy and h >= limit:
                self._fill(order, max(limit, o), when, "stop" if order.triggered else "target")
                return True
        return False

    def square_off_mis(self, at: float) -> int:
        """Close MIS positions and cancel MIS orders on exchanges whose square-off time this is."""
        sec = sec_of_day(at)
        closed = 0
        for order in list(self.pending):
            session = session_for(order.exchange)
            if order.product == "MIS" and session.square_off_sec == sec:
                order.status = CANCELLED
                order.reason = "cancelled at the MIS square-off time"
        self.pending = [o for o in self.pending if o.status in (OPEN, TRIGGER_PENDING)]
        for pos in list(self.positions.values()):
            session = session_for(pos.exchange)
            if pos.product == "MIS" and pos.qty and session.square_off_sec == sec:
                closed += self._close_position(pos, "square-off")
        return closed

    def _close_position(self, pos: Position, tag: str) -> int:
        q = pos.qty
        if not q:
            return 0
        ltp = self.market.get(pos.symbol, pos.exchange).price_at(self.now())
        if ltp is None:
            ltp = pos.average_price
        action = "SELL" if q > 0 else "BUY"
        order = Order(
            orderid=self._next_id(),
            strategy="",
            symbol=pos.symbol,
            exchange=pos.exchange,
            action=action,
            quantity=abs(q),
            price_type="MARKET",
            product=pos.product,
            price=0.0,
            trigger_price=0.0,
            placed_at=self.now(),
            status=OPEN,
            tag=tag,
        )
        self.orders[order.orderid] = order
        self._fill(order, self.costs.slip(action, ltp), self.now(), tag)
        return 1

    def end_of_test(self) -> None:
        for order in self.pending:
            order.status = CANCELLED
            order.reason = "cancelled at the end of the backtest"
        self.pending = []
        for pos in list(self.positions.values()):
            self._close_position(pos, "end of backtest")

    # ---------- fills and round trips ----------
    def _fill(self, order: Order, price: float, when: float, tag: str) -> None:
        price = float(price)
        order.status = COMPLETE
        order.average_price = price
        order.filled_at = when
        cost = self.costs.for_fill(order.action, order.quantity, price)
        self.total_costs += cost
        self.fills.append(
            {
                "action": order.action,
                "symbol": order.symbol,
                "exchange": order.exchange,
                "orderid": order.orderid,
                "product": order.product,
                "quantity": str(order.quantity),
                "average_price": round(price, 4),
                "trade_value": round(price * order.quantity, 2),
                "timestamp": f"{ist_datetime(when):%d-%b-%Y %H:%M:%S}",
                "epoch": when,
                "costs": round(cost, 2),
            }
        )

        pos = self.position(order.symbol, order.exchange, order.product)
        signed = order.quantity if order.action == "BUY" else -order.quantity
        exit_cpu = cost / order.quantity
        remaining = signed
        while remaining and pos.lots and (pos.lots[0].qty > 0) != (remaining > 0):
            lot = pos.lots[0]
            m = min(abs(remaining), abs(lot.qty))
            direction = 1 if lot.qty > 0 else -1
            gross = (price - lot.price) * m * direction
            costs = (lot.cost_per_unit + exit_cpu) * m
            self.realized += gross
            self.trades.append(
                {
                    "symbol": order.symbol,
                    "exchange": order.exchange,
                    "product": order.product,
                    "side": "LONG" if direction > 0 else "SHORT",
                    "quantity": m,
                    "entry_time": lot.opened_at,
                    "entry_price": round(lot.price, 4),
                    "exit_time": when,
                    "exit_price": round(price, 4),
                    "gross_pnl": round(gross, 2),
                    "costs": round(costs, 2),
                    "pnl": round(gross - costs, 2),
                    "exit_reason": tag,
                }
            )
            lot.qty -= m * direction
            remaining += m * direction
            if lot.qty == 0:
                pos.lots.popleft()
        if remaining:
            pos.lots.append(Lot(remaining, price, when, exit_cpu))

    # ---------- books ----------
    def order_status(self, order_id) -> dict:
        order = self.orders.get(str(order_id))
        if order is None:
            return _error(f"Order {order_id} not found")
        return {"status": "success", "data": order.as_dict()}

    def orderbook(self) -> dict:
        orders = [o.as_dict() for o in self.orders.values()]
        stats = {
            "total_buy_orders": sum(o["action"] == "BUY" for o in orders),
            "total_sell_orders": sum(o["action"] == "SELL" for o in orders),
            "total_completed_orders": sum(o["order_status"] == COMPLETE for o in orders),
            "total_open_orders": sum(o["order_status"] in (OPEN, TRIGGER_PENDING) for o in orders),
            "total_rejected_orders": sum(o["order_status"] == REJECTED for o in orders),
        }
        return {"status": "success", "data": {"orders": orders, "statistics": stats}}

    def tradebook(self) -> dict:
        return {
            "status": "success",
            "data": [
                {k: v for k, v in f.items() if k not in ("epoch", "costs")} for f in self.fills
            ],
        }

    def positionbook(self) -> dict:
        rows = []
        for pos in self.positions.values():
            q = pos.qty
            avg = pos.average_price
            ltp = self.mark(pos.symbol, pos.exchange, avg)
            rows.append(
                {
                    "symbol": pos.symbol,
                    "exchange": pos.exchange,
                    "product": pos.product,
                    "quantity": str(q),
                    "average_price": f"{avg:.2f}",
                    "ltp": f"{ltp:.2f}",
                    "pnl": f"{q * (ltp - avg):.2f}",
                }
            )
        return {"status": "success", "data": rows}

    def funds(self) -> dict:
        used = self.exposure()
        equity = self.equity()
        return {
            "status": "success",
            "data": {
                "availablecash": f"{equity - used:.2f}",
                "collateral": "0.00",
                "m2mrealized": f"{self.realized - self.total_costs:.2f}",
                "m2munrealized": f"{self.unrealized():.2f}",
                "utiliseddebits": f"{used:.2f}",
            },
        }

    def orders_report(self) -> list[dict]:
        out = []
        for o in self.orders.values():
            row = o.as_dict()
            row["placed_at"] = o.placed_at
            row["filled_at"] = o.filled_at
            row["reason"] = o.reason
            row["day"] = day_of(o.placed_at)
            out.append(row)
        return out


def _error(message: str) -> dict:
    return {"status": "error", "message": message}
