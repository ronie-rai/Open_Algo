"""``openalgo.api`` as a backtested strategy sees it.

Method names, keyword arguments and response shapes follow the real SDK
client closely enough that a strategy written for it runs unchanged. Market
data comes from the replay, orders go to the simulated broker, and a short
list of read-only metadata calls (symbol, search, expiry, intervals) is passed
to the real app. Anything else answers with an error a strategy can log, and
streaming calls report that they did not connect.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("python_backtest")

_STREAMING = ("connect", "disconnect", "subscribe_", "unsubscribe_")


class BacktestClient:
    def __init__(self, engine, market, broker, provider, api_key=None, host=None, **_ignored):
        self._engine = engine
        self._market = market
        self._broker = broker
        self._provider = provider
        self.api_key = api_key
        self.host = host
        self._symbol_cache: dict = {}
        self._warned: set[str] = set()

    # ---------- plumbing ----------
    def _check(self) -> float:
        """The simulated now; ends a loop that keeps calling after the test is over."""
        return self._engine.note_call()

    def _ensure_data(self, symbol, exchange):
        series = self._market.get(symbol, exchange)
        if not self._market.has_any_data():
            self._engine.finish(
                "failed",
                f"No historical data was found for {series.symbol} on {series.exchange} "
                f"between {self._market.start} and {self._market.end}. Download it in "
                "Historify, or log in to your broker so it can be fetched, then run the "
                "backtest again.",
            )
        return series

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        streaming = name.startswith(_STREAMING)

        def unavailable(*_args, **_kwargs):
            if name not in self._warned:
                self._warned.add(name)
                logger.info(f"{name} is not available in a backtest; the strategy was told so")
            if streaming:
                return False
            return {"status": "error", "message": f"{name} is not available in backtest"}

        return unavailable

    # ---------- market data ----------
    def history(self, *, symbol, exchange, interval, start_date, end_date, source="api", **_kw):
        now = self._check()
        self._ensure_data(symbol, exchange)
        return self._market.history(symbol, exchange, interval, start_date, end_date, now)

    def _quote(self, symbol, exchange) -> dict | None:
        now = self._check()
        series = self._ensure_data(symbol, exchange)
        ltp = series.price_at(now)
        if ltp is None:
            return None
        bar = series.today_bar(now)
        o, h, lo, _c, v, oi = bar if bar is not None else (ltp, ltp, ltp, ltp, 0.0, 0.0)
        prev = series.prev_close(now)
        return {
            "open": o,
            "high": h,
            "low": lo,
            "ltp": ltp,
            "ask": ltp,
            "bid": ltp,
            "prev_close": prev if prev is not None else o,
            "volume": int(v),
            "oi": int(oi),
        }

    def quotes(self, *, symbol, exchange, **_kw):
        data = self._quote(symbol, exchange)
        if data is None:
            return {"status": "error", "message": f"No price for {symbol} at this time"}
        return {"status": "success", "data": data}

    def multiquotes(self, *, symbols, **_kw):
        results = []
        for item in symbols or []:
            sym, exch = item.get("symbol"), item.get("exchange")
            data = self._quote(sym, exch)
            if data is None:
                results.append({"symbol": sym, "exchange": exch, "error": "No price at this time"})
            else:
                results.append({"symbol": sym, "exchange": exch, "data": data})
        return {"status": "success", "results": results}

    def depth(self, *, symbol, exchange, **_kw):
        data = self._quote(symbol, exchange)
        if data is None:
            return {"status": "error", "message": f"No price for {symbol} at this time"}
        level = [{"price": data["ltp"], "quantity": 0}] + [{"price": 0, "quantity": 0}] * 4
        return {
            "status": "success",
            "data": {
                **data,
                "ltq": 0,
                "totalbuyqty": 0,
                "totalsellqty": 0,
                "asks": [dict(x) for x in level],
                "bids": [dict(x) for x in level],
            },
        }

    def symbol(self, *, symbol, exchange, **kwargs):
        self._check()
        key = (str(symbol).upper(), str(exchange).upper())
        if key not in self._symbol_cache:
            resp = self._provider.call_readonly(
                "symbol", symbol=symbol, exchange=exchange, **kwargs
            )
            if not (isinstance(resp, dict) and resp.get("status") == "success"):
                resp = {
                    "status": "success",
                    "data": {
                        "symbol": key[0],
                        "exchange": key[1],
                        "tick_size": 0.05,
                        "lotsize": 1,
                    },
                }
            self._symbol_cache[key] = resp
        return self._symbol_cache[key]

    def _readonly(self, method, **kwargs):
        self._check()
        resp = self._provider.call_readonly(method, **kwargs)
        if resp is None:
            return {"status": "error", "message": f"{method} is not available in backtest"}
        return resp

    def search(self, *, query, exchange=None, **kwargs):
        return self._readonly("search", query=query, exchange=exchange, **kwargs)

    def expiry(self, *, symbol, exchange, instrumenttype, **kwargs):
        return self._readonly(
            "expiry", symbol=symbol, exchange=exchange, instrumenttype=instrumenttype, **kwargs
        )

    def intervals(self, **kwargs):
        return self._readonly("intervals", **kwargs)

    def interval(self):
        return self.intervals()

    def ping(self, **_kw):
        self._check()
        return {"status": "success", "data": {"broker": "backtest", "message": "pong"}}

    # ---------- orders ----------
    @staticmethod
    def _price_type(price_type, kwargs):
        return kwargs.pop("pricetype", None) or price_type

    def placeorder(
        self,
        *,
        strategy="Python",
        symbol,
        action,
        exchange,
        price_type="MARKET",
        product="MIS",
        quantity=1,
        **kwargs,
    ):
        self._check()
        self._ensure_data(symbol, exchange)
        return self._broker.place(
            strategy=strategy,
            symbol=symbol,
            exchange=exchange,
            action=action,
            quantity=quantity,
            price_type=self._price_type(price_type, kwargs),
            product=product,
            price=kwargs.get("price", 0),
            trigger_price=kwargs.get("trigger_price", 0),
        )

    def placesmartorder(
        self,
        *,
        strategy="Python",
        symbol,
        action,
        exchange,
        price_type="MARKET",
        product="MIS",
        quantity=1,
        position_size,
        **kwargs,
    ):
        self._check()
        self._ensure_data(symbol, exchange)
        return self._broker.place_smart(
            position_size=position_size,
            strategy=strategy,
            symbol=symbol,
            exchange=exchange,
            action=action,
            quantity=quantity,
            price_type=self._price_type(price_type, kwargs),
            product=product,
            price=kwargs.get("price", 0),
            trigger_price=kwargs.get("trigger_price", 0),
        )

    def basketorder(self, *, strategy="Python", orders, **_kw):
        self._check()
        results = []
        for item in orders or []:
            item = dict(item)
            sym = item.pop("symbol", "")
            resp = self.placeorder(
                strategy=strategy,
                symbol=sym,
                action=item.pop("action", ""),
                exchange=item.pop("exchange", ""),
                price_type=item.pop("pricetype", item.pop("price_type", "MARKET")),
                product=item.pop("product", "MIS"),
                quantity=item.pop("quantity", 0),
                **item,
            )
            results.append(
                {
                    "symbol": sym,
                    "status": resp.get("status"),
                    "orderid": resp.get("orderid"),
                    "message": resp.get("message"),
                }
            )
        return {"status": "success", "results": results}

    def splitorder(
        self,
        *,
        strategy="Python",
        symbol,
        action,
        exchange,
        quantity,
        splitsize,
        price_type="MARKET",
        product="MIS",
        **kwargs,
    ):
        self._check()
        try:
            total, size = int(float(quantity)), int(float(splitsize))
        except (TypeError, ValueError):
            return {"status": "error", "message": "quantity and splitsize must be numbers"}
        if total <= 0 or size <= 0:
            return {"status": "error", "message": "quantity and splitsize must be above zero"}
        results, left, n = [], total, 0
        while left > 0:
            n += 1
            part = min(size, left)
            resp = self.placeorder(
                strategy=strategy,
                symbol=symbol,
                action=action,
                exchange=exchange,
                price_type=price_type,
                product=product,
                quantity=part,
                **kwargs,
            )
            results.append(
                {
                    "order_num": n,
                    "quantity": part,
                    "status": resp.get("status"),
                    "orderid": resp.get("orderid"),
                }
            )
            left -= part
        return {
            "status": "success",
            "results": results,
            "split_size": size,
            "total_quantity": total,
        }

    def orderstatus(self, *, order_id, strategy="Python", **_kw):
        self._check()
        return self._broker.order_status(order_id)

    def openposition(self, *, strategy="Python", symbol, exchange, product, **_kw):
        self._check()
        qty = self._broker.net_qty(
            str(symbol).strip().upper(), str(exchange).strip().upper(), str(product).upper()
        )
        return {"status": "success", "quantity": str(qty)}

    def modifyorder(self, *, order_id, strategy="Python", **kwargs):
        self._check()
        if "pricetype" in kwargs and "price_type" not in kwargs:
            kwargs["price_type"] = kwargs.pop("pricetype")
        return self._broker.modify(order_id, **kwargs)

    def cancelorder(self, *, order_id, strategy="Python", **_kw):
        self._check()
        return self._broker.cancel(order_id)

    def cancelallorder(self, *, strategy="Python", **_kw):
        self._check()
        return self._broker.cancel_all()

    def closeposition(self, *, strategy="Python", **_kw):
        self._check()
        self._broker.close_all(tag="strategy")
        return {
            "status": "success",
            "message": "All Open Positions Squared Off",
            "mode": "backtest",
        }

    # ---------- account ----------
    def funds(self, **_kw):
        self._check()
        return self._broker.funds()

    def orderbook(self, **_kw):
        self._check()
        return self._broker.orderbook()

    def tradebook(self, **_kw):
        self._check()
        return self._broker.tradebook()

    def positionbook(self, **_kw):
        self._check()
        return self._broker.positionbook()

    def holdings(self, **_kw):
        self._check()
        return {
            "status": "success",
            "data": {
                "holdings": [],
                "statistics": {
                    "totalholdingvalue": 0,
                    "totalinvvalue": 0,
                    "totalpnlpercentage": 0,
                    "totalprofitandloss": 0,
                },
            },
        }

    def analyzerstatus(self, **_kw):
        self._check()
        return {
            "status": "success",
            "data": {"analyze_mode": True, "mode": "backtest", "total_logs": 0},
        }

    def analyzertoggle(self, mode=None, **_kw):
        self._check()
        return {"status": "error", "message": "analyzertoggle is not available in backtest"}

    def telegram(self, *, username=None, message=None, priority=5, **_kw):
        self._check()
        return {"status": "success", "message": "Telegram alert skipped in backtest"}

    def close(self):
        return None
