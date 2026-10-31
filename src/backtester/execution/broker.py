import logging
from .execution_model import SimpleExecutionModel
from .commission import Commission, NoCommission
from ..orders.order import Order, Position, Trade
from ..engine.calendar import WorkingDayTradingCalendar

logger = logging.getLogger(__name__)


class Broker:
    def __init__(
        self,
        cash=10_000,
        execution_model=None,
        commission_model=None,
        intra_bar_fill_model="CONSERVATIVE",
        margin_ratio: float = 0.5,
    ):
        self.orders = {}
        self.positions = {}  # key: asset, value: Position
        self.trades = {}
        self.cash = cash
        self.commission = 0
        self.commission_model = commission_model or NoCommission()
        self.current_prices = {}  # key: asset, value: latest known market price
        self.current_ts = (
            None  # Set by the backtester on each bar before processing orders
        )
        self.trading_calendar = WorkingDayTradingCalendar()
        self.ex_model = execution_model or SimpleExecutionModel()
        self.intra_bar_fill_model = intra_bar_fill_model.upper()
        if self.intra_bar_fill_model not in ("POLARITY", "CONSERVATIVE"):
            raise ValueError(
                f"intra_bar_fill_model must be 'POLARITY' or 'CONSERVATIVE', got '{intra_bar_fill_model}'"
            )
        self.margin_ratio = margin_ratio

    @property
    def equity(self) -> float:
        """
        Total portfolio Net Asset Value (NAV) marked to market:
        Cash + sum of open position values based on latest market price.
        Falls back to avg_price if current market price is not yet available.
        """
        pos_val = sum(
            pos.size * self.current_prices.get(asset, pos.avg_price)
            for asset, pos in self.positions.items()
            if pos.size != 0
        )
        return self.cash + pos_val

    def get_available_cash(self) -> float:
        """
        Calculates cash available to enter new positions (free unencumbered cash).
        Total cash minus short liabilities and required short margin collateral,
        marked to market using the latest available price.
        """
        short_reserve = sum(
            abs(pos.size) * self.current_prices.get(pos.asset, pos.avg_price) * (1.0 + self.margin_ratio)
            for pos in self.positions.values()
            if pos.size < 0
        )
        return max(0.0, self.cash - short_reserve)

    @property
    def market_close_time(self):
        return self.trading_calendar.market_close_time

    @property
    def market_open_time(self):
        return self.trading_calendar.market_open_time

    def add_execution_model(self, execution_model):
        self.ex_model = execution_model

    def place_order(
        self,
        asset: str,
        size: int,
        side: str,
        order_type: str = "MARKET",
        price: float = None,
        trigger_price: float = None,
        stop_price: float = None,
        oco_group_id: str = None,
    ) -> Order:
        order = Order(
            size=size,
            asset=asset,
            side=side,
            price=price,
            type=order_type,
            created_at=self.current_ts,
            trigger_price=trigger_price,
            stop_price=stop_price,
            oco_group_id=oco_group_id,
        )
        self.orders[order.id] = order
        logger.info(f"Order placed: {order}")
        return order

    def get_order_history(self):
        return self.orders

    def cancel_order(self, order_id):
        if order_id not in self.orders:
            raise ValueError(f"Order ID {order_id} not found.")

        order = self.orders[order_id]
        order.cancel()

    def remove_zero_positions(self, asset):
        if asset in self.positions and self.positions[asset].size == 0:
            del self.positions[asset]

    def _cancel_oco_siblings(self, order):
        """Cancels any sibling order in the same OCO group or competing exit when position is closed."""
        for oid, o in list(self.orders.items()):
            if o.id == order.id or o.status not in ("CREATED", "PARTIALLY_FILLED"):
                continue
            # Explicit OCO group cancellation
            if order.oco_group_id and o.oco_group_id == order.oco_group_id:
                o.cancel()
                logger.info(f"OCO sibling order {o.id} cancelled due to fill of {order.id}")
            # Competing exit order on same asset when position is now closed
            elif (
                o.asset == order.asset
                and o.side == order.side
                and o.type.upper() in ("LIMIT", "SL", "STOPLIMIT")
                and (order.asset not in self.positions or self.positions[order.asset].size == 0)
            ):
                o.cancel()
                logger.info(f"Competing exit order {o.id} cancelled as position in {order.asset} is closed")

    def _sort_orders_for_execution(self, orders_list, data):
        """
        Groups and prioritizes active orders for execution on the current bar.
        If an asset has both a LIMIT (Target) and SL (Stop Loss) exit order that both trigger,
        resolves execution order according to intra_bar_fill_model ('POLARITY' or 'CONSERVATIVE').
        """
        if len(orders_list) <= 1:
            return orders_list

        by_asset = {}
        for o in orders_list:
            by_asset.setdefault(o.asset, []).append(o)

        sorted_orders = []
        for asset, asset_orders in by_asset.items():
            if len(asset_orders) > 1 and asset in data and data[asset] is not None:
                bar = data[asset]
                pos = self.get_position(asset)

                limit_orders = [o for o in asset_orders if o.type.upper() == "LIMIT"]
                sl_orders = [o for o in asset_orders if o.type.upper() == "SL"]

                if limit_orders and sl_orders and pos.size != 0:
                    tgt = limit_orders[0]
                    sl = sl_orders[0]

                    tgt_triggered, _ = self._should_execute_order(tgt, bar)
                    sl_triggered, _ = self._should_execute_order(sl, bar)

                    if tgt_triggered and sl_triggered:
                        first = self._resolve_exit_conflict(pos, tgt, sl, bar)
                        second = tgt if first == sl else sl
                        other = [o for o in asset_orders if o not in (tgt, sl)]
                        sorted_orders.extend([first, second] + other)
                        continue

            sorted_orders.extend(asset_orders)

        return sorted_orders

    def _resolve_exit_conflict(self, pos, tgt, sl, bar):
        """
        Resolves whether Target (tgt) or Stop Loss (sl) executes first on an ambiguous bar.
        'CONSERVATIVE': Prudential / Worst-Case (Stop Loss always hit first).
        'POLARITY': Candle Polarity & Open Gap Trajectory Modeling (Default).
        """
        if self.intra_bar_fill_model == "CONSERVATIVE":
            return sl

        # 'POLARITY': Intra-bar price path reconstruction
        ts, o, h, l, c, v = bar

        if pos.size > 0:  # Long position
            # Gap at Open takes absolute priority
            if o <= sl.stop_price:
                return sl
            if o >= tgt.price:
                return tgt
            # Candle polarity: Bullish (c >= o) visited Low first; Bearish visited High first
            return sl if c >= o else tgt
        else:  # Short position
            if o >= sl.stop_price:
                return sl
            if o <= tgt.price:
                return tgt
            return tgt if c >= o else sl

    def process_orders(self, ts, data):
        # Update current market prices from the incoming bar data
        if data:
            for asset, bar in data.items():
                if bar is not None and len(bar) >= 5:
                    self.current_prices[asset] = float(bar[4])  # close price

        executed_trades = []
        active_orders = [
            order
            for order in self.orders.values()
            if order.status in ("CREATED", "PARTIALLY_FILLED")
        ]
        sorted_orders = self._sort_orders_for_execution(active_orders, data)

        for order in sorted_orders:
            if order.status not in ("CREATED", "PARTIALLY_FILLED"):
                continue
            trade = self._process_order(order, data)
            if trade:
                executed_trades.append(trade)
                self._cancel_oco_siblings(order)
            self.remove_zero_positions(order.asset)
        return executed_trades

    def get_commission(self, size, price):
        if self.commission_model is not None and not isinstance(self.commission_model, NoCommission):
            return self.commission_model.calculate(abs(size), price)
        if self.commission:
            return float(self.commission)
        return 0.0

    def check_for_funds(self):
        pass

    def _process_order(self, order, data):
        """
        Processes all orders with status 'CREATED' based on the latest market data.
        Executes orders if their conditions are met and updates positions accordingly.
        """
        if order.asset not in data:
            return None

        bar = data[order.asset]
        ts, o, h, l, c, v = bar

        execute, exec_price = self._should_execute_order(order, bar)
        if not execute:
            return None

        # Determine fill price and size via execution model (handle slippage/volume limits)
        fill_price, fill_size = self.ex_model.execute(exec_price, order, bar)

        if (fill_price is None or fill_price <= 0) or (not fill_size or fill_size <= 0):
            return None

        asset = order.asset
        if asset not in self.positions:
            self.positions[asset] = Position(asset=asset)
        position = self.positions[asset]

        # Positive size for BUY, negative for SELL
        size = fill_size if order.side == "BUY" else -fill_size
        commission = self.get_commission(fill_size, fill_price)

        curr_pos_size = position.size

        # Check account solvency before entering or increasing exposure
        if self.equity <= 0 and (
            (order.side == "BUY" and curr_pos_size >= 0)
            or (order.side == "SELL" and curr_pos_size <= 0)
        ):
            logger.warning(
                f"Order {order.id} REJECTED: Account is insolvent (equity: {self.equity:.2f} <= 0). "
                f"Cannot open or add to positions."
            )
            order.status = "REJECTED"
            return None

        # Solvency & Margin Validation for BUY and SELL orders
        if order.side == "BUY":
            if curr_pos_size >= 0:
                # Opening or adding to a LONG position
                required_capital = (fill_size * fill_price) + commission
                if required_capital > self.get_available_cash():
                    logger.warning(
                        f"Order {order.id} REJECTED: Insufficient available cash for BUY "
                        f"(required: {required_capital:.2f}, available: {self.get_available_cash():.2f})"
                    )
                    order.status = "REJECTED"
                    return None
            else:
                # Covering a SHORT position (partially, fully, or flipping to long)
                short_held = abs(curr_pos_size)
                covering_qty = min(fill_size, short_held)
                flipping_qty = max(0, fill_size - short_held)

                covering_cost = (covering_qty * fill_price) + commission
                if covering_cost > self.cash:
                    logger.warning(
                        f"Order {order.id} REJECTED: Insufficient cash to cover short "
                        f"(cost: {covering_cost:.2f}, cash: {self.cash:.2f})"
                    )
                    order.status = "REJECTED"
                    return None

                if flipping_qty > 0:
                    flipping_cost = flipping_qty * fill_price
                    # Reserve for short_held is released upon cover (marked to market)
                    curr_asset_price = self.current_prices.get(order.asset, position.avg_price)
                    released_reserve = short_held * curr_asset_price * (1.0 + self.margin_ratio)
                    est_avail_cash = (self.get_available_cash() + released_reserve) - covering_cost
                    if flipping_cost > est_avail_cash:
                        logger.warning(
                            f"Order {order.id} REJECTED: Insufficient cash to flip from short to long "
                            f"(required: {flipping_cost:.2f}, available: {est_avail_cash:.2f})"
                        )
                        order.status = "REJECTED"
                        return None
        else:  # SELL
            if curr_pos_size > 0:
                # Closing part or all of a LONG position (or flipping to short)
                if fill_size > curr_pos_size:
                    short_qty = fill_size - curr_pos_size
                    required_margin = (short_qty * fill_price * self.margin_ratio) + commission
                    proceeds_from_long = curr_pos_size * fill_price
                    est_avail_cash = self.get_available_cash() + proceeds_from_long
                    if required_margin > est_avail_cash:
                        logger.warning(
                            f"Order {order.id} REJECTED: Insufficient margin to flip to short "
                            f"(required: {required_margin:.2f}, available: {est_avail_cash:.2f})"
                        )
                        order.status = "REJECTED"
                        return None
            else:
                # Opening or adding to a SHORT position
                required_margin = (fill_size * fill_price * self.margin_ratio) + commission
                if required_margin > self.get_available_cash():
                    logger.warning(
                        f"Order {order.id} REJECTED: Insufficient margin for short sale "
                        f"(required: {required_margin:.2f}, available: {self.get_available_cash():.2f})"
                    )
                    order.status = "REJECTED"
                    return None

        # Actual update to the position and finalize the trade
        old_avg_price = position.avg_price
        old_size = position.size
        old_comm = getattr(position, "total_commission", 0.0)
        opened, closed, new_size, new_avg_price = position.update(size, fill_price)

        # Calculate entry commission allocated to the closed portion
        entry_comm_closed = 0.0
        if closed > 0 and abs(old_size) > 0:
            entry_comm_closed = old_comm * (closed / abs(old_size))

        # Commission for this order fill allocated to closing vs opening
        fill_abs = abs(size)
        if fill_abs > 0:
            exit_comm = commission * (closed / fill_abs)
            open_comm = commission * (opened / fill_abs)
        else:
            exit_comm = 0.0
            open_comm = 0.0

        total_trade_comm = entry_comm_closed + exit_comm

        # Update position's tracked commission
        if new_size == 0:
            position.total_commission = 0.0
        elif opened > 0 and closed > 0:
            # Flipped position
            position.total_commission = open_comm
        else:
            position.total_commission = (old_comm - entry_comm_closed) + open_comm

        realized_pnl = self._calculate_realized_pnl(
            order, old_avg_price, fill_price, closed, commission=total_trade_comm
        )

        _trade = Trade(
            asset,
            size,
            fill_price,
            order.side,
            commission,
            realized_pnl,
            self.current_ts,
            order.id,
        )
        self.trades[_trade.id] = _trade
        order.add_fill(_trade)
        self.current_prices[order.asset] = fill_price

        # Deduct or inject cash into the account
        if order.side == "BUY":
            self.cash -= (fill_size * fill_price) + commission
        else:  # SELL
            self.cash += (fill_size * fill_price) - commission

        return _trade

    def _calculate_realized_pnl(
        self, order, avg_price, exec_price, closed, commission=0.0
    ):
        """
        Calculate realized PnL for a closing trade (net of commission).
        """
        if closed == 0:
            return 0.0
        # If we were BUYing, it means we are closing a SHORT.
        # PnL = (Entry - Exit) * Size. Entry is avg_price, Exit is exec_price.
        if order.side == "BUY":
            gross = (avg_price - exec_price) * abs(closed)
        else:  # We were SELLING to close a LONG.
            # PnL = (Exit - Entry) * Size. Exit is exec_price, Entry is avg_price.
            gross = (exec_price - avg_price) * abs(closed)
        return gross - commission

    def _should_execute_order(self, order, bar):
        ts, o, h, l, c, v = bar
        order_type = order.type.upper()

        if order_type == "MARKET":
            # Market orders fill at the Open of the bar (the earliest available price)
            return True, o

        elif order_type == "LIMIT":
            # Fill if target price is within bar's High-Low range
            if order.side == "BUY" and l <= order.price:
                return True, min(o, order.price)
            if order.side == "SELL" and h >= order.price:
                return True, max(o, order.price)

        elif order_type == "SL":
            # Long stop: Sell when price drops below stop_price
            # Short stop: Buy when price rises above stop_price
            if order.side == "SELL" and l <= order.stop_price:  # GAP down
                return True, min(o, order.stop_price)

            if order.side == "BUY" and h >= order.stop_price:
                return True, max(o, order.stop_price)

        elif order_type == "STOPLIMIT":
            # If already triggered on an earlier bar, it acts as a standing LIMIT order
            if getattr(order, "is_triggered", False):
                if order.side == "BUY" and l <= order.price:
                    return True, min(o, order.price)
                if order.side == "SELL" and h >= order.price:
                    return True, max(o, order.price)
                return False, None

            # First time: check if trigger price was reached on this bar
            triggered = False
            if order.side == "BUY" and h >= order.trigger_price:
                triggered = True
            elif order.side == "SELL" and l <= order.trigger_price:
                triggered = True

            if triggered:
                order.is_triggered = True
                logger.info(
                    f"Order {order.id} STOPLIMIT triggered at trigger_price {order.trigger_price}; "
                    f"now active as standing LIMIT order at {order.price}"
                )
                # Check if limit price is also hit within this same bar
                if order.side == "BUY" and l <= order.price:
                    return True, order.price
                if order.side == "SELL" and h >= order.price:
                    return True, order.price

            return False, None

        else:
            logger.warning(f"Unknown order type: {order_type}")

        return False, None

    def get_position(self, asset):
        return self.positions.get(asset, Position(asset, size=0, avg_price=0))
