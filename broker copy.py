from execution_model import SimpleExecutionModel
from order import Order, Position, Trade
from trading_calendar import WorkingDayTradingCalendar
import logging

logger = logging.getLogger(__name__)


class Broker:
    def __init__(self, cash=10_000, execution_model=None):
        self.orders = {}
        self.positions = {}  # key: asset, value: Position
        self.trades = {}
        self.cash = cash
        self.commission = 0
        self.current_ts = (
            None  # Set by the backtester on each bar before processing orders
        )
        self.trading_calendar = WorkingDayTradingCalendar()
        self.ex_model = execution_model or SimpleExecutionModel()

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

    def process_orders(self, ts, data):
        executed_trades = []
        for order_id, order in self.orders.items():
            if order.status not in ("CREATED", "PARTIALLY_FILLED"):
                continue
            trade = self._process_order(order, data)
            if trade:
                executed_trades.append(trade)
            self.remove_zero_positions(order.asset)
        return executed_trades

    def get_commission(self, size, price):
        return self.commission

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
        timestamp = ts

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

        # Check if we are opening a NEW position to see if we have enough cash
        opened, closed, _, _ = position.pseudo_update(size, fill_price)

        # Update cash based on the trade
        trade_value = fill_size * fill_price
        commission = self.get_commission(fill_size, fill_price)

        # calc commission on positions
        open_value = opened * fill_price
        close_value = closed * fill_price
        commission = self.get_commission(fill_size, fill_price)

        cash_in = close_value
        cash_out = open_value + commission

        if opened > 0 and open_value > self.cash:
            order.status = "REJECTED"
            return None

        self.cash += cash_in  # reinject value of closed positons
        self.cash -= cash_out  # deduct open positions

        # if order.side == "BUY":
        #     total_cost = trade_value + commission
        #     if opened > 0 and total_cost > self.cash:
        #         order.status = "REJECTED"
        #         logger.info(f"Insufficient cash for order {order.id}.")
        #         return None
        #     self.cash -= total_cost
        # else:  # side == 'SELL'
        #     total_receive = trade_value - commission
        #     # Note: For short selling, we should ideally check margin, but here we just credit cash
        #     self.cash += total_receive

        # Actually update the position and finalize the trade
        old_avg_price = position.avg_price
        opened, closed, new_size, new_avg_price = position.update(size, fill_price)
        realized_pnl = self._calculate_realized_pnl(
            order, old_avg_price, fill_price, closed
        )

        order.add_fill()

        # if fill_size >= order.remaining_size:
        #     # Full fill of remaining portion
        #     order.execute(price=fill_price, pnl=realized_pnl, dt=timestamp)
        #     logger.info(
        #         f"Order FULLY executed {order.side}: {order.id} | Size: {fill_size} | Price: {fill_price} | PnL: {realized_pnl}"
        #     )
        # else:
        #     # Partial fill — keep the order alive for retry on the next bar
        #     order.filled_size += fill_size
        #     order.remaining_size -= fill_size
        #     order.status = "PARTIALLY_FILLED"
        #     order.last_modified = timestamp
        #     logger.info(
        #         f"Order PARTIALLY filled {order.side}: {order.id} | Filled: {fill_size} | Remaining: {order.remaining_size} | Price: {fill_price}"
        #     )

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

        return _trade

    def _calculate_realized_pnl(self, order, avg_price, exec_price, closed):
        """
        Calculate realized PnL for a closing trade.
        """
        if closed == 0:
            return 0
        # If we were BUYing, it means we are closing a SHORT.
        # PnL = (Entry - Exit) * Size. Entry is avg_price, Exit is exec_price.
        if order.side == "BUY":
            return (avg_price - exec_price) * abs(closed)
        else:  # We were SELLING to close a LONG.
            # PnL = (Exit - Entry) * Size. Exit is exec_price, Entry is avg_price.
            return (exec_price - avg_price) * abs(closed)

    def _should_execute_order(self, order, bar):
        # Note: LIMIT order execution gets you best price within the bar
        # note: SL order execution gets you worst price within the bar

        ts, o, h, l, c, v = bar
        order_type = order.type.upper()

        if order_type == "MARKET":
            # Market orders fill at the Open of the bar (the earliest available price)
            return True, o

        elif order_type == "LIMIT":
            # Fill if target price is within bar's High-Low range
            # min or max is used to get realistic fill price
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
            # Trigger price must be hit within the bar
            if (order.side == "BUY" and h >= order.trigger_price) or (
                order.side == "SELL" and l <= order.trigger_price
            ):
                # Then check if limit price is hit within the same bar (simplified)
                if (order.side == "BUY" and l <= order.price) or (
                    order.side == "SELL" and h >= order.price
                ):
                    return True, order.price

        else:
            logger.warning(f"Unknown order type: {order_type}")

        return False, None

    def get_position(self, asset):
        return self.positions.get(asset, Position(asset, size=0, avg_price=0))
