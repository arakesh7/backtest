from typing import Optional
from order import Order
import pandas as pd
from history import BarHistory
from collections import defaultdict
from datetime import datetime, timedelta
from sizer import FixedSizeSizer

# bar_hist = lambda size: defaultdict(lambda: BarHistory(size=size))

import logging
logger = logging.getLogger(__name__)


class Strategy:
	"""Base Strategy class to be subclassed for custom strategies.

	Subclasses should implement an `on_bar` or `on_data` method and call
	`self.buy(...)` / `self.sell(...)` to create orders.
	"""

	def __init__(self, broker, sizer=None, bar_history=100):
		self.broker = broker
		self.sizer = sizer or FixedSizeSizer(default_size=1)
		self.bar_history = defaultdict(lambda: BarHistory(size=bar_history)) #BarHistory(size=bar_history)
		#self._hist = defaultdict(lambda: BarHistory(size=bar_history))
		self.wait_for_cool_off_period = False
		self.next_method = None  # 'on_bar' or 'on_data'
		self.market_close_time = (datetime.strptime(self.broker.market_close_time, "%H:%M:%S") - timedelta(minutes=5)).time()
		self.market_open_time = datetime.strptime(self.broker.market_open_time, "%H:%M:%S").time()
	
	def next(self, ts, data):
		all_warmed_up = True
		for symbol in data:
			# Always add the latest bar to the history for each symbol.
			self.bar_history[symbol].add(data[symbol])
			if self.bar_history[symbol].count < self.bar_history[symbol].size:
				all_warmed_up = False  # This symbol isn't ready yet; don't exit, let others accumulate

		if not all_warmed_up:
			return

		# Always call on_data. The strategy itself is responsible for
		# checking if it has enough historical data to proceed.
		self.on_data(ts, data)
		
	def close_on_eod(self, ts: pd.Timestamp):
		"""
		Checks if the current timestamp is the market close time and closes all open positions.
		This method should be called by intraday strategies within on_data.
		"""
		if self.market_close_time and ts.time() == self.market_close_time:
			logger.info(f"Market close time reached at {ts}. Closing all open positions.")
			self._close_all_positions()
		

	def _close_all_positions(self):
		"""
		Iterates through all open positions and places market orders to close them.
		"""
		# Create a copy of position items to avoid issues with modifying the dictionary while iterating
		for asset in self.broker.positions:
			self.close(asset)

	def close(self, asset: str):
		"""
		Closes any open position for the given asset by placing a market order.
		"""
		position = self.broker.get_position(asset)
		if position.size > 0:  # Long position
			self.sell(asset, size=position.size)
		elif position.size < 0:  # Short position
			self.buy(asset, size=abs(position.size))

	def on_data(self, ts, data):
		"""Override in subclass. Called by the backtesting loop with market data."""
		raise NotImplementedError()

	def buy(self, asset: str, size: int = None, order_type: str = 'MARKET', price: Optional[float] = None,
			trigger_price: Optional[float] = None, stop_price: Optional[float] = None,
			entry_price: float = None, stop_loss_price: float = None) -> Optional[Order]:
		"""Place a buy order via the broker. Supported types: MARKET, LIMIT, STOPLIMIT, SL.

		If `size` is None, the configured sizer will be used to determine the order size.
		Returns the created Order instance.
		"""
		if size is None:
			if entry_price is None:
				raise ValueError("`entry_price` must be provided to the buy method when `size` is not specified for sizer calculation.")
			size = self.sizer.get_size(self.broker, asset, entry_price, stop_loss_price, side='BUY')
		
		if not size or size <= 0:
			return None

		return self.broker.place_order(asset=asset, size=size, side='BUY', order_type=order_type,
									price=price, trigger_price=trigger_price, stop_price=stop_price)

	def sell(self, asset: str, size: int = None, order_type: str = 'MARKET', price: Optional[float] = None,
			 trigger_price: Optional[float] = None, stop_price: Optional[float] = None,
			 entry_price: float = None, stop_loss_price: float = None) -> Optional[Order]:
		"""Place a sell order via the broker. Supported types: MARKET, LIMIT, STOPLIMIT, SL."""
		if size is None:
			if entry_price is None:
				raise ValueError("`entry_price` must be provided to the sell method when `size` is not specified for sizer calculation.")
			size = self.sizer.get_size(self.broker, asset, entry_price, stop_loss_price, side='SELL')
		
		if not size or size <= 0:
			return None

		return self.broker.place_order(asset=asset, size=size, side='SELL', order_type=order_type,
									price=price, trigger_price=trigger_price, stop_price=stop_price)

	def has_open_position(self, asset=None) -> bool:
		if asset is None:
			return len(self.broker.positions) > 0
		else:
			position = self.broker.get_position(asset)
			return position.size != 0




class Intraday(Strategy):
    def is_trades_taken_today(self, ts):
        """Returns True if any trade was executed on the same date as ts."""
        return any(
            t.executed_at.date() == ts.date()
            for t in self.broker.trades.values()
        )
    

class IntradayMixin:
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.market_close_time = (datetime.strptime(self.broker.market_close_time, "%H:%M:%S") - timedelta(minutes=5)).time()
		self._current_day = None

	def is_new_day(self, ts):
		return self._current_day != ts.date()

	def on_new_day(self, ts):
		self._current_day = ts.date()
		# Reset any daily state variables here
		logger.info(f"New day {self._current_day}. Resetting daily state.")
