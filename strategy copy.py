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

	def __init__(self, broker, sizer=None, bar_history=100, timeframes: dict = None):
		self.broker = broker
		self.sizer = sizer or FixedSizeSizer(default_size=1)
		self.bar_history = defaultdict(lambda: BarHistory(size=bar_history)) #BarHistory(size=bar_history)
		self.wait_for_cool_off_period = False
		self.next_method = None  # 'on_bar' or 'on_data'
		self.market_close_time = (datetime.strptime(self.broker.market_close_time, "%H:%M:%S") - timedelta(minutes=5)).time()
		self.market_open_time = datetime.strptime(self.broker.market_open_time, "%H:%M:%S").time()

		# Multi-timeframe support
		# timeframes format: {"5m": history_size, "15m": history_size}
		self.timeframes = timeframes or {}
		self.resampled_history = {}
		self._tf_accumulator = {}

		for tf_str, size in self.timeframes.items():
			if not tf_str.endswith('m'):
				raise ValueError(f"Only minutes ('Xm') timeframes are supported, got '{tf_str}'.")
			minutes = int(tf_str[:-1])
			self.resampled_history[tf_str] = defaultdict(lambda: BarHistory(size=size))
			self._tf_accumulator[tf_str] = {
				"minutes": minutes,
				"symbols": defaultdict(dict),
				"is_new_bar": False
			}
	
	def is_new_bar(self, timeframe: str) -> bool:
		"""Returns True if the specified timeframe's accumulated bar was just finalized on this tick."""
		if timeframe not in self._tf_accumulator:
			return False
		return self._tf_accumulator[timeframe]["is_new_bar"]

	def get_resampled_history(self, timeframe: str, symbol: str) -> BarHistory:
		"""Retrieve the aggregated BarHistory for a given timeframe and symbol."""
		return self.resampled_history.get(timeframe, {}).get(symbol)

	def next(self, ts, data):
		all_warmed_up = True
		for symbol in data:
			# Always add the latest bar to the history for each symbol.
			self.bar_history[symbol].add(data[symbol])
			if self.bar_history[symbol].count < self.bar_history[symbol].size:
				all_warmed_up = False  # This symbol isn't ready yet; don't exit, let others accumulate

		# Process multi-timeframe aggregations
		import numpy as np
		for tf_str, acc_data in self._tf_accumulator.items():
			minutes = acc_data["minutes"]
			# E.g. for 5m, the bar is completed when the minute ends in 4 or 9 (e.g. 09:19, 09:24...)
			is_completed = (ts.minute % minutes) == (minutes - 1)
			acc_data["is_new_bar"] = is_completed

			for symbol, bar in data.items():
				acc = acc_data["symbols"][symbol]
				# Assuming standard indices [ts, open, high, low, close, volume]
				# or [open, high, low, close, volume, ...]
				# Let's map safely based on IDX = 1, 2, 3, 4, 5
				o, h, low, c, v = float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4]), float(bar[5])

				if not acc:
					acc['open'] = o
					acc['high'] = h
					acc['low'] = low
					acc['volume'] = v
				else:
					acc['high'] = max(acc['high'], h)
					acc['low'] = min(acc['low'], low)
					acc['volume'] += v
				
				acc['close'] = c

				if is_completed:
					b_agg = np.array([ts, acc['open'], acc['high'], acc['low'], acc['close'], acc['volume']], dtype=object)
					self.resampled_history[tf_str][symbol].add(b_agg)
					acc.clear()

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
