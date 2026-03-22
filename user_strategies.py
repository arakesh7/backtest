import logging
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import talib

from strategy import Strategy

logger = logging.getLogger(__name__)


class SimpleMAStrategy(Strategy):
    def __init__(self, broker, sizer=None, fast_period=10, slow_period=20):
        super().__init__(broker, sizer=sizer, bar_history=slow_period + 1)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_data(self, ts, data):
        for symbol in ["ADANIPORTS", "HAL", "ADANIGREEN", "TATAMOTORS"]:
            if symbol not in data:
                continue

            # Get historical closes
            entry_price = data[symbol][4]  # close price
            bars = self.bar_history[symbol]

            # More performant way to get closes. Index 4 is 'close' in (ts, o, h, l, c, v).
            closes = np.stack(bars[:])[:, 4].astype(float)

            # Calculate MAs using TA-Lib
            fast_ma = talib.SMA(closes, timeperiod=self.fast_period)
            slow_ma = talib.SMA(closes, timeperiod=self.slow_period)

            # Get current position
            position = self.broker.get_position(symbol)

            # Trading logic
            if fast_ma[-1] > slow_ma[-1] and position.size == 0:
                self.buy(symbol, entry_price=entry_price)
            elif fast_ma[-1] < slow_ma[-1] and position.size > 0:
                # Close the full position
                self.close(symbol)

        self.close_on_eod(ts)  # Ensure positions are closed at end of day


class Intraday(Strategy):
    def is_trades_taken_today(self, ts):
        """Returns True if any trade was executed on the same date as ts."""
        return any(
            t.executed_at.date() == ts.date()
            for t in self.broker.trades.values()
        )


class OHLCReversalStrategy(Intraday):
    """
    A daily mean-reversion strategy that trades intraday based on the previous day's OHLC.

    - Buy Signal: If the previous day was a "down day" (close < open) and
      today's open is below yesterday's low, we buy, expecting a reversal.
    - Sell Signal: If the previous day was an "up day" (close > open) and
      today's open is above yesterday's high, we sell, expecting a reversal.

    This is an intraday strategy: it trades only once at market open and closes
    all positions at the end of the day. It correctly aggregates daily OHLC
    even when running on intraday (e.g., 1-minute) data.
    """

    def __init__(self, broker, sizer=None):
        super().__init__(broker, sizer=sizer, bar_history=1)  # Only need current bar
        self._current_day = None
        self._prev_day_ohlc = defaultdict(dict)
        self._daily_agg = defaultdict(
            lambda: {"high": -np.inf, "low": np.inf, "open": None, "close": None}
        )

    def _update_daily_data(self, ts, data):
        """
        Handles the transition between days and aggregates the current bar's data.
        This must be called on every bar.
        """
        if self._current_day != ts.date():
            # Day has changed. Finalize the previous day's aggregation.
            if self._current_day is not None:
                self._prev_day_ohlc = self._daily_agg.copy()
                logger.debug(
                    f"[{ts.date()}] New day. Previous day's OHLC finalized and stored."
                )
            # Reset for the new day.
            self._daily_agg.clear()
            self._current_day = ts.date()

        # Aggregate the current bar's data for the current day.
        for symbol, bar in data.items():
            symbol_agg = self._daily_agg[symbol]
            if symbol_agg["open"] is None:
                symbol_agg["open"] = bar[1]  # Open
            symbol_agg["high"] = max(symbol_agg["high"], bar[2])  # High
            symbol_agg["low"] = min(symbol_agg["low"], bar[3])  # Low
            symbol_agg["close"] = bar[4]  # Close

    def on_data(self, ts, data):
        # First, update the daily aggregation. This ensures _prev_day_ohlc is set correctly on a new day.
        self._update_daily_data(ts, data)

        # Always close positions at the end of the day for an intraday strategy.
        self.close_on_eod(ts)

        # Trading logic only runs once per day at market open.
        if ts.time() == self.market_open_time:
            for symbol in data:
                prev_day = self._prev_day_ohlc.get(symbol)
                if not prev_day or any(
                    k not in prev_day for k in ["open", "high", "low", "close"]
                ):
                    continue  # Not enough data to trade

                current_open = data[symbol][1]  # Current bar's open price

                if (
                    prev_day["close"] < prev_day["open"]  # Previous day was a down day
                    and current_open
                    < prev_day[
                        "low"
                    ]  # Current open gapped down below previous day's low
                ):
                    logger.info(
                        f"[{ts}] {symbol}: BUY condition met! Prev C({prev_day['close']:.2f}) < Prev O({prev_day['open']:.2f}) AND Curr O({current_open:.2f}) < Prev L({prev_day['low']:.2f})"
                    )
                    self.buy(
                        symbol,
                        entry_price=current_open,
                        stop_loss_price=current_open * 0.98,
                    )
                elif (
                    prev_day["close"] > prev_day["open"]  # Previous day was an up day
                    and current_open
                    > prev_day[
                        "high"
                    ]  # Current open gapped up above previous day's high
                ):
                    logger.info(
                        f"[{ts}] {symbol}: SELL condition met! Prev C({prev_day['close']:.2f}) > Prev O({prev_day['open']:.2f}) AND Curr O({current_open:.2f}) > Prev H({prev_day['high']:.2f})"
                    )
                    self.sell(
                        symbol,
                        entry_price=current_open,
                        stop_loss_price=current_open * 1.02,
                    )
                else:
                    logger.debug(
                        f"[{ts}] {symbol}: No trade condition met. Prev Day OHLC: O={prev_day['open']:.2f}, H={prev_day['high']:.2f}, L={prev_day['low']:.2f}, C={prev_day['close']:.2f}. Current Open: {current_open:.2f}"
                    )


class OpeningRangeBreakoutStrategy(Intraday):
    """
    An intraday strategy based on the breakout of an opening range.

    - Logic: Identifies the high and low of the first N minutes (e.g., 30) of trading.
    - Buy Signal: Price breaks above the opening range high.
    - Sell Signal: Price breaks below the opening range low.
    - Risk Management:
        - Stop-Loss: The opposite side of the opening range.
        - Take-Profit: 2x the size of the opening range.
        - EOD close for any remaining open positions.
    """

    def __init__(
        self, broker, sizer=None, opening_range_minutes=30, take_profit_factor=2.0
    ):
        super().__init__(broker, sizer=sizer)
        self.opening_range_minutes = opening_range_minutes
        self.take_profit_factor = take_profit_factor

        # Calculate opening range end time
        market_open_dt = datetime.strptime(self.broker.market_open_time, "%H:%M:%S")
        self.opening_range_end_time = (
            market_open_dt + timedelta(minutes=self.opening_range_minutes)
        ).time()

        # Daily state variables
        self._current_day = None
        self._opening_range = defaultdict(
            lambda: {"high": -np.inf, "low": np.inf, "established": False}
        )
        self._trade_taken = defaultdict(bool)
        self._position_details = {}  # Stores SL/TP for open positions

    def _reset_daily_state(self):
        """Resets the state at the start of a new trading day."""
        self._opening_range.clear()
        self._trade_taken.clear()
        self._position_details.clear()
        logger.info(f"New day {self._current_day}. Resetting ORB daily state.")

    def on_data(self, ts, data):
        # --- Daily State Management ---
        if self._current_day != ts.date():
            self._current_day = ts.date()
            self._reset_daily_state()

        # --- EOD Position Closing ---
        self.close_on_eod(ts)

        # --- Position and Risk Management (Check for SL/TP) ---
        for symbol, details in list(self._position_details.items()):
            if symbol not in data or symbol not in self.broker.positions:
                continue

            current_price = data[symbol][4]  # Using close price for check
            if details["side"] == "BUY" and (
                current_price >= details["take_profit"]
                or current_price <= details["stop_loss"]
            ):
                logger.info(
                    f"Closing {symbol} BUY position due to SL/TP hit at {current_price}."
                )
                self.close(symbol)
                del self._position_details[symbol]
            elif details["side"] == "SELL" and (
                current_price <= details["take_profit"]
                or current_price >= details["stop_loss"]
            ):
                logger.info(
                    f"Closing {symbol} SELL position due to SL/TP hit at {current_price}."
                )
                self.close(symbol)
                del self._position_details[symbol]

        # --- Main Strategy Logic ---
        for symbol in data:
            bar = data[symbol]
            current_high, current_low = bar[2], bar[3]

            # 1. Establish Opening Range
            if ts.time() <= self.opening_range_end_time:
                self._opening_range[symbol]["high"] = max(
                    self._opening_range[symbol]["high"], current_high
                )
                self._opening_range[symbol]["low"] = min(
                    self._opening_range[symbol]["low"], current_low
                )
                if ts.time() == self.opening_range_end_time:
                    self._opening_range[symbol]["established"] = True
                continue

            # 2. Check for Breakout/Breakdown after range is set
            orb = self._opening_range[symbol]
            if (
                orb["established"]
                and not self._trade_taken[symbol]
                and symbol not in self.broker.positions
            ):
                range_size = orb["high"] - orb["low"]
                if range_size == 0:
                    continue

                if current_high > orb["high"]:  # Buy breakout
                    entry_price = orb["high"]
                    stop_loss_price = orb["low"]
                    if not self.buy(
                        symbol, entry_price=entry_price, stop_loss_price=stop_loss_price
                    ):
                        continue
                    self._trade_taken[symbol] = True
                    self._position_details[symbol] = {
                        "side": "BUY",
                        "stop_loss": orb["low"],
                        "take_profit": entry_price
                        + (range_size * self.take_profit_factor),
                    }
                    logger.info(
                        f"ORB BUY for {symbol} at {entry_price}. SL: {self._position_details[symbol]['stop_loss']}, TP: {self._position_details[symbol]['take_profit']}"
                    )

                elif current_low < orb["low"]:  # Sell breakdown
                    entry_price = orb["low"]
                    stop_loss_price = orb["high"]
                    if not self.sell(
                        symbol, entry_price=entry_price, stop_loss_price=stop_loss_price
                    ):
                        continue
                    self._trade_taken[symbol] = True
                    self._position_details[symbol] = {
                        "side": "SELL",
                        "stop_loss": orb["high"],
                        "take_profit": entry_price
                        - (range_size * self.take_profit_factor),
                    }
                    logger.info(
                        f"ORB SELL for {symbol} at {entry_price}. SL: {self._position_details[symbol]['stop_loss']}, TP: {self._position_details[symbol]['take_profit']}"
                    )


class VWAPPullbackStrategy(Strategy):
    """
    An intraday strategy that trades pullbacks to the Volume-Weighted Average Price (VWAP).

    - Entry Logic:
        - Long: Enters a long position if the price is above VWAP and pulls back to touch it.
        - Short: Enters a short position if the price is below VWAP and pulls back to touch it.
    - Risk Management:
        - Uses a percentage-based stop-loss and take-profit from the entry price.
        - Closes any open positions at the end of the trading day.
    """

    def __init__(
        self,
        broker,
        sizer=None,
        history_size=120,
        stop_loss_pct=0.003,
        target_pct=0.006,
        vwap_period=15,
    ):
        super().__init__(broker, sizer=sizer, bar_history=history_size)

        # Risk/Reward settings
        if not 0 < stop_loss_pct < 1:
            raise ValueError("stop_loss_pct must be a float between 0 and 1.")
        if not 0 < target_pct < 1:
            raise ValueError("target_pct must be a float between 0 and 1.")

        self.stop_loss_pct = stop_loss_pct
        self.target_pct = target_pct
        self.vwap_period = vwap_period

        # Stores entry price, SL, and TP for active trades
        self._position_details = {}

    def on_data(self, ts, data):
        # Ensure all positions are closed at the end of the day
        self.close_on_eod(ts)

        for symbol, bar in data.items():
            # The base `Strategy.next` method handles warming up the bar history
            if self.bar_history[symbol].count < self.vwap_period:
                continue

            # Unpack bar data for clarity
            o, h, l, c, v = bar[1:6]

            # Calculate VWAP over stored history
            # More performant VWAP calculation by avoiding stacking the full history.
            # We extract only the 'close' (index 4) and 'volume' (index 5) from each bar.
            # Note: The bar format from data_manager is (ts, o, h, l, c, v)
            cv = np.array(
                [[b[4], b[5]] for b in self.bar_history[symbol][:]], dtype=float
            )
            vwap = np.sum(cv[:, 0] * cv[:, 1]) / np.sum(cv[:, 1])

            position = self.broker.get_position(symbol)

            # --- Risk Management for Open Positions ---
            if position.size != 0 and symbol in self._position_details:
                details = self._position_details[symbol]
                if (
                    position.size > 0 and (c <= details["sl"] or c >= details["tp"])
                ) or (position.size < 0 and (c >= details["sl"] or c <= details["tp"])):
                    logger.info(f"Closing {symbol} position due to SL/TP hit at {c}.")
                    self.close(symbol)
                    del self._position_details[symbol]
                continue  # Don't open a new trade if one is already active

            # --- Entry Logic ---
            if position.size == 0:
                # Long entry: price is above VWAP and pulls back to touch it
                if c > vwap and l <= vwap:
                    entry_price = c
                    stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                    if self.buy(
                        symbol, entry_price=entry_price, stop_loss_price=stop_loss_price
                    ):
                        self._position_details[symbol] = {
                            "sl": stop_loss_price,
                            "tp": entry_price * (1 + self.target_pct),
                        }
                        logger.info(
                            f"VWAP BUY for {symbol} at {c}. SL: {self._position_details[symbol]['sl']:.2f}, TP: {self._position_details[symbol]['tp']:.2f}"
                        )
                # Short entry: price is below VWAP and pulls back to touch it
                elif c < vwap and h >= vwap:
                    entry_price = c
                    stop_loss_price = entry_price * (1 + self.stop_loss_pct)
                    if self.sell(
                        symbol, entry_price=entry_price, stop_loss_price=stop_loss_price
                    ):
                        self._position_details[symbol] = {
                            "sl": stop_loss_price,
                            "tp": entry_price * (1 - self.target_pct),
                        }
                        logger.info(
                            f"VWAP SELL for {symbol} at {c}. SL: {self._position_details[symbol]['sl']:.2f}, TP: {self._position_details[symbol]['tp']:.2f}"
                        )


class VWAPEMAStrategy(Intraday):
    """
    An intraday strategy that buys when the price is above both its daily VWAP and a long-term EMA,
    and sells when the price falls below both.

    - Entry Logic:
        - A single long trade is initiated per day if the closing price crosses above both the
          intraday VWAP and the 200-period EMA.
    - Exit Logic:
        - The position is closed if the price falls below both the VWAP and the EMA.
        - Any open position is automatically closed at the end of the trading day.
    """

    def __init__(self, broker, sizer=None, ema_period=200):
        # We need enough history for the EMA calculation
        super().__init__(broker, sizer=sizer, bar_history=ema_period + 1)
        self.ema_period = ema_period

        # Daily state variables
        self._current_day = None
        # defaultdict to hold daily state for each symbol
        self._daily_state = defaultdict(
            lambda: {"cum_pv": 0.0, "cum_vol": 0.0, "trade_taken": False}
        )

    def _reset_daily_state(self):
        """Resets the state at the start of a new trading day."""
        self._daily_state.clear()
        logger.info(f"New day {self._current_day}. Resetting VWAP/EMA daily state.")

    def on_data(self, ts, data):
        # --- Daily State Management ---
        if self._current_day != ts.date():
            self._current_day = ts.date()
            self._reset_daily_state()

        # --- EOD Position Closing ---
        self.close_on_eod(ts)

        for symbol in data:
            # --- Data Warm-up & Prep ---
            if self.bar_history[symbol].count < self.ema_period:
                continue  # Not enough data for EMA

            bar = data[symbol]
            current_close = bar[4]
            current_volume = bar[5]

            # --- Indicator Calculation ---

            # 1. Update and calculate daily VWAP
            symbol_state = self._daily_state[symbol]
            symbol_state["cum_pv"] += current_close * current_volume
            symbol_state["cum_vol"] += current_volume
            vwap = (
                symbol_state["cum_pv"] / symbol_state["cum_vol"]
                if symbol_state["cum_vol"] != 0
                else current_close
            )

            # 2. Calculate EMA
            closes = np.array([b[4] for b in self.bar_history[symbol][:]], dtype=float)
            ema = talib.EMA(closes, timeperiod=self.ema_period)

            # --- Trading Logic ---
            position = self.broker.get_position(symbol)

            buy_condition = current_close > vwap and current_close > ema[-1]
            sell_condition = current_close < vwap and current_close < ema[-1]

            if position.size == 0:  # No open position
                # Check for buy signal and if a trade has already been taken today for this symbol
                if buy_condition and not symbol_state["trade_taken"]:
                    if self.buy(symbol, entry_price=current_close):
                        symbol_state["trade_taken"] = True
                        logger.info(f"VWAP_EMA BUY for {symbol} at {current_close:.2f}")

            elif position.size > 0:  # Have an open long position
                # Check for exit signal
                if sell_condition:
                    self.close(symbol)
                    logger.info(f"VWAP_EMA CLOSE for {symbol} at {current_close:.2f}")


class DualMACrossoverStrategy(Strategy):
    """
    A classic trend-following strategy based on the crossover of two moving averages.
    This strategy is designed to hold positions across multiple days to capture trends.

    - Buy Signal: Fast MA crosses above Slow MA.
    - Sell Signal: Fast MA crosses below Slow MA.
    """

    def __init__(
        self, broker, sizer=None, symbol: str = "BSE", fast_period=50, slow_period=200
    ):
        # We need enough history for the slow MA. Add a buffer.
        super().__init__(broker, sizer=sizer, bar_history=slow_period + 5)
        if fast_period >= slow_period:
            raise ValueError("`fast_period` must be less than `slow_period`.")

        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_data(self, ts, data):
        symbol = self.symbol
        if symbol not in data:
            return

        # Get historical close prices from the bar history
        closes = np.array([b[4] for b in self.bar_history[symbol][:]], dtype=float)

        # Calculate MAs using TA-Lib
        fast_ma = talib.SMA(closes, timeperiod=self.fast_period)
        slow_ma = talib.SMA(closes, timeperiod=self.slow_period)

        position = self.broker.get_position(symbol)

        # Buy signal: Fast MA crosses above Slow MA
        if (
            fast_ma[-1] > slow_ma[-1]
            and fast_ma[-2] <= slow_ma[-2]
            and position.size <= 0
        ):
            self.close(symbol)  # Close any existing short position
            self.buy(symbol, entry_price=data[symbol][4])
            logger.info(f"MA Crossover BUY for {symbol} at {data[symbol][4]}")
        # Sell signal: Fast MA crosses below Slow MA
        elif (
            fast_ma[-1] < slow_ma[-1]
            and fast_ma[-2] >= slow_ma[-2]
            and position.size >= 0
        ):
            self.close(symbol)  # Close any existing long position
            self.sell(symbol, entry_price=data[symbol][4])
            logger.info(f"MA Crossover SELL for {symbol} at {data[symbol][4]}")


class RsiMeanReversionStrategy(Intraday):
    """
    An intraday mean-reversion strategy that trades pullbacks within a larger trend.

    - Trend Filter: A long-term EMA (e.g., 200-period) determines the intraday trend.
    - Entry Signal: An RSI indicator identifies over-extended pullbacks.
        - Buy: If trend is up (Price > EMA) and RSI is oversold (e.g., < 30).
        - Sell: If trend is down (Price < EMA) and RSI is overbought (e.g., > 70).
    - Risk Management:
        - Fixed percentage-based stop-loss and take-profit.
        - Only one trade per symbol per day.
        - EOD close for any remaining open positions.
    """

    def __init__(
        self,
        broker,
        sizer=None,
        ema_period=200,
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        stop_loss_pct=0.5,
        take_profit_pct=1.0,
    ):
        # Need enough history for the longest indicator (EMA)
        super().__init__(broker, sizer=sizer, bar_history=ema_period + 1)
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.stop_loss_pct = stop_loss_pct / 100  # Convert to decimal
        self.take_profit_pct = take_profit_pct / 100  # Convert to decimal

        # Daily state variables
        self._current_day = None
        self._trade_taken = defaultdict(bool)
        self._position_details = {}  # Stores SL/TP for open positions

    def _reset_daily_state(self):
        """Resets the state at the start of a new trading day."""
        self._trade_taken.clear()
        self._position_details.clear()
        logger.info(
            f"New day {self._current_day}. Resetting RSI Mean Reversion daily state."
        )

    def on_data(self, ts, data):
        # --- Daily State Management ---
        if self._current_day != ts.date():
            self._current_day = ts.date()
            self._reset_daily_state()

        # --- EOD Position Closing ---
        self.close_on_eod(ts)

        # --- Position and Risk Management (Check for SL/TP) ---
        for symbol, details in list(self._position_details.items()):
            if symbol not in data or symbol not in self.broker.positions:
                continue

            current_price = data[symbol][4]  # Using close price for check
            if details["side"] == "BUY" and (
                current_price >= details["tp"] or current_price <= details["sl"]
            ):
                logger.info(
                    f"Closing {symbol} BUY position due to SL/TP hit at {current_price:.2f}."
                )
                self.close(symbol)
                del self._position_details[symbol]
            elif details["side"] == "SELL" and (
                current_price <= details["tp"] or current_price >= details["sl"]
            ):
                logger.info(
                    f"Closing {symbol} SELL position due to SL/TP hit at {current_price:.2f}."
                )
                self.close(symbol)
                del self._position_details[symbol]

        # --- Main Strategy Logic ---
        for symbol in data:
            if (
                self.bar_history[symbol].count < self.ema_period
                or self._trade_taken[symbol]
            ):
                continue

            closes = np.array([b[4] for b in self.bar_history[symbol][:]], dtype=float)
            current_close = closes[-1]

            ema = talib.EMA(closes, timeperiod=self.ema_period)[-1]
            rsi = talib.RSI(closes, timeperiod=self.rsi_period)[-1]

            # Buy Signal: Trend is up, and price pulls back to an oversold level
            if current_close > ema and rsi < self.oversold_threshold:
                sl = current_close * (1 - self.stop_loss_pct)
                if self.buy(symbol, entry_price=current_close, stop_loss_price=sl):
                    self._trade_taken[symbol] = True
                    self._position_details[symbol] = {
                        "side": "BUY",
                        "sl": sl,
                        "tp": current_close * (1 + self.take_profit_pct),
                    }
                    logger.info(
                        f"RSI_MR BUY for {symbol} at {current_close:.2f}. SL: {self._position_details[symbol]['sl']:.2f}, TP: {self._position_details[symbol]['tp']:.2f}"
                    )

            # Sell Signal: Trend is down, and price rallies to an overbought level
            elif current_close < ema and rsi > self.overbought_threshold:
                sl = current_close * (1 + self.stop_loss_pct)
                if self.sell(symbol, entry_price=current_close, stop_loss_price=sl):
                    self._trade_taken[symbol] = True
                    self._position_details[symbol] = {
                        "side": "SELL",
                        "sl": sl,
                        "tp": current_close * (1 - self.take_profit_pct),
                    }
                    logger.info(
                        f"RSI_MR SELL for {symbol} at {current_close:.2f}. SL: {self._position_details[symbol]['sl']:.2f}, TP: {self._position_details[symbol]['tp']:.2f}"
                    )


class BollingerMeanReversionStrategy(Intraday):
    """
    Intraday Volatility Mean Reversion Strategy.

    Logic:
    - Long: Price pierces Lower Bollinger Band, RSI is oversold (<30), and Volume surges (exhaustion).
    - Short: Price pierces Upper Bollinger Band, RSI is overbought (>70), and Volume surges.
    - Exit: Target the Middle Band (Mean) or fixed SL.
    """

    def __init__(
        self, broker, sizer=None, bb_period=20, bb_std=2, rsi_period=14, vol_period=20
    ):
        super().__init__(
            broker, sizer=sizer, bar_history=max(bb_period, rsi_period, vol_period) + 5
        )
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.vol_period = vol_period
        self._current_day = None
        self._position_details = {}

    def on_data(self, ts, data):
        if self._current_day != ts.date():
            self._current_day = ts.date()
            self._position_details.clear()

        self.close_on_eod(ts)

        for symbol, bar in data.items():
            if self.bar_history[symbol].count < self.vol_period:
                continue

            # Check SL/TP for existing positions
            if symbol in self.broker.positions and symbol in self._position_details:
                details = self._position_details[symbol]
                curr_price = bar[4]
                if (
                    details["side"] == "BUY"
                    and (curr_price >= details["tp"] or curr_price <= details["sl"])
                ) or (
                    details["side"] == "SELL"
                    and (curr_price <= details["tp"] or curr_price >= details["sl"])
                ):
                    self.close(symbol)
                    del self._position_details[symbol]
                continue

            # Calculate Indicators
            closes = np.array([b[4] for b in self.bar_history[symbol][:]], dtype=float)
            volumes = np.array([b[5] for b in self.bar_history[symbol][:]], dtype=float)

            upper, middle, lower = talib.BBANDS(
                closes,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
            )
            rsi = talib.RSI(closes, timeperiod=self.rsi_period)
            vol_sma = talib.SMA(volumes, timeperiod=self.vol_period)

            curr_close = closes[-1]
            curr_vol = volumes[-1]
            curr_rsi = rsi[-1]

            # Entry Conditions
            # Volume spike (> 1.2x average) often signals exhaustion at the bands
            vol_spike = curr_vol > (vol_sma[-1] * 1.2)

            if curr_close < lower[-1] and curr_rsi < 30 and vol_spike:
                sl = curr_close * 0.995  # 0.5% SL
                if self.buy(symbol, entry_price=curr_close, stop_loss_price=sl):
                    self._position_details[symbol] = {
                        "side": "BUY",
                        "sl": sl,
                        "tp": middle[-1],  # Target the mean
                    }
                    logger.info(
                        f"BMR BUY: {symbol} at {curr_close:.2f} | RSI: {curr_rsi:.1f} | Target: {middle[-1]:.2f}"
                    )

            elif curr_close > upper[-1] and curr_rsi > 70 and vol_spike:
                sl = curr_close * 1.005  # 0.5% SL
                if self.sell(symbol, entry_price=curr_close, stop_loss_price=sl):
                    self._position_details[symbol] = {
                        "side": "SELL",
                        "sl": sl,
                        "tp": middle[-1],  # Target the mean
                    }
                    logger.info(
                        f"BMR SELL: {symbol} at {curr_close:.2f} | RSI: {curr_rsi:.1f} | Target: {middle[-1]:.2f}"
                    )
