"""
PositionalEMACrossoverStrategy
==============================
A positional (swing / multi-day) strategy that operates on 1-minute data but
makes trading decisions once per day using *daily* aggregated OHLC figures.

Logic
-----
- At the close of each trading day the strategy computes:
    - EMA(fast) and EMA(slow) on the last N *daily* closes it has collected.
    - A trailing-stop level based on a multiple of the recent ATR.
- Signals
    - LONG  : fast EMA crosses above slow EMA (golden cross on daily bars).
    - EXIT  : fast EMA crosses below slow EMA  OR  price hits trailing stop.
    - SHORT : (optional, disabled by default – set `allow_short=True`).

The strategy intentionally does NOT call `close_on_eod`, so positions are
held overnight / across multiple days (true positional behaviour).

Data source
-----------
CSV files from  G:/Projects/pystox_nov_edit/src/pystox/data/1minute/
Available symbols: ANANTRAJ, APOLLO, JWL, KAYNES, TEXRAIL
"""

import logging
import os
import sys
from collections import defaultdict

import numpy as np
import talib

# Ensure the backtesting root is on sys.path so `strategy` can be imported
# regardless of whether this file is loaded directly or as a sub-package.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategy import Strategy

logger = logging.getLogger(__name__)

# Bar column indexes (as stored in data_cache numpy arrays after CSVDataLoader):
#   index 0 → open | 1 → high | 2 → low | 3 → close | 4 → volume | 5 → ts
# Wait – CSVDataLoader reads cols ['open','high','low','close','volume','ts']
# and sets df.index = df['ts'], so the numpy array columns are:
#   0=open, 1=high, 2=low, 3=close, 4=volume, 5=ts
# BUT the strategy receives bar = data_cache[symbol][idx] which is a raw numpy row.
# Looking at other strategies: bar[1]=open, bar[2]=high, bar[3]=low, bar[4]=close, bar[5]=volume
# (index 0 is 'open' column value, but let's check – CSVDataLoader uses:
#   cols = ['open','high','low','close','volume','ts']  → positions 0-5 in the df
#   df.index = df['ts']   → ts is the index, not a column in to_numpy()
# So to_numpy() gives columns: open=0, high=1, low=2, close=3, volume=4, ts=5 ... NO:
# df.index is 'ts', to_numpy() gives the DATA columns only → open,high,low,close,volume,ts
# Actually 'ts' is still in the columns because usecols includes it and it's also set as index → double.
# Empirically all existing strategies use bar[4] for close → let's follow that convention.
# bar = (open[1], high[2], low[3], close[4], volume[5])  with index 0 unused/ts-string.

IDX_OPEN   = 1
IDX_HIGH   = 2
IDX_LOW    = 3
IDX_CLOSE  = 4
IDX_VOLUME = 5


class PositionalEMACrossoverStrategy(Strategy):
    """
    Positional trend-following strategy using EMA crossover on daily closes
    aggregated from 1-minute bars. Trades survive overnight / across sessions.

    Parameters
    ----------
    fast_ema   : int   – period for fast EMA (default 9 days)
    slow_ema   : int   – period for slow EMA (default 21 days)
    atr_period : int   – ATR period for trailing stop (default 14 days)
    atr_mult   : float – trailing stop = entry - atr_mult * ATR  (default 2.0)
    allow_short: bool  – whether to take short positions (default False)
    """

    def __init__(
        self,
        broker,
        sizer=None,
        fast_ema: int = 9,
        slow_ema: int = 21,
        atr_period: int = 14,
        atr_mult: float = 2.0,
        allow_short: bool = False,
    ):
        # Keep a large minute-bar history so we can aggregate many daily closes.
        # Worst case ~375 bars/day * (slow_ema + atr_period + buffer)
        history_minutes = 375 * (slow_ema + atr_period + 10)
        super().__init__(broker, sizer=sizer, bar_history=history_minutes)

        self.fast_ema   = fast_ema
        self.slow_ema   = slow_ema
        self.atr_period = atr_period
        self.atr_mult   = atr_mult
        self.allow_short = allow_short

        # Daily OHLC accumulators  {symbol: {date: {o,h,l,c,v}}}
        self._daily_agg: dict = defaultdict(dict)
        # Ordered list of completed daily bars  {symbol: [{'date','o','h','l','c'}...]}
        self._daily_bars: dict = defaultdict(list)
        # Track the last day we acted on, to fire logic exactly once per day
        self._last_acted_day: dict = defaultdict(lambda: None)
        # Active trailing stop per symbol
        self._trailing_stop: dict = {}
        # Side of the active position ('LONG' / 'SHORT')
        self._position_side: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_daily_agg(self, ts, data):
        """Accumulate 1-min bars into daily OHLC buckets."""
        today = ts.date()
        for symbol, bar in data.items():
            o, h, lo, c, v = (
                float(bar[IDX_OPEN]),
                float(bar[IDX_HIGH]),
                float(bar[IDX_LOW]),
                float(bar[IDX_CLOSE]),
                float(bar[IDX_VOLUME]),
            )
            if today not in self._daily_agg[symbol]:
                self._daily_agg[symbol][today] = {
                    "open":  o, "high": h, "low": lo, "close": c, "volume": v
                }
            else:
                agg = self._daily_agg[symbol][today]
                agg["high"]   = max(agg["high"], h)
                agg["low"]    = min(agg["low"],  lo)
                agg["close"]  = c   # latest minute = day's running close
                agg["volume"] += v

    def _finalise_previous_day(self, ts, symbol):
        """
        When the day changes, move yesterday's accumulator into the
        completed daily-bars list (in chronological order).
        """
        today = ts.date()
        for day, ohlcv in sorted(self._daily_agg[symbol].items()):
            if day < today:  # day is complete
                already_stored = any(
                    b["date"] == day for b in self._daily_bars[symbol]
                )
                if not already_stored:
                    self._daily_bars[symbol].append({
                        "date":  day,
                        "open":  ohlcv["open"],
                        "high":  ohlcv["high"],
                        "low":   ohlcv["low"],
                        "close": ohlcv["close"],
                    })
                    logger.debug(
                        f"[{symbol}] Finalised daily bar {day}: "
                        f"O={ohlcv['open']:.2f} H={ohlcv['high']:.2f} "
                        f"L={ohlcv['low']:.2f} C={ohlcv['close']:.2f}"
                    )

    def _compute_indicators(self, symbol):
        """
        Return (fast_ema, slow_ema, atr) arrays computed from completed daily bars.
        Returns None if not enough history yet.
        """
        bars = self._daily_bars[symbol]
        needed = self.slow_ema + self.atr_period + 5
        if len(bars) < needed:
            return None

        closes = np.array([b["close"] for b in bars], dtype=np.float64)
        highs  = np.array([b["high"]  for b in bars], dtype=np.float64)
        lows   = np.array([b["low"]   for b in bars], dtype=np.float64)

        fast = talib.EMA(closes, timeperiod=self.fast_ema)
        slow = talib.EMA(closes, timeperiod=self.slow_ema)
        atr  = talib.ATR(highs, lows, closes, timeperiod=self.atr_period)

        return fast, slow, atr

    # ------------------------------------------------------------------
    # Main strategy logic
    # ------------------------------------------------------------------

    def on_data(self, ts, data):
        # Step 1 – always accumulate minute bars into daily buckets
        self._update_daily_agg(ts, data)

        # Step 2 – run end-of-day logic once per symbol, at market close bar
        if ts.time() != self.market_close_time:
            return

        today = ts.date()

        for symbol in list(data.keys()):
            # Finalise any complete (past) day bars
            self._finalise_previous_day(ts, symbol)

            # Fire trading logic at most once per symbol per day
            if self._last_acted_day[symbol] == today:
                continue
            self._last_acted_day[symbol] = today

            # --- Check trailing stop for existing position ---
            self._check_trailing_stop(symbol, data)

            # --- Compute indicators ---
            indicators = self._compute_indicators(symbol)
            if indicators is None:
                logger.debug(
                    f"[{symbol}] Not enough daily bars yet "
                    f"({len(self._daily_bars[symbol])} / "
                    f"{self.slow_ema + self.atr_period + 5} needed)."
                )
                continue

            fast, slow, atr = indicators
            current_close = float(data[symbol][IDX_CLOSE])

            # Require at least 2 valid data points for crossover detection
            valid = ~np.isnan(fast) & ~np.isnan(slow)
            if np.sum(valid) < 2:
                continue

            f_curr, f_prev = fast[-1], fast[-2]
            s_curr, s_prev = slow[-1], slow[-2]
            atr_val = atr[-1] if not np.isnan(atr[-1]) else (current_close * 0.015)

            position = self.broker.get_position(symbol)

            # -------------------------------------------------------
            # LONG logic
            # -------------------------------------------------------
            golden_cross = f_curr > s_curr and f_prev <= s_prev
            death_cross  = f_curr < s_curr and f_prev >= s_prev

            if position.size == 0:
                if golden_cross:
                    sl = current_close - self.atr_mult * atr_val
                    order = self.buy(
                        symbol,
                        entry_price=current_close,
                        stop_loss_price=sl,
                    )
                    if order:
                        self._trailing_stop[symbol] = sl
                        self._position_side[symbol]  = "LONG"
                        logger.info(
                            f"[{ts.date()}] {symbol}: POSITIONAL BUY @ {current_close:.2f} | "
                            f"SL={sl:.2f} | fast={f_curr:.2f} slow={s_curr:.2f}"
                        )

                elif self.allow_short and death_cross:
                    sl = current_close + self.atr_mult * atr_val
                    order = self.sell(
                        symbol,
                        entry_price=current_close,
                        stop_loss_price=sl,
                    )
                    if order:
                        self._trailing_stop[symbol] = sl
                        self._position_side[symbol]  = "SHORT"
                        logger.info(
                            f"[{ts.date()}] {symbol}: POSITIONAL SELL (short) @ {current_close:.2f} | "
                            f"SL={sl:.2f}"
                        )

            elif position.size > 0:  # In a long position
                # Exit on death cross (trend reversal signal)
                if death_cross:
                    logger.info(
                        f"[{ts.date()}] {symbol}: EXIT LONG (death cross) @ {current_close:.2f}"
                    )
                    self.close(symbol)
                    self._trailing_stop.pop(symbol, None)
                    self._position_side.pop(symbol, None)
                else:
                    # Update trailing stop upwards only
                    new_sl = current_close - self.atr_mult * atr_val
                    if new_sl > self._trailing_stop.get(symbol, -np.inf):
                        self._trailing_stop[symbol] = new_sl
                        logger.debug(
                            f"[{ts.date()}] {symbol}: Trailing stop raised to {new_sl:.2f}"
                        )

            elif position.size < 0 and self.allow_short:  # In a short position
                if golden_cross:
                    logger.info(
                        f"[{ts.date()}] {symbol}: EXIT SHORT (golden cross) @ {current_close:.2f}"
                    )
                    self.close(symbol)
                    self._trailing_stop.pop(symbol, None)
                    self._position_side.pop(symbol, None)
                else:
                    new_sl = current_close + self.atr_mult * atr_val
                    if new_sl < self._trailing_stop.get(symbol, np.inf):
                        self._trailing_stop[symbol] = new_sl

    def _check_trailing_stop(self, symbol, data):
        """Check and act on trailing stop hits using the current bar's price."""
        if symbol not in self._trailing_stop or symbol not in data:
            return

        position = self.broker.get_position(symbol)
        if position.size == 0:
            self._trailing_stop.pop(symbol, None)
            self._position_side.pop(symbol, None)
            return

        current_low  = float(data[symbol][IDX_LOW])
        current_high = float(data[symbol][IDX_HIGH])
        stop_level   = self._trailing_stop[symbol]

        if position.size > 0 and current_low <= stop_level:
            logger.info(
                f"[{symbol}] Trailing stop hit for LONG position. "
                f"Low={current_low:.2f} <= Stop={stop_level:.2f}. Closing."
            )
            self.close(symbol)
            self._trailing_stop.pop(symbol, None)
            self._position_side.pop(symbol, None)

        elif position.size < 0 and current_high >= stop_level:
            logger.info(
                f"[{symbol}] Trailing stop hit for SHORT position. "
                f"High={current_high:.2f} >= Stop={stop_level:.2f}. Closing."
            )
            self.close(symbol)
            self._trailing_stop.pop(symbol, None)
            self._position_side.pop(symbol, None)
