"""
VWAPMeanReversionStrategy
=========================
A disciplined intraday mean-reversion strategy. Instead of chasing breakouts,
it waits for price to *over-extend* away from VWAP and then FADES back to it.

Logic
-----
For each of the top-2 selected stocks (chosen by OR momentum score):

  LONG setup (buy the oversold dip):
    1. Today's trend is UP: price spent most of OR period above VWAP.
    2. Price pulls back BELOW VWAP by at least `vwap_band` (e.g. 0.3%).
    3. RSI(14) is oversold (< 35) confirming the stretch.
    4. Price reclaims VWAP (close crosses back above) → ENTRY.
    5. SL  = entry – 1.5 × ATR(14).
    6. TP  = entry + 2.5 × ATR(14)  (≈ 1.67:1 after slippage).

  SHORT setup (sell the overbought rally):
    1. Today's trend is DOWN: price spent most of OR period below VWAP.
    2. Price rallies ABOVE VWAP by at least `vwap_band`.
    3. RSI(14) is overbought (> 65).
    4. Price loses VWAP (close crosses back below) → ENTRY.
    5. SL  = entry + 1.5 × ATR(14).
    6. TP  = entry – 2.5 × ATR(14).

Risk management
---------------
- **One trade per symbol per day** — no re-entries.
- **Hard EOD close** at 15:25.
- **Max 2 symbols per day** (ranked by OR momentum score).
- **Position size**: risk a fixed % of cash using ATR-based stop distance,
  so a wider stop automatically produces a smaller position.

Why this beats the ORB strategy
---------------------------------
ORB chases price AFTER it has already moved — entering at extended levels
with wide stops (full OR range). This strategy does the opposite: it waits
for price to exhaust and then captures the mean-reversion move back to VWAP,
which is the most reliable intraday attractor.
"""

import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import talib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategy import Strategy  # noqa: E402

logger = logging.getLogger(__name__)

# Bar column indexes (confirmed: ts=0, open=1, high=2, low=3, close=4, vol=5)
IDX_OPEN   = 1
IDX_HIGH   = 2
IDX_LOW    = 3
IDX_CLOSE  = 4
IDX_VOLUME = 5


class VWAPMeanReversionStrategy(Strategy):
    """
    Intraday VWAP mean-reversion — trades the top-2 momentum stocks.

    Parameters
    ----------
    opening_range_minutes : int   OR build period (default 15).
    vwap_band_pct         : float Min % price must deviate from VWAP to qualify
                                  for a trade setup (default 0.25%).
    rsi_period            : int   RSI period (default 14).
    rsi_oversold          : float RSI threshold for long setups (default 35).
    rsi_overbought        : float RSI threshold for short setups (default 65).
    atr_period            : int   ATR period for SL/TP sizing (default 14).
    atr_sl_mult           : float SL = atr_sl_mult × ATR (default 1.5).
    atr_tp_mult           : float TP = atr_tp_mult × ATR (default 2.5).
    max_stocks_per_day    : int   Max symbols to trade per day (default 2).
    allow_short           : bool  Allow short setups (default True).
    vol_lookback          : int   Bars for avg-volume in scoring (default 1875).
    risk_pct              : float % of available cash to risk per trade (default 1.5).
    max_position_pct      : float Hard cap: position value ≤ X% of cash (default 25%).
    """

    def __init__(
        self,
        broker,
        sizer=None,
        opening_range_minutes: int = 15,
        vwap_band_pct: float = 0.25,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        atr_period: int = 14,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 2.5,
        max_stocks_per_day: int = 2,
        allow_short: bool = True,
        vol_lookback: int = 5 * 375,
        risk_pct: float = 1.5,
        max_position_pct: float = 25.0,
    ):
        warmup = max(vol_lookback, rsi_period + atr_period + 10) + 50
        super().__init__(broker, sizer=sizer, bar_history=warmup)

        self.opening_range_minutes = opening_range_minutes
        self.vwap_band_pct    = vwap_band_pct / 100.0
        self.rsi_period       = rsi_period
        self.rsi_oversold     = rsi_oversold
        self.rsi_overbought   = rsi_overbought
        self.atr_period       = atr_period
        self.atr_sl_mult      = atr_sl_mult
        self.atr_tp_mult      = atr_tp_mult
        self.max_stocks_per_day = max_stocks_per_day
        self.allow_short      = allow_short
        self.vol_lookback     = vol_lookback
        self.risk_pct         = risk_pct / 100.0
        self.max_position_pct = max_position_pct / 100.0

        market_open_dt   = datetime.strptime(self.broker.market_open_time, "%H:%M:%S")
        self.or_end_time = (market_open_dt + timedelta(minutes=opening_range_minutes)).time()

        # ── Per-day state ────────────────────────────────────────────────────
        self._current_day = None

        # OR accumulators
        self._or: dict = defaultdict(lambda: {
            "high": -np.inf, "low": np.inf, "open": None,
            "close_at_or_end": None, "volume": 0.0,
            "established": False, "bars_above_vwap": 0, "bars_total": 0,
        })

        self._selected: list = []
        self._pos_details: dict = {}     # {symbol: {sl, tp, side}}
        self._prev_close: dict = {}      # yesterday's close per symbol
        self._trade_taken: dict = defaultdict(bool)

        # Intraday VWAP state
        self._vwap_cum_pv: dict = defaultdict(float)   # cumulative price*vol
        self._vwap_cum_v:  dict = defaultdict(float)   # cumulative vol

        # Track previous close relative to VWAP for crossover detection
        self._prev_above_vwap: dict = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _reset_daily_state(self):
        for symbol, orb in self._or.items():
            if orb["close_at_or_end"] is not None:
                self._prev_close[symbol] = orb["close_at_or_end"]

        self._or.clear()
        self._selected.clear()
        self._pos_details.clear()
        self._trade_taken.clear()
        self._vwap_cum_pv.clear()
        self._vwap_cum_v.clear()
        self._prev_above_vwap.clear()
        logger.info(f"[{self._current_day}] New day — state reset.")

    def _update_vwap(self, symbol, bar):
        """Update cumulative VWAP for the symbol with the current bar."""
        tp = (float(bar[IDX_HIGH]) + float(bar[IDX_LOW]) + float(bar[IDX_CLOSE])) / 3.0
        v  = float(bar[IDX_VOLUME])
        self._vwap_cum_pv[symbol] += tp * v
        self._vwap_cum_v[symbol]  += v
        if self._vwap_cum_v[symbol] > 0:
            return self._vwap_cum_pv[symbol] / self._vwap_cum_v[symbol]
        return float(bar[IDX_CLOSE])

    def _get_atr_rsi(self, symbol):
        """Return (atr, rsi) from bar history, or (None, None) if not enough data."""
        hist  = self.bar_history[symbol]
        needed = self.atr_period + self.rsi_period + 5
        if hist.count < needed:
            return None, None
        bars   = list(hist[:])
        closes = np.array([float(b[IDX_CLOSE]) for b in bars], dtype=float)
        highs  = np.array([float(b[IDX_HIGH])  for b in bars], dtype=float)
        lows   = np.array([float(b[IDX_LOW])   for b in bars], dtype=float)
        atr_arr = talib.ATR(highs, lows, closes, timeperiod=self.atr_period)
        rsi_arr = talib.RSI(closes, timeperiod=self.rsi_period)
        atr = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else None
        rsi = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else None
        return atr, rsi

    def _calc_size(self, entry_price: float, sl_price: float) -> int:
        """
        ATR-based position sizing:
          size = (cash × risk_pct) / |entry - sl|
        Hard cap at max_position_pct of cash.
        """
        risk_per_share = abs(entry_price - sl_price)
        if risk_per_share <= 0:
            return 0
        cash       = self.broker.cash
        risk_money = cash * self.risk_pct
        raw_size   = int(risk_money / risk_per_share)
        # Hard cap on total position value
        max_size = int((cash * self.max_position_pct) / entry_price)
        return max(0, min(raw_size, max_size))

    def _score_symbol(self, symbol) -> float:
        orb = self._or[symbol]
        if not orb["established"] or orb["open"] is None:
            return 0.0
        prev_c = self._prev_close.get(symbol)
        gap_pct = abs((orb["open"] - prev_c) / prev_c * 100) if prev_c and prev_c > 0 else 0.0
        range_pct = (orb["high"] - orb["low"]) / orb["open"] * 100 if orb["open"] > 0 else 0.0
        vol_surge = 0.0
        hist = self.bar_history[symbol]
        if hist.count >= self.vol_lookback:
            volumes   = np.array([float(b[IDX_VOLUME]) for b in hist[:]], dtype=float)
            avg_vol   = float(np.mean(volumes[-self.vol_lookback:]))
            avg_or_vol = avg_vol * self.opening_range_minutes
            vol_surge = orb["volume"] / avg_or_vol if avg_or_vol > 0 else 1.0
        return gap_pct * 0.4 + range_pct * 0.3 + vol_surge * 0.3

    def _select_top_symbols(self, data):
        scores  = {s: self._score_symbol(s) for s in data if self._or[s]["established"]}
        ranked  = sorted(scores, key=scores.get, reverse=True)
        self._selected = ranked[: self.max_stocks_per_day]
        logger.info(
            f"[{self._current_day}] Selected: {self._selected} "
            f"(scores: {[f'{s}={scores[s]:.3f}' for s in self._selected]})"
        )

    # ── main loop ─────────────────────────────────────────────────────────────

    def on_data(self, ts, data):
        today     = ts.date()
        curr_time = ts.time()

        # 1. New-day reset
        if self._current_day != today:
            self._current_day = today
            self._reset_daily_state()

        # 2. Hard EOD close
        self.close_on_eod(ts)

        # 3. Update VWAP for all symbols (do this every bar all day)
        vwap_now = {}
        for symbol, bar in data.items():
            vwap_now[symbol] = self._update_vwap(symbol, bar)

        # 4. Opening Range accumulation
        if curr_time <= self.or_end_time:
            for symbol, bar in data.items():
                orb = self._or[symbol]
                o  = float(bar[IDX_OPEN])
                h  = float(bar[IDX_HIGH])
                lo = float(bar[IDX_LOW])
                c  = float(bar[IDX_CLOSE])
                v  = float(bar[IDX_VOLUME])
                if orb["open"] is None:
                    orb["open"] = o
                orb["high"]   = max(orb["high"], h)
                orb["low"]    = min(orb["low"],  lo)
                orb["volume"] += v
                orb["close_at_or_end"] = c
                orb["bars_total"] += 1
                if c > vwap_now.get(symbol, c):
                    orb["bars_above_vwap"] += 1
                if curr_time == self.or_end_time:
                    orb["established"] = True

            if curr_time == self.or_end_time:
                self._select_top_symbols(data)
            return

        if not self._selected:
            return

        # 5. SL / TP management for open positions
        for symbol in list(self._pos_details.keys()):
            if symbol not in data:
                continue
            pos = self.broker.get_position(symbol)
            if pos.size == 0:
                del self._pos_details[symbol]
                continue
            details   = self._pos_details[symbol]
            bar       = data[symbol]
            curr_high = float(bar[IDX_HIGH])
            curr_low  = float(bar[IDX_LOW])
            hit = (
                (details["side"] == "BUY"  and (curr_low  <= details["sl"] or curr_high >= details["tp"])) or
                (details["side"] == "SELL" and (curr_high >= details["sl"] or curr_low  <= details["tp"]))
            )
            if hit:
                self.close(symbol)
                logger.info(f"[{ts}] {symbol}: SL/TP exit. Closing.")
                del self._pos_details[symbol]

        # 6. Entry logic — only for the selected top-2
        for symbol in self._selected:
            if symbol not in data:
                continue
            if self._trade_taken[symbol]:
                continue
            if self.has_open_position(symbol):
                continue

            bar   = data[symbol]
            close = float(bar[IDX_CLOSE])
            vwap  = vwap_now[symbol]
            orb   = self._or[symbol]

            atr, rsi = self._get_atr_rsi(symbol)
            if atr is None or rsi is None or atr <= 0:
                continue

            # Trend bias from OR: majority of OR bars above/below VWAP
            bars_total = orb["bars_total"] or 1
            or_uptrend = (orb["bars_above_vwap"] / bars_total) >= 0.6

            # Detect VWAP crossover this bar
            prev_above = self._prev_above_vwap.get(symbol)
            curr_above = close > vwap
            self._prev_above_vwap[symbol] = curr_above

            deviation = abs(close - vwap) / vwap

            # ── LONG: uptrend, stretched below VWAP, now reclaiming ──────────
            if (
                or_uptrend
                and prev_above is not None
                and not prev_above       # was below VWAP last bar
                and curr_above           # now back above VWAP → reclaim crossover
                and deviation >= self.vwap_band_pct   # valid band stretch
                and rsi < self.rsi_oversold
            ):
                entry = close
                sl    = entry - self.atr_sl_mult * atr
                tp    = entry + self.atr_tp_mult * atr
                size  = self._calc_size(entry, sl)
                if size > 0:
                    order = self.buy(symbol, size=size)
                    if order:
                        self._trade_taken[symbol]   = True
                        self._pos_details[symbol]   = {"side": "BUY", "sl": sl, "tp": tp}
                        logger.info(
                            f"[{ts}] {symbol}: VWAP LONG @ {entry:.2f} | "
                            f"VWAP={vwap:.2f} RSI={rsi:.1f} ATR={atr:.2f} "
                            f"SL={sl:.2f} TP={tp:.2f} Size={size}"
                        )

            # ── SHORT: downtrend, stretched above VWAP, now losing it ────────
            elif (
                self.allow_short
                and not or_uptrend
                and prev_above is not None
                and prev_above           # was above VWAP last bar
                and not curr_above       # now back below VWAP → breakdown
                and deviation >= self.vwap_band_pct
                and rsi > self.rsi_overbought
            ):
                entry = close
                sl    = entry + self.atr_sl_mult * atr
                tp    = entry - self.atr_tp_mult * atr
                size  = self._calc_size(entry, sl)
                if size > 0:
                    order = self.sell(symbol, size=size)
                    if order:
                        self._trade_taken[symbol]   = True
                        self._pos_details[symbol]   = {"side": "SELL", "sl": sl, "tp": tp}
                        logger.info(
                            f"[{ts}] {symbol}: VWAP SHORT @ {entry:.2f} | "
                            f"VWAP={vwap:.2f} RSI={rsi:.1f} ATR={atr:.2f} "
                            f"SL={sl:.2f} TP={tp:.2f} Size={size}"
                        )
