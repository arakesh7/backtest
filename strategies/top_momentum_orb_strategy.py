"""
TopMomentumORBStrategy
======================
An intraday Opening Range Breakout strategy with a daily stock-selection filter.

Each trading day it:
 1. Builds a 15-minute Opening Range (OR) for every symbol in the universe.
 2. Scores each symbol by **momentum** (overnight gap + range-volume surge).
 3. Selects the **top 2** scoring symbols for the day.
 4. Trades breakouts on those 2 stocks only:
      - Long  → price closes above OR High
      - Short → price closes below OR Low  (if allow_short=True)
 5. Manages risk with a fixed SL (opposite side of OR) and TP (rr_ratio × range).
 6. Closes every open position at 15:25 (5 min before market close).

Data source
-----------
CSV files — the same pystox 1-minute folder used by the positional strategy:
    G:/Projects/pystox_nov_edit/src/pystox/data/1minute/
Available symbols: ANANTRAJ, APOLLO, JWL, KAYNES, TEXRAIL

Bar column mapping (confirmed from CSVDataLoader + existing strategies):
    bar[0] = ts-string (index)
    bar[1] = open
    bar[2] = high
    bar[3] = low
    bar[4] = close
    bar[5] = volume
"""

import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

# ── path fix so `strategy` (in parent dir) can always be imported ──────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategy import Strategy  # noqa: E402

logger = logging.getLogger(__name__)

# Bar column indexes
IDX_OPEN = 1
IDX_HIGH = 2
IDX_LOW = 3
IDX_CLOSE = 4
IDX_VOLUME = 5


class TopMomentumORBStrategy(Strategy):
    """
    Intraday Opening Range Breakout — trades only the top-2 momentum stocks.

    Parameters
    ----------
    opening_range_minutes : int
        How many minutes from market open form the Opening Range (default 15).
    rr_ratio : float
        Reward-to-risk ratio for the take-profit target (default 1.5).
        TP = entry ± rr_ratio × OR_size.
    max_stocks_per_day : int
        Maximum number of stocks to trade on any given day (default 2).
    allow_short : bool
        Whether to take short (sell) positions on OR breakdowns (default True).
    vol_lookback : int
        Number of previous bars used to compute average volume for the surge
        score (default 5 × 375 = last 5 trading days).
    """

    def __init__(
        self,
        broker,
        sizer=None,
        opening_range_minutes: int = 15,
        rr_ratio: float = 1.5,
        max_stocks_per_day: int = 2,
        allow_short: bool = True,
        vol_lookback: int = 5 * 375,
    ):
        # Warm up enough bars to compute the volume lookback
        super().__init__(broker, sizer=sizer, bar_history=vol_lookback + 50)

        self.opening_range_minutes = opening_range_minutes
        self.rr_ratio = rr_ratio
        self.max_stocks_per_day = max_stocks_per_day
        self.allow_short = allow_short
        self.vol_lookback = vol_lookback

        # Compute the time at which the Opening Range ends
        market_open_dt = datetime.strptime(self.broker.market_open_time, "%H:%M:%S")
        self.or_end_time = (
            market_open_dt + timedelta(minutes=opening_range_minutes)
        ).time()

        # ── Per-day state ────────────────────────────────────────────────────
        self._current_day = None

        # Opening Range accumulators: {symbol: {'high', 'low', 'open', 'volume', 'established'}}
        self._or: dict = defaultdict(
            lambda: {
                "high": -np.inf,
                "low": np.inf,
                "open": None,
                "close_at_or_end": None,
                "volume": 0.0,
                "established": False,
            }
        )

        # Which symbols are tradeable today (top-2 selected after OR build)
        self._selected: list = []

        # Position details for risk management: {symbol: {'side','sl','tp'}}
        self._pos_details: dict = {}

        # Previous day's close for gap calculation: {symbol: float}
        self._prev_close: dict = {}

        # Track whether a trade has been taken for each symbol today
        self._trade_taken: dict = defaultdict(bool)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _reset_daily_state(self):
        """Called once at the start of each new trading day."""
        # Capture yesterday's close before wiping state
        for symbol, orb in self._or.items():
            if orb["close_at_or_end"] is not None:
                self._prev_close[symbol] = orb["close_at_or_end"]

        self._or.clear()
        self._selected.clear()
        self._pos_details.clear()
        self._trade_taken.clear()
        logger.info(f"[{self._current_day}] New day — state reset.")

    def _score_symbol(self, symbol) -> float:
        """
        Momentum score used to pick the top-2 stocks for the day.

        Score = abs(gap_pct) × 0.4  +  range_pct × 0.3  +  vol_surge × 0.3

        - gap_pct   : overnight gap from previous close to today's open (%)
        - range_pct : OR size as % of open price
        - vol_surge : OR volume relative to average per-bar volume over lookback
        """
        orb = self._or[symbol]
        if not orb["established"] or orb["open"] is None:
            return 0.0

        prev_c = self._prev_close.get(symbol)
        gap_pct = 0.0
        if prev_c and prev_c > 0:
            gap_pct = abs((orb["open"] - prev_c) / prev_c * 100)

        range_pct = 0.0
        if orb["open"] > 0:
            range_pct = (orb["high"] - orb["low"]) / orb["open"] * 100

        # Average per-bar volume over the lookback window
        vol_surge = 0.0
        hist = self.bar_history[symbol]
        if hist.count >= self.vol_lookback:
            volumes = np.array([b[IDX_VOLUME] for b in hist[:]], dtype=float)
            avg_vol = (
                np.mean(volumes[-self.vol_lookback :])
                if len(volumes) >= self.vol_lookback
                else 1.0
            )
            bars_in_or = self.opening_range_minutes
            avg_or_vol = avg_vol * bars_in_or
            vol_surge = orb["volume"] / avg_or_vol if avg_or_vol > 0 else 1.0

        score = gap_pct * 0.4 + range_pct * 0.3 + vol_surge * 0.3
        logger.debug(
            f"[{symbol}] Score={score:.4f} | gap={gap_pct:.2f}% "
            f"range={range_pct:.2f}% vol_surge={vol_surge:.2f}x"
        )
        return score

    def _select_top_symbols(self, data):
        """Score all symbols with an established OR and pick the top N."""
        scores = {s: self._score_symbol(s) for s in data if self._or[s]["established"]}
        ranked = sorted(scores, key=scores.get, reverse=True)
        self._selected = ranked[: self.max_stocks_per_day]
        logger.info(
            f"[{self._current_day}] Selected stocks: {self._selected} "
            f"(scores: {[f'{s}={scores[s]:.3f}' for s in self._selected]})"
        )

    def _check_sl_tp(self, symbol, bar):
        """Exit a position if stop-loss or take-profit is hit."""
        if symbol not in self._pos_details:
            return
        position = self.broker.get_position(symbol)
        if position.size == 0:
            del self._pos_details[symbol]
            return

        details = self._pos_details[symbol]
        curr_high = float(bar[IDX_HIGH])
        curr_low = float(bar[IDX_LOW])

        hit = False
        if details["side"] == "BUY":
            if curr_low <= details["sl"] or curr_high >= details["tp"]:
                hit = True
        elif details["side"] == "SELL":
            if curr_high >= details["sl"] or curr_low <= details["tp"]:
                hit = True

        if hit:
            price = float(bar[IDX_CLOSE])
            logger.info(
                f"[{self._current_day}] {symbol}: SL/TP hit at {price:.2f}. Closing."
            )
            self.close(symbol)
            del self._pos_details[symbol]

    # ── main loop ─────────────────────────────────────────────────────────────

    def on_data(self, ts, data):
        today = ts.date()
        curr_time = ts.time()

        # ── 1. New-day reset ─────────────────────────────────────────────────
        if self._current_day != today:
            self._current_day = today
            self._reset_daily_state()

        # ── 2. EOD close (5 min before market close) ─────────────────────────
        self.close_on_eod(ts)

        # ── 3. Opening Range accumulation phase ──────────────────────────────
        if curr_time <= self.or_end_time:
            for symbol, bar in data.items():
                orb = self._or[symbol]
                o = float(bar[IDX_OPEN])
                h = float(bar[IDX_HIGH])
                lo = float(bar[IDX_LOW])
                c = float(bar[IDX_CLOSE])
                v = float(bar[IDX_VOLUME])

                if orb["open"] is None:  # first bar of the day
                    orb["open"] = o
                orb["high"] = max(orb["high"], h)
                orb["low"] = min(orb["low"], lo)
                orb["volume"] += v
                orb["close_at_or_end"] = c

                if curr_time == self.or_end_time:
                    orb["established"] = True

            # Select top-2 the moment the OR is established
            if curr_time == self.or_end_time:
                self._select_top_symbols(data)
            return  # No trades during the OR build

        # ── 4. Nothing to do until symbols are selected ──────────────────────
        if not self._selected:
            return

        # ── 5. Risk management for open positions ─────────────────────────────
        for symbol in list(self._pos_details.keys()):
            if symbol in data:
                self._check_sl_tp(symbol, data[symbol])

        # ── 6. Entry logic (only for selected stocks, one trade per symbol/day) ─
        for symbol in self._selected:
            if symbol not in data:
                continue
            if self._trade_taken[symbol]:
                continue
            if self.has_open_position(symbol):
                continue

            bar = data[symbol]
            orb = self._or[symbol]

            curr_high = float(bar[IDX_HIGH])
            curr_low = float(bar[IDX_LOW])
            curr_close = float(bar[IDX_CLOSE])

            or_high = orb["high"]
            or_low = orb["low"]
            or_size = or_high - or_low

            if or_size <= 0:
                continue

            # ── Long entry: bar closes above the OR High ─────────────────────
            if curr_high > or_high:
                # continue
                entry = or_high  # theoretical entry at breakout level
                sl = or_low  # stop at the bottom of the range
                tp = entry + self.rr_ratio * or_size

                order = self.buy(
                    symbol,
                    entry_price=entry,
                    stop_loss_price=sl,
                )
                if order:
                    self._trade_taken[symbol] = True
                    self._pos_details[symbol] = {"side": "BUY", "sl": sl, "tp": tp}
                    logger.info(
                        f"[{ts}] {symbol}: ORB LONG @ {entry:.2f} | "
                        f"OR=[{or_low:.2f},{or_high:.2f}] SL={sl:.2f} TP={tp:.2f}"
                    )

            # ── Short entry: bar closes below the OR Low ──────────────────────
            elif self.allow_short and curr_low < or_low:
                entry = or_low
                sl = or_high
                tp = entry - self.rr_ratio * or_size

                order = self.sell(
                    symbol,
                    entry_price=entry,
                    stop_loss_price=sl,
                )
                if order:
                    self._trade_taken[symbol] = True
                    self._pos_details[symbol] = {"side": "SELL", "sl": sl, "tp": tp}
                    logger.info(
                        f"[{ts}] {symbol}: ORB SHORT @ {entry:.2f} | "
                        f"OR=[{or_low:.2f},{or_high:.2f}] SL={sl:.2f} TP={tp:.2f}"
                    )
