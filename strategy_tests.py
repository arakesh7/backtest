import time
from broker import Broker
from backtester import Backtester
from user_strategies import (
    OHLCReversalStrategy,
    OpeningRangeBreakoutStrategy,
    VWAPEMAStrategy,
    VWAPPullbackStrategy,
    DualMACrossoverStrategy,
    RsiMeanReversionStrategy,
)
from sizer import FixedRiskPercentSizer, AllInSizer
from dataclasses import asdict
from tabulate import tabulate
import pprint

from execution_model import SimpleExecutionModel
from slippage import DefaultSlippage


# ---------------------------------------------------------------------------
# Slippage model
# ---------------------------------------------------------------------------
class PercentSlippage:
    def __init__(self, percent=0.0005):  # 0.05% slippage
        self.percent = percent

    def calculate_slippage(self, order, price):
        if order.side == "BUY":
            return price * (1 + self.percent)
        else:
            return price * (1 - self.percent)


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------
PYSTOX_DATA_DIR = r"G:\Projects\pystox_nov_edit\src\pystox\data\1minute"
PYSTOX_SYMBOLS = ["ANANTRAJ", "APOLLO", "JWL", "KAYNES", "TEXRAIL"]


def pystox_universe(ts):
    """All 5 symbols from the pystox 1-minute data folder."""
    return PYSTOX_SYMBOLS


def intraday_universe(ts):
    """Universe for strategies in the backtest_data folder."""
    return ["ADANIPORTS", "HAL", "ADANIGREEN", "TATAMOTORS", "BSE"]


# ---------------------------------------------------------------------------
# Active strategy selection
# ---------------------------------------------------------------------------
# Uncomment whichever strategy you want to run.
# All intraday strategies below use PYSTOX_DATA_DIR and pystox_universe.
#
# ── Intraday: VWAP Mean Reversion (ACTIVE) ───────────────────────────────
from strategies.vwap_mean_reversion_5m import VWAPMeanReversion5MinStrategy

broker = Broker(cash=100_000)  # ₹1 lakh starting capital
broker.commission = 20
broker.add_execution_model(SimpleExecutionModel(slippage_model=PercentSlippage()))

strategy = VWAPMeanReversion5MinStrategy(
    broker,
    opening_range_minutes=15,  # 15-min OR to establish trend bias
    vwap_band_pct=0.15,  # price must deviate >= 0.15% from VWAP
    rsi_period=14,
    rsi_oversold=40,  # long only when RSI < 40
    rsi_overbought=60,  # short only when RSI > 60
    atr_period=14,
    atr_sl_mult=1.5,  # SL = 1.5 x ATR from entry
    atr_tp_mult=3.0,  # TP = 3.0 x ATR from entry  (2:1 R:R)
    max_stocks_per_day=2,  # trade at most 2 stocks / day
    allow_short=True,
    risk_pct=1.5,  # risk 1.5% of available cash per trade
    max_position_pct=25.0,  # position value capped at 25% of cash
)

bt = Backtester(
    start_date="2025-01-01",
    end_date="2026-04-03",
    strategy=strategy,
    frequency="1min",
    universe_fn=pystox_universe,
    data_dir=PYSTOX_DATA_DIR,
)

# ── Intraday: ORB (commented out — account blow-up risk at high risk%) ────
# from strategies.top_momentum_orb_strategy import TopMomentumORBStrategy
# broker = Broker(cash=100_000)
# broker.commission = 20
# broker.add_execution_model(SimpleExecutionModel(slippage_model=PercentSlippage()))
# strategy = TopMomentumORBStrategy(broker, opening_range_minutes=15,
#     rr_ratio=1.5, max_stocks_per_day=2, allow_short=True,
#     risk_pct=1.5, max_position_pct=20.0)

# ── Positional: EMA Crossover (commented out) ─────────────────────────────
# from strategies.positional_ema_strategy import PositionalEMACrossoverStrategy
# broker = Broker(cash=500000)
# broker.commission = 20
# broker.add_execution_model(SimpleExecutionModel(slippage_model=PercentSlippage()))
# sizer = FixedRiskPercentSizer(risk_percent=1.5)
# strategy = PositionalEMACrossoverStrategy(broker, sizer=sizer,
#     fast_ema=9, slow_ema=21, atr_period=14, atr_mult=2.0, allow_short=False)
# bt = Backtester(start_date="2024-01-01", end_date="2025-01-03",
#     strategy=strategy, frequency="1min", universe_fn=pystox_universe,
#     data_dir=PYSTOX_DATA_DIR)

# ── Intraday (legacy, uses backtest_data folder) ──────────────────────────
# broker = Broker(cash=100000)
# broker.commission = 20
# broker.add_execution_model(SimpleExecutionModel(slippage_model=PercentSlippage()))
# sizer = FixedRiskPercentSizer(risk_percent=2.0)
# strategy = OHLCReversalStrategy(broker, sizer=sizer)
# bt = Backtester(start_date="2024-01-01", end_date="2025-01-03",
#     strategy=strategy, frequency="1min", universe_fn=intraday_universe)

# ---------------------------------------------------------------------------
# Run the backtest
# ---------------------------------------------------------------------------
start_time = time.time()
results = bt.run()
end_time = time.time()

trades = bt._broker.trades
trade_list = [asdict(trade) for trade in trades.values()]

print(f"\ntotal time : {end_time - start_time:.2f}s")
print(f"final cash : {bt._broker.cash:,.2f}")
print(f"positions  : {bt._broker.positions}")

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
default_analyzer_results = results.get("DefaultAnalyzer", {})
equity_curve = default_analyzer_results.pop("Equity Curve", None)

if equity_curve is not None:
    print("\n--- Equity Curve (first 5 rows) ---")
    print(equity_curve.head())

print(f"\n--- Trades ({len(trade_list)} total) ---")
print(tabulate(trade_list, headers="keys", tablefmt="grid"))
print("\n--- Performance Metrics ---")
pprint.pprint(default_analyzer_results)
