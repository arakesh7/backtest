import time
from dataclasses import asdict
import pprint
from tabulate import tabulate

from backtester import (
    Backtester,
    Broker,
    FixedRiskPercentSizer,
    AllInSizer,
    SimpleExecutionModel,
    DefaultSlippage,
)
from strategies.user_strategies import (
    OHLCReversalStrategy,
    OpeningRangeBreakoutStrategy,
    VWAPEMAStrategy,
    VWAPPullbackStrategy,
    DualMACrossoverStrategy,
    RsiMeanReversionStrategy,
)


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


if __name__ == "__main__":
    broker = Broker(cash=100_000)
    broker.commission = 20
    broker.add_execution_model(SimpleExecutionModel(slippage_model=PercentSlippage()))
    sizer = FixedRiskPercentSizer(risk_percent=2.0)
    strategy = OHLCReversalStrategy(broker, sizer=sizer)

    bt = Backtester(
        start_date="2024-01-01",
        end_date="2024-01-05",
        strategy=strategy,
        frequency="1min",
        universe_fn=intraday_universe,
    )

    print("Running Backtest...")
    start_time = time.time()
    results = bt.run()
    end_time = time.time()

    trades = bt._broker.trades
    trade_list = [asdict(trade) for trade in trades.values()]

    print(f"\nTotal time : {end_time - start_time:.2f}s")
    print(f"Final cash : {bt._broker.cash:,.2f}")
    print(f"Positions  : {bt._broker.positions}")

    default_analyzer_results = results.get("DefaultAnalyzer", {})
    equity_curve = default_analyzer_results.pop("Equity Curve", None)

    if equity_curve is not None:
        print("\n--- Equity Curve (first 5 rows) ---")
        print(equity_curve.head())

    print(f"\n--- Trades ({len(trade_list)} total) ---")
    print(tabulate(trade_list, headers="keys", tablefmt="grid"))
    print("\n--- Performance Metrics ---")
    pprint.pprint(default_analyzer_results)
