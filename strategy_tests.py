import time
from broker import Broker
from backtester import Backtester
from user_strategies import  OHLCReversalStrategy, OpeningRangeBreakoutStrategy, VWAPEMAStrategy, VWAPPullbackStrategy, DualMACrossoverStrategy, RsiMeanReversionStrategy
from sizer import FixedRiskPercentSizer, AllInSizer
from dataclasses import asdict
from tabulate import tabulate
import pprint

def universe(ts):
    # It's often better to run intraday strategies on a wider universe of liquid stocks.
    return ['ADANIPORTS', 'HAL', 'ADANIGREEN', 'TATAMOTORS', 'BSE']
    

from execution_model import SimpleExecutionModel
from slippage import DefaultSlippage

class PercentSlippage:
    def __init__(self, percent=0.0005): # 0.05% slippage
        self.percent = percent
    def calculate_slippage(self, order, price):
        if order.side == 'BUY':
            return price * (1 + self.percent)
        else:
            return price * (1 - self.percent)

# --- Configuration ---
broker = Broker(cash=100000) 
broker.commission = 20 # Flat ₹20 per trade
broker.add_execution_model(SimpleExecutionModel(slippage_model=PercentSlippage()))

# Using a sizer that risks only 2% of capital per trade is much safer.
sizer = FixedRiskPercentSizer(risk_percent=2.0)

# --- Select your strategy here ---
# strategy = OHLCReversalStrategy(broker, sizer=sizer)
# strategy = OpeningRangeBreakoutStrategy(broker, sizer=sizer, opening_range_minutes=30)
# strategy = VWAPPullbackStrategy(broker, sizer=sizer)
# strategy = VWAPEMAStrategy(broker, sizer=sizer)
# strategy = DualMACrossoverStrategy(broker, sizer=sizer, symbol="BSE", fast_period=50, slow_period=200)
# strategy = RsiMeanReversionStrategy(broker, sizer=sizer)
from user_strategies import BollingerMeanReversionStrategy
strategy = BollingerMeanReversionStrategy(broker, sizer=sizer)
bt = Backtester(start_date="2024-01-01", end_date="2025-01-03", strategy=strategy, frequency="1min", universe_fn=universe)

start_time = time.time()
results  = bt.run()
end_time = time.time()
trades = bt._broker.trades
trades = [asdict(trade) for trade in trades.values()]
#print(tabulate(trades, headers="keys", tablefmt="fancy_grid"))
print(f"final value: {bt._broker.cash}")
print(f"total taken time: {end_time - start_time:2f} seconds | final cash :{bt._broker.cash} | positions: {bt._broker.positions}    ")

#----------------------
# Pretty print the results from the DefaultAnalyzer
default_analyzer_results = results.get('DefaultAnalyzer', {})
equity_curve = default_analyzer_results.pop('Equity Curve', None) # Remove df for cleaner printing



if equity_curve is not None:
    print("\n--- Equity Curve ---")
    print(equity_curve.head())
    # You could also plot the equity curve here:
    # equity_curve['equity'].plot(title='Portfolio Equity Over Time').get_figure().savefig('equity_curve.png')

print(tabulate(trades, headers="keys", tablefmt="fancy_grid"))
print("\n--- Performance Metrics ---")
pprint.pprint(default_analyzer_results)