"""
Quick-start example demonstrating the backtester package API.
"""

from backtester import (
    Backtester,
    Strategy,
    Broker,
    FixedSizeSizer,
)


class DemoBuyAndHoldStrategy(Strategy):
    def on_data(self, ts, data):
        for symbol in data:
            if not self.has_open_position(symbol):
                self.buy(symbol, size=10)


def main():
    print("Welcome to Backtester Institutional Quantitative Engine!")
    print("Package successfully imported and configured.")


if __name__ == "__main__":
    main()
