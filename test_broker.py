import unittest
import pandas as pd
from order import Position
from broker import Broker

# Helper: a minimal OHLCV bar tuple that process_orders expects: (ts, o, h, l, c, v)
def make_bar(price, ts="2024-01-01 10:00:00"):
    return (ts, price, price * 1.01, price * 0.99, price, 100_000)


class TestBroker(unittest.TestCase):
    def setUp(self):
        self.broker = Broker(cash=10000)
        # Assign a dummy current_ts so place_order & process_orders don't fail
        self.broker.current_ts = pd.Timestamp("2024-01-01 10:00:00")

    def _make_data(self, asset, price):
        return {asset: make_bar(price)}

    def test_place_order(self):
        order = self.broker.place_order(asset="AAPL", size=10, side="BUY")
        self.assertIn(order.id, self.broker.orders)

    def test_buy_order_executes_and_deducts_cash(self):
        order = self.broker.place_order(asset="AAPL", size=10, side="BUY")
        data = self._make_data("AAPL", 100)
        self.broker.process_orders(self.broker.current_ts, data)
        self.assertEqual(order.status, "EXECUTED")
        self.assertEqual(self.broker.positions["AAPL"].size, 10)
        self.assertEqual(self.broker.cash, 10000 - 1000)

    def test_insufficient_cash(self):
        order = self.broker.place_order(asset="AAPL", size=200, side="BUY")
        data = self._make_data("AAPL", 100)
        self.broker.process_orders(self.broker.current_ts, data)
        self.assertEqual(order.status, "REJECTED")
        self.assertNotIn("AAPL", self.broker.positions)

    def test_sell_order_injects_cash(self):
        # First buy 10 shares at 100
        buy_order = self.broker.place_order(asset="AAPL", size=10, side="BUY")
        self.broker.process_orders(self.broker.current_ts, self._make_data("AAPL", 100))
        self.assertEqual(buy_order.status, "EXECUTED")

        # Then sell 10 shares at 110
        self.broker.current_ts = pd.Timestamp("2024-01-01 11:00:00")
        sell_order = self.broker.place_order(asset="AAPL", size=10, side="SELL")
        self.broker.process_orders(self.broker.current_ts, self._make_data("AAPL", 110))
        self.assertEqual(sell_order.status, "EXECUTED")
        self.assertNotIn("AAPL", self.broker.positions)
        self.assertEqual(self.broker.cash, 10000 - 1000 + 1100)

    def test_order_history(self):
        order = self.broker.place_order(asset="AAPL", size=5, side="BUY")
        history = self.broker.get_order_history()
        self.assertIn(order.id, history)


if __name__ == "__main__":
    unittest.main()
