import unittest
from order import Order, Position, Trade
from broker import Broker

class TestBroker(unittest.TestCase):
    def setUp(self):
        self.broker = Broker(cash=10000)

    def test_place_order(self):
        order = Order(asset='AAPL', size=10, side='BUY')
        self.broker.place_order(order)
        self.assertIn(order.id, self.broker.orders)

    def test_buy_order_executes_and_deducts_cash(self):
        order = Order(asset='AAPL', size=10, side='BUY', price=100, type='MARKET')
        self.broker.place_order(order)
        market_data = {'price': 100, 'timestamp': '2024-01-01T10:00:00'}
        self.broker.process_orders(market_data)
        self.assertEqual(order.status, 'EXECUTED')
        self.assertEqual(self.broker.positions['AAPL'].size, 10)
        self.assertEqual(self.broker.cash, 10000 - 1000)

    def test_insufficient_cash(self):
        order = Order(asset='AAPL', size=200, side='BUY', price=100, type='MARKET')
        self.broker.place_order(order)
        market_data = {'price': 100, 'timestamp': '2024-01-01T10:00:00'}
        self.broker.process_orders(market_data)
        self.assertEqual(order.status, 'REJECTED')  # Should not execute
        self.assertEqual(self.broker.positions.get('AAPL', None), None)

    def test_sell_order_injects_cash(self):
        # First buy
        buy_order = Order(asset='AAPL', size=10, side='BUY', price=100, type='MARKET')
        self.broker.place_order(buy_order)
        self.broker.process_orders({'price': 100, 'timestamp': '2024-01-01T10:00:00'})
        # Then sell
        sell_order = Order(asset='AAPL', size=10, side='SELL', price=110, type='MARKET')
        self.broker.place_order(sell_order)
        self.broker.process_orders({'price': 110, 'timestamp': '2024-01-01T11:00:00'})
        self.assertEqual(sell_order.status, 'EXECUTED')
        self.assertIsNone(self.broker.positions.get('AAPL', None))
        self.assertEqual(self.broker.cash, 10000 - 1000 + 1100)

    def test_order_history(self):
        order = Order(asset='AAPL', size=5, side='BUY')
        self.broker.place_order(order)
        history = self.broker.get_order_history()
        self.assertIn(order.id, history)

if __name__ == '__main__':
    unittest.main()
