import unittest
import pandas as pd
from backtester import Position, Broker, Backtester, Strategy, BarHistory, FixedRiskPercentSizer, FlatCommission, PercentageCommission

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

    def test_partial_sell_injects_cash(self):
        # Buy 10 shares at 100 -> cash = 9000
        self.broker.place_order(asset="AAPL", size=10, side="BUY")
        self.broker.process_orders(self.broker.current_ts, self._make_data("AAPL", 100))
        self.assertEqual(self.broker.cash, 9000)

        # Partially sell 5 shares at 120 -> cash should be 9000 + (5 * 120) = 9600
        self.broker.current_ts = pd.Timestamp("2024-01-01 11:00:00")
        sell_order = self.broker.place_order(asset="AAPL", size=5, side="SELL")
        trades = self.broker.process_orders(self.broker.current_ts, self._make_data("AAPL", 120))
        self.assertEqual(sell_order.status, "EXECUTED")
        self.assertEqual(self.broker.positions["AAPL"].size, 5)
        self.assertEqual(self.broker.positions["AAPL"].avg_price, 100)
        self.assertEqual(self.broker.cash, 9600)
        self.assertEqual(trades[0].pnl, 100)  # (120 - 100) * 5

    def test_position_flip_long_to_short(self):
        # Buy 10 shares at 100 -> cash = 9000
        self.broker.place_order(asset="AAPL", size=10, side="BUY")
        self.broker.process_orders(self.broker.current_ts, self._make_data("AAPL", 100))

        # Sell 15 shares at 120 -> closes 10 longs, opens 5 shorts
        self.broker.current_ts = pd.Timestamp("2024-01-01 11:00:00")
        sell_order = self.broker.place_order(asset="AAPL", size=15, side="SELL")
        trades = self.broker.process_orders(self.broker.current_ts, self._make_data("AAPL", 120))
        self.assertEqual(sell_order.status, "EXECUTED")
        self.assertEqual(self.broker.positions["AAPL"].size, -5)
        self.assertEqual(self.broker.positions["AAPL"].avg_price, 120)  # Short entered at 120
        self.assertEqual(self.broker.cash, 9000 + (15 * 120))  # 10800
        self.assertEqual(trades[0].pnl, 200)  # (120 - 100) * 10

    def test_short_selling_and_covering(self):
        # Short 10 shares at 100 -> cash = 10000 + 1000 = 11000
        self.broker.place_order(asset="AAPL", size=10, side="SELL")
        self.broker.process_orders(self.broker.current_ts, self._make_data("AAPL", 100))
        self.assertEqual(self.broker.cash, 11000)
        self.assertEqual(self.broker.positions["AAPL"].size, -10)

        # Partially cover 4 shares at 80 -> cash = 11000 - (4 * 80) = 10680
        self.broker.current_ts = pd.Timestamp("2024-01-01 11:00:00")
        self.broker.place_order(asset="AAPL", size=4, side="BUY")
        trades = self.broker.process_orders(self.broker.current_ts, self._make_data("AAPL", 80))
        self.assertEqual(self.broker.positions["AAPL"].size, -6)
        self.assertEqual(self.broker.cash, 10680)
        self.assertEqual(trades[0].pnl, 80)  # (100 - 80) * 4

    def test_stop_loss_order_execution(self):
        # Buy 10 shares at 100
        self.broker.place_order(asset="AAPL", size=10, side="BUY")
        self.broker.process_orders(self.broker.current_ts, self._make_data("AAPL", 100))

        # Place SL sell order at stop_price 95
        sl_order = self.broker.place_order(asset="AAPL", size=10, side="SELL", order_type="SL", stop_price=95)
        # Bar opens at 99, low reaches 94 -> triggers at 95
        bar = {"AAPL": (self.broker.current_ts, 99.0, 100.0, 94.0, 96.0, 50000)}
        trades = self.broker.process_orders(self.broker.current_ts, bar)
        self.assertEqual(sl_order.status, "EXECUTED")
        self.assertEqual(trades[0].price, 95.0)
        self.assertEqual(trades[0].pnl, -50.0)

    def test_bar_history_newest_to_oldest_slicing(self):
        bh = BarHistory(size=5)
        for i in range(7):
            bh.add(f"b{i}")
        # Buffer holds last 5 bars from newest to oldest: b6, b5, b4, b3, b2
        self.assertEqual(bh[0], "b6")    # Latest bar
        self.assertEqual(bh[-1], "b5")   # 1 bar before latest
        self.assertEqual(bh[-2], "b4")   # 2 bars before latest
        self.assertEqual(list(bh[:]), ["b6", "b5", "b4", "b3", "b2"])  # Newest to oldest slice
        self.assertEqual(list(bh[:3]), ["b6", "b5", "b4"])             # 3 most recent bars

    def test_bar_history_chronological(self):
        bh = BarHistory(size=5)
        for i in range(7):
            bh.add(f"b{i}")
        # Chronological returns oldest-to-newest for indicators: b2, b3, b4, b5, b6
        self.assertEqual(list(bh.chronological()), ["b2", "b3", "b4", "b5", "b6"])

    def test_bar_history_boundary_check(self):
        bh = BarHistory(size=5)
        bh.add(10)
        bh.add(20)
        # count is 2. Valid indices: 0 (latest=20), -1 (1 bar ago=10).
        self.assertEqual(bh[0], 20)
        self.assertEqual(bh[-1], 10)
        with self.assertRaises(IndexError):
            _ = bh[-2]
        with self.assertRaises(IndexError):
            _ = bh[2]

    def test_fwd01_conflict_resolution_approach_conservative(self):
        # Broker configured with intra_bar_fill_model='CONSERVATIVE' (Stop Loss always takes priority)
        broker = Broker(cash=10000, intra_bar_fill_model="CONSERVATIVE")
        broker.current_ts = pd.Timestamp("2024-01-01 10:00:00")
        broker.place_order(asset="AAPL", size=10, side="BUY")
        broker.process_orders(broker.current_ts, {"AAPL": (broker.current_ts, 100, 100, 100, 100, 1000)})

        # Place competing exit orders: Target at 110, Stop Loss at 90
        tgt = broker.place_order(asset="AAPL", size=10, side="SELL", order_type="LIMIT", price=110)
        sl = broker.place_order(asset="AAPL", size=10, side="SELL", order_type="SL", stop_price=90)

        # Ambiguous bar: Open=100, High=115 (hits target), Low=85 (hits stop), Close=112 (Bullish)
        bar = {"AAPL": (broker.current_ts, 100.0, 115.0, 85.0, 112.0, 50000)}
        trades = broker.process_orders(broker.current_ts, bar)

        # CONSERVATIVE forces Stop Loss to execute first
        self.assertEqual(len(trades), 1)
        self.assertEqual(sl.status, "EXECUTED")
        self.assertEqual(tgt.status, "CANCELLED")
        self.assertEqual(trades[0].price, 90.0)
        self.assertEqual(broker.get_position("AAPL").size, 0)
        self.assertNotIn("AAPL", broker.positions)

    def test_fwd01_conflict_resolution_polarity_bullish_candle(self):
        # Broker configured with intra_bar_fill_model='POLARITY' (Default: Trajectory based on candle polarity)
        broker = Broker(cash=10000, intra_bar_fill_model="POLARITY")
        broker.current_ts = pd.Timestamp("2024-01-01 10:00:00")
        broker.place_order(asset="AAPL", size=10, side="BUY")
        broker.process_orders(broker.current_ts, {"AAPL": (broker.current_ts, 100, 100, 100, 100, 1000)})

        tgt = broker.place_order(asset="AAPL", size=10, side="SELL", order_type="LIMIT", price=110)
        sl = broker.place_order(asset="AAPL", size=10, side="SELL", order_type="SL", stop_price=90)

        # Bullish bar (Close 112 >= Open 100): Trajectory Open -> Low -> High -> Close
        # Low (85) was visited BEFORE High (115) -> Stop Loss hit first
        bar = {"AAPL": (broker.current_ts, 100.0, 115.0, 85.0, 112.0, 50000)}
        trades = broker.process_orders(broker.current_ts, bar)

        self.assertEqual(len(trades), 1)
        self.assertEqual(sl.status, "EXECUTED")
        self.assertEqual(tgt.status, "CANCELLED")
        self.assertEqual(trades[0].price, 90.0)
        self.assertEqual(broker.get_position("AAPL").size, 0)
        self.assertNotIn("AAPL", broker.positions)

    def test_fwd01_conflict_resolution_polarity_bearish_candle(self):
        # Broker with POLARITY on Bearish bar
        broker = Broker(cash=10000, intra_bar_fill_model="POLARITY")
        broker.current_ts = pd.Timestamp("2024-01-01 10:00:00")
        broker.place_order(asset="AAPL", size=10, side="BUY")
        broker.process_orders(broker.current_ts, {"AAPL": (broker.current_ts, 100, 100, 100, 100, 1000)})

        tgt = broker.place_order(asset="AAPL", size=10, side="SELL", order_type="LIMIT", price=110)
        sl = broker.place_order(asset="AAPL", size=10, side="SELL", order_type="SL", stop_price=90)

        # Bearish bar (Close 88 < Open 100): Trajectory Open -> High -> Low -> Close
        # High (115) was visited BEFORE Low (85) -> Target hit first
        bar = {"AAPL": (broker.current_ts, 100.0, 115.0, 85.0, 88.0, 50000)}
        trades = broker.process_orders(broker.current_ts, bar)

        self.assertEqual(len(trades), 1)
        self.assertEqual(tgt.status, "EXECUTED")
        self.assertEqual(sl.status, "CANCELLED")
        self.assertEqual(trades[0].price, 110.0)
        self.assertEqual(broker.get_position("AAPL").size, 0)
        self.assertNotIn("AAPL", broker.positions)

    def test_fwd01_conflict_resolution_polarity_gap_open(self):
        broker = Broker(cash=10000, intra_bar_fill_model="POLARITY")
        broker.current_ts = pd.Timestamp("2024-01-01 10:00:00")
        broker.place_order(asset="AAPL", size=10, side="BUY")
        broker.process_orders(broker.current_ts, {"AAPL": (broker.current_ts, 100, 100, 100, 100, 1000)})

        tgt = broker.place_order(asset="AAPL", size=10, side="SELL", order_type="LIMIT", price=110)
        sl = broker.place_order(asset="AAPL", size=10, side="SELL", order_type="SL", stop_price=90)

        # Gap Down at Open (Open 88 <= stop_price 90): Gap takes absolute priority -> Stop fills at Open 88
        bar = {"AAPL": (broker.current_ts, 88.0, 112.0, 85.0, 110.0, 50000)}
        trades = broker.process_orders(broker.current_ts, bar)

        self.assertEqual(len(trades), 1)
        self.assertEqual(sl.status, "EXECUTED")
        self.assertEqual(tgt.status, "CANCELLED")
        self.assertEqual(trades[0].price, 88.0)
        self.assertEqual(broker.get_position("AAPL").size, 0)
        self.assertNotIn("AAPL", broker.positions)

    def test_backtester_intra_bar_fill_model_switch(self):
        class DummyStrategy(Strategy):
            def on_data(self, ts, data):
                pass

        # Verify Backtester defaults to CONSERVATIVE
        strat_def = DummyStrategy(broker=Broker())
        bt_def = Backtester(
            start_date="2024-01-01",
            end_date="2024-01-05",
            strategy=strat_def,
            universe_fn=lambda ts: [],
        )
        self.assertEqual(bt_def.intra_bar_fill_model, "CONSERVATIVE")
        self.assertEqual(bt_def._broker.intra_bar_fill_model, "CONSERVATIVE")

        # Verify Backtester accepts POLARITY switch
        strat_a = DummyStrategy(broker=Broker())
        bt_a = Backtester(
            start_date="2024-01-01",
            end_date="2024-01-05",
            strategy=strat_a,
            universe_fn=lambda ts: [],
            intra_bar_fill_model="POLARITY",
        )
        self.assertEqual(bt_a.intra_bar_fill_model, "POLARITY")
        self.assertEqual(bt_a._broker.intra_bar_fill_model, "POLARITY")
    def test_short_selling_insufficient_margin(self):
        # Initial cash is 1000. Try to short 200 shares at 100 (notional 20,000).
        # Margin required with margin_ratio=0.5 is 10,000, which exceeds 1000.
        broker = Broker(cash=1000, margin_ratio=0.5)
        broker.current_ts = pd.Timestamp("2024-01-01 10:00:00")
        order = broker.place_order(asset="AAPL", size=200, side="SELL")
        broker.process_orders(broker.current_ts, self._make_data("AAPL", 100))
        self.assertEqual(order.status, "REJECTED")
        self.assertNotIn("AAPL", broker.positions)
        self.assertEqual(broker.cash, 1000)

    def test_fixed_risk_sizer_with_stop_loss(self):
        # Broker equity = 10,000. Risk = 1% = $100.
        # Entry = 100, Stop = 98 -> Risk per share = $2.
        # Expected size = 100 / 2 = 50 shares.
        broker = Broker(cash=10000)
        sizer = FixedRiskPercentSizer(risk_percent=1.0)
        size = sizer.get_size(broker, asset="AAPL", entry_price=100.0, stop_loss_price=98.0, side="BUY")
        self.assertEqual(size, 50)

    def test_fixed_risk_sizer_without_stop_loss(self):
        # Broker equity = 10,000. Risk = 2% = $200.
        # Entry = 100, Stop = None -> Fallback to capital allocation: 200 / 100 = 2 shares.
        broker = Broker(cash=10000)
        sizer = FixedRiskPercentSizer(risk_percent=2.0)
        size = sizer.get_size(broker, asset="AAPL", entry_price=100.0, stop_loss_price=None, side="BUY")
        self.assertEqual(size, 2)

    def test_dynamic_universe_retains_held_positions(self):
        # Strategy buys AAPL on day 1. On day 2, AAPL drops out of the universe.
        # The engine must retain AAPL data so stop-loss / exit orders execute properly.
        from backtester.data.loader import BaseDataLoader
        from backtester.data.manager import DataManager

        class MockLoader(BaseDataLoader):
            def load(self, symbol: str) -> pd.DataFrame:
                dates = pd.date_range("2024-01-01 09:15:00", periods=5, freq="D")
                return pd.DataFrame({
                    "open": [100.0, 95.0, 90.0, 85.0, 80.0],
                    "high": [102.0, 96.0, 91.0, 86.0, 81.0],
                    "low": [98.0, 93.0, 89.0, 84.0, 79.0],
                    "close": [99.0, 94.0, 90.0, 85.0, 80.0],
                    "volume": [1000, 1000, 1000, 1000, 1000],
                    "ts": [str(d) for d in dates]
                }, index=[str(d) for d in dates])

        loader = MockLoader()
        dm = DataManager(loader)
        # Day 1: Universe has AAPL
        dm.load_data(["AAPL"])
        self.assertIn("AAPL", dm.data_cache)

        # Day 2: AAPL drops out of universe, but broker holds AAPL position
        # With protected_symbols=["AAPL"], AAPL must not be evicted!
        dm.load_data(["MSFT"], protected_symbols=["AAPL"])
        self.assertIn("AAPL", dm.data_cache)
        self.assertIn("MSFT", dm.data_cache)

    def test_broker_equity_mark_to_market(self):
        broker = Broker(cash=10000)
        broker.current_ts = pd.Timestamp("2024-01-01 10:00:00")
        broker.place_order(asset="AAPL", size=10, side="BUY")
        broker.process_orders(broker.current_ts, {"AAPL": make_bar(100)})
        # Cost basis is 100, cash is 9000, equity is 10000
        self.assertEqual(broker.equity, 10000.0)

        # Bar moves price to 120
        broker.current_ts = pd.Timestamp("2024-01-01 10:01:00")
        broker.process_orders(broker.current_ts, {"AAPL": make_bar(120)})
        # Equity is marked to market: 9000 + 10 * 120 = 10200
        self.assertEqual(broker.equity, 10200.0)

        # Bar moves price down to 90
        broker.current_ts = pd.Timestamp("2024-01-01 10:02:00")
        broker.process_orders(broker.current_ts, {"AAPL": make_bar(90)})
        # Equity is marked to market: 9000 + 10 * 90 = 9900
        self.assertEqual(broker.equity, 9900.0)

    def test_commission_model_and_net_realized_pnl(self):
        # Commission is flat 5 per trade
        comm_model = FlatCommission(fee=5.0)
        broker = Broker(cash=10000, commission_model=comm_model)
        broker.current_ts = pd.Timestamp("2024-01-01 10:00:00")

        # Buy 10 AAPL @ 100 -> Cost = 1000 + 5 = 1005. Cash = 8995.
        broker.place_order(asset="AAPL", size=10, side="BUY")
        trades_buy = broker.process_orders(broker.current_ts, {"AAPL": make_bar(100)})
        self.assertEqual(broker.cash, 8995.0)
        self.assertEqual(trades_buy[0].commission, 5.0)
        self.assertEqual(trades_buy[0].pnl, 0.0)

        # Sell 10 AAPL @ 110 -> Gross gain = (110 - 100) * 10 = 100.
        # Total roundtrip commission = 5 (entry) + 5 (exit) = 10.
        # Net realized PnL = 100 - 10 = 90.
        # Cash = 8995 + (1100 - 5) = 10090. Net cash gain = +90.
        broker.current_ts = pd.Timestamp("2024-01-01 10:01:00")
        broker.place_order(asset="AAPL", size=10, side="SELL")
        trades_sell = broker.process_orders(broker.current_ts, {"AAPL": make_bar(110)})
        self.assertEqual(trades_sell[0].commission, 5.0)
        self.assertEqual(trades_sell[0].pnl, 90.0)
        self.assertEqual(broker.cash, 10090.0)

    def test_short_reserve_marked_to_market_and_insolvency_check(self):
        # Initial cash: 10,000, margin_ratio: 0.5
        broker = Broker(cash=10000, margin_ratio=0.5)
        broker.current_ts = pd.Timestamp("2024-01-01 10:00:00")

        # Short 20 shares at 100 -> cash becomes 10,000 + 2000 = 12,000
        broker.place_order(asset="AAPL", size=20, side="SELL")
        broker.process_orders(broker.current_ts, {"AAPL": (broker.current_ts, 100, 100, 100, 100, 1000)})
        self.assertEqual(broker.cash, 12000)
        # At price 100: reserve = 20 * 100 * 1.5 = 3000. Available cash = 12,000 - 3000 = 9000
        self.assertEqual(broker.get_available_cash(), 9000.0)

        # Price rallies to 200 on next bar
        broker.current_ts = pd.Timestamp("2024-01-01 10:01:00")
        broker.process_orders(broker.current_ts, {"AAPL": (broker.current_ts, 200, 205, 195, 200, 1000)})
        # At price 200: reserve = 20 * 200 * 1.5 = 6000. Available cash = 12,000 - 6000 = 6000
        self.assertEqual(broker.get_available_cash(), 6000.0)

        # Price skyrockets to 500 (account equity = 12,000 - 20 * 500 = 2,000; reserve = 20 * 500 * 1.5 = 15,000 > cash)
        broker.current_ts = pd.Timestamp("2024-01-01 10:02:00")
        broker.process_orders(broker.current_ts, {"AAPL": (broker.current_ts, 500, 510, 490, 500, 1000)})
        # Available cash should clamp to 0.0
        self.assertEqual(broker.get_available_cash(), 0.0)

        # Attempt to open a new BUY order when available cash is 0 should be REJECTED
        new_order = broker.place_order(asset="MSFT", size=5, side="BUY")
        broker.process_orders(broker.current_ts, {"MSFT": (broker.current_ts, 100, 105, 95, 100, 1000)})
        self.assertEqual(new_order.status, "REJECTED")

        # Now price surges to 700 -> account equity = 12,000 - 20 * 700 = -2000 (insolvent)
        broker.current_ts = pd.Timestamp("2024-01-01 10:03:00")
        broker.process_orders(broker.current_ts, {"AAPL": (broker.current_ts, 700, 710, 690, 700, 1000)})
        self.assertLess(broker.equity, 0)
        insolvent_order = broker.place_order(asset="GOOG", size=1, side="BUY")
        broker.process_orders(broker.current_ts, {"GOOG": (broker.current_ts, 50, 55, 45, 50, 1000)})
        self.assertEqual(insolvent_order.status, "REJECTED")

    def test_stoplimit_order_execution_across_subsequent_bars(self):
        broker = Broker(cash=10000)
        broker.current_ts = pd.Timestamp("2024-01-01 10:00:00")

        # Place BUY STOPLIMIT: Trigger = 100, Limit = 99 (buy on pullback after 100 trigger)
        order = broker.place_order(
            asset="AAPL",
            size=10,
            side="BUY",
            order_type="STOPLIMIT",
            trigger_price=100.0,
            price=99.0,
        )
        self.assertFalse(order.is_triggered)

        # Bar 1: Price touches 100.5 (trigger breached), but low is 99.5 (limit 99 not reached)
        # Bar 1 OHLCV: open 98.0, high 100.5, low 99.5, close 100.0
        bar1 = {"AAPL": (broker.current_ts, 98.0, 100.5, 99.5, 100.0, 1000)}
        trades1 = broker.process_orders(broker.current_ts, bar1)
        self.assertEqual(len(trades1), 0)
        self.assertTrue(order.is_triggered)
        self.assertEqual(order.status, "CREATED")

        # Bar 2: Price opens at 99.2, high 99.4, low 98.8, close 99.0
        # Notice: Bar 2 never reaches trigger 100.0! But order is already triggered, so it acts as LIMIT at 99.0
        broker.current_ts = pd.Timestamp("2024-01-01 10:01:00")
        bar2 = {"AAPL": (broker.current_ts, 99.2, 99.4, 98.8, 99.0, 1000)}
        trades2 = broker.process_orders(broker.current_ts, bar2)
        self.assertEqual(len(trades2), 1)
        self.assertEqual(order.status, "EXECUTED")
        self.assertEqual(trades2[0].price, 99.0)
        self.assertEqual(broker.get_position("AAPL").size, 10)

    def test_stoplimit_sell_order_execution_across_subsequent_bars(self):
        broker = Broker(cash=10000)
        broker.current_ts = pd.Timestamp("2024-01-01 10:00:00")
        # Buy 10 shares at 100 first
        broker.place_order(asset="AAPL", size=10, side="BUY")
        broker.process_orders(broker.current_ts, {"AAPL": make_bar(100)})

        # Place SELL STOPLIMIT (exit order): Trigger = 95.0, Limit = 95.5 (sell on bounce after breakdown)
        order = broker.place_order(
            asset="AAPL",
            size=10,
            side="SELL",
            order_type="STOPLIMIT",
            trigger_price=95.0,
            price=95.5,
        )

        # Bar 1: Price drops to 94.0 (triggers at 95.0), but high only reaches 95.2 (limit 95.5 not touched)
        broker.current_ts = pd.Timestamp("2024-01-01 10:01:00")
        bar1 = {"AAPL": (broker.current_ts, 95.0, 95.2, 94.0, 94.5, 1000)}
        trades1 = broker.process_orders(broker.current_ts, bar1)
        self.assertEqual(len(trades1), 0)
        self.assertTrue(order.is_triggered)

        # Bar 2: Price bounces: open 94.5, high 96.0, low 94.0, close 95.8
        # High touches 96.0 >= 95.5 limit price. Order fills!
        broker.current_ts = pd.Timestamp("2024-01-01 10:02:00")
        bar2 = {"AAPL": (broker.current_ts, 94.5, 96.0, 94.0, 95.8, 1000)}
        trades2 = broker.process_orders(broker.current_ts, bar2)
        self.assertEqual(len(trades2), 1)
        self.assertEqual(order.status, "EXECUTED")
        self.assertEqual(trades2[0].price, 95.5)
        self.assertEqual(broker.get_position("AAPL").size, 0)


if __name__ == "__main__":
    unittest.main()
