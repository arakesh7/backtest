from datetime import timedelta
import logging
import pandas as pd

from ..analytics.analyzer import DefaultAnalyzer
from ..execution.broker import Broker
from ..utils.logging_config import setup_logging
from ..data.manager import DataManager
from ..data.loader import CSVDataLoader
from .calendar import WorkingDayTradingCalendar

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Event-driven backtester that loops over a date range with a configurable frequency.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        strategy,
        frequency: str = "D",
        universe_fn=None,
        analyzers=None,
        data_dir=None,
        data_loader=None,
        data_manager=None,
        intra_bar_fill_model: str = "CONSERVATIVE",
    ):
        """
        :param start_date: Start date as 'YYYY-MM-DD'
        :param end_date: End date as 'YYYY-MM-DD'
        :param frequency: Frequency string (e.g., 'D' for daily, 'W' for weekly, '1min' for minutely)
        :param data_loader: Optional BaseDataLoader instance (e.g. QDataLoader, CSVDataLoader)
        :param data_manager: Optional DataManager instance
        :param intra_bar_fill_model: 'CONSERVATIVE' (Prudential Stop-Loss Priority, default) or 'POLARITY' (Candle Polarity Trajectory)
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.frequency = frequency
        self.trading_calendar = WorkingDayTradingCalendar()
        self.strategy = strategy
        if data_manager is not None:
            self.data_manager = data_manager
        elif data_loader is not None:
            self.data_manager = DataManager(data_loader)
        else:
            self.data_manager = DataManager(CSVDataLoader(data_dir=data_dir or r"G:\backtest_data"))
        self.universe_fn = universe_fn
        self.universe_refresh_interval = timedelta(days=2)  # Default refresh interval
        self.last_refresh_time = None
        self.intra_bar_fill_model = intra_bar_fill_model.upper()
        self._broker = self.strategy.broker or Broker(intra_bar_fill_model=self.intra_bar_fill_model)
        if hasattr(self._broker, "intra_bar_fill_model"):
            self._broker.intra_bar_fill_model = self.intra_bar_fill_model
        self.current_universe = []
        self.analyzers = analyzers or [DefaultAnalyzer()]
        self._attach_components()

    def _attach_components(self):
        """Attach backtester reference to its components."""
        for analyzer in self.analyzers:
            analyzer.attach_backtester(self)

    def run(self):
        if self.universe_fn is None:
            raise ValueError("Universe must be provided to run the backtester.")

        for analyzer in self.analyzers:
            analyzer.on_start()

        # retrieve all the time indexes based on the frequency during the backtest period.
        time_index = self.trading_calendar.get_trading_bars(
            self.start_date, self.end_date, freq=self.frequency
        )
        for ts in time_index:
            active_position_symbols = {
                asset for asset, pos in self._broker.positions.items() if pos.size != 0
            }
            active_order_symbols = {
                order.asset
                for order in self._broker.orders.values()
                if order.status in ("CREATED", "PARTIALLY_FILLED")
            }
            held_symbols = active_position_symbols | active_order_symbols

            if self._should_refresh_universe(ts):
                new_universe = self.universe_fn(ts)
                if set(new_universe) != set(self.current_universe):
                    self.current_universe = new_universe
                    logger.info(f"Universe updated for {ts}: {self.current_universe}")
                    self.data_manager.load_data(self.current_universe, protected_symbols=held_symbols)
            elif held_symbols:
                # Ensure any newly held/ordered symbols are loaded in data_manager
                missing = [s for s in held_symbols if s not in self.data_manager.data_cache]
                if missing:
                    self.data_manager.load_data(list(set(self.current_universe) | held_symbols))

            tracking_symbols = list(set(self.current_universe) | held_symbols)
            data = self.data_manager.get_data(tracking_symbols, str(ts))
            self._broker.current_ts = ts
            executed_trades = self._broker.process_orders(ts, data)

            for trade in executed_trades:
                for analyzer in self.analyzers:
                    analyzer.on_trade(trade)

            for analyzer in self.analyzers:
                analyzer.on_bar(ts, data)

            self.strategy.next(ts, data)

        return self.get_results()

    def _should_refresh_universe(self, current_time) -> bool:
        """Determines if the universe of tradable assets needs to be refreshed.

        Returns True if:
            - No previous refresh, or
            - Current time exceeds last_refresh_time + interval
        """
        if self.last_refresh_time is None:
            self.last_refresh_time = current_time
            return True

        nxt_refresh_time = self.last_refresh_time + self.universe_refresh_interval
        if current_time > nxt_refresh_time:
            self.last_refresh_time = current_time
            return True
        return False

    def get_results(self):
        """Collects and returns results from all analyzers."""
        all_results = {}
        for analyzer in self.analyzers:
            all_results[analyzer.__class__.__name__] = analyzer.get_results()
        return all_results
