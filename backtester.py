import logging
from datetime import  timedelta
from analyzer import DefaultAnalyzer
import pandas as pd

from broker import Broker
from logging_config import  setup_logging
from data_manager import DataManager, CSVDataLoader
from trading_calendar import WorkingDayTradingCalendar


setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtester:
    """
    Event-driven backtester that loops over a date range with a configurable frequency.
    """
    def __init__(self, start_date: str, end_date: str, strategy, frequency: str = 'D', universe_fn=None, analyzers=None, data_dir=None):
        """
        :param start_date: Start date as 'YYYY-MM-DD'
        :param end_date: End date as 'YYYY-MM-DD'
        :param frequency: Frequency string (e.g., 'D' for daily, 'W' for weekly, 'T' for minutely)
        :param on_bar: Callback function to execute on each bar (date)
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.frequency = frequency
        self.trading_calendar = WorkingDayTradingCalendar()
        self.strategy = strategy
        self.data_manager =  DataManager(CSVDataLoader(data_dir=r'G:\backtest_data'))
        self.universe_fn = universe_fn
        self.universe_refresh_interval = timedelta(days=2)  # Default refresh interval
        self.last_refresh_time = None
        self._broker = self.strategy.broker or Broker()
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

        time_index = self.trading_calendar.get_trading_bars(self.start_date, self.end_date, freq=self.frequency)
        for ts in time_index:
            if self._should_refresh_universe(ts):
                new_universe = self.universe_fn(ts)
                if set(new_universe) != set(self.current_universe):
                    self.current_universe = new_universe
                    logger.info(f"Universe updated for {ts}: {self.current_universe}")
                    self.data_manager.load_data(self.current_universe)

            data = self.data_manager.get_data(self.current_universe, str(ts))
            # print(f"Data for {ts}: {data}")
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
        Args:
            current_time (pd.Timestamp): Current timestamp in the simulation

        Returns:
            bool: True if universe should be refreshed, False otherwise
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