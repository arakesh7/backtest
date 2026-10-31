"""
Backtester: Institutional Event-Driven Quantitative Backtesting Framework.
"""

from .engine.backtester import Backtester
from .engine.calendar import TradingCalendar, WorkingDayTradingCalendar
from .strategy.base import Strategy, Intraday, IntradayMixin
from .execution.broker import Broker
from .execution.execution_model import (
    ExecutionModel,
    SimpleExecutionModel,
    PercentVolumeExecutionModel,
)
from .execution.slippage import Slippage, DefaultSlippage
from .execution.commission import (
    Commission,
    NoCommission,
    FlatCommission,
    PercentageCommission,
)
from .orders.order import Order, BuyOrder, SellOrder, Position, Trade
from .orders.sizer import (
    Sizer,
    FixedRiskPercentSizer,
    FixedSizeSizer,
    AllInSizer,
)
from .data.manager import DataManager
from .data.loader import BaseDataLoader, CSVDataLoader
from .data.history import BarHistory
from .analytics.analyzer import Analyzer, DefaultAnalyzer
from .utils.logging_config import setup_logging

__version__ = "0.1.0"

__all__ = [
    "Backtester",
    "TradingCalendar",
    "WorkingDayTradingCalendar",
    "Strategy",
    "Intraday",
    "IntradayMixin",
    "Broker",
    "ExecutionModel",
    "SimpleExecutionModel",
    "PercentVolumeExecutionModel",
    "Slippage",
    "DefaultSlippage",
    "Commission",
    "NoCommission",
    "FlatCommission",
    "PercentageCommission",
    "Order",
    "BuyOrder",
    "SellOrder",
    "Position",
    "Trade",
    "Sizer",
    "FixedRiskPercentSizer",
    "FixedSizeSizer",
    "AllInSizer",
    "DataManager",
    "BaseDataLoader",
    "CSVDataLoader",
    "BarHistory",
    "Analyzer",
    "DefaultAnalyzer",
    "setup_logging",
]
