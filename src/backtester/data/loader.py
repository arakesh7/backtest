import os
from abc import ABC, abstractmethod
import pandas as pd


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self, symbol: str) -> pd.DataFrame:
        """Load data for a symbol and return canonical OHLCV DataFrame."""
        raise NotImplementedError


class CSVDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load(self, symbol: str) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No data found for symbol '{symbol}': {file_path}")
        cols = ['open', 'high', 'low', 'close', 'volume', 'ts']
        df = pd.read_csv(file_path, usecols=cols, parse_dates=['ts'])
        
        if df['ts'].dt.tz is None:
            df['ts'] = df['ts'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        else:
            df['ts'] = df['ts'].dt.tz_convert('Asia/Kolkata')

        df['ts'] = df['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df.index = df['ts']
        return df


class QDataLoader(BaseDataLoader):
    """
    Data loader integrating high-performance qdata parquet store.
    Supports both daily ('1d') and intraday ('1min') market data.
    """

    def __init__(
        self,
        data_dir: str = r"G:\qdata_data",
        timeframe: str = "1d",
        tz: str = "Asia/Kolkata",
    ):
        from pathlib import Path

        path = Path(data_dir)
        # Automatically resolve to child data folder if raw store is nested (e.g. G:\qdata_data\data)
        if not (path / "raw").exists() and (path / "data" / "raw").exists():
            path = path / "data"
        elif not (path / "catalog").exists() and (path / "data" / "catalog").exists():
            path = path / "data"

        self.data_dir = str(path)
        self.timeframe = timeframe.lower()
        self.tz = tz

        from qdata import DataReader
        self.reader = DataReader(data_dir=self.data_dir)

    def load(self, symbol: str) -> pd.DataFrame:
        df = self.reader.load(symbol, timeframe=self.timeframe)
        if df is None or df.empty:
            raise FileNotFoundError(
                f"No data found in qdata for symbol '{symbol}' with timeframe '{self.timeframe}' in {self.data_dir}"
            )
        df = df.copy()

        # Format canonical timestamps matching the backtester's calendar
        if self.timeframe in ("1d", "d", "daily"):
            df["ts"] = df.index.strftime("%Y-%m-%d 00:00:00")
        else:
            if df.index.tz is not None:
                ts_series = df.index.tz_convert(self.tz).tz_localize(None)
            else:
                ts_series = df.index
            df["ts"] = ts_series.strftime("%Y-%m-%d %H:%M:%S")

        df.index = df["ts"]
        cols = ["ts", "open", "high", "low", "close", "volume"]
        return df[cols]
