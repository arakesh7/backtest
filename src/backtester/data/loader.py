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
