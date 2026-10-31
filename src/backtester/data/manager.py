import logging
from typing import Dict, List, Optional
import pandas as pd
from .loader import BaseDataLoader, CSVDataLoader

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self, loader: BaseDataLoader):
        self.loader = loader
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.universe_refresh_frequency = 1  # days
        self.index_map_dict = {}

    def clear_cache(self, symbols=None):
        """
        Clear the data cache.
        """
        if symbols is None:
            self.data_cache.clear()
            self.index_map_dict.clear()
        else:
            for symbol in symbols:
                if symbol in self.data_cache:
                    del self.data_cache[symbol]
                    del self.index_map_dict[symbol]

    def _to_evict(self, symbols):
        old_symbols = list(self.data_cache.keys())
        symbols_to_evict = list(set(old_symbols) - set(symbols))
        return symbols_to_evict

    def load_data(self, symbols, protected_symbols=None):
        """
        Load data for the given symbols into cache.
        Evicts symbols that are neither in `symbols` nor in `protected_symbols`.
        """
        protected = set(protected_symbols or [])
        keep_symbols = set(symbols) | protected
        old_symbols = list(self.data_cache.keys())
        symbols_to_evict = list(set(old_symbols) - keep_symbols)
        self.clear_cache(symbols_to_evict)
        self._load_data(list(keep_symbols))

    def _load_data(self, symbols):
        for symbol in symbols:
            if symbol not in self.data_cache:
                try:
                    raw_data = self.loader.load(symbol)
                    self.index_map_dict[symbol] = self._create_ts_to_index(raw_data)
                    self.data_cache[symbol] = raw_data.to_numpy()
                except FileNotFoundError as e:
                    logger.warning(e)

    def get_data(self, symbols, ts):
        if not isinstance(symbols, list):
            symbols = [symbols]

        result = {}
        for symbol in symbols:
            idx = self._get_index_from_timestamp(symbol, ts)
            if idx is not None:  # Ensure index 0 is not falsely evaluated as False
                result[symbol] = self.data_cache[symbol][idx]
        return result

    def get_data_of_symbol(self, symbol, ts):
        idx = self._get_index_from_timestamp(symbol, ts)
        if idx is not None:
            return self.data_cache[symbol][idx]

    def _create_ts_to_index(self, raw_data) -> Dict[str, int]:
        n_data_array = raw_data.index
        return {str(key): i for i, key in enumerate(n_data_array)}

    def _get_index_from_timestamp(self, symbol, timestamp: str) -> Optional[int]:
        """
        Get the index of the given timestamp in the data cache.
        Returns None if the symbol was never loaded or the timestamp doesn't exist.
        """
        symbol_map = self.index_map_dict.get(symbol)
        if symbol_map is None:
            return None
        return symbol_map.get(timestamp, None)
