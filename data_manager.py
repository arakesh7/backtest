import pandas as pd
import os
from typing import Dict, Optional

"""
for date in dates:
    df = DataManager.load_data(date)
    data = DataManager.get_data(df, 'AAPL', '2020-01-01', '2020-12-31')
    strategy.on_data(data)


"""

class CSVDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load(self, symbol) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        print(f"Loading data from {file_path}")
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




class DataManager:
    def __init__(self, loader):
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

    def load_data(self, symbols):
        old_symbols = list(self.data_cache.keys())
        symbols_to_evict = list(set(old_symbols) - set(symbols))
        self.clear_cache(symbols_to_evict)
        self._load_data(symbols)

    def _load_data(self, symbols):
        for symbol in symbols:
            if symbol not in self.data_cache:
                try:
                    raw_data = self.loader.load(symbol)
                    self.index_map_dict[symbol] = self._create_ts_to_index(raw_data)
                    self.data_cache[symbol] = raw_data.to_numpy()
                except FileNotFoundError as e:
                    print(e)
        
    def get_data(self, symbols, ts):
        if not isinstance(symbols, list):
            symbols = [symbols]
   
        result = {}
        for symbol in symbols:
            idx = self._get_index_from_timestamp(symbol, ts)
            if idx is not None:  # `if idx:` would incorrectly skip index 0 (the first row)
                result[symbol] = self.data_cache[symbol][idx]
        return result            

    def get_data_of_symbol(self, symbol, ts):
        idx = self._get_index_from_timestamp(symbol, ts)
        if idx is not None:
            return self.data_cache[symbol][idx]
    
    def _create_ts_to_index(self, raw_data) -> Optional[int]:
        n_data_array= raw_data.index
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
        

# dm = DataManager(loader)
# dm.load_data(['ADANIGREEN', 'ADANIPORTS'])
# dm._create_ts_to_index()
# print(dm.index_map_dict)
# print(dm._get_index_from_timestamp('ADANIGREEN', '2024-01-03 15:02:00'))
# print(dm.get_data('ADANIGREEN', '2024-01-03 15:02:00'))
# print(dm.get_data_of_symbol('ADANIGREEN', '2024-08-01 03:45:00'))
# print(dm.get_data(['ADANIGREEN', 'ADANIPORTS'], '2024-08-01 03:45:00'))



# # Timeit performance for accessing a specific row
# stmt = "dm.data_cache['ADANIGREEN'].loc['2024-01-01 03:46:00']"
# setup = "from __main__ import dm"
# execution_time = timeit.timeit(stmt, setup=setup, number=100000)
# print(f"Average time for 1000 accesses: {execution_time:.6f} seconds")

# index_keys = dm.data_cache['ADANIGREEN'].index
# index_map_dict = {str(key): i for i, key in enumerate(index_keys)}
# print(list(index_map_dict.items())[:5])
# dm_numpy = dm.data_cache['ADANIGREEN'].to_numpy()

# # Timeit performance for accessing a specific row
# stmt = "dm.data_cache['ADANIGREEN'].iloc[index_map_dict[str(datetime.strptime('2024-01-01 03:46:00', '%Y-%m-%d %H:%M:%S')  + timedelta(minutes=20))]]"
# setup = "from datetime import datetime, timedelta; from __main__ import index_map_dict, dm, dm_numpy"
# execution_time = timeit.timeit(stmt, setup=setup, number=100000)
# print(f"Average time for 1000 accesses: {execution_time:.6f} seconds")

# # Timeit performance for accessing a specific row
# stmt = "dm_numpy[index_map_dict[str(datetime.strptime('2024-01-01 03:46:00', '%Y-%m-%d %H:%M:%S')  + timedelta(minutes=20))]]"
# setup = "from datetime import datetime, timedelta; from __main__ import index_map_dict, dm, dm_numpy"
# execution_time = timeit.timeit(stmt, setup=setup, number=100000)
# print(f"Average time for 1000 accesses: {execution_time:.6f} seconds")

# # print(dm.data_cache['ADANIGREEN'].loc['2024-01-01 03:46:00']) #
# # print(dm.data_cache['ADANIGREEN'].to_numpy())
# from datetime import datetime, timedelta
# print(dm_numpy[index_map_dict[str(datetime.strptime('2024-01-01 03:46:00', '%Y-%m-%d %H:%M:%S')  + timedelta(minutes=20))]])

# # Create a raw dictionary with index keys as key and incremental numeric count as value
