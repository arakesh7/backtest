import os
import timeit
from datetime import datetime
from backtester import DataManager, CSVDataLoader

DATA_DIR = r'G:\backtest_data'

if not os.path.exists(DATA_DIR):
    print(f"Data directory {DATA_DIR} does not exist. Skipping profiling.")
    exit(0)

symbols = [os.path.splitext(f)[0] for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

load_st_time = datetime.now()
loader = CSVDataLoader(data_dir=DATA_DIR)
dm = DataManager(loader)
dm.load_data(symbols)
load_et_time = datetime.now()
print(f"Data loaded in {load_et_time - load_st_time} seconds")


def execute():
    count = 0
    if not dm.index_map_dict:
        return
    first_symbol = list(dm.index_map_dict.keys())[0]
    timestamps = dm.index_map_dict[first_symbol].keys()
    for ts in timestamps:
        bar_data = dm.get_data(symbols, ts)
        for symbol in bar_data:
            _ = bar_data[symbol]
            count += 1


if __name__ == "__main__":
    num_calls = 5
    execution_time = timeit.timeit(execute, number=num_calls)
    print(f"Average time for {num_calls} calls: {execution_time:.6f} seconds")
    print(f"Symbols loaded: {symbols}")
