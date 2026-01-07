import timeit
from datetime import datetime
import os
from data_manager import DataManager, CSVDataLoader


symbols = [os.path.splitext(f)[0] for f in os.listdir(r'G:\backtest_data') if f.endswith('.csv')]#[:10]

load_st_time = datetime.now()
loader = CSVDataLoader(data_dir=r'G:\backtest_data')
dm = DataManager(loader)
dm.load_data(symbols)
load_et_time = datetime.now()
print(f"Data loaded in {load_et_time - load_st_time} seconds")

# print(len(dm.index_map_dict))
print(len(dm.index_map_dict['ADANIGREEN']))
print(len(dm.index_map_dict['ADANIPORTS']))

def execute():
    count = 0
    timestamps = dm.index_map_dict['ADANIGREEN'].keys()
    for ts in timestamps:
        bar_data = dm.get_data(symbols, ts)         
        for symbol in bar_data:
            _ = bar_data[symbol]
            count += 1
    # for ts in timestamps:
    #     for symbol in symbols:
    #         bar_data = dm.get_data_of_symbol(symbol, ts) #12 seconds for 10 calls
    #         count += 1

    print(f"count: {count}")

num_calls = 10
execution_time = timeit.timeit(execute, number=num_calls)
print(f"Average time for {num_calls} calls: {execution_time:.6f} seconds")


print(f"Symbols loaded: {symbols}")
