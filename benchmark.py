import pandas as pd
import time
import os
import random
from data_manager import DataManager, CSVDataLoader

# Configuration
DATA_DIR = r'G:\backtest_data'
SYMBOL = 'ADANIPORTS'

if not os.path.exists(os.path.join(DATA_DIR, f"{SYMBOL}.csv")):
    print(f"Error: Could not find {SYMBOL}.csv in {DATA_DIR}")
    exit()

# Setup DataManager
loader = CSVDataLoader(data_dir=DATA_DIR)
dm = DataManager(loader)
dm.load_data([SYMBOL])

# Get the raw dataframe for comparison
raw_df = loader.load(SYMBOL)

# Get a list of unique timestamps from the data to simulate a real backtest loop
all_timestamps = list(raw_df.index)
# We'll use 5,000 different timestamps to ensure we bypass simple row/object caching
test_timestamps = all_timestamps[:5000] if len(all_timestamps) >= 5000 else all_timestamps

print(f"Benchmarking with {len(test_timestamps)} unique lookups to avoid CPU/Object caching...")

def pandas_benchmark():
    start = time.perf_counter()
    for ts in test_timestamps:
        _ = raw_df.loc[ts]
    return time.perf_counter() - start

def datamanager_benchmark():
    start = time.perf_counter()
    for ts in test_timestamps:
        _ = dm.get_data(SYMBOL, ts)
    return time.perf_counter() - start

# Run Benchmark
t_pandas = pandas_benchmark()
t_dm = datamanager_benchmark()

print(f"\n--- Results for {len(test_timestamps)} unique lookups ---")
print(f"Pandas .loc total:       {t_pandas:.4f}s (Avg: {t_pandas/len(test_timestamps)*1000:.4f}ms per lookup)")
print(f"DataManager total:       {t_dm:.4f}s (Avg: {t_dm/len(test_timestamps)*1000:.4f}ms per lookup)")
print(f"Speedup:                {t_pandas / t_dm:.1f}x faster")
