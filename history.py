import numpy as np

class BarHistory:
    def __init__(self, size=120):
        self.size = size
        self.data = np.empty(size, dtype=object)
        self.head = -1
        self.count = 0

    def add(self, bar):
        self.head = (self.head + 1) % self.size
        self.data[self.head] = bar
        self.count = min(self.count + 1, self.size)

    def get(self, idx):
        if idx > 0:
            raise IndexError("Index out of bounds")
        
        physical = self._get_physical_idx(idx)
        return self.data[physical]

    def _get_physical_idx(self, idx):
        return (self.head + idx) % self.size

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get(key)
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.count)
            #idxs = [(self.head + i) % self.size for i in range(start, stop, step)]
            idxs = (self.head + np.arange(start, stop, step)) % self.size #faster than list comprehension
            return self.data[idxs]

    def __setitem__(self, idx, bar):
        physical = self._get_physical_idx(idx)
        self.data[physical] = bar

    def __iter__(self):
        """Iterate over elements from newest to oldest"""
        for i in range(self.count):
            yield self.get(-i)
#---------------------------------------------
import sys
import timeit
from pathlib import Path
import numpy as np

def benchmark_bar_history():
    """Run performance benchmarks for BarHistory operations"""
    
    # Test configurations
    HISTORY_SIZE = 100_000
    NUM_OPERATIONS = 100_000
    
    # Setup code for timeit
    setup = f"""
from __main__ import BarHistory
import numpy as np
history = BarHistory(size={HISTORY_SIZE})
for i in range({HISTORY_SIZE}):
    history.add(i)
"""

    # Test cases with their statements
    tests = {
        "append": "history.add(42)",
        "get_current": "history[0]",
        "get_previous": "history[-1]",
        "get_oldest": f"history[-{HISTORY_SIZE}]",
        "slice_last_10": "history[-10:]",
        "slice_last_100": "history[-100:]",
        "slice_last_1000": "history[-1000:]",
        "slice_with_step_100": "history[-100::2]",
        "slice_with_step_1000": "history[-1000::2]",
        #"slice_with_step_10000": "history[-10000::2]",
        "set_value": "history[-1] = 42",
    }

    results = {}
    for name, stmt in tests.items():
        # Run each test case
        time_taken = timeit.timeit(stmt=stmt, setup=setup, number=NUM_OPERATIONS)
        ops_per_sec = NUM_OPERATIONS / time_taken
        results[name] = {
            "total_time": time_taken,
            "ops_per_second": ops_per_sec,
            "avg_time_us": (time_taken / NUM_OPERATIONS) * 1e6  # Convert to microseconds
        }

    # Print results in a formatted table
    print("\nBarHistory Performance Benchmarks")
    print(f"History Size: {HISTORY_SIZE}, Operations per test: {NUM_OPERATIONS:,}")
    print("\n{:<20} {:>12} {:>15} {:>12}".format(
        "Operation", "Total Time", "Ops/sec", "Avg µs/op"
    ))
    print("-" * 60)
    
    for name, metrics in results.items():
        print("{:<20} {:>12.3f} {:>15,.0f} {:>12.2f}".format(
            name,
            metrics["total_time"],
            metrics["ops_per_second"],
            metrics["avg_time_us"]
        ))

def memory_usage_test():
    """Test memory usage with large histories"""
    from memory_profiler import profile
    
    @profile
    def create_and_fill_large_history():
        history = BarHistory(size=100000)
        for i in range(200000):  # Overflow the history twice
            history.add(np.random.random(100))  # Add large arrays as bars
        return history

    create_and_fill_large_history()

if __name__ == "__main__":
    # history = BarHistory(size=5)
    # for i in range(5):
    #     history.add(i)
    
    # print(history.data)
    # history[1] = 5
    # print(history[0])  # bar_0
    # print(history[-2])  # bar_1
    # print(history[-1])  # bar_4
    # print(history[-12:])  # [bar_0, bar_1, bar_2]
    # # print(history[1:4])  # [bar_1, bar_2, bar_3]
    # # print(history[::2])  # [bar_0, bar_2, bar_4]
    # print(history.data)
    benchmark_bar_history()
    # memory_usage_test()