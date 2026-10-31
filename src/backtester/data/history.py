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
            raise IndexError(
                f"BarHistory uses 0 (newest) and negative indices only. Got positive index: {idx}"
            )
        if self.count == 0:
            raise IndexError("BarHistory is empty.")
        if idx <= -self.count:
            raise IndexError(
                f"Index {idx} is out of range for history with {self.count} bars."
            )
        physical = self._get_physical_idx(idx)
        return self.data[physical]

    def _get_physical_idx(self, idx):
        return (self.head + idx) % self.size

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get(key)
        elif isinstance(key, slice):
            if self.count == 0:
                return np.empty(0, dtype=object)
            start, stop, step = key.indices(self.count)
            # 0 is latest (head), 1 is 1 bar before (head - 1), etc.
            idxs = (self.head - np.arange(start, stop, step)) % self.size
            return self.data[idxs]

    def __setitem__(self, idx, bar):
        physical = self._get_physical_idx(idx)
        self.data[physical] = bar

    def chronological(self):
        """Returns all stored bars in chronological order (oldest to newest)."""
        if self.count == 0:
            return np.empty(0, dtype=object)
        idxs = (self.head - np.arange(self.count - 1, -1, -1)) % self.size
        return self.data[idxs]

    def __iter__(self):
        """Iterate over elements from newest to oldest"""
        for i in range(self.count):
            yield self.get(-i)
