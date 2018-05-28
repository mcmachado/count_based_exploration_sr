import numpy as np

class CircularBuffer:
    def __init__(self, maxlen, shape, dtype, fetch_wrap=True):
        self._fetch_wrap = fetch_wrap

        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.empty((maxlen,) + shape, dtype=dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if np.abs(idx) >= self.length:
                raise KeyError("Index out of bounds")
        elif isinstance(idx, np.ndarray):
            if (np.abs(idx) >= self.length).any():
                raise KeyError("Index out of bounds")
        if self._fetch_wrap:
            return self.data.take(self.start + idx, mode='wrap', axis=0)
        return self.data.take(idx, axis=0)

    def __array__(self):
        if self._fetch_wrap:
            return self.data.take(np.arange(self.start, self.start + self.length), mode='wrap', axis=0)
        return self.data

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            raise RuntimeError()

        self.data[(self.start + self.length - 1) % self.maxlen] = v
