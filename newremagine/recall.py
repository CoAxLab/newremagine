# AUTOGENERATED! DO NOT EDIT! File to edit: recall.ipynb (unless otherwise specified).

__all__ = ['Recall', 'PriorityRecall']

# Cell
import random
import sys
import numpy as np

# Cell
class Recall(object):
    def __init__(self, capacity=1e5):
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0

    def encode(self, x):
        """Saves a memory x."""
        # Pad out
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # Remember
        self.memory[self.position] = x
        self.position = int((self.position + 1) % self.capacity)

    def __call__(self, *args):
        self.encode(*args)

    def sample(self, n):
        return random.sample(self.memory, n)[0]

    def __len__(self):
        return len(self.memory)

# Cell
class PriorityRecall(object):
    def __init__(self, capacity=1e5):
        self.capacity = int(capacity)
        self.memory = []
        self.priority = []
        self.probs = None
        self.position = 0
        # Prevents div by 0 problems
        self.eps = sys.float_info.min

    def encode(self, weight, x):
        """Saves a weight and a memory x."""
        # Sanity
        weight = float(weight)
        if np.isclose(weight, 0.0):
            raise ValueError("w must be > 0")
        # Pad out
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priority.append(None)
        # Remember
        self.priority[self.position] = weight
        self.memory[self.position] = x
        self.position = int((self.position + 1) % self.capacity)

    def __call__(self, weight, *args):
        self.encode(weight, *args)

    def sample(self, n):
        # Est probs from priority weights
        summed = sum(self.weight) + self.eps
        self.probs = [w / summed for w in self.priority]
        # Wieghted sample
        return np.random.choice(self.memory, size=n, p=self.probs).tolist()

    def __len__(self):
        return len(self.memory)