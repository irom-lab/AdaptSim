import numpy as np
from collections import deque
import torch
import itertools


class ReplayMemory(object):

    def __init__(self, capacity, seed):
        self.reset(capacity)
        self.capacity = capacity
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    def reset(self, capacity=None):
        if capacity is None:
            capacity = self.capacity
        self.memory = deque(maxlen=capacity)

    def set_array(self, memory):
        if len(memory) > self.capacity:
            memory = deque(
                itertools.islice(
                    memory,
                    len(memory) - self.capacity, len(memory)
                )
            )  # take the recent ones
        self.memory = memory

    def update(self, transition):
        self.memory.appendleft(transition)  # pop from right if full

    def sample(self, batch_size, recent_size=None):
        length = len(self.memory)
        if recent_size is not None:
            length = min(length, recent_size)
        indices = self.rng.integers(low=0, high=length, size=(batch_size,))
        return [self.memory[i] for i in indices], None  # dummy for nxt

    def save(self, path):
        """Use torch.save() to make sure compatible across devices."""
        torch.save({'deque': self.memory}, f=path)

    def __len__(self):
        return len(self.memory)


class ReplayMemoryMeta(ReplayMemory):

    def __init__(self, capacity, seed):
        super().__init__(capacity, seed)
        self.memory_meta = []

    def set_meta_array(self, memory):
        """Can be a deque"""
        self.memory_meta = memory

    def move_memory_to_meta(self, keep_meta_size=False):
        if keep_meta_size:
            # remove the same amount from end, and add from memory
            for item in reversed(self.memory):
                self.memory_meta.pop()
                self.memory_meta.appendleft(item)
        else:
            # Make meta larger, and then add
            self.memory_meta = deque(
                self.memory_meta, maxlen=len(self.memory_meta) + len(self)
            )
            for item in reversed(self.memory):
                self.memory_meta.appendleft(item)

        # reset
        self.reset()

    # Overwrites
    def sample(self, batch_size, online_weight=0.5):
        """weighted sampling from online and meta buffer"""
        length_online = len(self.memory)
        length_meta = len(self.memory_meta)
        batch_size_meta = min(
            length_meta, int((1-online_weight) * batch_size)
        )  # in case initially meta has no sample
        batch_size_online = batch_size - batch_size_meta
        indices_online = self.rng.integers(
            low=0, high=length_online, size=(batch_size_online,)
        )
        indices_meta = self.rng.integers(
            low=0, high=length_meta, size=(batch_size_meta,)
        )

        samples = [self.memory[i] for i in indices_online] + \
                  [self.memory_meta[i] for i in indices_meta]
        return samples, None  # dummy for nxt
