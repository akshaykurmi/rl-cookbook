import collections
from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    def __init__(self, buffer_size, name):
        self.buffer_size = buffer_size
        self.name = name

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def record(self, transition):
        pass

    @abstractmethod
    def compute(self):
        pass


class AverageReturn(Metric):
    def __init__(self, buffer_size, name='average_return'):
        super().__init__(buffer_size, name)
        self.reward = collections.deque(maxlen=self.buffer_size)
        self.done = collections.deque(maxlen=self.buffer_size)

    def reset(self):
        self.reward.clear()
        self.done.clear()

    def record(self, transition):
        self.reward.append(transition['reward'])
        self.done.append(transition['done'])

    def compute(self):
        returns = [0.0]
        for i, (r, d) in enumerate(zip(self.reward, self.done)):
            returns[-1] += r
            if d and i != len(self.reward) - 1:
                returns.append(0.0)
        return np.mean(returns)


class AverageEpisodeLength(Metric):
    def __init__(self, buffer_size, name='average_episode_length'):
        super().__init__(buffer_size, name)
        self.done = collections.deque(maxlen=self.buffer_size)

    def reset(self):
        self.done.clear()

    def record(self, transition):
        self.done.append(transition['done'])

    def compute(self):
        lengths = [0.0]
        for i, d in enumerate(self.done):
            lengths[-1] += 1
            if d and i != len(self.done) - 1:
                lengths.append(0.0)
        return np.mean(lengths)
