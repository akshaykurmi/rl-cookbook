from abc import ABC, abstractmethod

import numpy as np

from rl.utils import RingBuffer


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
    def __init__(self, buffer_size=10, name='average_return'):
        super().__init__(buffer_size, name)
        self.returns = RingBuffer(self.buffer_size, (), np.float32)
        self.current_return = 0.0

    def reset(self):
        self.returns.purge()
        self.current_return = 0.0

    def record(self, transition):
        self.current_return += transition['reward']
        if transition['done']:
            self.returns.append(self.current_return)
            self.current_return = 0

    def compute(self):
        return np.mean(self.returns[:])


class AverageEpisodeLength(Metric):
    def __init__(self, buffer_size=10, name='average_episode_length'):
        super().__init__(buffer_size, name)
        self.episode_lengths = RingBuffer(self.buffer_size, (), np.float32)
        self.current_length = 0.0

    def reset(self):
        self.episode_lengths.purge()
        self.current_length = 0.0

    def record(self, transition):
        self.current_length += 1
        if transition['done']:
            self.episode_lengths.append(self.current_length)
            self.current_length = 0.0

    def compute(self):
        return np.mean(self.episode_lengths[:])
