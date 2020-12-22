import collections

import numpy as np

from rl.metrics.core import Metric


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
