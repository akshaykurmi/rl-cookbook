import collections

import numpy as np

from rl.metrics.core import Metric


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
