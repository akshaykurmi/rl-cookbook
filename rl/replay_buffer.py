from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from rl.utils import RingBuffer, discounted_cumsum


@dataclass
class ReplayField:
    name: str
    dtype: np.dtype = np.float32
    shape: tuple = ()


@dataclass
class ComputeField(ABC, ReplayField):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Advantage(ComputeField):
    def __init__(self, gamma=1.0, lambda_=1.0, reward_field='reward', value_field='value',
                 value_next_field='value_next', name='advantage'):
        super().__init__(name=name)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.reward_field = reward_field
        self.value_field = value_field
        self.value_next_field = value_next_field

    def __call__(self, buffers, head, tail, *args, **kwargs):
        rewards = buffers[self.reward_field][head:tail]
        values = buffers[self.value_field][head:tail]
        value_nexts = buffers[self.value_next_field][head:tail]
        deltas = rewards + self.gamma * value_nexts - values
        buffers[self.name][head:tail] = discounted_cumsum(deltas, self.gamma * self.lambda_)


class RewardToGo(ComputeField):
    def __init__(self, gamma=1.0, reward_field='reward', name='reward_to_go'):
        super().__init__(name=name)
        self.gamma = gamma
        self.reward_field = reward_field

    def __call__(self, buffers, head, tail, *args, **kwargs):
        rewards = buffers[self.reward_field][head:tail]
        buffers[self.name][head:tail] = discounted_cumsum(rewards, self.gamma)


class EpisodeReturn(ComputeField):
    def __init__(self, reward_field='reward', name='episode_return'):
        super().__init__(name=name)
        self.reward_field = reward_field

    def __call__(self, buffers, head, tail, *args, **kwargs):
        episode_return = np.sum(buffers['reward'][head:tail])
        buffers[self.name][head:tail] = episode_return


class EpisodeLength(ComputeField):
    def __init__(self, name='episode_length'):
        super().__init__(name=name)

    def __call__(self, buffers, head, tail, *args, **kwargs):
        buffers[self.name][head:tail] = tail - head + 1


class ReplayBuffer(ABC):
    def __init__(self, buffer_size, store_fields, compute_fields):
        self.buffer_size = buffer_size
        self.store_fields = store_fields
        self.compute_fields = compute_fields
        self.buffers = {f.name: RingBuffer(self.buffer_size, f.shape, f.dtype)
                        for f in self.store_fields + self.compute_fields}
        self.current_size, self.compute_head = 0, 0

    @abstractmethod
    def as_dataset(self, *args, **kwargs):
        self._compute()

    def purge(self):
        for buffer in self.buffers.values():
            buffer.purge()
        self.current_size, self.compute_head = 0, 0

    def store_transition(self, transition):
        for f in self.store_fields:
            self.buffers[f.name].append(transition[f.name])
        for f in self.compute_fields:
            self.buffers[f.name].append(np.zeros(f.shape))
        if self.current_size == self.buffer_size:
            self.compute_head = max(self.compute_head - 1, 0)
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def _compute(self):
        if self.compute_head == self.current_size:
            return

        indices = np.arange(self.compute_head, self.current_size)
        tail_indices = 1 + indices[self.buffers['done'][self.compute_head:]]
        # If the last index is not done, add it to the tail_indices
        if self.buffers['done'][-1] is False:
            tail_indices = np.concatenate((tail_indices, [self.current_size]))

        for compute_tail in tail_indices:
            for f in self.compute_fields:
                f(self.buffers, self.compute_head, compute_tail)
            # If the last index is not done, do not move the compute_head
            if compute_tail < self.current_size or self.buffers['done'][-1]:
                self.compute_head = compute_tail


class OnePassReplayBuffer(ReplayBuffer):
    def as_dataset(self, batch_size=32):
        def data_generator():
            for i in np.random.default_rng().choice(self.current_size, size=self.current_size, replace=False):
                yield {f.name: self.buffers[f.name][i.item()]
                       for f in self.store_fields + self.compute_fields}

        super().as_dataset()
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_types={f.name: tf.as_dtype(f.dtype) for f in self.store_fields + self.compute_fields},
            output_shapes={f.name: f.shape for f in self.store_fields + self.compute_fields}
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


class UniformReplayBuffer(ReplayBuffer):
    def as_dataset(self, batch_size=32):
        def data_generator():
            i = np.random.randint(self.current_size)
            yield {k: buf[i] for k, buf in self.buffers.items()}

        super().as_dataset()
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_types={f.name: tf.as_dtype(f.dtype) for f in self.store_fields + self.compute_fields},
            output_shapes={f.name: f.shape for f in self.store_fields + self.compute_fields}
        )
        dataset = dataset.repeat(-1)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
