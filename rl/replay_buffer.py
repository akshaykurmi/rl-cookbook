import collections
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import tensorflow as tf

ReplayField = collections.namedtuple(
    'ReplayField',
    ['name', 'dtype', 'shape'],
    defaults=[np.float32, ()]
)


class ReplayBuffer(ABC):
    def __init__(self, buffer_size, store_fields, compute_fields, gamma=1.0, lambda_=1.0):
        self.buffer_size = buffer_size
        self.store_fields = store_fields
        self.compute_fields = compute_fields
        self.gamma = gamma
        self.lambda_ = lambda_

        self.compute_config = {
            'advantage': {
                'func': self._compute_advantage,
                'dependencies': {'done', 'reward', 'value', 'value_next'}
            },
            'reward_to_go': {
                'func': self._compute_reward_to_go,
                'dependencies': {'done', 'reward'}
            },
            'episode_return': {
                'func': self._compute_episode_return,
                'dependencies': {'done', 'reward'}
            },
            'episode_length': {
                'func': self._compute_episode_length,
                'dependencies': {'done'}
            },
        }
        store_field_names = {f.name for f in self.store_fields}
        for f in self.compute_fields:
            dependencies = self.compute_config[f.name]['dependencies']
            if not dependencies.issubset(store_field_names):
                raise ValueError(f'Compute field {f.name} requires store fields {dependencies}')

        self.store_buffers, self.compute_buffers = None, None
        self.store_head, self.compute_head, self.current_size = None, None, None
        self.purge()

    @abstractmethod
    def as_dataset(self, *args, **kwargs):
        self._compute()

    def purge(self):
        def create_empty_buffers(fields):
            return {f.name: np.empty((self.buffer_size, *f.shape), dtype=f.dtype)
                    for f in fields}

        if self.store_buffers is None and self.compute_buffers is None:
            self.store_buffers = create_empty_buffers(self.store_fields)
            self.compute_buffers = create_empty_buffers(self.compute_fields)
        self.store_head, self.compute_head, self.current_size = 0, 0, 0

    def store_transition(self, transition):
        for f in self.store_fields:
            self.store_buffers[f.name][self.store_head] = transition[f.name]
        self.store_head = self._increment(self.store_head)
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def _compute(self):
        if self.compute_head == self.store_head:
            return

        indices = self._circular_indices(self.compute_head, self._decrement(self.store_head))
        tail_indices = indices[self.store_buffers['done'][indices]]
        if self.store_buffers['done'][indices[-1]] is False:  # Add the last index which might not be done
            tail_indices = np.concatenate((tail_indices, [indices[-1]]))

        for i, compute_tail in enumerate(tail_indices):
            for f in self.compute_fields:
                self.compute_config[f.name]['func'](compute_tail)
            # If the last index is not done, do not move the compute_head
            if i < len(tail_indices) - 1 or self.store_buffers['done'][compute_tail]:
                self.compute_head = self._increment(compute_tail)

    def _compute_advantage(self, compute_tail):
        indices = self._circular_indices(self.compute_head, compute_tail)
        rewards = self.store_buffers['reward'][indices]
        values = self.store_buffers['value'][indices]
        value_nexts = self.store_buffers['value_next'][indices]
        deltas = rewards + self.gamma * value_nexts - values
        self.compute_buffers['advantage'][indices] = self._discounted_cumsum(deltas, self.gamma * self.lambda_)

    def _compute_reward_to_go(self, compute_tail):
        indices = self._circular_indices(self.compute_head, compute_tail)
        rewards = self.store_buffers['reward'][indices]
        self.compute_buffers['reward_to_go'][indices] = self._discounted_cumsum(rewards, self.gamma)

    def _compute_episode_return(self, compute_tail):
        indices = self._circular_indices(self.compute_head, compute_tail)
        episode_return = np.sum(self.store_buffers['reward'][indices])
        self.compute_buffers['episode_return'][indices] = episode_return

    def _compute_episode_length(self, compute_tail):
        length = self._circular_length(self.compute_head, compute_tail)
        indices = self._circular_indices(self.compute_head, compute_tail)
        self.compute_buffers['episode_length'][indices] = length

    def _circular_length(self, head, tail):
        length = tail - head + 1
        if head > tail:
            length += self.buffer_size
        return length

    def _circular_indices(self, head, tail):
        length = self._circular_length(head, tail)
        return np.arange(head, head + length) % self.buffer_size

    @staticmethod
    def _discounted_cumsum(values, discount):
        """
        Example:
        values = [1,2,3], discount = 0.9
        returns = [1 * 0.9^0 + 2 * 0.9^1 + 3 * 0.9^3,
                   2 * 0.9^0 + 3 * 0.9^1,
                   3 * 0.9^0]
        """
        return sp.signal.lfilter([1], [1, float(-discount)], values[::-1], axis=0)[::-1]

    def _increment(self, ptr, n=1):
        n = n % self.buffer_size
        return (ptr + n) % self.buffer_size

    def _decrement(self, ptr, n=1):
        n = n % self.buffer_size
        ptr = ptr - n
        if ptr < 0:
            ptr += self.buffer_size
        return ptr


class DeterministicReplayBuffer(ReplayBuffer):
    def as_dataset(self, num_batches=1, batch_size=None):
        super().as_dataset()
        batch_size = batch_size or self.current_size

        def data_generator():
            nonlocal batch_size
            fetch_head = self.store_head if self.current_size == self.buffer_size else 0
            for _ in range(num_batches * batch_size):
                store_data = {k: buf[fetch_head] for k, buf in self.store_buffers.items()}
                compute_data = {k: buf[fetch_head] for k, buf in self.compute_buffers.items()}
                yield {**store_data, **compute_data}
                fetch_head = self._increment(fetch_head)

        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_types={f.name: tf.as_dtype(f.dtype) for f in self.store_fields + self.compute_fields},
            output_shapes={f.name: f.shape for f in self.store_fields + self.compute_fields}
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


class UniformReplayBuffer(ReplayBuffer):
    def as_dataset(self, num_batches=1, batch_size=None):
        super().as_dataset()
        batch_size = batch_size or self.current_size

        def data_generator():
            nonlocal batch_size
            for _ in range(num_batches * batch_size):
                idx = np.random.randint(0, self.current_size)
                store_data = {k: buf[idx] for k, buf in self.store_buffers.items()}
                compute_data = {k: buf[idx] for k, buf in self.compute_buffers.items()}
                yield {**store_data, **compute_data}

        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_types={f.name: tf.as_dtype(f.dtype) for f in self.store_fields + self.compute_fields},
            output_shapes={f.name: f.shape for f in self.store_fields + self.compute_fields}
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
