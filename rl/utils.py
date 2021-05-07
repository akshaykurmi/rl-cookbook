import numpy as np
import scipy.signal
import tensorflow as tf


class RingBuffer:
    def __init__(self, buffer_size, shape, dtype):
        self.buffer_size = buffer_size
        self.buffer = np.empty((self.buffer_size, *shape), dtype=dtype)
        self.head, self.tail = 0, -1

    def __len__(self):
        return self._circular_length(self.head, self.tail)

    def __getitem__(self, key):
        if isinstance(key, int):
            index = self._translate_index(key)
            return self.buffer[index]
        if isinstance(key, slice):
            indices = self._translate_slice(key)
            return self.buffer[indices]
        raise IndexError('Indices must be an integer or a slice')

    def __setitem__(self, key, value):
        if isinstance(key, int):
            index = self._translate_index(key)
            self.buffer[index] = value
        elif isinstance(key, slice):
            indices = self._translate_slice(key)
            self.buffer[indices] = value
        else:
            raise IndexError('Indices must be an integer or a slice')

    def purge(self):
        self.head, self.tail = 0, -1

    def append(self, value):
        new_tail = (self.tail + 1) % self.buffer_size
        if self.tail >= 0 and self.head == new_tail:
            self.head = (self.head + 1) % self.buffer_size
        self.tail = new_tail
        self.buffer[self.tail] = value

    def _translate_index(self, i):
        i = self._positivify_index(i)
        if not 0 <= i < len(self):
            raise IndexError(f'Index {i} is out of bounds')
        return (self.head + i) % self.buffer_size

    def _translate_slice(self, s):
        current_size = len(self)
        start = self._positivify_index(s.start) if s.start is not None else -1
        stop = self._positivify_index(s.stop) - 1 if s.stop is not None else current_size
        start = np.clip(start, -1, current_size)
        stop = np.clip(stop, -1, current_size)
        if stop < start or current_size == 0 or (start == stop == -1) or (start == stop == current_size):
            return []
        start = np.clip(start, 0, current_size - 1)
        stop = np.clip(stop, 0, current_size - 1)
        start = (self.head + start) % self.buffer_size
        stop = (self.head + stop) % self.buffer_size
        return self._circular_indices(start, stop, s.step)

    def _positivify_index(self, i):
        return i + len(self) if i < 0 else i

    def _circular_length(self, head, tail):
        if tail == -1:
            return 0
        return tail - head + 1 + (self.buffer_size if head > tail else 0)

    def _circular_indices(self, head, tail, step=1):
        length = self._circular_length(head, tail)
        return np.arange(head, head + length, step) % self.buffer_size


class GradientAccumulator:
    def __init__(self):
        self._gradients = []
        self._steps = 0

    def add(self, gradients, steps=1):
        if not self._gradients:
            self._gradients.extend([tf.Variable(tf.zeros_like(gradient), trainable=False)
                                    for gradient in gradients])
        if len(gradients) != len(self._gradients):
            raise ValueError(f'Expected {len(self._gradients)} gradients, but got {len(gradients)}')
        for acc, gradient in zip(self._gradients, gradients):
            acc.assign_add(gradient, read_value=False)
        self._steps += steps

    def gradients(self):
        if not self._gradients:
            raise ValueError('No gradients have been accumulated yet')
        grads = list(gradient.value() / tf.cast(self._steps, tf.float32) for gradient in self._gradients)
        return grads


class MeanAccumulator:
    def __init__(self):
        self._loss = 0.0
        self._steps = 0

    def add(self, losses):
        self._loss += tf.reduce_sum(losses)
        self._steps += tf.size(losses)

    def value(self):
        return self._loss / tf.cast(self._steps, tf.float32)


@tf.function
def tf_standardize(x):
    x -= tf.reduce_mean(x)
    x /= tf.math.reduce_std(x) + 1e-10
    return x


def discounted_cumsum(values, discount):
    """
    Example:
    values = [1,2,3], discount = 0.9
    returns = [1 * 0.9^0 + 2 * 0.9^1 + 3 * 0.9^3,
               2 * 0.9^0 + 3 * 0.9^1,
               3 * 0.9^0]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], values[::-1], axis=0)[::-1]
