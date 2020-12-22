import numpy as np


class Episode:
    def __init__(self, gamma, lambda_):
        self.gamma = gamma
        self.lambda_ = lambda_

        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.is_done = False

    @property
    def empty(self):
        return len(self.observations) == 0

    @property
    def length(self):
        return len(self.rewards)

    @property
    def advantages(self):
        deltas, coeffs, advantages = [], [], []
        for t in range(len(self.rewards)):
            deltas.append(self.rewards[t] + self.gamma * self.values[t + 1] - self.values[t])
            coeffs.append(np.power(self.gamma * self.lambda_, t))
        for t in range(len(self.rewards)):
            advantages.append(np.sum([coeffs[i] * deltas[t + i] for i in range(len(self.rewards) - t)]))
        return advantages

    @property
    def rewards_to_go(self):
        coeffs = [np.power(self.gamma, t) for t in range(len(self.rewards))]
        rewards_to_go = []
        for t in range(len(self.rewards)):
            rewards_to_go.append(np.sum([coeffs[i] * self.rewards[t + i] for i in range(len(self.rewards) - t)]))
        return rewards_to_go

    @property
    def total_rewards(self):
        episode_reward = np.sum(self.rewards)
        return [episode_reward] * len(self.rewards)


class ReplayBuffer:
    def __init__(self, gamma=1.0, lambda_=1.0):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.episodes = []

    def purge(self):
        self.episodes = []

    def store_transition(self, observation, action, reward, value=None):
        if len(self.episodes) == 0 or self.episodes[-1].is_done:
            self.episodes.append(Episode(self.gamma, self.lambda_))
        episode = self.episodes[-1]
        episode.observations.append(observation)
        episode.actions.append(action)
        episode.rewards.append(reward)
        if value is not None:
            episode.values.append(value)

    def terminate_episode(self, observation, value=None):
        episode = self.episodes[-1]
        episode.is_done = True
        episode.observations.append(observation)
        if value is not None:
            episode.values.append(value)

    def get(self, fields):
        if not all([e.is_done for e in self.episodes]):
            raise RuntimeError('Cannot get. Some episodes have not terminated.')

        data = {}

        if 'observations' in fields:
            data['observations'] = []
            for e in self.episodes:
                data['observations'].extend(e.observations[:-1])  # remove terminal observation

        if 'actions' in fields:
            data['actions'] = []
            for e in self.episodes:
                data['actions'].extend(e.actions)

        if 'rewards' in fields:
            data['rewards'] = []
            for e in self.episodes:
                data['rewards'].extend(e.rewards)

        if 'values' in fields:
            data['values'] = []
            for e in self.episodes:
                data['values'].extend(e.values[:-1])  # remove terminal value

        if 'advantages' in fields:
            data['advantages'] = []
            for e in self.episodes:
                data['advantages'].extend(e.advantages)

        if 'rewards_to_go' in fields:
            data['rewards_to_go'] = []
            for e in self.episodes:
                data['rewards_to_go'].extend(e.rewards_to_go)

        if 'total_rewards' in fields:
            data['total_rewards'] = []
            for e in self.episodes:
                data['total_rewards'].extend(e.total_rewards)

        if 'episode_lengths' in fields:
            data['episode_lengths'] = [e.length for e in self.episodes]

        return data


class UniformReplayBuffer:
    def __init__(self, buffer_size, store_fields, compute_fields, gamma=None, lambda_=None):
        self.max_size = buffer_size
        self.store_fields = store_fields
        self.compute_fields = compute_fields
        self.gamma = gamma
        self.lambda_ = lambda_
        self.buffers, self.store_head, self.compute_head, self.size = None, None, None, None
        self.purge()

    def purge(self):
        self.buffers = {name: [None] * self.max_size for name in self.store_fields + self.compute_fields}
        self.store_head, self.compute_head, self.size = 0, 0, 0

    def store_transition(self, transition):
        if not all([k in self.store_fields for k in transition.keys()]):
            raise ValueError('Invalid transition')
        for name, data in transition.items():
            self.buffers[name][self.store_head] = data
        self.store_head = self._increment(self.store_head)
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        self._compute()
        indexes = np.random.randint(0, self.size, size=batch_size)
        batch = {name: [] for name in self.buffers.keys()}
        for i in indexes:
            for name in self.buffers.keys():
                batch[name].append(self.buffers[name][i])
        return batch

    def fetch_all(self):
        self._compute()
        return {name: buffer[name][:self.size]
                for name, buffer in self.buffers.items()}

    def _compute(self):
        if self.compute_head == self.store_head:
            return

        while True:
            compute_tail = self.compute_head
            while self.buffers['done'][compute_tail] is False:
                next_tail = self._increment(compute_tail)
                if next_tail == self.store_head:
                    break
                compute_tail = next_tail

            if 'advantage' in self.compute_fields:
                # TODO: calculate advantage with rewards to go?
                deltas, coeffs = [], []
                for t, buf_i in enumerate(self._crange(self.compute_head, compute_tail)):
                    deltas.append(self.buffers['reward'][buf_i] +
                                  self.gamma * self.buffers['value_next'][buf_i] -
                                  self.buffers['value'][buf_i])
                    coeffs.append(np.power(self.gamma * self.lambda_, t))
                for t, buf_i in enumerate(self._crange(self.compute_head, compute_tail)):
                    self.buffers['advantage'][buf_i] = np.sum([
                        coeffs[i] * deltas[t + i]
                        for i in range(len(deltas) - t)
                    ])

            if 'reward_to_go' in self.compute_fields:
                rewards, coeffs = [], []
                for t, buf_i in enumerate(self._crange(self.compute_head, compute_tail)):
                    rewards.append(self.buffers['reward'][buf_i])
                    coeffs.append(np.power(self.gamma, t))
                for t, buf_i in enumerate(self._crange(self.compute_head, compute_tail)):
                    self.buffers['reward_to_go'][buf_i] = np.sum([
                        coeffs[i] * rewards[t + i]
                        for i in range(len(rewards) - t)
                    ])

            if 'episode_return' in self.compute_fields:
                episode_return = 0.0
                for buf_i in self._crange(self.compute_head, compute_tail):
                    episode_return += self.buffers['reward'][buf_i]
                for buf_i in self._crange(self.compute_head, compute_tail):
                    self.buffers['episode_return'][buf_i] = episode_return

            if self._increment(compute_tail) == self.store_head:
                if self.buffers['done'][compute_tail] is True:
                    self.compute_head = self.store_head
                break
            self.compute_head = self._increment(compute_tail)

    def _increment(self, ptr):
        return (ptr + 1) % self.max_size

    def _decrement(self, ptr):
        return ptr - 1 if ptr > 0 else self.max_size - 1

    def _crange(self, start, end):
        while start != self._increment(end):
            yield start
            start = self._increment(start)
