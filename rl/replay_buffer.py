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
