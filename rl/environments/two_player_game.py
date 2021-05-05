from abc import ABC, abstractmethod


class TwoPlayerGame(ABC):
    # Set these in subclasses
    metadata = {'render.modes': []}
    action_space = None
    observation_space = None

    @abstractmethod
    def reset(self, canonical=True):
        raise NotImplementedError

    @abstractmethod
    def step(self, action, canonical=True):
        raise NotImplementedError

    @abstractmethod
    def valid_actions(self, canonical=True):
        raise NotImplementedError

    @abstractmethod
    def observation(self, canonical=True):
        raise NotImplementedError

    @abstractmethod
    def score(self, canonical=True):
        raise NotImplementedError

    @abstractmethod
    def is_over(self):
        raise NotImplementedError

    @abstractmethod
    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

    def __str__(self):
        return f'<{type(self).__name__} instance>'
