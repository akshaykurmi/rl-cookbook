from enum import Enum

import gym
import numpy as np
import tensorflow as tf

from rl.environments.two_player_game import TwoPlayerGame


class TicTacToe(TwoPlayerGame):
    GameStatus = Enum('GameStatus', 'X_WON, O_WON, DRAW, IN_PROGRESS')
    Players = Enum('Players', {'X': 1, 'O': -1})

    metadata = {'render.modes': ['human']}
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)
    action_space = gym.spaces.Discrete(9)

    def __init__(self):
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.turn = TicTacToe.Players.X

    def reset(self):
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.turn = TicTacToe.Players.X

    def step(self, action):
        assert action in self.valid_actions()
        col, row = action % 3, action // 3
        self.state[row, col] = self.turn.value
        self.turn = TicTacToe.Players(-self.turn.value)

    def valid_actions(self):
        actions = np.argwhere(self.state == 0)
        actions = actions[:, 0] * 3 + actions[:, 1]
        return actions

    def observation(self, canonical=True):
        if canonical and self.turn == TicTacToe.Players.O:
            return self.state * -1
        return self.state.copy()

    def score(self):
        status = self._game_status(self.state)
        if status in {TicTacToe.GameStatus.DRAW, TicTacToe.GameStatus.IN_PROGRESS}:
            return 0
        return {
            TicTacToe.GameStatus.X_WON: 1,
            TicTacToe.GameStatus.O_WON: -1,
        }[status]

    def is_over(self):
        status = self._game_status(self.state)
        return status != TicTacToe.GameStatus.IN_PROGRESS

    def render(self, mode='human'):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        status = self._game_status(self.state)
        result = {
            TicTacToe.GameStatus.IN_PROGRESS: f'TURN : {symbols[self.turn.value]}',
            TicTacToe.GameStatus.DRAW: f'Draw!',
            TicTacToe.GameStatus.X_WON: f'X Won!',
            TicTacToe.GameStatus.O_WON: f'O Won!',
        }[status]
        result = result.center(13)
        result += '\n┌' + ('───┬' * 3)[:-1] + '┐\n'
        for i, row in enumerate(self.state):
            for v in row:
                result += f'| {symbols[v]} '
            result += '|\n'
            if i < 2:
                result += '├' + ('───┼' * 3)[:-1] + '┤\n'
        result += '└' + ('───┴' * 3)[:-1] + '┘'
        return result

    @staticmethod
    def _game_status(state):
        def unique_elements_along_positions():
            yield np.unique(np.diagonal(state))
            yield np.unique(np.diagonal(np.fliplr(state)))
            for i in range(3):
                yield np.unique(state[:, i])
                yield np.unique(state[i, :])

        for elements in unique_elements_along_positions():
            if elements.size == 1:
                if elements[0] == 1:
                    return TicTacToe.GameStatus.X_WON
                if elements[0] == -1:
                    return TicTacToe.GameStatus.O_WON
        if np.count_nonzero(state) == 9:
            return TicTacToe.GameStatus.DRAW
        return TicTacToe.GameStatus.IN_PROGRESS


class PolicyAndValueFunctionNetwork(tf.keras.Model):
    def __init__(self, observation_shape, n_actions, l2):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=observation_shape)
        self.dense1 = tf.keras.layers.Dense(32, 'relu', kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.dense2 = tf.keras.layers.Dense(32, 'relu', kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.dense_pi = tf.keras.layers.Dense(16, 'relu', kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.dense_v = tf.keras.layers.Dense(16, 'relu', kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.pi = tf.keras.layers.Dense(n_actions, 'softmax', kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.v = tf.keras.layers.Dense(1, 'tanh', kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.call(tf.ones((1, *observation_shape)))

    def get_config(self):
        super().get_config()

    def call(self, observations, **kwargs):
        x = self.flatten(observations)
        x = self.dense1(x)
        x = self.dense2(x)
        pi = self.dense_pi(x)
        v = self.dense_v(x)
        pi = self.pi(pi)
        v = self.v(v)
        return pi, v
