from enum import Enum

import gym
import numpy as np
import tensorflow as tf
from colr import color

from rl.environments.two_player_game import TwoPlayerGame


class Connect4(TwoPlayerGame):
    GameStatus = Enum('GameStatus', 'BLUE_WON, YELLOW_WON, DRAW, IN_PROGRESS')
    Players = Enum('Players', {'BLUE': 1, 'YELLOW': -1})

    metadata = {'render.modes': ['human']}
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(6, 7, 1), dtype=np.int8)
    action_space = gym.spaces.Discrete(7)

    _TOP_ROW = np.uint64(int('1000000_1000000_1000000_1000000_1000000_1000000_1000000', 2))
    _GRID_MASKS = [np.uint64(1) << np.uint64(i + 7 * j) for i in range(5, -1, -1) for j in range(7)]

    def __init__(self):
        self.state, self.turn = None, None
        self.reset()

    def reset(self):
        self.state = {
            Connect4.Players.BLUE: np.uint64(0),
            Connect4.Players.YELLOW: np.uint64(0),
            'height': np.arange(7, dtype=np.uint64) * 7
        }
        self.turn = Connect4.Players.BLUE

    def step(self, action):
        assert action in self.valid_actions()
        move = np.uint64(1) << self.state['height'][action]
        self.state[self.turn] ^= move
        self.state['height'][action] += 1
        self.turn = Connect4.Players(-self.turn.value)

    def valid_actions(self):
        actions = []
        for col in range(7):
            if (self._TOP_ROW & (np.uint64(1) << self.state['height'][col])) == 0:
                actions.append(col)
        return np.array(actions, dtype=np.int8)

    def observation(self, canonical=True):
        grid = self._bitboard_to_grid()
        if canonical:
            return grid[..., None] * self.turn.value
        return grid[..., None]

    def score(self):
        status = self._game_status()
        if status in {Connect4.GameStatus.DRAW, Connect4.GameStatus.IN_PROGRESS}:
            return 0
        return {
            Connect4.GameStatus.BLUE_WON: 1,
            Connect4.GameStatus.YELLOW_WON: -1,
        }[status]

    def is_over(self):
        return self._game_status() != Connect4.GameStatus.IN_PROGRESS

    def render(self, mode='human'):
        grid = self._bitboard_to_grid()
        symbols = {0: ' ', 1: color('●', fore='#8e86ff'), -1: color('●', fore='#cfff00')}
        turns = {1: 'BLUE', -1: 'YELLOW'}
        status = self._game_status()
        result = {
            Connect4.GameStatus.IN_PROGRESS: f'TURN : {turns[self.turn.value]}',
            Connect4.GameStatus.DRAW: f'Draw!',
            Connect4.GameStatus.BLUE_WON: f'BLUE Won!',
            Connect4.GameStatus.YELLOW_WON: f'YELLOW Won!',
        }[status]
        result = result.center(29)
        result += '\n┌' + ('───┬' * 7)[:-1] + '┐\n'
        for i, row in enumerate(grid):
            for v in row:
                result += f'| {symbols[v]} '
            result += '|\n'
            if i < 5:
                result += '├' + ('───┼' * 7)[:-1] + '┤\n'
        result += '└' + ('───┴' * 7)[:-1] + '┘\n'
        result += ''.join([f'  {i} ' for i in range(7)]) + '\n'
        return result

    def _game_status(self):
        def is_won(bitboard):
            if bitboard & (bitboard >> np.uint64(6)) & (bitboard >> np.uint64(12)) & (bitboard >> np.uint64(18)) != 0:
                return True  # diagonal left-right
            if bitboard & (bitboard >> np.uint64(8)) & (bitboard >> np.uint64(16)) & (bitboard >> np.uint64(24)) != 0:
                return True  # diagonal right-left
            if bitboard & (bitboard >> np.uint64(7)) & (bitboard >> np.uint64(14)) & (bitboard >> np.uint64(21)) != 0:
                return True  # horizontal
            if bitboard & (bitboard >> np.uint64(1)) & (bitboard >> np.uint64(2)) & (bitboard >> np.uint64(3)) != 0:
                return True  # vertical
            return False

        if is_won(self.state[Connect4.Players.BLUE]):
            return Connect4.GameStatus.BLUE_WON
        if is_won(self.state[Connect4.Players.YELLOW]):
            return Connect4.GameStatus.YELLOW_WON
        if len(self.valid_actions()) == 0:
            return Connect4.GameStatus.DRAW
        return Connect4.GameStatus.IN_PROGRESS

    def _bitboard_to_grid(self):
        def get_value_at_position(mask):
            if self.state[Connect4.Players.BLUE] & mask != 0:
                return Connect4.Players.BLUE.value
            elif self.state[Connect4.Players.YELLOW] & mask != 0:
                return Connect4.Players.YELLOW.value
            return 0

        grid = np.array([get_value_at_position(mask) for mask in self._GRID_MASKS], dtype=np.int8)
        return np.reshape(grid, (6, 7))


class PolicyAndValueFunctionNetwork(tf.keras.Model):
    def __init__(self, observation_shape, n_actions, l2):
        super().__init__()
        self.entry = tf.keras.layers.InputLayer(input_shape=observation_shape)
        self.conv1 = tf.keras.layers.Conv2D(16, (4, 4), activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.conv2 = tf.keras.layers.Conv2D(16, (2, 2), activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, 'relu', kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.dense_pi = tf.keras.layers.Dense(16, 'relu', kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.dense_v = tf.keras.layers.Dense(16, 'relu', kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.pi = tf.keras.layers.Dense(n_actions, 'softmax', kernel_regularizer=tf.keras.regularizers.L2(l2))
        self.v = tf.keras.layers.Dense(1, 'tanh', kernel_regularizer=tf.keras.regularizers.L2(l2))

    def get_config(self):
        super().get_config()

    def call(self, observations, **kwargs):
        x = self.entry(tf.cast(observations, tf.float32))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        pi = self.dense_pi(x)
        v = self.dense_v(x)
        pi = self.pi(pi)
        v = self.v(v)
        return pi, v
