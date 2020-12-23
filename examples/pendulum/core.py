import argparse
import os
import shutil
from time import sleep

import tensorflow as tf
import tensorflow_probability as tfp


class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_shape, output_dim, env_action_max, env_action_min):
        super().__init__()
        self.env_action_min = env_action_min
        self.env_action_max = env_action_max
        self.action_min, self.action_max = -1.0, 1.0

        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu', input_shape=input_shape)
        self.dense2 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=output_dim, activation='tanh')

        self.call(tf.ones((1, *input_shape)))

    def get_config(self):
        super().get_config()

    def call(self, observations, **kwargs):
        x = self.dense1(observations)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def sample(self, observations, noise=None, noise_clip=None, **kwargs):
        actions = self.call(observations)
        actions = (actions - self.action_min) / (self.action_max - self.action_min)
        actions = actions * (self.env_action_max - self.env_action_min) + self.env_action_min
        if noise:
            epsilon = tf.random.normal(actions.shape, mean=0.0, stddev=noise)
            if noise_clip:
                epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            actions += epsilon
        actions = tf.clip_by_value(actions, self.env_action_min, self.env_action_max)
        return actions


class QFunctionNetwork(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu', input_shape=input_shape)
        self.dense2 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=1, activation='linear')

        self.call(tf.ones((1, *input_shape)))

    def get_config(self):
        super().get_config()

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def compute(self, observations, actions):
        x = tf.concat([observations, actions], axis=-1)
        x = self.call(x)
        return tf.squeeze(x)


class PolicyNetworkSAC(tf.keras.Model):
    def __init__(self, input_shape, output_dim, env_action_max, env_action_min):
        super().__init__()
        self.env_action_min = env_action_min
        self.env_action_max = env_action_max

        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu', input_shape=input_shape)
        self.dense2 = tf.keras.layers.Dense(units=32, activation='relu')
        self.mean = tf.keras.layers.Dense(units=output_dim)
        self.log_std = tf.keras.layers.Dense(units=output_dim)

        self.call(tf.ones((1, *input_shape)))

    def get_config(self):
        super().get_config()

    def call(self, observations, **kwargs):
        x = self.dense1(observations)
        x = self.dense2(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = tf.clip_by_value(log_std, 2, -20)
        std = tf.exp(log_std)
        return mean, std

    def sample(self, observations, deterministic=False, return_entropy=True, **kwargs):
        mean, std = self.call(observations)
        distribution = tfp.distributions.Normal(loc=mean, scale=std)
        unscaled_actions = distribution.mean() if deterministic else distribution.sample()
        actions = tf.tanh(unscaled_actions)
        actions = (actions + 1) / 2  # scale to [0,1]
        actions = actions * (self.env_action_max - self.env_action_min) + self.env_action_min
        actions = tf.clip_by_value(actions, self.env_action_min, self.env_action_max)

        if return_entropy:
            entropy = distribution.log_prob(unscaled_actions)
            entropy = tf.reduce_sum(entropy, axis=-1)
            entropy -= tf.reduce_sum(2 * (tf.math.log(2.0) -
                                          unscaled_actions -
                                          tf.math.softplus(-2 * unscaled_actions)
                                          ), axis=1)  # rescale due to tanh squashing
            return actions, entropy
        return actions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True, help='Train or evaluate the agent?')
    return parser.parse_args()


def get_output_dirs(name, overwrite):
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'ckpt', name)
    log_dir = os.path.join(os.path.dirname(__file__), 'log', name)
    if overwrite:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        shutil.rmtree(log_dir, ignore_errors=True)
    return ckpt_dir, log_dir


def evaluate_policy(env, policy):
    observation = env.reset()
    env.render()
    done = False
    while not done:
        sleep(0.005)
        action = policy.sample(observation.reshape(1, -1), deterministic=True, return_entropy=False).numpy()[0]
        observation, reward, done, info = env.step(action)
        env.render()
    env.close()
