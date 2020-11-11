import argparse
import os
import shutil

import gym
import tensorflow as tf
import tensorflow_probability as tfp

from rl.agents.vpg.vpg import VPG


class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=20, activation='relu', input_shape=input_shape)
        self.dense2 = tf.keras.layers.Dense(units=10, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=output_dim, activation='linear')

    def get_config(self):
        pass

    def call(self, inputs, **kwarg):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def step_and_sample(self, observations):
        logits = self.call(observations)
        return tfp.distributions.Categorical(logits=logits).sample()

    def calculate_log_probs(self, observations, actions):
        logits = self.call(observations)
        return tfp.distributions.Categorical(logits=logits).log_prob(actions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'simulate'], required=True, help='Train or simulate the agent?')
    args = parser.parse_args()

    env = gym.make('CartPole-v0')
    model = PolicyNetwork(env.observation_space.shape, env.action_space.n)
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'ckpt', 'vpg')
    log_dir = os.path.join(os.path.dirname(__file__), 'log', 'vpg')
    if args.mode == 'train':
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    agent = VPG(
        env=env,
        model=model,
        lr=1e-3,
        epochs=150,
        episodes_per_epoch=64,
        max_episode_length=250,
        ckpt_epochs=10,
        log_epochs=1,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir
    )

    if args.mode == 'train':
        agent.train()
    if args.mode == 'simulate':
        agent.simulate()
