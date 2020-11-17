from time import sleep

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class VPGGAE:
    def __init__(self, env, policy_model, vf_model, lr_policy, lr_vf, gamma, lambda_, epochs, episodes_per_epoch,
                 max_episode_length, vf_update_iterations, ckpt_epochs, log_epochs, ckpt_dir, log_dir):
        self.env = env
        self.policy_model = policy_model
        self.vf_model = vf_model
        self.lr_policy = lr_policy
        self.lr_vf = lr_vf
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.max_episode_length = max_episode_length
        self.vf_update_iterations = vf_update_iterations
        self.ckpt_epochs = ckpt_epochs
        self.log_epochs = log_epochs
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.epochs_done = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_policy)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_vf)
        self.ckpt = tf.train.Checkpoint(policy_model=self.policy_model, vf_model=self.vf_model,
                                        policy_optimizer=self.policy_optimizer, vf_optimizer=self.vf_optimizer,
                                        epochs_done=self.epochs_done)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=1)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()

    def simulate(self):
        observation = self.env.reset()
        self.env.render()
        done = False
        while not done:
            sleep(0.005)
            action = self.policy_model.step_and_sample(observation.reshape(1, -1)).numpy()[0]
            observation, reward, done, info = self.env.step(action)
            self.env.render()
        self.env.close()

    def train(self):
        summary_writer = tf.summary.create_file_writer(self.log_dir)
        with tqdm(total=self.epochs, desc='Training', unit='epoch') as pbar:
            pbar.update(self.ckpt.epochs_done.numpy())
            while self.ckpt.epochs_done.numpy() <= self.epochs:
                all_observations, all_actions, all_returns, all_values, all_advantages = [], [], [], [], []
                episode_rewards, episode_lengths = [], []
                for episode in range(self.episodes_per_epoch):
                    observations, actions, rewards, values = [], [], [], []
                    observation = self.env.reset()
                    for step in range(self.max_episode_length):
                        observations.append(observation)
                        action = self.policy_model.step_and_sample(observation.reshape(1, -1)).numpy()[0]
                        value = self.vf_model.compute_value(observation.reshape(1, -1)).numpy()[0]
                        observation, reward, done, _ = self.env.step(action)
                        actions.append(action)
                        rewards.append(reward)
                        values.append(value)
                        if done:
                            break
                    # Add final state and value. observations & values will contain 1 additional element
                    observations.append(observation)
                    value = self.vf_model.compute_value(observation.reshape(1, -1)).numpy()[0]
                    values.append(value)
                    advantages = self._calculate_advantages(rewards, values)
                    returns = self._compute_rewards_to_go(rewards)
                    all_observations.extend(observations[:-1])
                    all_actions.extend(actions)
                    all_returns.extend(returns)
                    all_values.extend(values[:-1])
                    all_advantages.extend(advantages)
                    episode_rewards.append(sum(rewards))
                    episode_lengths.append(len(rewards))

                policy_loss = self._update_policy_model(
                    observations=tf.convert_to_tensor(all_observations, tf.float32),
                    actions=tf.convert_to_tensor(all_actions, tf.int32),
                    weights=tf.convert_to_tensor(all_advantages, tf.float32)
                )
                vf_loss = self._update_vf_model(
                    observations=tf.convert_to_tensor(all_observations, tf.float32),
                    returns=tf.convert_to_tensor(all_returns, tf.float32),
                    iterations=self.vf_update_iterations
                )

                e = self.ckpt.epochs_done.numpy()
                if e % self.ckpt_epochs == 0 or e == self.epochs:
                    self.ckpt_manager.save()
                if e % self.log_epochs == 0 or e == self.epochs:
                    with summary_writer.as_default(), tf.name_scope('training'):
                        tf.summary.scalar('avg_episode_reward', tf.reduce_mean(episode_rewards),
                                          step=self.ckpt.epochs_done)
                        tf.summary.scalar('avg_episode_length', tf.reduce_mean(episode_lengths),
                                          step=self.ckpt.epochs_done)
                        tf.summary.scalar('policy_loss', policy_loss, step=self.ckpt.epochs_done)
                        tf.summary.scalar('vf_loss', vf_loss, step=self.ckpt.epochs_done)

                self.ckpt.epochs_done.assign_add(1)
                pbar.update(1)
            self.env.close()

    def _calculate_advantages(self, rewards, values):
        deltas, coeffs, advantages = [], [], []
        for t in range(len(rewards)):
            deltas.append(rewards[t] + self.gamma * values[t + 1] - values[t])
            coeffs.append(np.power(self.gamma * self.lambda_, t))
        for t in range(len(rewards)):
            advantages.append(np.sum([coeffs[i] * deltas[t + i] for i in range(len(rewards) - t)]))
        return advantages

    def _compute_rewards_to_go(self, rewards):
        coeffs = [np.power(self.gamma, t) for t in range(len(rewards))]
        rewards_to_go = []
        for t in range(len(rewards)):
            rewards_to_go.append(np.sum([coeffs[i] * rewards[t + i] for i in range(len(rewards) - t)]))
        return rewards_to_go

    @tf.function(experimental_relax_shapes=True)
    def _update_policy_model(self, observations, actions, weights):
        weights -= tf.reduce_mean(weights)
        weights /= tf.math.reduce_std(weights) + 1e-32
        with tf.GradientTape() as tape:
            log_probs = self.policy_model.calculate_log_probs(observations, actions)
            loss = -tf.reduce_sum(log_probs * weights) / self.episodes_per_epoch
            gradients = tape.gradient(loss, self.policy_model.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(gradients, self.policy_model.trainable_variables))
        return loss

    @tf.function(experimental_relax_shapes=True)
    def _update_vf_model(self, observations, returns, iterations):
        losses = []
        for i in range(iterations):
            with tf.GradientTape() as tape:
                values = self.vf_model.compute_value(observations)
                loss = tf.keras.losses.mean_squared_error(returns, tf.squeeze(values))
                gradients = tape.gradient(loss, self.vf_model.trainable_variables)
                self.vf_optimizer.apply_gradients(zip(gradients, self.vf_model.trainable_variables))
                losses.append(loss)
        return tf.reduce_mean(losses)
