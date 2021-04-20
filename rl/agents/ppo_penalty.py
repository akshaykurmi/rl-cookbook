import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rl.replay_buffer import OnePassReplayBuffer, ReplayField
from rl.utils import LossAccumulator, GradientAccumulator


class PPOPenalty:
    def __init__(self, env, policy_fn, vf_fn, lr_policy, lr_vf, gamma, lambda_, beta, kl_target,
                 kl_tolerance, beta_update_factor, vf_update_iterations, policy_update_iterations,
                 policy_update_batch_size, vf_update_batch_size, replay_buffer_size):
        self.env = env
        self.policy = policy_fn(env.observation_space.shape, env.action_space.n)
        self.policy_old = policy_fn(env.observation_space.shape, env.action_space.n)
        self.policy_old.set_weights(self.policy.get_weights())
        self.vf = vf_fn(env.observation_space.shape)
        self.lr_policy = lr_policy
        self.lr_vf = lr_vf
        self.gamma = gamma
        self.lambda_ = lambda_
        self.beta = beta
        self.kl_target = kl_target
        self.kl_tolerance = kl_tolerance
        self.beta_update_factor = beta_update_factor
        self.vf_update_iterations = vf_update_iterations
        self.policy_update_iterations = policy_update_iterations
        self.policy_update_batch_size = policy_update_batch_size
        self.vf_update_batch_size = vf_update_batch_size

        self.replay_buffer = OnePassReplayBuffer(
            buffer_size=replay_buffer_size,
            store_fields=[
                ReplayField('observation', shape=self.env.observation_space.shape),
                ReplayField('action', dtype=np.int32),
                ReplayField('reward'),
                ReplayField('value'),
                ReplayField('value_next'),
                ReplayField('done', dtype=np.bool),
            ],
            compute_fields=[
                ReplayField('advantage'),
                ReplayField('reward_to_go'),
            ],
            gamma=gamma,
            lambda_=lambda_,
        )
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_policy)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_vf)

    def variables_to_checkpoint(self):
        return {'policy': self.policy, 'policy_old': self.policy_old, 'vf': self.vf,
                'policy_optimizer': self.policy_optimizer, 'vf_optimizer': self.vf_optimizer}

    def step(self, previous_transition=None, training=False):
        observation = previous_transition['observation_next'] if previous_transition else self.env.reset()
        value = previous_transition['value_next'] if previous_transition else \
            self.vf.compute(tf.expand_dims(observation, axis=0)).numpy()[0, 0]
        action = self.policy.sample(tf.expand_dims(observation, axis=0)).numpy()[0]
        observation_next, reward, done, _ = self.env.step(action)
        value_next = self.vf.compute(tf.expand_dims(observation_next, axis=0)).numpy()[0, 0]
        transition = {'observation': observation, 'observation_next': observation_next,
                      'action': action, 'reward': reward, 'value': value, 'value_next': value_next, 'done': done}
        if training:
            self.replay_buffer.store_transition(transition)
        return transition

    def update(self):
        result = {
            'policy_loss': self._update_policy(self.replay_buffer.as_dataset(self.policy_update_batch_size)),
            'vf_loss': self._update_vf(self.replay_buffer.as_dataset(self.vf_update_batch_size)),
        }
        self.replay_buffer.purge()
        return result

    def _update_policy(self, dataset):
        loss_acc = LossAccumulator()
        for i in range(self.policy_update_iterations):
            for data in dataset:
                gradients, loss = self._update_policy_step(data)
                self.policy_optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
                loss_acc.add(loss)

        kl_acc = LossAccumulator()
        for data in dataset:
            distribution_old = self.policy_old.distribution(data['observation'])
            distribution = self.policy.distribution(data['observation'])
            kl = tfp.distributions.kl_divergence(distribution_old, distribution)
            kl_acc.add(kl)

        if kl_acc.loss() < self.kl_target / self.kl_tolerance:
            self.beta /= self.beta_update_factor
        elif kl_acc.loss() > self.kl_target * self.kl_tolerance:
            self.beta *= self.beta_update_factor

        self.policy_old.set_weights(self.policy.get_weights())
        return loss_acc.loss()

    @tf.function(experimental_relax_shapes=True)
    def _update_policy_step(self, data):
        observation, action, advantage = data['observation'], data['action'], data['advantage']
        advantage -= tf.reduce_mean(advantage)
        advantage /= tf.math.reduce_std(advantage) + 1e-8
        distribution_old = self.policy_old.distribution(observation)
        log_probs_old = distribution_old.log_prob(action)
        with tf.GradientTape() as tape:
            distribution = self.policy.distribution(observation)
            log_probs = distribution.log_prob(action)
            importance_sampling_weight = tf.exp(log_probs - log_probs_old)
            kl = tf.reduce_mean(tfp.distributions.kl_divergence(distribution_old, distribution))
            loss = -tf.reduce_mean(importance_sampling_weight * advantage - self.beta * kl)
            gradients = tape.gradient(loss, self.policy.trainable_variables)
        return gradients, loss

    def _update_vf(self, dataset):
        loss_acc = LossAccumulator()
        for i in range(self.vf_update_iterations):
            gradient_acc = GradientAccumulator()
            for data in dataset:
                gradients, loss = self._update_vf_step(data)
                gradient_acc.add(gradients, tf.size(loss))
                loss_acc.add(loss)
            self.vf_optimizer.apply_gradients(zip(gradient_acc.gradients(), self.vf.trainable_variables))
        return loss_acc.loss()

    @tf.function(experimental_relax_shapes=True)
    def _update_vf_step(self, data):
        observation, reward_to_go = data['observation'], data['reward_to_go']
        with tf.GradientTape() as tape:
            values = self.vf.compute(observation)
            loss = tf.math.squared_difference(reward_to_go, tf.squeeze(values))
            gradients = tape.gradient(loss, self.vf.trainable_variables)
        return gradients, loss
