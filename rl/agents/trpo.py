import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rl.replay_buffer import ReplayField, OnePassReplayBuffer
from rl.utils import MeanAccumulator, GradientAccumulator, tf_standardize


class TRPO:
    def __init__(self, env, policy_fn, vf_fn, lr_vf, gamma, lambda_, delta, replay_buffer_size,
                 policy_update_batch_size, vf_update_batch_size, vf_update_iterations, conjugate_gradient_iterations,
                 conjugate_gradient_tol, line_search_iterations, line_search_coefficient):
        self.env = env
        self.policy = policy_fn(env.observation_space.shape, env.action_space.n)
        self.vf = vf_fn(env.observation_space.shape)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta = delta
        self.vf_update_iterations = vf_update_iterations
        self.policy_update_batch_size = policy_update_batch_size
        self.vf_update_batch_size = vf_update_batch_size
        self.conjugate_gradient_iterations = conjugate_gradient_iterations
        self.conjugate_gradient_tol = conjugate_gradient_tol
        self.line_search_iterations = line_search_iterations
        self.line_search_coefficient = line_search_coefficient

        self.replay_buffer = OnePassReplayBuffer(
            buffer_size=replay_buffer_size,
            store_fields=[
                ReplayField('observation', shape=self.env.observation_space.shape,
                            dtype=self.env.observation_space.dtype),
                ReplayField('action', shape=self.env.action_space.shape,
                            dtype=self.env.action_space.dtype),
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
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_vf)

    def variables_to_checkpoint(self):
        return {'policy': self.policy, 'vf': self.vf, 'vf_optimizer': self.vf_optimizer}

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
        loss_acc = MeanAccumulator()
        for data in dataset:  # TODO: is batching here correct?
            observation, action, advantage = data['observation'], data['action'], data['advantage']
            advantage = tf_standardize(advantage)
            log_probs_old = self.policy.log_prob(observation, action)
            distribution_old = self.policy.distribution(observation)

            with tf.GradientTape() as tape:
                loss_old = self._surrogate_loss(observation, action, advantage, log_probs_old)
                gradients = tape.gradient(loss_old, self.policy.trainable_variables)
                gradients = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)

            Ax = lambda v: self._fisher_vector_product(v, observation, distribution_old)
            step_direction = self._conjugate_gradient(Ax, gradients)
            loss = self._line_search(observation, action, advantage, Ax, step_direction,
                                     distribution_old, log_probs_old, loss_old)
            loss_acc.add(loss)
        return loss_acc.value()

    def _surrogate_loss(self, observation, action, advantage, log_probs_old):
        log_probs = self.policy.log_prob(observation, action)
        importance_sampling_weight = tf.exp(log_probs - log_probs_old)
        return -tf.reduce_mean(importance_sampling_weight * advantage)

    def _kl_divergence(self, observation, distribution_old):
        distribution = self.policy.distribution(observation)
        return tf.reduce_mean(tfp.distributions.kl_divergence(distribution_old, distribution))

    def _fisher_vector_product(self, v, observation, distribution_old):
        with tf.GradientTape(persistent=True) as tape:
            kl = self._kl_divergence(observation, distribution_old)
            gradients = tape.gradient(kl, self.policy.trainable_variables)
            gradients = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)
            grad_vector_product = tf.reduce_sum(gradients * v)
            hessian_vector_product = tape.gradient(grad_vector_product, self.policy.trainable_variables)
            hessian_vector_product = tf.concat([tf.reshape(g, [-1]) for g in hessian_vector_product], axis=0)
        return hessian_vector_product

    def _conjugate_gradient(self, Ax, b):
        x = tf.zeros_like(b)
        r = tf.identity(b)
        p = tf.identity(b)
        r2 = tf.tensordot(r, r, 1)
        for _ in tf.range(self.conjugate_gradient_iterations):
            z = Ax(p)
            alpha = r2 / (tf.tensordot(p, z, 1) + 1e-8)
            x += alpha * p
            r -= alpha * z
            r2_i = tf.tensordot(r, r, 1)
            p = r + (r2_i / r2) * p
            r2 = r2_i
            if r2 < self.conjugate_gradient_tol:
                break
        return x

    def _line_search(self, observation, action, advantage, Ax, step_direction,
                     distribution_old, log_probs_old, loss_old):
        sAs = tf.tensordot(step_direction, Ax(step_direction), 1)
        beta = tf.math.sqrt((2 * self.delta) / (sAs + 1e-8))

        theta_old = self.policy.get_weights()
        shapes = [w.shape for w in theta_old]
        step_direction = tf.split(step_direction, [tf.reduce_prod(s) for s in shapes])
        step_direction = [tf.reshape(sd, s) for sd, s in zip(step_direction, shapes)]

        for i in range(self.line_search_iterations):
            theta = [w - beta * sd * (self.line_search_coefficient ** i) for w, sd in zip(theta_old, step_direction)]
            self.policy.set_weights(theta)
            kl = self._kl_divergence(observation, distribution_old)
            loss = self._surrogate_loss(observation, action, advantage, log_probs_old)
            if kl <= self.delta and loss <= loss_old:
                return loss
            if i == self.line_search_iterations - 1:
                self.policy.set_weights(theta_old)
        return loss_old

    def _update_vf(self, dataset):
        loss_acc = MeanAccumulator()
        for i in range(self.vf_update_iterations):
            gradient_acc = GradientAccumulator()
            for data in dataset:
                gradients, loss = self._update_vf_step(data)
                gradient_acc.add(gradients, tf.size(loss))
                loss_acc.add(loss)
            self.vf_optimizer.apply_gradients(zip(gradient_acc.gradients(), self.vf.trainable_variables))
        return loss_acc.value()

    @tf.function(experimental_relax_shapes=True)
    def _update_vf_step(self, data):
        observation, reward_to_go = data['observation'], data['reward_to_go']
        with tf.GradientTape() as tape:
            values = self.vf.compute(observation)
            loss = tf.math.squared_difference(reward_to_go, tf.squeeze(values))
            gradients = tape.gradient(loss, self.vf.trainable_variables)
        return gradients, loss
