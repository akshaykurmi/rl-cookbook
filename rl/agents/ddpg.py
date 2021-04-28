import numpy as np
import tensorflow as tf

from rl.replay_buffer import UniformReplayBuffer, ReplayField
from rl.utils import MeanAccumulator


class DDPG:
    def __init__(self, env, policy_fn, qf_fn, lr_policy, lr_qf, gamma, polyak, action_noise,
                 update_iterations, update_batch_size, replay_buffer_size):
        self.env = env
        self.policy = policy_fn()
        self.qf = qf_fn()
        self.qf_target = qf_fn()
        self.qf_target.set_weights(self.qf.get_weights())
        self.lr_policy = lr_policy
        self.lr_qf = lr_qf
        self.gamma = gamma
        self.polyak = polyak
        self.action_noise = action_noise
        self.update_iterations = update_iterations
        self.update_batch_size = update_batch_size

        self.replay_buffer = UniformReplayBuffer(
            buffer_size=replay_buffer_size,
            store_fields=[
                ReplayField('observation', shape=self.env.observation_space.shape,
                            dtype=self.env.observation_space.dtype),
                ReplayField('observation_next', shape=self.env.observation_space.shape,
                            dtype=self.env.observation_space.dtype),
                ReplayField('action', shape=self.env.action_space.shape,
                            dtype=self.env.action_space.dtype),
                ReplayField('reward'),
                ReplayField('done', dtype=np.bool),
            ],
            compute_fields=[],
        )
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_policy)
        self.qf_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_qf)

    def variables_to_checkpoint(self):
        return {'policy': self.policy, 'qf': self.qf, 'qf_target': self.qf_target,
                'policy_optimizer': self.policy_optimizer, 'qf_optimizer': self.qf_optimizer}

    def step(self, previous_transition=None, training=False, random_action=False):
        observation = previous_transition['observation_next'] if previous_transition else self.env.reset()
        action = self.env.action_space.sample() if random_action else \
            self.policy.sample(observation.reshape(1, -1), noise=self.action_noise).numpy()[0]
        observation_next, reward, done, _ = self.env.step(action)
        transition = {'observation': observation, 'observation_next': observation_next,
                      'action': action, 'reward': reward, 'done': done}
        if training:
            self.replay_buffer.store_transition(transition)
        return transition

    def update(self):
        dataset = self.replay_buffer.as_dataset(self.update_batch_size).take(self.update_iterations)
        policy_loss_acc, qf_loss_acc = MeanAccumulator(), MeanAccumulator()
        for data in dataset:
            qf_loss_acc.add(self._update_qf(data))
            policy_loss_acc.add(self._update_policy(data))
            self._update_qf_target()
        return {
            'policy_loss': policy_loss_acc.value(),
            'qf_loss': qf_loss_acc.value(),
        }

    @tf.function(experimental_relax_shapes=True)
    def _update_qf(self, data):
        observation, observation_next = data['observation'], data['observation_next']
        action, reward, done = data['action'], data['reward'], data['done']
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.qf.trainable_variables)
            q = self.qf.compute(observation, action)
            q_target = self.qf_target.compute(observation_next, self.policy.sample(observation_next))
            bellman_backup = reward + self.gamma * (1 - done) * q_target
            loss = tf.keras.losses.mean_squared_error(q, bellman_backup)
            gradients = tape.gradient(loss, self.qf.trainable_variables)
            self.qf_optimizer.apply_gradients(zip(gradients, self.qf.trainable_variables))
        return loss

    @tf.function(experimental_relax_shapes=True)
    def _update_policy(self, data):
        observation = data['observation']
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.policy.trainable_variables)
            a = self.policy.sample(observation)
            q = self.qf.compute(observation, a)
            loss = -tf.reduce_mean(q)
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        return loss

    @tf.function(experimental_relax_shapes=True)
    def _update_qf_target(self):
        for v1, v2 in zip(self.qf.trainable_variables, self.qf_target.trainable_variables):
            v2.assign(tf.multiply(v2, self.polyak))
            v2.assign_add(tf.multiply(v1, 1 - self.polyak))
