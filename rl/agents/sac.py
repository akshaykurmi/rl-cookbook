import numpy as np
import tensorflow as tf

from rl.replay_buffer import UniformReplayBuffer, ReplayField
from rl.utils import MeanAccumulator


class SAC:
    def __init__(self, env, policy_fn, qf_fn, lr_policy, lr_qf, gamma, polyak, alpha,
                 update_iterations, update_batch_size, replay_buffer_size):
        self.env = env
        self.policy = policy_fn()
        self.qf1 = qf_fn()
        self.qf2 = qf_fn()
        self.policy_target = policy_fn()
        self.qf1_target = qf_fn()
        self.qf2_target = qf_fn()
        self.policy_target.set_weights(self.policy.get_weights())
        self.qf1_target.set_weights(self.qf1.get_weights())
        self.qf2_target.set_weights(self.qf2.get_weights())
        self.lr_policy = lr_policy
        self.lr_qf = lr_qf
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
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
        return {'policy': self.policy, 'qf1': self.qf1, 'qf2': self.qf2,
                'policy_target': self.policy_target, 'qf1_target': self.qf1_target, 'qf2_target': self.qf2_target,
                'policy_optimizer': self.policy_optimizer, 'qf_optimizer': self.qf_optimizer}

    def step(self, previous_transition=None, training=False, random_action=False):
        observation = previous_transition['observation_next'] if previous_transition else self.env.reset()
        action = self.env.action_space.sample() if random_action else \
            self.policy.sample(observation.reshape(1, -1), return_entropy=False).numpy()[0]
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
            self._update_targets()
        return {
            'policy_loss': policy_loss_acc.value(),
            'qf_loss': qf_loss_acc.value(),
        }

    @tf.function(experimental_relax_shapes=True)
    def _update_qf(self, data):
        observation, observation_next = data['observation'], data['observation_next']
        action, reward, done = data['action'], data['reward'], tf.cast(data['done'], tf.float32)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.qf1.trainable_variables)
            tape.watch(self.qf2.trainable_variables)
            q1 = self.qf1.compute(observation, action)
            q2 = self.qf2.compute(observation, action)
            target_action, target_action_entropy = self.policy.sample(observation_next)
            q1_target = self.qf1_target.compute(observation_next, target_action)
            q2_target = self.qf2_target.compute(observation_next, target_action)
            q_target = tf.minimum(q1_target, q2_target)
            bellman_backup = reward + self.gamma * (1 - done) * (q_target - self.alpha * target_action_entropy)
            q1_loss = tf.keras.losses.mean_squared_error(q1, bellman_backup)
            q2_loss = tf.keras.losses.mean_squared_error(q2, bellman_backup)
            loss = q1_loss + q2_loss
            variables = self.qf1.trainable_variables + self.qf2.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.qf_optimizer.apply_gradients(zip(gradients, variables))
        return loss

    @tf.function(experimental_relax_shapes=True)
    def _update_policy(self, data):
        observation = data['observation']
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.policy.trainable_variables)
            a, e = self.policy.sample(observation)
            q1 = self.qf1.compute(observation, a)
            q2 = self.qf2.compute(observation, a)
            q = tf.minimum(q1, q2)
            loss = -tf.reduce_mean(q - self.alpha * e)
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        return loss

    @tf.function(experimental_relax_shapes=True)
    def _update_targets(self):
        for active, target in [(self.policy, self.policy_target),
                               (self.qf1, self.qf1_target),
                               (self.qf2, self.qf2_target)]:
            for v1, v2 in zip(active.trainable_variables, target.trainable_variables):
                v2.assign(tf.multiply(v2, self.polyak))
                v2.assign_add(tf.multiply(v1, 1 - self.polyak))
