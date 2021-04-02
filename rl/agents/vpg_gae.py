import numpy as np
import tensorflow as tf

from rl.replay_buffer import ReplayField, DeterministicReplayBuffer


class VPGGAE:
    def __init__(self, env, policy_fn, vf_fn, lr_policy, lr_vf, vf_update_iterations,
                 gamma, lambda_, replay_buffer_size):
        self.env = env
        self.policy = policy_fn(env.observation_space.shape, env.action_space.n)
        self.vf = vf_fn(env.observation_space.shape)
        self.vf_update_iterations = vf_update_iterations

        self.replay_buffer = DeterministicReplayBuffer(
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
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_policy)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_vf)

    def variables_to_checkpoint(self):
        return {'policy': self.policy, 'vf': self.vf,
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
        dataset = self.replay_buffer.as_dataset(num_batches=1, batch_size=None)
        data = next(iter(dataset))
        result = {
            'policy_loss': self._update_policy(data['observation'], data['action'], data['advantage']),
            'vf_loss': self._update_vf(data['observation'], data['reward_to_go']),
        }
        self.replay_buffer.purge()
        return result

    @tf.function(experimental_relax_shapes=True)
    def _update_policy(self, observation, action, advantage):
        advantage -= tf.reduce_mean(advantage)
        advantage /= tf.math.reduce_std(advantage) + 1e-8
        with tf.GradientTape() as tape:
            log_probs = self.policy.log_prob(observation, action)
            loss = -tf.reduce_mean(log_probs * advantage)
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        return loss

    @tf.function(experimental_relax_shapes=True)
    def _update_vf(self, observation, reward_to_go):
        losses = []
        for i in range(self.vf_update_iterations):
            with tf.GradientTape() as tape:
                values = self.vf.compute(observation)
                loss = tf.keras.losses.mean_squared_error(reward_to_go, tf.squeeze(values))
                gradients = tape.gradient(loss, self.vf.trainable_variables)
                self.vf_optimizer.apply_gradients(zip(gradients, self.vf.trainable_variables))
                losses.append(loss)
        return tf.reduce_mean(losses)
