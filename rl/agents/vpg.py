import numpy as np
import tensorflow as tf

from rl.replay_buffer import DeterministicReplayBuffer, ReplayField


class VPG:
    def __init__(self, env, policy_fn, lr, replay_buffer_size):
        self.env = env
        self.policy = policy_fn(env.observation_space.shape, env.action_space.n)
        self.lr = lr

        self.replay_buffer = DeterministicReplayBuffer(
            buffer_size=replay_buffer_size,
            store_fields=[
                ReplayField('observation', shape=self.env.observation_space.shape),
                ReplayField('action', dtype=np.int32),
                ReplayField('reward'),
                ReplayField('done', dtype=np.bool),
            ],
            compute_fields=[
                ReplayField('episode_return'),
            ],
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def variables_to_checkpoint(self):
        return {'policy': self.policy, 'optimizer': self.optimizer}

    def step(self, previous_transition=None, training=False):
        observation = previous_transition['observation_next'] if previous_transition else self.env.reset()
        action = self.policy.sample(tf.expand_dims(observation, axis=0)).numpy()[0]
        observation_next, reward, done, _ = self.env.step(action)
        transition = {'observation': observation, 'observation_next': observation_next,
                      'action': action, 'reward': reward, 'done': done}
        if training:
            self.replay_buffer.store_transition(transition)
        return transition

    def update(self):
        dataset = self.replay_buffer.as_dataset(num_batches=1, batch_size=None)
        data = next(iter(dataset))
        result = {
            'policy_loss': self._update_policy(data['observation'], data['action'], data['episode_return']),
        }
        self.replay_buffer.purge()
        return result

    @tf.function(experimental_relax_shapes=True)
    def _update_policy(self, observation, action, episode_return):
        episode_return -= tf.reduce_mean(episode_return)
        episode_return /= tf.math.reduce_std(episode_return) + 1e-8
        with tf.GradientTape() as tape:
            log_probs = self.policy.log_prob(observation, action)
            loss = -tf.reduce_mean(log_probs * episode_return)
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        return loss
