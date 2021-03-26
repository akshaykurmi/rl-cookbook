import tensorflow as tf
from tqdm import tqdm

from rl.metrics import AverageEpisodeLength
from rl.metrics import AverageReturn
from rl.replay_buffer import UniformReplayBuffer


class DDPG:
    def __init__(self, env, policy_fn, qf_fn, lr_policy, lr_qf, gamma, polyak, episodes, max_episode_length,
                 replay_buffer_size, initial_random_episodes, update_every_steps, update_iterations, update_batch_size,
                 action_noise, ckpt_episodes, log_episodes, ckpt_dir, log_dir):
        self.env = env
        self.policy = policy_fn()
        self.qf = qf_fn()
        self.qf_target = qf_fn()
        self.qf_target.set_weights(self.qf.get_weights())
        self.lr_policy = lr_policy
        self.lr_qf = lr_qf
        self.gamma = gamma
        self.polyak = polyak
        self.episodes = episodes
        self.max_episode_length = max_episode_length
        self.replay_buffer_size = replay_buffer_size
        self.initial_random_episodes = initial_random_episodes
        self.update_every_steps = update_every_steps
        self.update_iterations = update_iterations
        self.update_batch_size = update_batch_size
        self.action_noise = action_noise
        self.ckpt_episodes = ckpt_episodes
        self.log_episodes = log_episodes
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir

        self.replay_buffer = UniformReplayBuffer(
            buffer_size=self.replay_buffer_size,
            store_fields=['observation', 'action', 'reward', 'observation_next', 'done'],
            compute_fields=[]
        )
        self.metrics = [
            AverageReturn(self.log_episodes * self.max_episode_length),
            AverageEpisodeLength(self.log_episodes * self.max_episode_length)
        ]
        self.episodes_done = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_policy)
        self.qf_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_qf)
        self.ckpt = tf.train.Checkpoint(policy=self.policy, qf=self.qf, qf_target=self.qf_target,
                                        policy_optimizer=self.policy_optimizer, qf_optimizer=self.qf_optimizer,
                                        episodes_done=self.episodes_done)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=1)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()

    def train(self):
        summary_writer = tf.summary.create_file_writer(self.log_dir)
        with tqdm(total=self.episodes, desc='Training', unit='episode') as pbar:
            pbar.update(self.ckpt.episodes_done.numpy())
            global_step = 0
            while self.ckpt.episodes_done.numpy() <= self.episodes:
                observation = self.env.reset()
                for step in range(self.max_episode_length):
                    global_step += 1
                    if self.ckpt.episodes_done.numpy() <= self.initial_random_episodes:
                        action = self.env.action_space.sample()
                    else:
                        action = self.policy.sample(observation.reshape(1, -1), noise=self.action_noise).numpy()[0]
                    observation_next, reward, done, _ = self.env.step(action)
                    transition = {'observation': observation, 'action': action, 'reward': reward,
                                  'observation_next': observation_next, 'done': done}
                    observation = observation_next
                    self.replay_buffer.store_transition(transition)
                    for m in self.metrics:
                        m.record(transition)
                    if global_step % self.update_every_steps == 0:
                        policy_loss, qf_loss = self._update()
                    if done:
                        break

                e = self.ckpt.episodes_done.numpy()
                if e % self.ckpt_episodes == 0 or e == self.episodes:
                    self.ckpt_manager.save()
                if e % self.log_episodes == 0 or e == self.episodes:
                    with summary_writer.as_default(), tf.name_scope('training'):
                        for m in self.metrics:
                            tf.summary.scalar(m.name, m.compute(), step=self.ckpt.episodes_done)
                        tf.summary.scalar('policy_loss', policy_loss, step=self.ckpt.episodes_done)
                        tf.summary.scalar('qf_loss', qf_loss, step=self.ckpt.episodes_done)

                for m in self.metrics:
                    m.reset()
                self.ckpt.episodes_done.assign_add(1)
                pbar.update(1)
            self.env.close()

    def _update(self):
        policy_losses, qf_losses = [], []
        for _ in range(self.update_iterations):
            batch = self.replay_buffer.sample_batch(self.update_batch_size)
            observation = tf.convert_to_tensor(batch['observation'], tf.float32)
            action = tf.convert_to_tensor(batch['action'], tf.float32)
            reward = tf.convert_to_tensor(batch['reward'], tf.float32)
            observation_next = tf.convert_to_tensor(batch['observation_next'], tf.float32)
            done = tf.convert_to_tensor(batch['done'], tf.float32)
            qf_losses.append(self._update_qf(observation, action, reward, observation_next, done))
            policy_losses.append(self._update_policy(observation))
            self._update_qf_target()
        return tf.reduce_mean(policy_losses), tf.reduce_mean(qf_losses)

    @tf.function(experimental_relax_shapes=True)
    def _update_qf(self, observation, action, reward, observation_next, done):
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
    def _update_policy(self, observation):
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
