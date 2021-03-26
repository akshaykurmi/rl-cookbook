import tensorflow as tf
from tqdm import tqdm

from rl.replay_buffer import ReplayBuffer


class VPG:
    def __init__(self, env, policy_fn, lr, epochs, episodes_per_epoch, max_episode_length, ckpt_epochs, log_epochs,
                 ckpt_dir, log_dir):
        self.env = env
        self.policy = policy_fn(env.observation_space.shape, env.action_space.n)
        self.lr = lr
        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.max_episode_length = max_episode_length
        self.ckpt_epochs = ckpt_epochs
        self.log_epochs = log_epochs
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir

        self.replay_buffer = ReplayBuffer()
        self.epochs_done = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.ckpt = tf.train.Checkpoint(policy=self.policy, optimizer=self.optimizer, epochs_done=self.epochs_done)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=1)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()

    def train(self):
        summary_writer = tf.summary.create_file_writer(self.log_dir)
        with tqdm(total=self.epochs, desc='Training', unit='epoch') as pbar:
            pbar.update(self.ckpt.epochs_done.numpy())
            while self.ckpt.epochs_done.numpy() <= self.epochs:
                self.replay_buffer.purge()
                for episode in range(self.episodes_per_epoch):
                    observation = self.env.reset()
                    for step in range(self.max_episode_length):
                        action = self.policy.sample(observation.reshape(1, -1)).numpy()[0]
                        observation_next, reward, done, _ = self.env.step(action)
                        self.replay_buffer.store_transition(observation, action, reward)
                        observation = observation_next
                        if done:
                            self.replay_buffer.terminate_episode(observation)
                            break

                data = self.replay_buffer.get(['observations', 'actions', 'total_rewards', 'episode_lengths'])

                loss = self._update_policy(
                    observations=tf.convert_to_tensor(data['observations'], tf.float32),
                    actions=tf.convert_to_tensor(data['actions'], tf.int32),
                    total_rewards=tf.convert_to_tensor(data['total_rewards'], tf.float32)
                )

                e = self.ckpt.epochs_done.numpy()
                if e % self.ckpt_epochs == 0 or e == self.epochs:
                    self.ckpt_manager.save()
                if e % self.log_epochs == 0 or e == self.epochs:
                    with summary_writer.as_default(), tf.name_scope('training'):
                        tf.summary.scalar('total_rewards', tf.reduce_mean([tf.reduce_mean(t) for t in tf.split(
                            data['total_rewards'], data['episode_lengths']
                        )]), step=self.ckpt.epochs_done)
                        tf.summary.scalar('episode_lengths', tf.reduce_mean(data['episode_lengths']),
                                          step=self.ckpt.epochs_done)
                        tf.summary.scalar('policy_loss', loss, step=self.ckpt.epochs_done)

                self.ckpt.epochs_done.assign_add(1)
                pbar.update(1)
            self.env.close()

    @tf.function(experimental_relax_shapes=True)
    def _update_policy(self, observations, actions, total_rewards):
        total_rewards -= tf.reduce_mean(total_rewards)
        total_rewards /= tf.math.reduce_std(total_rewards) + 1e-8
        with tf.GradientTape() as tape:
            log_probs = self.policy.log_prob(observations, actions)
            loss = -tf.reduce_mean(log_probs * total_rewards)
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        return loss
