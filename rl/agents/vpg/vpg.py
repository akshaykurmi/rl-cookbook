from time import sleep

import tensorflow as tf
from tqdm import tqdm


class VPG:
    def __init__(self, env, model, lr, epochs, episodes_per_epoch, max_episode_length, ckpt_epochs, log_epochs,
                 ckpt_dir, log_dir):
        self.env = env
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.max_episode_length = max_episode_length
        self.ckpt_epochs = ckpt_epochs
        self.log_epochs = log_epochs
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.epochs_done = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, epochs_done=self.epochs_done)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=1)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()

    def simulate(self):
        observation = self.env.reset()
        self.env.render()
        done = False
        while not done:
            sleep(0.005)
            action = self.model.step_and_sample(observation.reshape(1, -1)).numpy()[0]
            observation, reward, done, info = self.env.step(action)
            self.env.render()
        self.env.close()

    def train(self):
        summary_writer = tf.summary.create_file_writer(self.log_dir)
        with tqdm(total=self.epochs, desc='Training', unit='epoch') as pbar:
            pbar.update(self.ckpt.epochs_done.numpy())
            while self.ckpt.epochs_done.numpy() <= self.epochs:
                all_observations, all_actions, all_rewards = [], [], []
                episode_rewards, episode_lengths = [], []
                for episode in range(self.episodes_per_epoch):
                    observations, actions, episode_reward, episode_length = [], [], 0, 0
                    observation = self.env.reset()
                    for step in range(self.max_episode_length):
                        observations.append(observation)
                        action = self.model.step_and_sample(observation.reshape(1, -1)).numpy()[0]
                        observation, reward, done, _ = self.env.step(action)
                        actions.append(action)
                        episode_reward += reward
                        episode_length += 1
                        if done:
                            break
                    all_observations.extend(observations)
                    all_actions.extend(actions)
                    all_rewards.extend([episode_reward] * episode_length)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)

                loss = self._train_step(
                    observations=tf.convert_to_tensor(all_observations, tf.float32),
                    actions=tf.convert_to_tensor(all_actions, tf.int32),
                    weights=tf.convert_to_tensor(all_rewards, tf.float32)
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
                        tf.summary.scalar('loss', loss, step=self.ckpt.epochs_done)

                self.ckpt.epochs_done.assign_add(1)
                pbar.update(1)
            self.env.close()

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, observations, actions, weights):
        weights -= tf.reduce_mean(weights)
        weights /= tf.math.reduce_std(weights) + 1e-32
        with tf.GradientTape() as tape:
            log_probs = self.model.calculate_log_probs(observations, actions)
            loss = -tf.reduce_sum(log_probs * weights) / self.episodes_per_epoch
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
