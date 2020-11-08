from time import sleep

import tensorflow as tf
from tqdm import tqdm


class VPG1:
    def __init__(self, env, model, alpha, max_steps, update_steps, ckpt_dir, log_dir):
        self.env = env
        self.model = model
        self.alpha = alpha
        self.max_steps = max_steps
        self.update_steps = update_steps
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.steps_done = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, steps_done=self.steps_done)
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
        with tqdm(total=self.max_steps, desc='Training', unit='step') as pbar:
            pbar.update(self.ckpt.steps_done.numpy())
            observations, actions, episode_rewards, episode_lengths = [], [], [], []
            episode_reward, episode_length = 0, 0
            observation = self.env.reset()
            for _ in range(self.max_steps - self.ckpt.steps_done.numpy()):
                observations.append(observation)
                action = self.model.step_and_sample(observation.reshape(1, -1)).numpy()[0]
                observation, reward, done, _ = self.env.step(action)
                actions.append(action)
                episode_reward += reward
                episode_length += 1
                if done:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    observation = self.env.reset()
                    episode_reward, episode_length = 0, 0
                    if len(observations) >= self.update_steps:
                        observations = tf.convert_to_tensor(observations, tf.float32)
                        actions = tf.convert_to_tensor(actions, tf.int32)
                        weights = tf.convert_to_tensor(tf.repeat(episode_rewards, episode_lengths), tf.float32)
                        loss = self._train_step(observations, actions, weights)
                        self.ckpt_manager.save()
                        with summary_writer.as_default(), tf.name_scope('training'):
                            tf.summary.scalar('avg_episode_reward', tf.reduce_mean(episode_rewards),
                                              step=self.ckpt.steps_done)
                            tf.summary.scalar('avg_episode_length', tf.reduce_mean(episode_lengths),
                                              step=self.ckpt.steps_done)
                            tf.summary.scalar('loss', loss, step=self.ckpt.steps_done)
                        observations, actions, episode_rewards, episode_lengths = [], [], [], []
                self.ckpt.steps_done.assign_add(1)
                pbar.update(1)
            self.env.close()

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, observations, actions, weights):
        weights -= tf.reduce_mean(weights)
        weights /= tf.math.reduce_std(weights) + 1e-32
        with tf.GradientTape() as tape:
            log_probs = self.model.calculate_log_probs(observations, actions)
            loss = -tf.reduce_mean(log_probs * weights)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
