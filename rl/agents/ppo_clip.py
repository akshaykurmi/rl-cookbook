import tensorflow as tf
from tqdm import tqdm

from rl.replay_buffer import ReplayBuffer


class PPOClip:
    def __init__(self, env, policy_fn, vf_fn, lr_policy, lr_vf, gamma, lambda_, delta, epsilon, epochs,
                 episodes_per_epoch, max_episode_length, vf_update_iterations, policy_update_iterations,
                 policy_update_batch_size, ckpt_epochs, log_epochs, ckpt_dir, log_dir):
        self.env = env
        self.policy = policy_fn(env.observation_space.shape, env.action_space.n)
        self.vf = vf_fn(env.observation_space.shape)
        self.lr_policy = lr_policy
        self.lr_vf = lr_vf
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta = delta
        self.epsilon = epsilon
        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.max_episode_length = max_episode_length
        self.vf_update_iterations = vf_update_iterations
        self.policy_update_iterations = policy_update_iterations
        self.policy_batch_size = policy_update_batch_size
        self.ckpt_epochs = ckpt_epochs
        self.log_epochs = log_epochs
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir

        self.replay_buffer = ReplayBuffer(self.gamma, self.lambda_)
        self.epochs_done = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_policy)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_vf)
        self.ckpt = tf.train.Checkpoint(policy=self.policy, vf=self.vf,
                                        policy_optimizer=self.policy_optimizer, vf_optimizer=self.vf_optimizer,
                                        epochs_done=self.epochs_done)
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
                        action = self.policy.sample(tf.expand_dims(observation, axis=0)).numpy()[0]
                        value = self.vf.compute(tf.expand_dims(observation, axis=0)).numpy()[0, 0]
                        observation_next, reward, done, _ = self.env.step(action)
                        self.replay_buffer.store_transition(observation, action, reward, value)
                        observation = observation_next
                        if done:
                            value = self.vf.compute(tf.expand_dims(observation, axis=0)).numpy()[0, 0]
                            self.replay_buffer.terminate_episode(observation, value)
                            break

                data = self.replay_buffer.get(['observations', 'actions', 'advantages', 'rewards_to_go',
                                               'total_rewards', 'episode_lengths'])

                policy_loss = self._update_policy(
                    observations=tf.convert_to_tensor(data['observations'], tf.float32),
                    actions=tf.convert_to_tensor(data['actions'], tf.int32),
                    advantages=tf.convert_to_tensor(data['advantages'], tf.float32)
                )
                vf_loss = self._update_vf(
                    observations=tf.convert_to_tensor(data['observations'], tf.float32),
                    rewards_to_go=tf.convert_to_tensor(data['rewards_to_go'], tf.float32)
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
                        tf.summary.scalar('policy_loss', policy_loss, step=self.ckpt.epochs_done)
                        tf.summary.scalar('vf_loss', vf_loss, step=self.ckpt.epochs_done)

                self.ckpt.epochs_done.assign_add(1)
                pbar.update(1)
            self.env.close()

    @tf.function(experimental_relax_shapes=True)
    def _update_policy(self, observations, actions, advantages):
        advantages -= tf.reduce_mean(advantages)
        advantages /= tf.math.reduce_std(advantages) + 1e-8
        log_probs_old = self.policy.log_prob(observations, actions)

        dataset = tf.data.Dataset.from_tensor_slices({'observations': observations, 'actions': actions,
                                                      'advantages': advantages, 'log_probs_old': log_probs_old})
        dataset = dataset.shuffle(500).batch(self.policy_batch_size)

        losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        i = 0
        for _ in range(self.policy_update_iterations):
            for batch in dataset:
                with tf.GradientTape() as tape:
                    log_probs = self.policy.log_prob(batch['observations'], batch['actions'])
                    importance_sampling_weight = tf.exp(log_probs - batch['log_probs_old'])
                    clipped_importance_sampling_weight = tf.clip_by_value(importance_sampling_weight, 1 - self.epsilon,
                                                                          1 + self.epsilon)
                    loss = -tf.reduce_mean(tf.math.minimum(importance_sampling_weight * batch['advantages'],
                                                           clipped_importance_sampling_weight * batch['advantages']))
                    gradients = tape.gradient(loss, self.policy.trainable_variables)
                    self.policy_optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
                    losses.write(i, loss)
                    i += 1
        return tf.reduce_mean(losses.stack())

    @tf.function(experimental_relax_shapes=True)
    def _update_vf(self, observations, rewards_to_go):
        losses = []
        for i in range(self.vf_update_iterations):
            with tf.GradientTape() as tape:
                values = self.vf.compute(observations)
                loss = tf.keras.losses.mean_squared_error(rewards_to_go, tf.squeeze(values))
                gradients = tape.gradient(loss, self.vf.trainable_variables)
                self.vf_optimizer.apply_gradients(zip(gradients, self.vf.trainable_variables))
                losses.append(loss)
        return tf.reduce_mean(losses)
