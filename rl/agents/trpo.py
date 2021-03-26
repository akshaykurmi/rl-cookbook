import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from rl.replay_buffer import ReplayBuffer


class TRPO:
    def __init__(self, env, policy_fn, vf_fn, lr_vf, gamma, lambda_, delta, epochs, episodes_per_epoch,
                 max_episode_length, vf_update_iterations, conjugate_gradient_iterations, conjugate_gradient_tol,
                 line_search_iterations, line_search_coefficient, ckpt_epochs, log_epochs, ckpt_dir, log_dir):
        self.env = env
        self.policy = policy_fn(env.observation_space.shape, env.action_space.n)
        self.vf = vf_fn(env.observation_space.shape)
        self.lr_vf = lr_vf
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta = delta
        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.max_episode_length = max_episode_length
        self.vf_update_iterations = vf_update_iterations
        self.conjugate_gradient_iterations = conjugate_gradient_iterations
        self.conjugate_gradient_tol = conjugate_gradient_tol
        self.line_search_iterations = line_search_iterations
        self.line_search_coefficient = line_search_coefficient
        self.ckpt_epochs = ckpt_epochs
        self.log_epochs = log_epochs
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir

        self.replay_buffer = ReplayBuffer(self.gamma, self.lambda_)
        self.epochs_done = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_vf)
        self.ckpt = tf.train.Checkpoint(policy=self.policy, vf=self.vf,
                                        vf_optimizer=self.vf_optimizer, epochs_done=self.epochs_done)
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
                        value = self.vf.compute(observation.reshape(1, -1)).numpy()[0, 0]
                        observation_next, reward, done, _ = self.env.step(action)
                        self.replay_buffer.store_transition(observation, action, reward, value)
                        observation = observation_next
                        if done:
                            value = self.vf.compute(observation.reshape(1, -1)).numpy()[0, 0]
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

    def _update_policy(self, observations, actions, advantages):
        advantages -= tf.reduce_mean(advantages)
        advantages /= tf.math.reduce_std(advantages) + 1e-8
        log_probs_old = self.policy.log_prob(observations, actions)
        distribution_old = self.policy.distribution(observations)

        with tf.GradientTape() as tape:
            loss_old = self._surrogate_loss(observations, actions, advantages, log_probs_old)
            gradients = tape.gradient(loss_old, self.policy.trainable_variables)
            gradients = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)

        Ax = lambda v: self._fisher_vector_product(v, observations, distribution_old)
        step_direction = self._conjugate_gradient(Ax, gradients)

        loss = self._line_search(observations, actions, advantages, Ax, step_direction, distribution_old, log_probs_old,
                                 loss_old)
        return loss

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

    def _surrogate_loss(self, observations, actions, advantages, log_probs_old):
        log_probs = self.policy.log_prob(observations, actions)
        importance_sampling_weight = tf.exp(log_probs - log_probs_old)
        return -tf.reduce_mean(importance_sampling_weight * advantages)

    def _kl_divergence(self, observations, distribution_old):
        distribution = self.policy.distribution(observations)
        return tf.reduce_mean(tfp.distributions.kl_divergence(distribution_old, distribution))

    def _fisher_vector_product(self, v, observations, distribution_old):
        with tf.GradientTape(persistent=True) as tape:
            kl = self._kl_divergence(observations, distribution_old)
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

    def _line_search(self, observations, actions, advantages, Ax, step_direction, distribution_old, log_probs_old,
                     loss_old):
        sAs = tf.tensordot(step_direction, Ax(step_direction), 1)
        beta = tf.math.sqrt((2 * self.delta) / (sAs + 1e-8))

        theta_old = self.policy.get_weights()
        shapes = [w.shape for w in theta_old]
        step_direction = tf.split(step_direction, [tf.reduce_prod(s) for s in shapes])
        step_direction = [tf.reshape(sd, s) for sd, s in zip(step_direction, shapes)]

        for i in range(self.line_search_iterations):
            theta = [w - beta * sd * (self.line_search_coefficient ** i) for w, sd in zip(theta_old, step_direction)]
            self.policy.set_weights(theta)
            kl = self._kl_divergence(observations, distribution_old)
            loss = self._surrogate_loss(observations, actions, advantages, log_probs_old)
            if kl <= self.delta and loss <= loss_old:
                return loss
            if i == self.line_search_iterations - 1:
                self.policy.set_weights(theta_old)
        return loss_old
