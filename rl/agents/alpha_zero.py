import itertools
from copy import deepcopy

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from rl.replay_buffer import UniformReplayBuffer, ReplayField
from rl.utils import MeanAccumulator


class AlphaZero:
    def __init__(self, game, policy_and_vf_fn, lr, replay_buffer_size, ckpt_dir, log_dir):
        self.game = game
        self.policy_and_vf = policy_and_vf_fn()
        # self.policy_and_vf_old = policy_and_vf_fn()
        # self.policy_and_vf_old.set_weights(self.policy_and_vf.get_weights())
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir

        self.replay_buffer = UniformReplayBuffer(
            buffer_size=replay_buffer_size,
            store_fields=[
                ReplayField('observation', shape=self.game.observation_space.shape,
                            dtype=self.game.observation_space.dtype),
                ReplayField('p', shape=self.game.action_space.shape),
                ReplayField('reward'),
                ReplayField('done', dtype=np.bool),
            ],
            compute_fields=[
                ReplayField('episode_return'),
            ],
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.iterations_done = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.ckpt = tf.train.Checkpoint(iterations_done=self.iterations_done, optimizer=self.optimizer,
                                        policy_and_vf=self.policy_and_vf, policy_and_vf_old=self.policy_and_vf_old)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.ckpt_dir, max_to_keep=1, keep_checkpoint_every_n_hours=1)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()

    def train(self, n_iterations, n_self_play_games, n_eval_games,
              mcts_tau, mcts_n_steps, mcts_eta, mcts_epsilon, mcts_c_puct,
              update_batch_size, update_iterations,
              ckpt_every, log_every, eval_every):
        summary_writer = tf.summary.create_file_writer(self.log_dir)
        with tqdm(total=n_iterations, desc='Running train loop', unit='iteration') as pbar:
            pbar.update(self.ckpt.iterations_done.numpy())
            for i in range(self.ckpt.iterations_done.numpy(), n_iterations):

                for _ in range(n_self_play_games):
                    observation = self.game.reset()
                    mcts = MCTS(game=deepcopy(self.game), policy_and_vf=self.policy_and_vf, n_steps=mcts_n_steps,
                                tau=mcts_tau, eta=mcts_eta, epsilon=mcts_epsilon, c_puct=mcts_c_puct)
                    for step in itertools.count():
                        p = mcts.search(step)
                        action = int(tfp.distributions.Categorical(probs=p).sample())
                        observation_next, score, done, _, = self.game.step(action, canonical=True)
                        mcts.step(action)
                        transition = {'observation': observation, 'p': p, 'reward': score, 'done': done}
                        self.replay_buffer.store_transition(transition)
                        observation = observation_next
                        if done:
                            break

                losses = self.update(update_batch_size, update_iterations)

                if i % ckpt_every == 0 or i == n_iterations - 1:
                    self.ckpt_manager.save()
                if i % log_every == 0 or i == n_iterations - 1:
                    with summary_writer.as_default(), tf.name_scope('losses'):
                        for k, v in losses.items():
                            tf.summary.scalar(k, v, step=i)
                if i % eval_every == 0 or i == n_iterations - 1:
                    # evaluate
                    pass

                self.ckpt.iterations_done.assign_add(1)
                pbar.update(1)

        self.game.close()

    def update(self, update_batch_size, update_iterations):
        dataset = self.replay_buffer.as_dataset(update_batch_size).take(update_iterations)
        policy_loss_acc, vf_loss_acc, total_loss_acc = MeanAccumulator(), MeanAccumulator(), MeanAccumulator()
        for data in dataset:
            policy_loss, vf_loss, total_loss = self._update(data)
            policy_loss_acc.add(policy_loss)
            vf_loss_acc.add(vf_loss)
            total_loss_acc.add(total_loss)
        return {
            'policy_loss': policy_loss_acc.value(),
            'vf_loss': vf_loss_acc.value(),
            'total_loss': total_loss_acc.value(),
        }

    @tf.function
    def _update(self, data):
        observation, p, z = data['observation'], data['p'], data['episode_return']
        with tf.GradientTape as tape:
            pi, v = self.policy_and_vf(observation, training=True)
            policy_loss = tf.keras.losses.categorical_crossentropy(p, pi)
            vf_loss = tf.keras.losses.mean_squared_error(z, v)
            regularization_loss = self.policy_and_vf.losses  # change this
            total_loss = policy_loss + vf_loss + regularization_loss
            gradients = tape.gradient(total_loss, self.policy_and_vf.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.policy_and_vf.trainable_variables))
        return policy_loss, vf_loss, total_loss


class MCTS:
    def __init__(self, game, policy_and_vf, n_steps, tau, eta, epsilon, c_puct):
        self.policy_and_vf = policy_and_vf
        self.n_steps = n_steps
        self.tau = tau
        self.eta = eta
        self.epsilon = epsilon
        self.c_puct = c_puct
        self.root = MCTSNode(game)

    def search(self, step):
        self.root.add_dirichlet_noise(eta=self.eta, epsilon=self.epsilon)
        for _ in range(self.n_steps):
            leaf = self.root.traverse(self.c_puct)
            pi, v, is_game_over = leaf.evaluate(self.policy_and_vf)
            leaf.expand(pi)
            leaf.backup(v)
            if is_game_over:
                break
        p = np.zeros(self.root.game.action_space.shape, dtype=np.float32)
        if step <= self.tau:
            p[self.root.valid_actions] = self.root.N
        else:
            one_hot_N = np.zeros_like(self.root.N)
            one_hot_N[np.argwhere(self.root.N == np.max(self.root.N))] = 1
            p[self.root.valid_actions] = one_hot_N
        return p / np.sum(p)

    def step(self, action):
        self.root = self.root.children[action]


class MCTSNode:
    def __init__(self, game, parent=None, causing_action=None):
        self.game = game
        self.children = {}
        self.parent: 'MCTSNode' = parent
        self.causing_action = causing_action
        self.valid_actions = self.game.valid_actions(canonical=False)
        self.n = len(self.valid_actions)
        self.N = None
        self.Q = None
        self.P = None
        self.P_noise = lambda p: p

    def add_dirichlet_noise(self, eta, epsilon):
        self.P_noise = lambda p: (1 - epsilon) * p + \
                                 epsilon * np.random.default_rng().dirichlet(np.repeat(eta, self.n))

    def is_leaf(self):
        return len(self.children) == 0

    def traverse(self, c_puct):
        if self.is_leaf():
            return self
        U = c_puct * self.P_noise(self.P) * np.sqrt(np.sum(self.N)) / (1 + self.N)
        action = self.valid_actions[np.argmax(self.Q + U)]
        leaf = self.children[action].traverse(c_puct)
        return leaf

    def evaluate(self, policy_and_vf):
        pi, v = policy_and_vf(self.game.observation(canonical=True))  # evaluate a symmetry?
        if self.game.is_over():
            return pi, self.game.score(canonical=False), True
        return pi, v, False

    def expand(self, pi):
        self.N = np.zeros(self.n, dtype=np.int32)
        self.Q = np.zeros(self.n, dtype=np.float32)
        self.P = pi.numpy()[self.valid_actions]
        self.P /= np.sum(self.P)
        for action in self.valid_actions:
            game = deepcopy(self.game)
            game.step(action)
            self.children[action] = MCTSNode(game, parent=self, causing_action=action)

    def backup(self, v):
        if self.parent:
            self.parent._update(-v, self.causing_action)
            self.parent.backup(-v)

    def _update(self, v, action):
        self.N[action] += 1
        self.Q[action] = (v + self.Q[action] * (self.N[action] - 1)) / self.N[action]
