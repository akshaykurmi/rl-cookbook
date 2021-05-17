import itertools
import os
from copy import deepcopy
from time import sleep

import numpy as np
import ray
import tensorflow as tf
import tensorflow_probability as tfp

from rl.replay_buffer import UniformReplayBuffer, ReplayField, EpisodeReturn


@ray.remote
class SelfPlayJob:
    def __init__(self, game_fn, policy_and_vf_fn, mcts_n_steps, mcts_tau, mcts_eta, mcts_epsilon, mcts_c_puct):
        self.game = game_fn()
        self.policy_and_vf = policy_and_vf_fn()
        self.mcts_n_steps = mcts_n_steps
        self.mcts_tau = mcts_tau
        self.mcts_eta = mcts_eta
        self.mcts_epsilon = mcts_epsilon
        self.mcts_c_puct = mcts_c_puct

    def set_weights(self, weights):
        self.policy_and_vf.set_weights(weights)

    def simulate_game(self):
        episode = []
        self.game.reset()
        mcts = MCTS(game=deepcopy(self.game), policy_and_vf=self.policy_and_vf, n_steps=self.mcts_n_steps,
                    tau=self.mcts_tau, eta=self.mcts_eta, epsilon=self.mcts_epsilon, c_puct=self.mcts_c_puct)
        for step in itertools.count():
            pi = mcts.search(step)
            action = int(tfp.distributions.Categorical(probs=pi).sample())
            observation = self.game.observation(canonical=True)
            player = self.game.turn.value
            self.game.step(action)
            mcts.step(action)
            score = self.game.score()
            is_over = self.game.is_over()
            transition = {'observation': observation, 'player': player, 'pi': pi, 'score': score, 'done': is_over}
            episode.append(transition)
            if is_over:
                break
        self.game.close()
        return episode


@ray.remote
class ParameterServer:
    def __init__(self):
        self.weights = None

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights


@ray.remote
class ReplayBuffer:
    def __init__(self, game_fn, replay_buffer_size, batch_size):
        game = game_fn()
        self.buffer = UniformReplayBuffer(
            buffer_size=replay_buffer_size,
            store_fields=[
                ReplayField('observation', shape=game.observation_space.shape,
                            dtype=game.observation_space.dtype),
                ReplayField('pi', shape=(game.action_space.n,)),
                ReplayField('player', dtype=np.int8),
                ReplayField('score'),
                ReplayField('done', dtype=np.bool),
            ],
            compute_fields=[
                EpisodeReturn(reward_field='score', name='z'),
            ],
        )
        self.batch_size = batch_size

    def add_episode(self, episode):
        for transition in episode:
            self.buffer.store_transition(transition)

    def sample_batch(self):
        return next(iter(self.buffer.as_dataset(batch_size=self.batch_size).take(1)))

    def is_ready(self):
        return self.buffer.current_size > 0


@ray.remote
class SelfPlayDriver:
    def __init__(self, parameter_server, replay_buffer, game_fn, policy_and_vf_fn, n_self_play_workers, mcts_n_steps,
                 mcts_tau, mcts_eta, mcts_epsilon, mcts_c_puct):
        self.parameter_server = parameter_server
        self.replay_buffer = replay_buffer
        self.self_play_workers = [
            SelfPlayJob.remote(game_fn, policy_and_vf_fn, mcts_n_steps, mcts_tau, mcts_eta, mcts_epsilon, mcts_c_puct)
            for _ in range(n_self_play_workers)
        ]
        current_weights = self.parameter_server.get_weights.remote()
        ray.get([worker.set_weights.remote(current_weights) for worker in self.self_play_workers])

    def start(self):
        jobs = {worker.simulate_game.remote(): worker
                for worker in self.self_play_workers}
        while True:
            simulated_episode_id = ray.wait(list(jobs))[0][0]
            worker = jobs.pop(simulated_episode_id)
            self.replay_buffer.add_episode.remote(simulated_episode_id)
            worker.set_weights.remote(self.parameter_server.get_weights.remote())
            jobs[worker.simulate_game.remote()] = worker


@ray.remote
class ModelUpdateDriver:
    def __init__(self, parameter_server, replay_buffer, game_fn, policy_and_vf_fn, lr, ckpt_dir, log_dir,
                 n_iterations, ckpt_every, log_every, eval_every):
        self.parameter_server = parameter_server
        self.replay_buffer = replay_buffer
        self.game = game_fn()
        self.policy_and_vf = policy_and_vf_fn()
        self.n_iterations = n_iterations
        self.ckpt_every = ckpt_every
        self.log_every = log_every
        self.eval_every = eval_every

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.cce_loss = tf.keras.losses.CategoricalCrossentropy()
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.iterations_done = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.ckpt = tf.train.Checkpoint(iterations_done=self.iterations_done, optimizer=self.optimizer,
                                        policy_and_vf=self.policy_and_vf)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, os.path.join(ckpt_dir, 'model_update'), max_to_keep=1, keep_checkpoint_every_n_hours=1)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()

        ray.get(self.parameter_server.set_weights.remote(self.policy_and_vf.get_weights()))

    def start(self):
        while not ray.get(self.replay_buffer.is_ready.remote()):
            sleep(1)
        for i in range(self.ckpt.iterations_done.numpy(), self.n_iterations):
            batch = ray.get(self.replay_buffer.sample_batch.remote())
            losses = self._update(batch)

            if i % self.ckpt_every == 0 or i == self.n_iterations - 1:
                self.ckpt_manager.save()
            if i % self.log_every == 0 or i == self.n_iterations - 1:
                with self.summary_writer.as_default(), tf.name_scope('losses'):
                    for k, v in losses.items():
                        tf.summary.scalar(k, v, step=i)
            if i % self.eval_every == 0 or i == self.n_iterations - 1:
                self._evaluate()

            self.parameter_server.set_weights.remote(self.policy_and_vf.get_weights())
            self.ckpt.iterations_done.assign_add(1)

    @tf.function
    def _update(self, batch):
        observation, pi, z = batch['observation'], batch['pi'], batch['z'] * tf.cast(batch['player'], tf.float32)
        with tf.GradientTape() as tape:
            p, v = self.policy_and_vf(observation, training=True)
            policy_loss = self.cce_loss(pi, p)
            vf_loss = self.mse_loss(z, v)
            regularization_loss = tf.reduce_sum(self.policy_and_vf.losses)
            total_loss = policy_loss + vf_loss + regularization_loss
            gradients = tape.gradient(total_loss, self.policy_and_vf.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.policy_and_vf.trainable_variables))
        return {
            'policy_loss': policy_loss,
            'vf_loss': vf_loss,
            'regularization_loss': regularization_loss,
            'total_loss': total_loss,
        }

    def _evaluate(self):
        print('=============== Evaluating ===============')
        self.game.reset()
        print(self.game.render())
        while not self.game.is_over():
            valid_actions = self.game.valid_actions()
            p = self.policy_and_vf(self.game.observation(canonical=True)[None, ...])[0].numpy()[0]
            pi = np.zeros_like(p)
            pi[valid_actions] = p[valid_actions]
            print(f'All Actions   : {np.round(p, 2)}')
            print(f'Valid Actions : {np.round(pi, 2)}', end='\n\n')
            self.game.step(np.argmax(pi))
            print(self.game.render())
        print('============= Done Evaluating =============')


class AlphaZero:
    def __init__(self, game_fn, policy_and_vf_fn, lr, mcts_n_steps, mcts_tau, mcts_eta, mcts_epsilon, mcts_c_puct,
                 n_self_play_workers, update_iterations, update_batch_size, replay_buffer_size, ckpt_dir, log_dir,
                 ckpt_every, log_every, eval_every):
        self.game_fn = game_fn
        self.policy_and_vf_fn = policy_and_vf_fn
        self.lr = lr
        self.mcts_n_steps = mcts_n_steps
        self.mcts_tau = mcts_tau
        self.mcts_eta = mcts_eta
        self.mcts_epsilon = mcts_epsilon
        self.mcts_c_puct = mcts_c_puct
        self.n_self_play_workers = n_self_play_workers
        self.update_iterations = update_iterations
        self.update_batch_size = update_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.ckpt_every = ckpt_every
        self.log_every = log_every
        self.eval_every = eval_every

    def train(self):
        ray.init(ignore_reinit_error=True)
        replay_buffer = ReplayBuffer.remote(self.game_fn, self.replay_buffer_size, self.update_batch_size)
        parameter_server = ParameterServer.remote()
        model_update_driver = ModelUpdateDriver.remote(
            parameter_server, replay_buffer, self.game_fn, self.policy_and_vf_fn, self.lr, self.ckpt_dir, self.log_dir,
            self.update_iterations, self.ckpt_every, self.log_every, self.eval_every
        )
        self_play_driver = SelfPlayDriver.remote(
            parameter_server, replay_buffer, self.game_fn, self.policy_and_vf_fn, self.n_self_play_workers,
            self.mcts_n_steps, self.mcts_tau, self.mcts_eta, self.mcts_epsilon, self.mcts_c_puct
        )
        self_play_driver.start.remote()
        ray.get(model_update_driver.start.remote())
        ray.shutdown()


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
            if not is_game_over:
                leaf.expand(pi)
            leaf.backup(v)
        pi = np.zeros(self.root.game.action_space.n, dtype=np.float32)
        if step <= self.tau:
            pi[self.root.valid_actions] = self.root.N
        else:
            one_hot_N = np.zeros_like(self.root.N)
            one_hot_N[np.argwhere(self.root.N == np.max(self.root.N))] = 1
            pi[self.root.valid_actions] = one_hot_N
        return pi / np.sum(pi)

    def step(self, action):
        a = np.argwhere(self.root.valid_actions == action)[0][0]
        self.root = self.root.children[a]


class MCTSNode:
    def __init__(self, game, parent=None, causing_action=None):
        self.game = game
        self.children = {}
        self.parent: 'MCTSNode' = parent
        self.causing_action = causing_action
        self.valid_actions = self.game.valid_actions()
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
        a = np.argmax(self.Q + U)
        leaf = self.children[a].traverse(c_puct)
        return leaf

    def evaluate(self, policy_and_vf):
        if self.game.is_over():
            return None, self.game.score() * self.game.turn.value, True  # v is always -1 or 0 here
        pi, v = policy_and_vf(np.expand_dims(self.game.observation(canonical=True), axis=0))
        return pi, v, False

    def expand(self, pi):
        self.N = np.zeros(self.n, dtype=np.int32)
        self.Q = np.zeros(self.n, dtype=np.float32)
        self.P = pi.numpy()[0][self.valid_actions]
        self.P /= np.sum(self.P)
        for a in range(self.n):
            game = deepcopy(self.game)
            game.step(self.valid_actions[a])
            self.children[a] = MCTSNode(game, parent=self, causing_action=a)

    def backup(self, v):
        if self.parent:
            self.parent._update(-v, self.causing_action)
            self.parent.backup(-v)

    def _update(self, v, a):
        self.N[a] += 1
        self.Q[a] = (v + self.Q[a] * (self.N[a] - 1)) / self.N[a]
