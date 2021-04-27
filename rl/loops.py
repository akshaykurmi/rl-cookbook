import tensorflow as tf
from tqdm import tqdm


class EpisodeTrainLoop:
    def __init__(self, agent, n_episodes, max_episode_length, ckpt_dir, log_dir,
                 ckpt_every, log_every, update_every, metrics):
        self.agent = agent
        self.n_episodes = n_episodes
        self.max_episode_length = max_episode_length
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.ckpt_every = ckpt_every
        self.log_every = log_every
        self.update_every = update_every
        self.metrics = metrics

        self.episodes_done = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.ckpt = tf.train.Checkpoint(episodes_done=self.episodes_done, **agent.variables_to_checkpoint())
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.ckpt_dir, max_to_keep=1, keep_checkpoint_every_n_hours=1)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()

    def run(self):
        summary_writer = tf.summary.create_file_writer(self.log_dir)
        with tqdm(total=self.n_episodes, desc='Running train loop', unit='episode') as pbar:
            pbar.update(self.ckpt.episodes_done.numpy())
            self.agent.env.reset()
            while self.ckpt.episodes_done.numpy() < self.n_episodes:
                transition, losses = None, None
                for step in range(self.max_episode_length):
                    transition = self.agent.step(transition, training=True)
                    for m in self.metrics:
                        m.record(transition)
                    if transition['done']:
                        break
                e = self.ckpt.episodes_done.numpy()
                if e % self.update_every == 0 or e == self.n_episodes:
                    losses = self.agent.update()
                if e % self.ckpt_every == 0 or e == self.n_episodes:
                    self.ckpt_manager.save()
                if e % self.log_every == 0 or e == self.n_episodes:
                    if losses:
                        with summary_writer.as_default(), tf.name_scope('losses'):
                            for k, v in losses.items():
                                tf.summary.scalar(k, v, step=e)
                    with summary_writer.as_default(), tf.name_scope('metrics'):
                        for m in self.metrics:
                            tf.summary.scalar(m.name, m.compute(), step=e)

                self.ckpt.episodes_done.assign_add(1)
                pbar.update(1)
            self.agent.env.close()


class StepTrainLoop:
    def __init__(self, agent, n_steps, max_episode_length, ckpt_dir, log_dir,
                 ckpt_every, log_every, update_every, metrics):
        self.agent = agent
        self.n_steps = n_steps
        self.max_episode_length = max_episode_length
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.ckpt_every = ckpt_every
        self.log_every = log_every
        self.update_every = update_every
        self.metrics = metrics

        self.steps_done = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.ckpt = tf.train.Checkpoint(steps_done=self.steps_done, **agent.variables_to_checkpoint())
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.ckpt_dir, max_to_keep=1, keep_checkpoint_every_n_hours=1)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()

    def run(self):
        summary_writer = tf.summary.create_file_writer(self.log_dir)
        with tqdm(total=self.n_steps, desc='Running train loop', unit='step') as pbar:
            pbar.update(self.ckpt.steps_done.numpy())
            self.agent.env.reset()
            while self.ckpt.steps_done.numpy() < self.n_steps:
                transition, losses = None, None
                for step in range(self.max_episode_length):
                    transition = self.agent.step(transition, training=True)
                    for m in self.metrics:
                        m.record(transition)
                    e = self.ckpt.steps_done.numpy()
                    if e % self.update_every == 0 or e == self.n_steps:
                        losses = self.agent.update()
                    if e % self.ckpt_every == 0 or e == self.n_steps:
                        self.ckpt_manager.save()
                    if e % self.log_every == 0 or e == self.n_steps:
                        if losses:
                            with summary_writer.as_default(), tf.name_scope('losses'):
                                for k, v in losses.items():
                                    tf.summary.scalar(k, v, step=e)
                        with summary_writer.as_default(), tf.name_scope('metrics'):
                            for m in self.metrics:
                                tf.summary.scalar(m.name, m.compute(), step=e)
                    if transition['done'] or e == self.n_steps:
                        break
                    self.ckpt.steps_done.assign_add(1)
                    pbar.update(1)
            self.agent.env.close()
