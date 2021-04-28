import os

import gym

from rl.agents.ddpg import DDPG
from rl.loops import StepTrainLoop
from rl.metrics import AverageReturn, AverageEpisodeLength
from zoo.lunar_lander_continuous.core import PolicyNetwork, QFunctionNetwork
from zoo.utils import parse_args, get_output_dirs, evaluate_policy

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir = get_output_dirs(os.path.dirname(__file__), 'ddpg', args)

    env = gym.make('LunarLanderContinuous-v2')
    policy_fn = lambda: PolicyNetwork(env.observation_space.shape, env.action_space.shape[0], env.action_space.high,
                                      env.action_space.low)
    qf_fn = lambda: QFunctionNetwork((env.observation_space.shape[0] + env.action_space.shape[0],))
    agent = DDPG(
        env=env,
        policy_fn=policy_fn,
        qf_fn=qf_fn,
        lr_policy=1e-3,
        lr_qf=1e-3,
        gamma=0.99,
        polyak=0.995,
        replay_buffer_size=50_000,
        update_iterations=50,
        update_batch_size=32,
        action_noise=0.1,
    )

    train_loop = StepTrainLoop(
        agent=agent,
        n_steps=10_000 * 1000,
        max_episode_length=1000,
        initial_random_steps=20 * 1000,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
        ckpt_every=10 * 1000,
        log_every=1 * 1000,
        update_every=50,
        metrics=[AverageReturn(5), AverageEpisodeLength(5)],
    )

    if args.mode == 'train':
        train_loop.run()
    if args.mode == 'evaluate':
        evaluate_policy(agent.env, agent.policy)
