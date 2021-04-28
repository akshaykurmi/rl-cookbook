import os

import gym

from rl.agents.sac import SAC
from rl.loops import StepTrainLoop
from rl.metrics import AverageReturn
from zoo.pendulum.core import QFunctionNetwork, PolicyNetworkSAC
from zoo.utils import get_output_dirs, parse_args, evaluate_policy

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir = get_output_dirs(os.path.dirname(__file__), 'sac', args)

    env = gym.make('Pendulum-v0')
    policy_fn = lambda: PolicyNetworkSAC(env.observation_space.shape, env.action_space.shape[0], env.action_space.high,
                                         env.action_space.low)
    qf_fn = lambda: QFunctionNetwork((env.observation_space.shape[0] + env.action_space.shape[0],))
    agent = SAC(
        env=env,
        policy_fn=policy_fn,
        qf_fn=qf_fn,
        lr_policy=1e-3,
        lr_qf=1e-3,
        gamma=0.99,
        polyak=0.995,
        alpha=0.2,
        replay_buffer_size=5000,
        update_iterations=50,
        update_batch_size=32,
    )

    train_loop = StepTrainLoop(
        agent=agent,
        n_steps=500 * 200,
        max_episode_length=200,
        initial_random_steps=20 * 200,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
        ckpt_every=20 * 200,
        log_every=5 * 200,
        update_every=50,
        metrics=[AverageReturn(5)],
    )

    if args.mode == 'train':
        train_loop.run()
    if args.mode == 'evaluate':
        evaluate_policy(agent.env, agent.policy)
