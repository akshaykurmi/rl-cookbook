import os

import gym

from rl.agents.vpg import VPG
from rl.loops import EpisodeTrainLoop
from rl.metrics import AverageReturn, AverageEpisodeLength
from zoo.cartpole.core import PolicyNetwork, ValueFunctionNetwork
from zoo.utils import parse_args, get_output_dirs, evaluate_policy

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir = get_output_dirs(os.path.dirname(__file__), 'vpg', args)

    env = gym.make('CartPole-v0')
    policy_fn = lambda: PolicyNetwork(env.observation_space.shape, env.action_space.n)
    vf_fn = lambda: ValueFunctionNetwork(env.observation_space.shape)
    agent = VPG(
        env=env,
        policy_fn=policy_fn,
        lr=1e-3,
        replay_buffer_size=250 * 2,
        policy_update_batch_size=64,
    )
    train_loop = EpisodeTrainLoop(
        agent=agent,
        n_episodes=2000,
        max_episode_length=250,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
        ckpt_every=100,
        log_every=10,
        update_every=2,
        metrics=[AverageReturn(2), AverageEpisodeLength(2)]
    )

    if args.mode == 'train':
        train_loop.run()
    if args.mode == 'evaluate':
        evaluate_policy(agent.env, agent.policy)
