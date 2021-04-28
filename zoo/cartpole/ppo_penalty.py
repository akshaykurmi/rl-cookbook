import os

import gym

from rl.agents.ppo_penalty import PPOPenalty
from rl.loops import EpisodeTrainLoop
from rl.metrics import AverageEpisodeLength, AverageReturn
from zoo.cartpole.core import PolicyNetwork, ValueFunctionNetwork
from zoo.utils import parse_args, get_output_dirs, evaluate_policy

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir = get_output_dirs(os.path.dirname(__file__), 'ppo_penalty', args)

    env = gym.make('CartPole-v0')
    policy_fn = lambda: PolicyNetwork(env.observation_space.shape, env.action_space.n)
    vf_fn = lambda: ValueFunctionNetwork(env.observation_space.shape)
    agent = PPOPenalty(
        env=env,
        policy_fn=policy_fn,
        vf_fn=vf_fn,
        lr_vf=1e-3,
        lr_policy=1e-3,
        gamma=0.98,
        lambda_=0.96,
        beta=1.0,
        kl_target=0.001,
        kl_tolerance=1.5,
        beta_update_factor=2,
        vf_update_iterations=20,
        policy_update_iterations=5,
        policy_update_batch_size=64,
        vf_update_batch_size=64,
        replay_buffer_size=250 * 8,
    )

    train_loop = EpisodeTrainLoop(
        agent=agent,
        n_episodes=150 * 8,
        max_episode_length=250,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
        ckpt_every=100,
        log_every=10,
        update_every=8,
        metrics=[AverageReturn(8), AverageEpisodeLength(8)]
    )

    if args.mode == 'train':
        train_loop.run()
    if args.mode == 'evaluate':
        evaluate_policy(agent.env, agent.policy)
