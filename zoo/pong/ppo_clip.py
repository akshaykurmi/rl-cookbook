import os

from rl.agents.ppo_clip import PPOClip
from rl.loops import EpisodeTrainLoop
from rl.metrics import AverageReturn, AverageEpisodeLength
from zoo.pong.core import PolicyNetwork, ValueFunctionNetwork, PongEnvWrapper
from zoo.utils import parse_args, get_output_dirs, evaluate_policy

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir = get_output_dirs(os.path.dirname(__file__), 'ppo_clip', args)

    env = PongEnvWrapper()
    policy_fn = lambda: PolicyNetwork(env.observation_space.shape, env.action_space.n)
    vf_fn = lambda: ValueFunctionNetwork(env.observation_space.shape)
    agent = PPOClip(
        env=env,
        policy_fn=policy_fn,
        vf_fn=vf_fn,
        lr_policy=1e-3,
        lr_vf=1e-3,
        gamma=0.98,
        lambda_=0.96,
        epsilon=0.05,
        vf_update_iterations=20,
        policy_update_iterations=5,
        policy_update_batch_size=64,
        vf_update_batch_size=64,
        replay_buffer_size=100_000
    )

    train_loop = EpisodeTrainLoop(
        agent=agent,
        n_episodes=10_000,
        max_episode_length=100_000,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
        ckpt_every=10,
        log_every=1,
        update_every=1,
        metrics=[AverageReturn(1), AverageEpisodeLength(1)]
    )

    if args.mode == 'train':
        train_loop.run()
    if args.mode == 'evaluate':
        evaluate_policy(agent.env, agent.policy)
