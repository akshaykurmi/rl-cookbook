import gym

from rl.agents.trpo import TRPO
from rl.loops import EpisodeTrainLoop
from rl.metrics import AverageReturn, AverageEpisodeLength
from zoo.cartpole.core import PolicyNetwork, ValueFunctionNetwork, get_output_dirs, parse_args, evaluate_policy

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir = get_output_dirs('trpo', args.mode == 'train')

    agent = TRPO(
        env=gym.make('CartPole-v0'),
        policy_fn=PolicyNetwork,
        vf_fn=ValueFunctionNetwork,
        lr_vf=1e-3,
        gamma=0.98,
        lambda_=0.96,
        delta=0.001,
        replay_buffer_size=250 * 8,
        policy_update_batch_size=512,
        vf_update_batch_size=512,
        vf_update_iterations=20,
        conjugate_gradient_iterations=20,
        conjugate_gradient_tol=1e-5,
        line_search_iterations=10,
        line_search_coefficient=0.5,
    )
    train_loop = EpisodeTrainLoop(
        agent=agent,
        n_episodes=125 * 8,
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
