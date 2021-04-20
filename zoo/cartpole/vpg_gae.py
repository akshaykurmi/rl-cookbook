import gym

from rl.agents.vpg_gae import VPGGAE
from rl.loops import EpisodeTrainLoop
from rl.metrics import AverageReturn, AverageEpisodeLength
from zoo.cartpole.core import PolicyNetwork, ValueFunctionNetwork, get_output_dirs, parse_args, evaluate_policy

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir = get_output_dirs('vpg_gae', args.mode == 'train')

    agent = VPGGAE(
        env=gym.make('CartPole-v0'),
        policy_fn=PolicyNetwork,
        vf_fn=ValueFunctionNetwork,
        lr_policy=1e-3,
        lr_vf=1e-3,
        gamma=0.98,
        lambda_=0.96,
        vf_update_iterations=20,
        replay_buffer_size=250 * 2,
        policy_update_batch_size=256,
        vf_update_batch_size=256,
    )
    train_loop = EpisodeTrainLoop(
        agent=agent,
        n_episodes=1000,
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
