import gym

from examples.cartpole.core import PolicyNetwork, ValueFunctionNetwork, get_output_dirs, parse_args, evaluate_policy
from rl.agents.vpg.vpg_gae import VPGGAE

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
        epochs=500,
        episodes_per_epoch=2,
        max_episode_length=250,
        vf_update_iterations=20,
        ckpt_epochs=10,
        log_epochs=1,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir
    )

    if args.mode == 'train':
        agent.train()
    if args.mode == 'evaluate':
        evaluate_policy(agent.env, agent.policy)
