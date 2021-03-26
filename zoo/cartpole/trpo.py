import gym

from rl.agents.trpo import TRPO
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
        epochs=150,
        episodes_per_epoch=8,
        max_episode_length=250,
        vf_update_iterations=20,
        conjugate_gradient_iterations=20,
        conjugate_gradient_tol=1e-5,
        line_search_iterations=10,
        line_search_coefficient=0.5,
        ckpt_epochs=10,
        log_epochs=1,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir
    )

    if args.mode == 'train':
        agent.train()
    if args.mode == 'evaluate':
        evaluate_policy(agent.env, agent.policy)
