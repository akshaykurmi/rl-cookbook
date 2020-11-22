import gym

from examples.cartpole.core import PolicyNetwork, parse_args, get_output_dirs, evaluate_policy
from rl.agents.vpg.vpg import VPG

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir = get_output_dirs('vpg', args.mode == 'train')

    agent = VPG(
        env=gym.make('CartPole-v0'),
        policy_fn=PolicyNetwork,
        lr=1e-3,
        epochs=1000,
        episodes_per_epoch=2,
        max_episode_length=250,
        ckpt_epochs=10,
        log_epochs=1,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir
    )

    if args.mode == 'train':
        agent.train()
    if args.mode == 'evaluate':
        evaluate_policy(agent.env, agent.policy)
