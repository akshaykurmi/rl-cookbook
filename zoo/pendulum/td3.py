import gym

from rl.agents.td3 import TD3
from zoo.pendulum.core import parse_args, get_output_dirs, evaluate_policy, PolicyNetwork, \
    QFunctionNetwork

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir = get_output_dirs('td3', args.mode == 'train')

    env = gym.make('Pendulum-v0')
    policy_fn = lambda: PolicyNetwork(env.observation_space.shape, env.action_space.shape[0], env.action_space.high,
                                      env.action_space.low)
    qf_fn = lambda: QFunctionNetwork((env.observation_space.shape[0] + env.action_space.shape[0],))
    agent = TD3(
        env=env,
        policy_fn=policy_fn,
        qf_fn=qf_fn,
        lr_policy=1e-3,
        lr_qf=1e-3,
        gamma=0.99,
        polyak=0.995,
        episodes=10000,
        max_episode_length=250,
        replay_buffer_size=5000,
        initial_random_episodes=50,
        update_every_steps=50,
        update_iterations=50,
        update_batch_size=32,
        update_policy_delay=2,
        transition_action_noise=0.1,
        target_action_noise=0.2,
        target_action_noise_clip=0.5,
        ckpt_episodes=100,
        log_episodes=1,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir
    )

    if args.mode == 'train':
        agent.train()
    if args.mode == 'evaluate':
        evaluate_policy(agent.env, agent.policy)
