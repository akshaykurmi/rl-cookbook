import os

from rl.agents.alpha_zero import AlphaZero
from zoo.connect_4.core import Connect4, PolicyAndValueFunctionNetwork
from zoo.utils import parse_args, get_output_dirs

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir, replay_dir = get_output_dirs(os.path.dirname(__file__), 'alpha_zero', args,
                                                    ['ckpt', 'log', 'replay'])

    game_fn = lambda: Connect4()
    policy_and_vf_fn = lambda: PolicyAndValueFunctionNetwork(
        observation_shape=Connect4.observation_space.shape,
        n_actions=Connect4.action_space.n,
        l2=3e-4,
    )
    agent = AlphaZero(
        game_fn=game_fn,
        policy_and_vf_fn=policy_and_vf_fn,
        lr=1e-3,
        mcts_n_steps=500,
        mcts_tau=6 * 7,
        mcts_eta=0.03,
        mcts_epsilon=0.25,
        mcts_c_puct=1,
        n_self_play_workers=8,
        update_iterations=10_000_000,
        update_batch_size=64,
        replay_buffer_size=50_000,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
        replay_dir=replay_dir,
        ckpt_every=100,
        log_every=5,
        replay_save_every=50,
        eval_every=500,
    )

    if args.mode == 'train':
        agent.train()
    if args.mode == 'evaluate':
        pass
