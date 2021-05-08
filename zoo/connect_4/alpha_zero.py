import os

from rl.agents.alpha_zero import AlphaZero
from zoo.connect_4.core import Connect4, PolicyAndValueFunctionNetwork
from zoo.utils import parse_args, get_output_dirs

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir = get_output_dirs(os.path.dirname(__file__), 'alpha_zero', args)

    game = Connect4()
    policy_and_vf_fn = lambda: PolicyAndValueFunctionNetwork(
        observation_shape=game.observation_space.shape,
        n_actions=game.action_space.n,
        l2=1e-3,
    )
    agent = AlphaZero(
        game=game,
        policy_and_vf_fn=policy_and_vf_fn,
        lr=1e-3,
        replay_buffer_size=50_000,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
    )

    if args.mode == 'train':
        agent.train(
            n_iterations=100_000,
            n_self_play_games=1,
            mcts_tau=6 * 7,
            mcts_n_steps=500,
            mcts_eta=0.03,
            mcts_epsilon=0.25,
            mcts_c_puct=1,
            update_batch_size=32,
            update_iterations=1,
            ckpt_every=50,
            log_every=1,
            eval_every=20,
        )
    if args.mode == 'evaluate':
        pass
