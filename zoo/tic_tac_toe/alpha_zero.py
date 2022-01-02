import os

from rl.agents.alpha_zero import AlphaZero
from zoo.tic_tac_toe.core import TicTacToe, PolicyAndValueFunctionNetwork
from zoo.utils import parse_args, get_output_dirs

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir = get_output_dirs(os.path.dirname(__file__), 'alpha_zero', args)

    game_fn = lambda: TicTacToe()
    policy_and_vf_fn = lambda: PolicyAndValueFunctionNetwork(
        observation_shape=TicTacToe.observation_space.shape,
        n_actions=TicTacToe.action_space.n,
        l2=1e-3,
    )
    agent = AlphaZero(
        game_fn=game_fn,
        policy_and_vf_fn=policy_and_vf_fn,
        lr=1e-3,
        mcts_n_steps=100,
        mcts_tau=9,
        mcts_eta=0.03,
        mcts_epsilon=0.25,
        mcts_c_puct=1,
        n_self_play_workers=8,
        update_iterations=10_000,
        update_batch_size=64,
        replay_buffer_size=10_000,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
        ckpt_every=500,
        log_every=10,
        eval_every=500,
    )

    if args.mode == 'train':
        agent.train()
    if args.mode == 'evaluate':
        import numpy as np

        game.reset()
        print(game.render())
        while not game.is_over():
            action = int(input('Action: '))
            game.step(action)
            print(game.render())
            if game.is_over():
                break
            valid_actions = game.valid_actions()
            pi, v = agent.policy_and_vf(np.expand_dims(game.observation(canonical=True), axis=0))
            pi = pi.numpy()[0]
            p = np.zeros_like(pi)
            p[valid_actions] = pi[valid_actions]
            print('==================================')
            print(pi)
            print(p)
            action = np.argmax(p)
            game.step(action)
            print(game.render())
