import os

from rl.agents.alpha_zero import AlphaZero
from zoo.tic_tac_toe.core import TicTacToe, PolicyAndValueFunctionNetwork
from zoo.utils import parse_args, get_output_dirs

if __name__ == '__main__':
    args = parse_args()
    ckpt_dir, log_dir = get_output_dirs(os.path.dirname(__file__), 'alpha_zero', args)

    game = TicTacToe()
    policy_and_vf_fn = lambda: PolicyAndValueFunctionNetwork(
        observation_shape=game.observation_space.shape,
        n_actions=game.action_space.n,
        l2=1e-3,
    )
    agent = AlphaZero(
        game=game,
        policy_and_vf_fn=policy_and_vf_fn,
        lr=1e-3,
        replay_buffer_size=10_000,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
    )

    if args.mode == 'train':
        agent.train(
            n_iterations=5000,
            n_self_play_games=10,
            mcts_tau=9,
            mcts_n_steps=100,
            mcts_eta=0.03,
            mcts_epsilon=0.25,
            mcts_c_puct=1,
            update_batch_size=32,
            update_iterations=5,
            ckpt_every=50,
            log_every=1,
            eval_every=20,
        )
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
