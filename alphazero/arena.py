"""
Arena for pitting two models against each other.

Used to evaluate whether a newly trained model is stronger than the previous
best. The stronger model is kept for the next training iteration.
"""

import numpy as np
import torch

from hexo.game import HeXOGame
from alphazero.mcts import MCTS


def play_match(game, mcts1, mcts2, num_games=20, verbose=False):
    """
    Play `num_games` between two MCTS agents. Each agent plays half as P1.

    Args:
        game: HeXOGame instance.
        mcts1: MCTS instance for model 1.
        mcts2: MCTS instance for model 2.
        num_games: total games to play (half each side).
        verbose: print game results.

    Returns:
        (wins1, wins2, draws) tuple.
    """
    wins1, wins2 = 0, 0
    half = num_games // 2

    for game_idx in range(num_games):
        # Alternate who goes first
        if game_idx < half:
            p1_mcts, p2_mcts = mcts1, mcts2
            p1_label, p2_label = "Model1", "Model2"
        else:
            p1_mcts, p2_mcts = mcts2, mcts1
            p1_label, p2_label = "Model2", "Model1"

        state = game.get_initial_state()

        while True:
            current = state.current_player
            if current == 1:
                mcts_agent = p1_mcts
            else:
                mcts_agent = p2_mcts

            action_probs = mcts_agent.search(state)
            action = int(np.argmax(action_probs))  # greedy during evaluation

            state = game.get_next_state(state, action, current)
            value, is_terminal = game.get_value_and_terminated(state, action)

            if is_terminal:
                winner = state.winner
                if game_idx < half:
                    if winner == 1:
                        wins1 += 1
                    elif winner == -1:
                        wins2 += 1
                else:
                    # Sides are swapped
                    if winner == 1:
                        wins2 += 1
                    elif winner == -1:
                        wins1 += 1

                if verbose:
                    w = "Draw" if winner == 0 else (
                        p1_label if winner == 1 else p2_label
                    )
                    print(f"  Game {game_idx + 1}: {w} wins "
                          f"({state.total_moves} moves)")
                break

    draws = num_games - wins1 - wins2
    return wins1, wins2, draws


def evaluate_models(game, model_new, model_old, args, num_games=20,
                    win_threshold=0.55, verbose=True):
    """
    Evaluate new model against old model. Returns True if new model wins
    more than `win_threshold` fraction of games.
    """
    eval_args = dict(args)
    eval_args['dirichlet_epsilon'] = 0.0  # no noise during evaluation
    eval_args['num_searches'] = args.get('arena_num_searches',
                                          args.get('num_searches', 60))

    mcts_new = MCTS(game, eval_args, model_new)
    mcts_old = MCTS(game, eval_args, model_old)

    if verbose:
        print(f"\nArena: {num_games} games, threshold={win_threshold:.0%}")

    model_new.eval()
    model_old.eval()

    w_new, w_old, draws = play_match(
        game, mcts_new, mcts_old, num_games, verbose
    )

    if verbose:
        print(f"Results: New={w_new}, Old={w_old}, Draws={draws}")

    total_decisive = w_new + w_old
    if total_decisive == 0:
        return False
    return (w_new / total_decisive) >= win_threshold
