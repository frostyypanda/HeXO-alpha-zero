"""
Interactive HeXO game — supports 3 play modes:
  1. Human vs Human
  2. Human vs AI  (requires a trained model)
  3. AI vs AI     (requires a trained model)

Usage:
  python play.py --mode hvh          # Human vs Human
  python play.py --mode hva --model model.pt   # Human vs AI
  python play.py --mode ava --model model.pt   # AI vs AI
"""

import argparse
import sys
import os

from hexo.game import HeXOGame, HeXOState


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def parse_coord(text):
    """Parse 'q,r' or 'q r' into (int, int). Returns None on failure."""
    text = text.strip().replace(',', ' ')
    parts = text.split()
    if len(parts) != 2:
        return None
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return None


def human_move(game, state):
    """Prompt the human for a move and return the action index."""
    while True:
        player_name = "P1 (X)" if state.current_player == 1 else "P2 (O)"
        coord = input(f"{player_name} — enter move as 'q r' (or 'quit'): ").strip()
        if coord.lower() in ('quit', 'exit', 'q'):
            print("Game aborted.")
            sys.exit(0)

        parsed = parse_coord(coord)
        if parsed is None:
            print("Invalid format. Enter two integers like: 0 1")
            continue

        q, r = parsed
        if not game.is_valid_placement(state, q, r):
            print(f"({q}, {r}) is not a valid placement. Must be empty and within "
                  f"distance {game.max_placement_dist} of an existing hex.")
            continue

        action = game.axial_to_action(state, q, r)
        if action == -1:
            print("That cell is outside the current grid. Try closer to existing pieces.")
            continue

        return action


def ai_move(game, state, mcts):
    """Use MCTS to pick a move. Returns the action index."""
    import numpy as np

    action_probs = mcts.search(state)
    # During play, pick greedily (temperature -> 0)
    action = int(np.argmax(action_probs))
    q, r = game.action_to_axial(state, action)
    player_name = "P1 (X)" if state.current_player == 1 else "P2 (O)"
    print(f"{player_name} [AI] plays: ({q}, {r})")
    return action


def play_game(game, mode, mcts=None):
    """Run a full game loop."""
    state = game.get_initial_state()

    # Determine who is human / AI
    # mode: 'hvh' = both human, 'hva' = p1 human p2 AI, 'ava' = both AI
    def get_move(player):
        if mode == 'hvh':
            return human_move(game, state)
        elif mode == 'hva':
            if player == 1:
                return human_move(game, state)
            else:
                return ai_move(game, state, mcts)
        else:  # ava
            return ai_move(game, state, mcts)

    print("\n=== HeXO — Infinity Hexagonal Tic-Tac-Toe ===\n")
    print("Coordinate system: axial (q, r). Enter moves as 'q r'.")
    print("First move starts at the origin (0 0).\n")

    while True:
        # Show board
        last_moves = set()
        if state.move_history:
            last_entry = state.move_history[-1]
            last_moves.add((last_entry[0], last_entry[1]))
        print(game.render(state, highlight=last_moves))
        print()

        if state.winner != 0:
            break

        player = state.current_player
        moves_in_turn = state.moves_left_in_turn

        for i in range(moves_in_turn):
            if i > 0:
                # Show updated board between sub-moves
                last_entry = state.move_history[-1]
                last_moves = {(last_entry[0], last_entry[1])}
                print(game.render(state, highlight=last_moves))
                print()
                if state.winner != 0:
                    break

            action = get_move(player)
            state = game.get_next_state(state, action, player)

            if state.winner != 0:
                break

    # Final board
    print(game.render(state))
    print("\nGame over!")


def main():
    parser = argparse.ArgumentParser(description="Play HeXO")
    parser.add_argument('--mode', choices=['hvh', 'hva', 'ava'], default='hvh',
                        help='hvh=Human vs Human, hva=Human vs AI, ava=AI vs AI')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model checkpoint (required for AI modes)')
    parser.add_argument('--num-searches', type=int, default=800,
                        help='MCTS simulations per move (default: 800)')
    parser.add_argument('--time-limit', type=int, default=None,
                        help='Time limit per move in ms (overrides --num-searches)')
    args = parser.parse_args()

    game = HeXOGame()

    mcts = None
    if args.mode in ('hva', 'ava'):
        if args.model is None:
            print("Error: --model is required for AI modes.")
            sys.exit(1)

        # Import AI components (deferred to avoid torch import for hvh mode)
        import torch
        from alphazero.model import ResNet
        from alphazero.mcts import MCTS

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model = ResNet(game, num_resBlocks=10, num_hidden=128, device=device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        mcts_args = {
            'C': 2,
            'num_searches': args.num_searches,
            'dirichlet_epsilon': 0.0,   # no noise during play
            'dirichlet_alpha': 0.3,
        }
        if args.time_limit is not None:
            mcts_args['time_limit_ms'] = args.time_limit

        mcts = MCTS(game, mcts_args, model)

    play_game(game, args.mode, mcts)


if __name__ == '__main__':
    main()
