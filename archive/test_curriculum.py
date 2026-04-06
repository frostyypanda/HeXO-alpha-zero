"""Quick test: do wins happen at various curriculum settings with random play?"""
import sys
import time
import numpy as np
sys.path.insert(0, '.')

from hexo.game import HeXOGame


def test_random_games(win_length, dist, max_moves, num_games=50):
    """Play random games and report win rate."""
    game = HeXOGame(max_placement_dist=dist, win_length=win_length)
    wins = 0
    draws = 0
    total_moves = []

    for _ in range(num_games):
        state = game.get_initial_state()
        for move_num in range(max_moves):
            valid = game.get_valid_moves(state)
            legal = np.where(valid > 0)[0]
            if len(legal) == 0:
                break
            action = np.random.choice(legal)
            state = game.get_next_state(state, action, state.current_player)
            if state.winner != 0:
                wins += 1
                total_moves.append(move_num + 1)
                break
        else:
            draws += 1
            total_moves.append(max_moves)

    avg_moves = np.mean(total_moves)
    print(f"  win={win_length} dist={dist} max_moves={max_moves}: "
          f"{wins}/{num_games} wins ({100*wins/num_games:.0f}%), "
          f"avg moves={avg_moves:.1f}")
    return wins / num_games


def test_mcts_games(win_length, dist, max_moves, num_sims, num_games=3):
    """Play a few MCTS games with random model to check timing + wins."""
    import torch
    from alphazero.model import ResNet
    from alphazero.mcts import MCTS

    game = HeXOGame(max_placement_dist=dist, win_length=win_length)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(game, num_resBlocks=10, num_hidden=128, device=device)
    model.eval()

    mcts_args = {
        'C': 2,
        'num_searches': num_sims,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
        'temperature': 1.25,
    }
    mcts = MCTS(game, mcts_args, model)

    wins = 0
    for g in range(num_games):
        state = game.get_initial_state()
        t0 = time.time()
        for move_num in range(max_moves):
            action_probs = mcts.search(state)
            temp_probs = action_probs ** (1.0 / 1.25)
            t = temp_probs.sum()
            if t > 0:
                temp_probs /= t
            action = np.random.choice(len(temp_probs), p=temp_probs)
            state = game.get_next_state(state, action, state.current_player)
            if state.winner != 0:
                wins += 1
                break
        elapsed = time.time() - t0
        result = "WIN" if state.winner != 0 else "DRAW"
        print(f"  Game {g+1}: {move_num+1} moves, {elapsed:.1f}s [{result}]")

    return wins


print("=== Random play win rates (50 games each) ===")
configs = [
    (3, 1, 20), (3, 2, 24), (3, 3, 30),
    (4, 2, 40), (4, 3, 50), (4, 4, 60),
    (5, 3, 80), (5, 5, 120),
]
for wl, d, mm in configs:
    test_random_games(wl, d, mm)

print("\n=== MCTS games with random model (timing + win check) ===")
mcts_configs = [
    (3, 2, 24, 50),
    (4, 3, 50, 60),
]
for wl, d, mm, sims in mcts_configs:
    print(f"\nwin={wl} dist={d} sims={sims}:")
    test_mcts_games(wl, d, mm, sims, num_games=3)
