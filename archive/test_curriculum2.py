"""Test MCTS with settings that should produce wins."""
import sys
import time
import numpy as np
sys.path.insert(0, '.')

import torch
from hexo.game import HeXOGame
from alphazero.model import ResNet
from alphazero.mcts import MCTS


def test_mcts(win_length, dist, max_moves, num_sims, temperature,
              dirichlet_eps, num_games=5):
    game = HeXOGame(max_placement_dist=dist, win_length=win_length)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(game, num_resBlocks=10, num_hidden=128, device=device)
    model.eval()

    mcts_args = {
        'C': 2,
        'num_searches': num_sims,
        'dirichlet_epsilon': dirichlet_eps,
        'dirichlet_alpha': 0.5,
        'temperature': temperature,
    }
    mcts = MCTS(game, mcts_args, model)

    wins = 0
    times = []
    for g in range(num_games):
        state = game.get_initial_state()
        t0 = time.time()
        for move_num in range(max_moves):
            action_probs = mcts.search(state)
            temp_probs = action_probs ** (1.0 / temperature)
            t = temp_probs.sum()
            if t > 0:
                temp_probs /= t
            action = np.random.choice(len(temp_probs), p=temp_probs)
            state = game.get_next_state(state, action, state.current_player)
            if state.winner != 0:
                wins += 1
                break
        elapsed = time.time() - t0
        times.append(elapsed)
        result = "WIN" if state.winner != 0 else f"DRAW@{move_num+1}"
        print(f"    G{g+1}: {move_num+1}mv {elapsed:.1f}s [{result}]")

    avg_time = np.mean(times)
    print(f"  => {wins}/{num_games} wins, avg {avg_time:.1f}s/game")
    return wins, avg_time


print("=== Testing MCTS configs that should produce wins ===\n")

# Test 1: dist=1 (96% random win rate), various sims
print("--- win=3, dist=1, max_moves=20 ---")
for sims in [10, 25, 50]:
    print(f"\n  sims={sims}, temp=1.5, noise=0.5:")
    test_mcts(3, 1, 20, sims, temperature=1.5, dirichlet_eps=0.5, num_games=5)

# Test 2: dist=2, high noise
print("\n--- win=3, dist=2, max_moves=30 ---")
for sims in [10, 25]:
    print(f"\n  sims={sims}, temp=2.0, noise=0.5:")
    test_mcts(3, 2, 30, sims, temperature=2.0, dirichlet_eps=0.5, num_games=5)

# Test 3: what about very few sims (almost random)?
print("\n--- win=3, dist=2, sims=5 (near-random MCTS) ---")
print(f"\n  sims=5, temp=2.0, noise=0.75:")
test_mcts(3, 2, 30, 5, temperature=2.0, dirichlet_eps=0.75, num_games=5)
