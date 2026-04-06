"""Train on win=3/dist=1 then test if it transfers to dist=2."""
import sys
import time
import numpy as np
sys.path.insert(0, '.')

import torch
from hexo.game import HeXOGame
from alphazero.model import ResNet
from alphazero.mcts import MCTS
from alphazero.train import AlphaZero


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Phase 1: Train on dist=1
game1 = HeXOGame(max_placement_dist=1, win_length=3)
model = ResNet(game1, num_resBlocks=10, num_hidden=128, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

args = {
    'C': 2,
    'num_searches': 25,
    'dirichlet_epsilon': 0.5,
    'dirichlet_alpha': 0.5,
    'temperature': 1.5,
    'max_moves_per_game': 20,
    'batch_size': 64,
}

trainer = AlphaZero(model, optimizer, game1, args)

print("=== Training on win=3, dist=1 (15 iterations) ===")
for iteration in range(15):
    t0 = time.time()
    memory = []
    wins = 0
    model.eval()
    for g in range(15):
        samples = trainer.self_play()
        memory.extend(samples)
        if any(abs(s[2]) > 0.5 for s in samples):
            wins += 1
    model.train()
    for epoch in range(4):
        losses = trainer.train(memory)
    elapsed = time.time() - t0
    print(f"  Iter {iteration}: {wins}/15 wins, "
          f"p={losses['policy_loss']:.4f} v={losses['value_loss']:.4f} ({elapsed:.1f}s)")

# Now test on dist=2 with the trained model
print("\n=== Testing trained model on win=3, dist=2 ===")
game2 = HeXOGame(max_placement_dist=2, win_length=3)
model.eval()

for num_sims in [25, 50]:
    mcts_args = {
        'C': 2,
        'num_searches': num_sims,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
        'temperature': 1.25,
    }
    mcts = MCTS(game2, mcts_args, model)

    wins = 0
    for g in range(10):
        state = game2.get_initial_state()
        for move_num in range(30):
            action_probs = mcts.search(state)
            temp_probs = action_probs ** (1.0 / 1.25)
            t = temp_probs.sum()
            if t > 0:
                temp_probs /= t
            action = np.random.choice(len(temp_probs), p=temp_probs)
            state = game2.get_next_state(state, action, state.current_player)
            if state.winner != 0:
                wins += 1
                break

    print(f"  sims={num_sims}: {wins}/10 wins")

# Also test on win=4, dist=2
print("\n=== Testing trained model on win=4, dist=2 ===")
game3 = HeXOGame(max_placement_dist=2, win_length=4)
for num_sims in [25, 50]:
    mcts_args = {
        'C': 2,
        'num_searches': num_sims,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
        'temperature': 1.25,
    }
    mcts = MCTS(game3, mcts_args, model)

    wins = 0
    for g in range(10):
        state = game3.get_initial_state()
        for move_num in range(40):
            action_probs = mcts.search(state)
            temp_probs = action_probs ** (1.0 / 1.25)
            t = temp_probs.sum()
            if t > 0:
                temp_probs /= t
            action = np.random.choice(len(temp_probs), p=temp_probs)
            state = game3.get_next_state(state, action, state.current_player)
            if state.winner != 0:
                wins += 1
                break

    print(f"  sims={num_sims}: {wins}/10 wins")
