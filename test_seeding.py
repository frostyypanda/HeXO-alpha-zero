"""Test: seed model with random game data, then MCTS self-play."""
import sys
import time
import numpy as np
sys.path.insert(0, '.')

import torch
from hexo.game import HeXOGame
from alphazero.model import ResNet
from alphazero.mcts import MCTS
from alphazero.train import AlphaZero


def generate_random_games(game, num_games, max_moves):
    """Generate training data from purely random play."""
    all_samples = []
    wins = 0
    for _ in range(num_games):
        state = game.get_initial_state()
        memory = []
        for move_num in range(max_moves):
            valid = game.get_valid_moves(state)
            legal = np.where(valid > 0)[0]
            if len(legal) == 0:
                break
            # Uniform random policy
            action_probs = np.zeros(len(valid), dtype=np.float32)
            action_probs[legal] = 1.0 / len(legal)

            encoded = game.get_encoded_state(state)
            player = state.current_player
            memory.append((encoded, action_probs, player))

            action = np.random.choice(legal)
            state = game.get_next_state(state, action, state.current_player)

            if state.winner != 0:
                wins += 1
                last_player = state.move_history[-1][2]
                value = 1.0
                for enc, pi, pl in memory:
                    outcome = value if pl == last_player else -value
                    all_samples.append((enc, pi, outcome))
                break
        else:
            # Draw
            for enc, pi, pl in memory:
                all_samples.append((enc, pi, 0.0))

    return all_samples, wins


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Target: win=3, dist=2
game = HeXOGame(max_placement_dist=2, win_length=3)
model = ResNet(game, num_resBlocks=10, num_hidden=128, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

args = {
    'C': 2,
    'num_searches': 50,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'temperature': 1.25,
    'max_moves_per_game': 30,
    'batch_size': 64,
}
trainer = AlphaZero(model, optimizer, game, args)

# Step 1: Generate random game data
print("=== Generating random game data (500 games) ===")
t0 = time.time()
random_data, random_wins = generate_random_games(game, 500, 30)
print(f"  {len(random_data)} samples, {random_wins}/500 wins "
      f"({100*random_wins/500:.0f}%), {time.time()-t0:.1f}s")

# Step 2: Pre-train on random data
print("\n=== Pre-training on random data (8 epochs) ===")
model.train()
for epoch in range(8):
    losses = trainer.train(random_data)
    print(f"  Epoch {epoch}: p={losses['policy_loss']:.4f} v={losses['value_loss']:.4f}")

# Step 3: MCTS self-play with pre-trained model
print("\n=== MCTS self-play with pre-trained model ===")
for iteration in range(10):
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
