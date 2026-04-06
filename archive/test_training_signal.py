"""Quick training test: does value loss appear with win=3, dist=1?"""
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

game = HeXOGame(max_placement_dist=1, win_length=3)
model = ResNet(game, num_resBlocks=10, num_hidden=128, device=device)
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

trainer = AlphaZero(model, optimizer, game, args)

for iteration in range(5):
    t0 = time.time()
    memory = []
    wins = 0
    model.eval()
    for g in range(15):
        samples = trainer.self_play()
        memory.extend(samples)
        # Check if game had a winner (non-zero outcome)
        outcomes = [s[2] for s in samples]
        if any(abs(o) > 0.5 for o in outcomes):
            wins += 1

    model.train()
    for epoch in range(4):
        losses = trainer.train(memory)

    elapsed = time.time() - t0
    print(f"Iter {iteration}: {len(memory)} samples, {wins}/15 wins, "
          f"p_loss={losses['policy_loss']:.4f}, v_loss={losses['value_loss']:.4f}, "
          f"{elapsed:.1f}s")
