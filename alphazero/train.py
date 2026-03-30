"""
AlphaZero training loop for HeXO.

Single-threaded version: self-play generates games, then trains the model.
For parallel training, see pipeline.py which orchestrates multiple workers.

Handles variable-size states by using custom collation for batching.
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from hexo.game import HeXOGame
from alphazero.model import ResNet, collate_states, uncollate_policy
from alphazero.mcts import MCTS


class AlphaZero:
    """
    AlphaZero training loop adapted for HeXO.

    Handles:
      - HeXO turn structure (1/2/2/2 placements per turn).
      - Variable-size board encoding (dynamic bounding box).
      - Custom batching for variable-size states during training.
    """

    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def self_play(self):
        """
        Play one complete game using MCTS and return training samples.

        Returns:
            list of (encoded_state, action_probs, outcome) tuples.
            encoded_state: (3, H, W) numpy array.
            action_probs: (H*W,) numpy array (from MCTS visit counts).
            outcome: float in {-1, 0, 1} from that player's perspective.
        """
        memory = []
        state = self.game.get_initial_state()

        max_moves = self.args.get('max_moves_per_game', 200)

        while True:
            player = state.current_player

            # MCTS search from current player's perspective
            action_probs = self.mcts.search(state)

            # Store for training: (state, pi, player) — we'll fill in value later
            encoded = self.game.get_encoded_state(state)
            memory.append((encoded, action_probs, player))

            # Sample action with temperature
            temperature = self.args.get('temperature', 1.25)
            if temperature > 0 and temperature != 1.0:
                temp_probs = action_probs ** (1.0 / temperature)
                total = temp_probs.sum()
                if total > 0:
                    temp_probs /= total
                else:
                    temp_probs = action_probs
            else:
                temp_probs = action_probs

            action_size = self.game.get_action_size(state)
            action = np.random.choice(action_size, p=temp_probs)

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                # Assign outcomes: value is from perspective of player who just won
                last_player = state.move_history[-1][2]
                return_memory = []
                for hist_encoded, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == last_player else -value
                    return_memory.append((hist_encoded, hist_action_probs, hist_outcome))
                return return_memory

            # Safety: cap game length to avoid infinite games
            if state.total_moves >= max_moves:
                # Draw — value 0 for everyone
                return [(enc, pi, 0.0) for enc, pi, _ in memory]

    def train(self, memory):
        """
        Train the model on collected self-play data.

        Args:
            memory: list of (encoded_state, action_probs, value) tuples.
                    States may have different sizes.
        """
        # Filter oversized states to prevent VRAM spikes from padding
        max_dim = 60
        memory = [s for s in memory if s[0].shape[1] <= max_dim and s[0].shape[2] <= max_dim]
        random.shuffle(memory)
        batch_size = self.args.get('batch_size', 64)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for batch_idx in range(0, len(memory), batch_size):
            batch = memory[batch_idx:batch_idx + batch_size]
            if len(batch) < 2:
                continue

            states, policy_targets, value_targets = zip(*batch)

            # Collate variable-size states into padded batch
            batch_tensor, batch_valid, orig_sizes = collate_states(
                list(states), list(policy_targets)  # policy_targets used for size info
            )
            batch_tensor = batch_tensor.to(self.model.device)
            batch_valid = batch_valid.to(self.model.device)

            # Forward pass
            out_policy, out_value = self.model(batch_tensor, batch_valid)

            # Remap policy targets to padded space (vectorized)
            max_H, max_W = batch_tensor.shape[2], batch_tensor.shape[3]
            padded_targets = torch.zeros(
                len(batch), max_H * max_W,
                dtype=torch.float32, device=self.model.device
            )
            for i, (pi, (Hi, Wi)) in enumerate(zip(policy_targets, orig_sizes)):
                pi_arr = np.asarray(pi)
                nz = np.flatnonzero(pi_arr)
                if len(nz) > 0:
                    rows = nz // Wi
                    cols = nz % Wi
                    new_idx = rows * max_W + cols
                    padded_targets[i, new_idx] = torch.from_numpy(
                        pi_arr[nz].astype(np.float32)).to(self.model.device)

            value_targets_t = torch.tensor(
                np.array(value_targets, dtype=np.float32).reshape(-1, 1),
                dtype=torch.float32, device=self.model.device
            )

            # Losses
            # Manual cross-entropy: log_softmax handles -inf correctly,
            # but 0 * (-inf) = NaN in IEEE 754, so we nan_to_num.
            log_probs = F.log_softmax(out_policy, dim=1)
            policy_loss = -(torch.nan_to_num(padded_targets * log_probs, nan=0.0)
                            ).sum(dim=1).mean()
            value_loss = F.mse_loss(out_value, value_targets_t)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        if num_batches > 0:
            return {
                'policy_loss': total_policy_loss / num_batches,
                'value_loss': total_value_loss / num_batches,
            }
        return {'policy_loss': 0.0, 'value_loss': 0.0}

    def learn(self):
        """
        Full AlphaZero learning loop: iterate self-play -> training -> checkpoint.
        """
        save_dir = self.args.get('save_dir', 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)

        num_iterations = self.args.get('num_iterations', 10)
        num_self_play = self.args.get('num_selfPlay_iterations', 100)
        num_epochs = self.args.get('num_epochs', 4)

        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")

            # --- Self-play ---
            memory = []
            self.model.eval()
            print(f"Self-play: generating {num_self_play} games...")
            for sp in trange(num_self_play, desc="Self-play"):
                memory += self.self_play()

            print(f"Collected {len(memory)} training samples.")

            # --- Training ---
            self.model.train()
            for epoch in trange(num_epochs, desc="Training"):
                losses = self.train(memory)

            print(f"Final losses — policy: {losses['policy_loss']:.4f}, "
                  f"value: {losses['value_loss']:.4f}")

            # --- Checkpoint ---
            model_path = os.path.join(save_dir, f"model_{iteration}.pt")
            optim_path = os.path.join(save_dir, f"optimizer_{iteration}.pt")
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.optimizer.state_dict(), optim_path)
            print(f"Saved checkpoint: {model_path}")
