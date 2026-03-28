"""
Fully Convolutional ResNet for HeXO AlphaZero.

Accepts variable-size (3, H, W) inputs — no FC layers in the body.
Uses GroupNorm instead of BatchNorm (BatchNorm breaks with zero-padded
variable-size batches because padding zeros corrupt running statistics).

Policy head: conv -> spatial logits (B, H*W), masked before softmax.
Value head: conv -> masked global avg pool -> FC -> tanh -> scalar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block: Conv -> GroupNorm -> ReLU -> Conv -> GroupNorm + skip."""

    def __init__(self, num_hidden, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        x += residual
        return F.relu(x)


class ResNet(nn.Module):
    """
    Fully convolutional dual-head ResNet for AlphaZero.

    Args:
        game: HeXOGame instance (used only for config, not stored).
        num_resBlocks: number of residual blocks in the body.
        num_hidden: number of convolutional filters throughout.
        device: torch device.
        num_groups: GroupNorm groups (must divide num_hidden).
    """

    def __init__(self, game, num_resBlocks=10, num_hidden=128,
                 device=None, num_groups=8):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.num_hidden = num_hidden

        # --- Stem ---
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, num_hidden),
            nn.ReLU(),
        )

        # --- Residual backbone ---
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden, num_groups) for _ in range(num_resBlocks)]
        )

        # --- Policy head (fully convolutional) ---
        # Outputs (B, 1, H, W) -> flatten to (B, H*W) action logits
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # --- Value head ---
        # Conv down to 32 channels, then masked global avg pool -> FC -> tanh
        self.valueConv = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
        )
        self.valueFC = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.to(self.device)

    def forward(self, x, valid_moves_mask=None):
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) encoded board states.
            valid_moves_mask: optional (B, H*W) uint8/bool mask. If provided,
                illegal moves get -inf in policy logits. If None, no masking.

        Returns:
            policy: (B, H*W) log-softmax action probabilities.
            value: (B, 1) scalar values in [-1, 1].
        """
        B, C, H, W = x.shape

        # --- Body ---
        out = self.startBlock(x)
        for block in self.backBone:
            out = block(out)

        # --- Policy head ---
        policy = self.policyHead(out)          # (B, 1, H, W)
        policy = policy.view(B, H * W)         # (B, H*W)

        if valid_moves_mask is not None:
            # Mask illegal moves with -inf before softmax
            policy = policy.masked_fill(valid_moves_mask == 0, float('-inf'))

        # --- Value head with masked global average pooling ---
        val = self.valueConv(out)              # (B, 32, H, W)

        # Build spatial mask: a cell is "real" if any of the 3 input channels is nonzero,
        # OR if no mask provided, treat all cells as real.
        # For padded batches, padding cells have all-zero input channels.
        spatial_mask = (x.sum(dim=1, keepdim=True) != 0).float()  # (B, 1, H, W)
        # Ensure at least 1 cell is active (empty board edge case)
        mask_sum = spatial_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        val = (val * spatial_mask).sum(dim=(2, 3)) / mask_sum.squeeze(-1).squeeze(-1)  # (B, 32)

        value = self.valueFC(val)              # (B, 1)

        return policy, value


# ---------------------------------------------------------------------------
# Batching utilities for variable-size states
# ---------------------------------------------------------------------------

def collate_states(encoded_states, valid_moves_list):
    """
    Pad variable-size encoded states and valid-move masks to a common size
    for batched GPU inference.

    Args:
        encoded_states: list of (3, H_i, W_i) numpy arrays.
        valid_moves_list: list of (H_i * W_i,) numpy arrays (uint8).

    Returns:
        batch_tensor: (B, 3, max_H, max_W) float32 tensor.
        batch_valid: (B, max_H * max_W) uint8 tensor with remapped indices.
        orig_sizes: list of (H_i, W_i) tuples for uncollating.
    """
    import numpy as np

    max_H = max(s.shape[1] for s in encoded_states)
    max_W = max(s.shape[2] for s in encoded_states)

    B = len(encoded_states)
    batch_np = np.zeros((B, 3, max_H, max_W), dtype=np.float32)
    batch_valid_np = np.zeros((B, max_H * max_W), dtype=np.uint8)
    orig_sizes = []

    for i, (state, valid) in enumerate(zip(encoded_states, valid_moves_list)):
        _, Hi, Wi = state.shape
        orig_sizes.append((Hi, Wi))
        # Place state in top-left corner
        batch_np[i, :, :Hi, :Wi] = state

        # Remap valid moves: index j in (Hi*Wi) -> row*max_W + col in (max_H*max_W)
        valid_indices = np.where(valid > 0)[0]
        for idx in valid_indices:
            row = idx // Wi
            col = idx % Wi
            new_idx = row * max_W + col
            batch_valid_np[i, new_idx] = 1

    batch_tensor = torch.from_numpy(batch_np)
    batch_valid = torch.from_numpy(batch_valid_np)
    return batch_tensor, batch_valid, orig_sizes


def uncollate_policy(policy_logits, orig_sizes):
    """
    Extract per-state policy from padded batch output.

    Args:
        policy_logits: (B, max_H * max_W) tensor (log-softmax or raw logits).
        orig_sizes: list of (H_i, W_i) from collate_states.

    Returns:
        list of (H_i * W_i,) numpy arrays with remapped probabilities.
    """
    import numpy as np

    B = policy_logits.shape[0]
    max_W = max(w for _, w in orig_sizes)
    results = []

    for i in range(B):
        Hi, Wi = orig_sizes[i]
        out = np.zeros(Hi * Wi, dtype=np.float32)
        for row in range(Hi):
            src_start = row * max_W
            dst_start = row * Wi
            out[dst_start:dst_start + Wi] = policy_logits[i, src_start:src_start + Wi].detach().cpu().numpy()
        results.append(out)

    return results
