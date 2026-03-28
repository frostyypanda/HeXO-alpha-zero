"""Unit tests for the ResNet model and batching utilities."""

import numpy as np
import pytest
import torch

from hexo.game import HeXOGame
from alphazero.model import ResNet, ResBlock, collate_states, uncollate_policy


@pytest.fixture
def game():
    return HeXOGame()


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def model(game, device):
    return ResNet(game, num_resBlocks=2, num_hidden=32, device=device, num_groups=8)


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestResNet:
    def test_output_shapes(self, model, game):
        state = game.get_initial_state()
        encoded = game.get_encoded_state(state)
        x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        policy, value = model(x)
        _, _, H, W = x.shape
        assert policy.shape == (1, H * W)
        assert value.shape == (1, 1)

    def test_value_in_range(self, model, game):
        state = game.get_initial_state()
        encoded = game.get_encoded_state(state)
        x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        _, value = model(x)
        assert -1.0 <= value.item() <= 1.0

    def test_variable_size_input(self, model, game):
        """Model should accept different spatial dimensions."""
        state = game.get_initial_state()
        # 17x17 (empty board)
        enc1 = game.get_encoded_state(state)
        x1 = torch.tensor(enc1, dtype=torch.float32).unsqueeze(0)
        p1, v1 = model(x1)
        assert p1.shape == (1, 17 * 17)

        # Place pieces to grow the grid
        state = game.get_next_state(state, game.axial_to_action(state, 0, 0), 1)
        state = game.get_next_state(state, game.axial_to_action(state, 7, 0), -1)
        enc2 = game.get_encoded_state(state)
        x2 = torch.tensor(enc2, dtype=torch.float32).unsqueeze(0)
        p2, v2 = model(x2)
        _, _, H2, W2 = x2.shape
        assert p2.shape == (1, H2 * W2)
        assert H2 > 17 or W2 > 17  # grid grew

    def test_masking(self, model, game):
        """Masked illegal moves should get -inf."""
        state = game.get_initial_state()
        enc = game.get_encoded_state(state)
        valid = game.get_valid_moves(state)
        x = torch.tensor(enc, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(valid, dtype=torch.uint8).unsqueeze(0)
        policy, _ = model(x, mask)
        # Only 1 legal move on first turn (center)
        probs = torch.softmax(policy, dim=1).squeeze(0).detach().numpy()
        assert probs[valid == 1].sum() > 0.99

    def test_batch_forward(self, model, game):
        """Batched forward pass with same-size states."""
        state = game.get_initial_state()
        enc = game.get_encoded_state(state)
        batch = torch.tensor(np.stack([enc, enc]), dtype=torch.float32)
        p, v = model(batch)
        assert p.shape[0] == 2
        assert v.shape[0] == 2


# ---------------------------------------------------------------------------
# Collation tests
# ---------------------------------------------------------------------------

class TestCollation:
    def test_collate_same_size(self, game):
        state = game.get_initial_state()
        enc = game.get_encoded_state(state)
        valid = game.get_valid_moves(state)
        bt, bv, sizes = collate_states([enc, enc], [valid, valid])
        assert bt.shape == (2, 3, 17, 17)
        assert bv.shape == (2, 17 * 17)
        assert sizes == [(17, 17), (17, 17)]

    def test_collate_different_sizes(self, game):
        s1 = game.get_initial_state()
        enc1 = game.get_encoded_state(s1)
        valid1 = game.get_valid_moves(s1)

        s2 = game.get_initial_state()
        s2 = game.get_next_state(s2, game.axial_to_action(s2, 0, 0), 1)
        s2 = game.get_next_state(s2, game.axial_to_action(s2, 7, 0), -1)
        enc2 = game.get_encoded_state(s2)
        valid2 = game.get_valid_moves(s2)

        bt, bv, sizes = collate_states([enc1, enc2], [valid1, valid2])
        H1, W1 = sizes[0]
        H2, W2 = sizes[1]
        max_H = max(H1, H2)
        max_W = max(W1, W2)
        assert bt.shape == (2, 3, max_H, max_W)
        assert bv.shape == (2, max_H * max_W)

    def test_action_index_remap(self, game):
        """Valid move indices should be correctly remapped to padded grid."""
        s1 = game.get_initial_state()
        enc1 = game.get_encoded_state(s1)
        valid1 = game.get_valid_moves(s1)

        s2 = game.get_initial_state()
        s2 = game.get_next_state(s2, game.axial_to_action(s2, 0, 0), 1)
        s2 = game.get_next_state(s2, game.axial_to_action(s2, 7, 0), -1)
        enc2 = game.get_encoded_state(s2)
        valid2 = game.get_valid_moves(s2)

        _, bv, sizes = collate_states([enc1, enc2], [valid1, valid2])
        max_W = max(s[1] for s in sizes)

        # Check that all original valid moves map correctly
        for i, (valid, (Hi, Wi)) in enumerate(zip([valid1, valid2], sizes)):
            for idx in np.where(valid > 0)[0]:
                row = idx // Wi
                col = idx % Wi
                new_idx = row * max_W + col
                assert bv[i, new_idx] == 1, f"Remap failed: state {i}, idx {idx}"

    def test_uncollate_roundtrip(self, game, model):
        """collate -> model -> uncollate should give correct-sized outputs."""
        s1 = game.get_initial_state()
        enc1 = game.get_encoded_state(s1)
        valid1 = game.get_valid_moves(s1)

        bt, bv, sizes = collate_states([enc1], [valid1])
        bt = bt.to(model.device)
        bv = bv.to(model.device)
        policy, _ = model(bt, bv)
        probs = torch.softmax(policy, dim=1)
        result = uncollate_policy(probs, sizes)
        assert len(result) == 1
        assert result[0].shape == (17 * 17,)


# ---------------------------------------------------------------------------
# ResBlock test
# ---------------------------------------------------------------------------

class TestResBlock:
    def test_skip_connection(self):
        block = ResBlock(32, num_groups=8)
        x = torch.randn(1, 32, 10, 10)
        out = block(x)
        assert out.shape == x.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
