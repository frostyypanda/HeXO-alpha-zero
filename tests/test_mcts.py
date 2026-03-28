"""Unit tests for MCTS."""

import numpy as np
import pytest
import torch

from hexo.game import HeXOGame
from alphazero.model import ResNet
from alphazero.mcts import MCTS, Node


@pytest.fixture
def game():
    return HeXOGame()


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def model(game, device):
    m = ResNet(game, num_resBlocks=2, num_hidden=32, device=device, num_groups=8)
    m.eval()
    return m


@pytest.fixture
def mcts_args():
    return {
        'C': 2,
        'num_searches': 10,  # low for fast tests
        'dirichlet_epsilon': 0.0,
        'dirichlet_alpha': 0.3,
    }


class TestMCTS:
    def test_search_returns_valid_probs(self, game, model, mcts_args):
        mcts = MCTS(game, mcts_args, model)
        state = game.get_initial_state()
        probs = mcts.search(state)
        action_size = game.get_action_size(state)
        assert probs.shape == (action_size,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_search_only_legal_moves(self, game, model, mcts_args):
        mcts = MCTS(game, mcts_args, model)
        state = game.get_initial_state()
        probs = mcts.search(state)
        valid = game.get_valid_moves(state)
        # All probability should be on legal moves
        illegal_prob = probs[valid == 0].sum()
        assert illegal_prob < 1e-6

    def test_search_after_one_move(self, game, model, mcts_args):
        mcts = MCTS(game, mcts_args, model)
        state = game.get_initial_state()
        state = game.get_next_state(state, game.axial_to_action(state, 0, 0), 1)
        probs = mcts.search(state)
        action_size = game.get_action_size(state)
        assert probs.shape == (action_size,)
        assert probs.sum() > 0

    def test_dirichlet_noise(self, game, model):
        args_noise = {
            'C': 2,
            'num_searches': 5,
            'dirichlet_epsilon': 0.25,
            'dirichlet_alpha': 0.3,
        }
        mcts = MCTS(game, args_noise, model)
        state = game.get_initial_state()
        # Should not crash with noise enabled
        probs = mcts.search(state)
        assert probs.sum() > 0

    def test_time_limited_search(self, game, model):
        args_time = {
            'C': 2,
            'time_limit_ms': 100,  # 100ms
            'dirichlet_epsilon': 0.0,
            'dirichlet_alpha': 0.3,
        }
        mcts = MCTS(game, args_time, model)
        state = game.get_initial_state()
        probs = mcts.search(state)
        assert probs.sum() > 0


class TestNode:
    def test_ucb_unvisited_child(self, game, mcts_args):
        state = game.get_initial_state()
        parent = Node(game, mcts_args, state)
        parent.visit_count = 10
        child = Node(game, mcts_args, state, parent=parent, prior=0.5)
        ucb = parent._ucb(child)
        # Unvisited child should get high UCB due to prior
        assert ucb > 0

    def test_select_prefers_unvisited(self, game, mcts_args):
        state = game.get_initial_state()
        parent = Node(game, mcts_args, state)
        parent.visit_count = 5
        c1 = Node(game, mcts_args, state, parent=parent, prior=0.3)
        c1.visit_count = 3
        c1.value_sum = 0.5
        c2 = Node(game, mcts_args, state, parent=parent, prior=0.3)
        c2.visit_count = 0  # unvisited
        parent.children = [c1, c2]
        selected = parent.select()
        assert selected is c2  # should prefer unvisited


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
