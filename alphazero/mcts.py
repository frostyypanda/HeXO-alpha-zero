"""
Monte Carlo Tree Search for HeXO AlphaZero.

Key adaptations from the reference notebook:
  - Variable action space size (dynamic grid per state).
  - HeXO turn structure: a "move" in MCTS is a single hex placement.
    The multi-placement turn (1/2/2/2) is handled by the game engine —
    get_next_state advances the internal turn counter. MCTS just sees
    sequential single-hex placements, and the current_player in the state
    tells it whose perspective to evaluate from.
  - Supports both fixed simulation count and time-limited search.
  - Dirichlet noise at root for exploration during training.
"""

import math
import time
import numpy as np
import torch


class Node:
    """A node in the MCTS tree."""

    __slots__ = ('game', 'args', 'state', 'parent', 'action_taken',
                 'prior', 'children', 'visit_count', 'value_sum')

    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0.0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken  # axial (q, r) that led here
        self.prior = prior

        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def select(self):
        """Select the child with the highest UCB score."""
        best_child = None
        best_ucb = -math.inf

        for child in self.children:
            ucb = self._ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def _ucb(self, child):
        """Upper confidence bound: Q + C * prior * sqrt(parent_visits) / (1 + child_visits)."""
        if child.visit_count == 0:
            q_value = 0.0
        else:
            # Value is from child's perspective; we want parent's perspective
            q_value = 1.0 - ((child.value_sum / child.visit_count) + 1.0) / 2.0
        return q_value + self.args['C'] * child.prior * (
            math.sqrt(self.visit_count) / (1 + child.visit_count)
        )

    def expand(self, policy):
        """
        Expand node using the policy distribution.

        Args:
            policy: dict mapping action_index -> probability (only legal moves).
        """
        for action_idx, prob in policy.items():
            if prob > 0:
                child_state = self.game.get_next_state(
                    self.state, action_idx, self.state.current_player
                )
                # Store the axial coords for this action (for readable output)
                q, r = self.game.action_to_axial(self.state, action_idx)
                child = Node(
                    self.game, self.args, child_state,
                    parent=self, action_taken=action_idx, prior=prob,
                )
                self.children.append(child)

    def backpropagate(self, value):
        """Propagate the evaluation value up the tree."""
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            # Flip value for opponent
            self.parent.backpropagate(self.game.get_opponent_value(value))


class MCTS:
    """
    MCTS guided by a neural network.

    Args:
        game: HeXOGame instance.
        args: dict with keys:
            C: exploration constant (default 2).
            num_searches: number of simulations (used if time_limit_ms not set).
            time_limit_ms: optional time budget in milliseconds.
            dirichlet_epsilon: noise weight at root (0 = no noise).
            dirichlet_alpha: Dirichlet concentration parameter.
        model: ResNet instance.
    """

    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        """
        Run MCTS from the given state and return action probabilities.

        Returns:
            action_probs: numpy array of shape (action_size,) — visit count
                distribution over the current state's action space.
        """
        root = Node(self.game, self.args, state)
        root.visit_count = 1  # virtual visit for UCB denominator

        # --- Evaluate root ---
        policy = self._get_policy(state)
        valid_moves = self.game.get_valid_moves(state)
        action_size = len(valid_moves)

        # Mask and normalize
        policy = self._mask_and_normalize(policy, valid_moves)

        # Add Dirichlet noise at root for exploration
        eps = self.args.get('dirichlet_epsilon', 0.0)
        if eps > 0:
            noise = np.random.dirichlet(
                [self.args['dirichlet_alpha']] * len(policy)
            )
            noisy_policy = {}
            for i, (action_idx, prob) in enumerate(policy.items()):
                noisy_policy[action_idx] = (1 - eps) * prob + eps * noise[i]
            policy = noisy_policy

        root.expand(policy)

        # --- Run simulations ---
        time_limit_ms = self.args.get('time_limit_ms', None)
        num_searches = self.args.get('num_searches', 800)

        if time_limit_ms is not None:
            deadline = time.perf_counter() + time_limit_ms / 1000.0
            sim = 0
            while time.perf_counter() < deadline:
                self._simulate(root)
                sim += 1
        else:
            for _ in range(num_searches):
                self._simulate(root)

        # --- Collect visit counts ---
        action_probs = np.zeros(action_size, dtype=np.float32)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count

        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs

    def _simulate(self, root):
        """Run one simulation: select -> expand/evaluate -> backprop."""
        node = root

        # --- Selection ---
        while node.is_expanded():
            node = node.select()

        # --- Check terminal ---
        state = node.state
        value, is_terminal = self.game.get_value_and_terminated(state, node.action_taken)

        if is_terminal:
            # Value from perspective of the player who just moved (parent's player)
            node.backpropagate(value)
            return

        # --- Expansion ---
        policy = self._get_policy(state)
        valid_moves = self.game.get_valid_moves(state)
        policy = self._mask_and_normalize(policy, valid_moves)

        value = self._get_value(state)
        node.expand(policy)

        # Value is from current player's perspective
        node.backpropagate(value)

    def _get_policy(self, state):
        """Get raw policy logits from the model, return as numpy array."""
        encoded = self.game.get_encoded_state(state)
        tensor = torch.tensor(encoded, dtype=torch.float32,
                              device=self.model.device).unsqueeze(0)
        policy_logits, _ = self.model(tensor)
        return torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

    def _get_value(self, state):
        """Get value estimate from the model."""
        encoded = self.game.get_encoded_state(state)
        tensor = torch.tensor(encoded, dtype=torch.float32,
                              device=self.model.device).unsqueeze(0)
        _, value = self.model(tensor)
        return value.item()

    def _mask_and_normalize(self, policy_array, valid_moves):
        """
        Mask illegal moves and normalize. Returns dict {action_idx: prob}.

        Args:
            policy_array: numpy array of shape (H*W,) with raw probs.
            valid_moves: numpy array of shape (H*W,) with 1 for legal.
        """
        masked = policy_array * valid_moves
        total = masked.sum()
        if total <= 0:
            # Fallback: uniform over legal moves
            legal_indices = np.where(valid_moves > 0)[0]
            return {int(idx): 1.0 / len(legal_indices) for idx in legal_indices}

        masked /= total
        result = {}
        for idx in np.where(masked > 0)[0]:
            result[int(idx)] = float(masked[idx])
        return result
