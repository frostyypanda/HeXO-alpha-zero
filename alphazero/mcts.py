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
  - **Lazy expansion**: child states are only created when first visited,
    not for all legal moves at once. Critical for HeXO where 300+ moves
    may be legal but only a handful are visited per search.
"""

import math
import time
import numpy as np
import torch


class Node:
    """A node in the MCTS tree. Uses lazy expansion for efficiency."""

    __slots__ = ('game', 'args', 'state', 'parent', 'action_taken',
                 'prior', 'children', 'visit_count', 'value_sum',
                 '_unexpanded_actions', '_action_priors')

    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0.0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = {}           # action_idx -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self._unexpanded_actions = []  # list of (action_idx, prior)
        self._action_priors = {}       # action_idx -> prior (for UCB on unexpanded)

    def set_policy(self, policy_dict):
        """Store the policy for lazy expansion.
        Args:
            policy_dict: {action_idx: probability}
        """
        self._action_priors = policy_dict
        self._unexpanded_actions = list(policy_dict.keys())

    def is_leaf(self):
        """True if this node has not been evaluated by the NN yet."""
        return not self._action_priors and not self.children

    def select_or_expand(self):
        """Select best child by UCB, creating it lazily if needed.
        Returns the selected child node."""
        best_action = None
        best_ucb = -math.inf
        C = self.args['C']
        sqrt_parent = math.sqrt(self.visit_count)

        # Score all actions (both expanded and unexpanded)
        for action_idx, prior in self._action_priors.items():
            child = self.children.get(action_idx)
            if child is not None:
                if child.visit_count == 0:
                    q_value = 0.0
                else:
                    q_value = 1.0 - ((child.value_sum / child.visit_count) + 1.0) / 2.0
                ucb = q_value + C * prior * sqrt_parent / (1 + child.visit_count)
            else:
                # Unexpanded: visit_count=0, so q=0
                ucb = C * prior * sqrt_parent  # / (1+0) = same
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action_idx

        if best_action is None:
            return None

        # Lazily create the child if not yet expanded
        if best_action not in self.children:
            child_state = self.game.get_next_state(
                self.state, best_action, self.state.current_player
            )
            child = Node(
                self.game, self.args, child_state,
                parent=self, action_taken=best_action,
                prior=self._action_priors[best_action],
            )
            self.children[best_action] = child

        return self.children[best_action]

    def backpropagate(self, value):
        """Propagate the evaluation value up the tree."""
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
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
        policy, value = self._evaluate(state)
        valid_moves = self.game.get_valid_moves(state)
        action_size = len(valid_moves)

        # Mask and normalize
        policy_dict = self._mask_and_normalize(policy, valid_moves)

        # Add Dirichlet noise at root for exploration
        eps = self.args.get('dirichlet_epsilon', 0.0)
        if eps > 0:
            noise = np.random.dirichlet(
                [self.args['dirichlet_alpha']] * len(policy_dict)
            )
            noisy = {}
            for i, (action_idx, prob) in enumerate(policy_dict.items()):
                noisy[action_idx] = (1 - eps) * prob + eps * noise[i]
            policy_dict = noisy

        root.set_policy(policy_dict)

        # --- Run simulations ---
        time_limit_ms = self.args.get('time_limit_ms', None)
        num_searches = self.args.get('num_searches', 800)

        if time_limit_ms is not None:
            deadline = time.perf_counter() + time_limit_ms / 1000.0
            while time.perf_counter() < deadline:
                self._simulate(root)
        else:
            for _ in range(num_searches):
                self._simulate(root)

        # --- Collect visit counts ---
        action_probs = np.zeros(action_size, dtype=np.float32)
        for action_idx, child in root.children.items():
            action_probs[action_idx] = child.visit_count

        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs

    def _simulate(self, root):
        """Run one simulation: select -> expand/evaluate -> backprop."""
        node = root

        # --- Selection: walk down the tree ---
        while not node.is_leaf():
            node = node.select_or_expand()
            if node is None:
                return

        # --- Check terminal ---
        state = node.state
        value, is_terminal = self.game.get_value_and_terminated(state, node.action_taken)

        if is_terminal:
            node.backpropagate(value)
            return

        # --- Evaluate and expand ---
        policy, value = self._evaluate(state)
        valid_moves = self.game.get_valid_moves(state)
        policy_dict = self._mask_and_normalize(policy, valid_moves)
        node.set_policy(policy_dict)

        node.backpropagate(value)

    def _evaluate(self, state):
        """Get policy and value from the model. Returns (policy_np, value_float)."""
        encoded = self.game.get_encoded_state(state)
        tensor = torch.tensor(encoded, dtype=torch.float32,
                              device=self.model.device).unsqueeze(0)
        policy_logits, value = self.model(tensor)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        return policy, value.item()

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
            legal_indices = np.where(valid_moves > 0)[0]
            return {int(idx): 1.0 / len(legal_indices) for idx in legal_indices}

        masked /= total
        result = {}
        for idx in np.where(masked > 0)[0]:
            result[int(idx)] = float(masked[idx])
        return result
