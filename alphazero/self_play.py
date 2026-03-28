"""
Parallel self-play worker for HeXO AlphaZero.

Each worker runs MCTS games on CPU, sending neural network evaluation
requests to the GPU inference server via queues.
"""

import numpy as np
import math
import time

from hexo.game import HeXOGame


class RemoteMCTSNode:
    """MCTS node with lazy expansion and remote NN evaluation."""

    __slots__ = ('game', 'args', 'state', 'parent', 'action_taken',
                 'prior', 'children', 'visit_count', 'value_sum',
                 '_action_priors')

    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0.0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self._action_priors = {}

    def set_policy(self, policy_dict):
        self._action_priors = policy_dict

    def is_leaf(self):
        return not self._action_priors and not self.children

    def select_or_expand(self):
        best_action = None
        best_ucb = -math.inf
        C = self.args['C']
        sqrt_parent = math.sqrt(self.visit_count)

        for action_idx, prior in self._action_priors.items():
            child = self.children.get(action_idx)
            if child is not None:
                if child.visit_count == 0:
                    q_value = 0.0
                else:
                    q_value = 1.0 - ((child.value_sum / child.visit_count) + 1.0) / 2.0
                ucb = q_value + C * prior * sqrt_parent / (1 + child.visit_count)
            else:
                ucb = C * prior * sqrt_parent
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action_idx

        if best_action is None:
            return None

        if best_action not in self.children:
            child_state = self.game.get_next_state(
                self.state, best_action, self.state.current_player
            )
            child = RemoteMCTSNode(
                self.game, self.args, child_state,
                parent=self, action_taken=best_action,
                prior=self._action_priors[best_action],
            )
            self.children[best_action] = child

        return self.children[best_action]

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(self.game.get_opponent_value(value))


def remote_evaluate(state, game, worker_id, request_queue, response_queue):
    """Send state to inference server and get back (policy, value)."""
    encoded = game.get_encoded_state(state)
    valid_moves = game.get_valid_moves(state)
    request_queue.put((worker_id, encoded, valid_moves))
    policy_array, value = response_queue.get()
    return policy_array, value


def mask_and_normalize(policy_array, valid_moves):
    """Mask illegal moves and normalize to dict."""
    masked = policy_array * valid_moves
    total = masked.sum()
    if total <= 0:
        legal = np.where(valid_moves > 0)[0]
        return {int(idx): 1.0 / len(legal) for idx in legal}
    masked /= total
    return {int(idx): float(masked[idx]) for idx in np.where(masked > 0)[0]}


def remote_mcts_search(state, game, args, worker_id, request_queue, response_queue):
    """Run MCTS using remote inference. Returns action_probs array."""
    root = RemoteMCTSNode(game, args, state)
    root.visit_count = 1

    # Evaluate root
    policy_array, _ = remote_evaluate(state, game, worker_id, request_queue, response_queue)
    valid_moves = game.get_valid_moves(state)
    policy = mask_and_normalize(policy_array, valid_moves)

    # Dirichlet noise
    eps = args.get('dirichlet_epsilon', 0.0)
    if eps > 0:
        noise = np.random.dirichlet([args['dirichlet_alpha']] * len(policy))
        noisy = {}
        for i, (aidx, prob) in enumerate(policy.items()):
            noisy[aidx] = (1 - eps) * prob + eps * noise[i]
        policy = noisy

    root.set_policy(policy)

    # Simulations
    num_searches = args.get('num_searches', 800)
    for _ in range(num_searches):
        node = root
        while not node.is_leaf():
            node = node.select_or_expand()
            if node is None:
                break
        if node is None:
            continue

        value, is_terminal = game.get_value_and_terminated(node.state, node.action_taken)
        if not is_terminal:
            pa, value = remote_evaluate(
                node.state, game, worker_id, request_queue, response_queue
            )
            vm = game.get_valid_moves(node.state)
            pol = mask_and_normalize(pa, vm)
            node.set_policy(pol)

        node.backpropagate(value)

    # Collect visit counts
    action_size = game.get_action_size(state)
    action_probs = np.zeros(action_size, dtype=np.float32)
    for action_idx, child in root.children.items():
        action_probs[action_idx] = child.visit_count
    total = action_probs.sum()
    if total > 0:
        action_probs /= total
    return action_probs


def self_play_worker(worker_id, game_args, mcts_args, request_queue,
                     response_queue, result_queue, num_games, stop_event):
    """
    Self-play worker process. Plays `num_games` games and puts training
    samples into result_queue.
    """
    game = HeXOGame(max_placement_dist=game_args.get('max_placement_dist', 8),
                    win_length=game_args.get('win_length', 5))
    max_moves = mcts_args.get('max_moves_per_game', 200)

    for game_num in range(num_games):
        if stop_event.is_set():
            break

        memory = []
        state = game.get_initial_state()

        while True:
            if stop_event.is_set():
                break

            player = state.current_player

            action_probs = remote_mcts_search(
                state, game, mcts_args, worker_id,
                request_queue, response_queue
            )

            encoded = game.get_encoded_state(state)
            memory.append((encoded, action_probs, player))

            # Sample with temperature
            temperature = mcts_args.get('temperature', 1.25)
            if temperature > 0 and temperature != 1.0:
                temp_probs = action_probs ** (1.0 / temperature)
                t = temp_probs.sum()
                if t > 0:
                    temp_probs /= t
                else:
                    temp_probs = action_probs
            else:
                temp_probs = action_probs

            action_size = game.get_action_size(state)
            action = np.random.choice(action_size, p=temp_probs)
            state = game.get_next_state(state, action, player)

            value, is_terminal = game.get_value_and_terminated(state, action)
            if is_terminal:
                last_player = state.move_history[-1][2]
                samples = []
                for enc, pi, pl in memory:
                    outcome = value if pl == last_player else -value
                    samples.append((enc, pi, outcome))
                result_queue.put(samples)
                break

            if state.total_moves >= max_moves:
                result_queue.put([(enc, pi, 0.0) for enc, pi, _ in memory])
                break
