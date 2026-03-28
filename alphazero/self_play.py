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
    """MCTS node that uses a remote inference server instead of a local model."""

    __slots__ = ('game', 'args', 'state', 'parent', 'action_taken',
                 'prior', 'children', 'visit_count', 'value_sum')

    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0.0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -math.inf
        for child in self.children:
            ucb = self._ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def _ucb(self, child):
        if child.visit_count == 0:
            q_value = 0.0
        else:
            q_value = 1.0 - ((child.value_sum / child.visit_count) + 1.0) / 2.0
        return q_value + self.args['C'] * child.prior * (
            math.sqrt(self.visit_count) / (1 + child.visit_count)
        )

    def expand(self, policy):
        for action_idx, prob in policy.items():
            if prob > 0:
                child_state = self.game.get_next_state(
                    self.state, action_idx, self.state.current_player
                )
                child = RemoteMCTSNode(
                    self.game, self.args, child_state,
                    parent=self, action_taken=action_idx, prior=prob,
                )
                self.children.append(child)

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

    root.expand(policy)

    # Simulations
    num_searches = args.get('num_searches', 800)
    for _ in range(num_searches):
        node = root
        while node.is_expanded():
            node = node.select()

        value, is_terminal = game.get_value_and_terminated(node.state, node.action_taken)
        if not is_terminal:
            pa, value = remote_evaluate(
                node.state, game, worker_id, request_queue, response_queue
            )
            vm = game.get_valid_moves(node.state)
            pol = mask_and_normalize(pa, vm)
            node.expand(pol)

        node.backpropagate(value)

    # Collect visit counts
    action_size = game.get_action_size(state)
    action_probs = np.zeros(action_size, dtype=np.float32)
    for child in root.children:
        action_probs[child.action_taken] = child.visit_count
    total = action_probs.sum()
    if total > 0:
        action_probs /= total
    return action_probs


def self_play_worker(worker_id, game_args, mcts_args, request_queue,
                     response_queue, result_queue, num_games, stop_event):
    """
    Self-play worker process. Plays `num_games` games and puts training
    samples into result_queue.

    Args:
        worker_id: unique int for this worker.
        game_args: dict with game config (max_placement_dist).
        mcts_args: dict with MCTS config (C, num_searches, etc.).
        request_queue: shared queue to send inference requests.
        response_queue: this worker's response queue.
        result_queue: queue to put completed game samples.
        num_games: number of games to play.
        stop_event: multiprocessing.Event to signal shutdown.
    """
    game = HeXOGame(**game_args)

    for game_num in range(num_games):
        if stop_event.is_set():
            break

        memory = []
        state = game.get_initial_state()
        max_moves = mcts_args.get('max_moves_per_game', 200)

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
