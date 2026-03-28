"""
HeXO game logic — Infinity Hexagonal Tic-Tac-Toe.

Rules:
  - Infinite hex grid, axial coordinates (q, r).
  - Player 1 places 1 hex (opening), then Player 2 places 2, then alternating 2 each.
  - Win: 6 in a row on any of the 3 hex axes.
  - New hex must be within distance 8 of any existing hex.

Compatible with the AlphaZero framework interface from the reference notebook.

Board representation: **Dynamic Bounding Box + Fully Convolutional Network (FCN).**
Instead of a fixed-size window, the board is encoded as a tight bounding box around
all placed pieces plus a padding of `max_placement_dist` (8) cells in each direction.
This guarantees:
  - 0% piece loss — every placed stone is always in the grid.
  - All legal moves are within the grid — the +8 padding covers the placement rule.
  - The grid grows organically with the game — small early, larger if pieces spread.

The NN must be fully convolutional (no FC layers in the body) to accept variable
spatial dimensions. The policy head outputs (1, H, W) logits; the value head uses
global average pooling to produce a fixed-size output regardless of grid size.
"""

import numpy as np
from hexo.hex_utils import (
    hex_distance, check_line_through, bounding_box, cells_within_distance,
    hex_neighbors,
)


# Padding added around the bounding box of all pieces.
# Equal to max_placement_dist so all legal moves are inside the grid.
GRID_PADDING = 8


class HeXOState:
    """Mutable game state for HeXO. Tracks all placed stones and turn info."""

    __slots__ = ('stones_p1', 'stones_p2', 'all_stones', 'move_history',
                 'current_player', 'moves_left_in_turn', 'total_moves', 'winner')

    def __init__(self):
        self.stones_p1 = set()       # set of (q, r) for player 1
        self.stones_p2 = set()       # set of (q, r) for player -1
        self.all_stones = set()      # union of both
        self.move_history = []       # list of (q, r, player)
        self.current_player = 1      # 1 or -1
        self.moves_left_in_turn = 1  # player 1 starts with 1 move
        self.total_moves = 0         # total individual hex placements
        self.winner = 0              # 0 = ongoing, 1 or -1 = that player won

    def copy(self):
        s = HeXOState()
        s.stones_p1 = set(self.stones_p1)
        s.stones_p2 = set(self.stones_p2)
        s.all_stones = set(self.all_stones)
        s.move_history = list(self.move_history)
        s.current_player = self.current_player
        s.moves_left_in_turn = self.moves_left_in_turn
        s.total_moves = self.total_moves
        s.winner = self.winner
        return s

    def get_stones(self, player):
        return self.stones_p1 if player == 1 else self.stones_p2


class HeXOGame:
    """
    HeXO game implementing the AlphaZero-compatible interface.

    The infinite board is encoded via a dynamic bounding box: the tight bounding
    box of all placed pieces, padded by `max_placement_dist` cells in every
    direction. This guarantees all pieces and all legal moves are in the grid.

    The grid dimensions (H, W) vary per state. The NN must be fully convolutional
    to handle this. For batching, states are padded to the max (H, W) in the batch.

    Parameters:
        max_placement_dist: max hex distance from any existing stone for a new
                           placement (default 8, per HeXO rules).
    """

    def __init__(self, max_placement_dist=8, win_length=6):
        self.max_placement_dist = max_placement_dist
        self.win_length = win_length

    # -------------------------------------------------------------------------
    # Dynamic grid helpers
    # -------------------------------------------------------------------------

    def get_grid_bounds(self, state):
        """Compute the grid origin and size for encoding this state.

        Returns (origin_q, origin_r, height, width) where:
          - origin_q, origin_r: axial coords of the top-left cell in the grid
          - height, width: dimensions of the grid

        For an empty board, returns a default 17x17 grid centered on (0,0)
        (= 1 cell + 8 padding on each side).
        """
        pad = self.max_placement_dist
        if not state.all_stones:
            # First move: center on origin, pad for legal moves
            return (-pad, -pad, 2 * pad + 1, 2 * pad + 1)

        min_q, max_q, min_r, max_r = bounding_box(state.all_stones)
        origin_q = min_q - pad
        origin_r = min_r - pad
        height = (max_q - min_q) + 2 * pad + 1
        width = (max_r - min_r) + 2 * pad + 1
        return (origin_q, origin_r, height, width)

    def _axial_to_grid(self, origin_q, origin_r, q, r):
        """Convert axial (q, r) to grid indices (gq, gr)."""
        return (q - origin_q, r - origin_r)

    def _grid_to_axial(self, origin_q, origin_r, gq, gr):
        """Convert grid indices (gq, gr) to axial (q, r)."""
        return (gq + origin_q, gr + origin_r)

    # -------------------------------------------------------------------------
    # AlphaZero interface methods
    # -------------------------------------------------------------------------

    def get_initial_state(self):
        """Return a fresh HeXOState."""
        return HeXOState()

    def get_next_state(self, state, action, player):
        """Place a hex for `player` at the action index. Returns the new state.

        `action` is an index into the flattened grid for the CURRENT state's
        bounding box. We convert it to axial coords using the current grid origin.
        """
        state = state.copy()
        oq, orr, h, w = self.get_grid_bounds(state)
        gq = action // w
        gr = action % w
        q, r = self._grid_to_axial(oq, orr, gq, gr)
        self._place_stone(state, q, r, player)
        return state

    def get_valid_moves(self, state):
        """Return a flat array of shape (H*W,) with 1 for legal moves.

        The array size matches the current state's dynamic grid dimensions.
        """
        oq, orr, h, w = self.get_grid_bounds(state)
        valid = np.zeros(h * w, dtype=np.uint8)

        if state.winner != 0:
            return valid

        legal_axial = self._get_legal_axial(state)

        for q, r in legal_axial:
            gq, gr = self._axial_to_grid(oq, orr, q, r)
            if 0 <= gq < h and 0 <= gr < w:
                valid[gq * w + gr] = 1

        return valid

    def get_action_size(self, state):
        """Return the action space size for this state (H * W of its grid)."""
        _, _, h, w = self.get_grid_bounds(state)
        return h * w

    def check_win(self, state, action):
        """Check if the last action was a winning move."""
        if not state.move_history:
            return False
        q, r, player = state.move_history[-1]
        stones = state.get_stones(player)
        return check_line_through(stones, q, r)

    def get_value_and_terminated(self, state, action):
        """Return (value, is_terminated).

        value is from the perspective of the player who just moved.
        +1 = win for that player, 0 = ongoing or draw.
        """
        if state.winner != 0:
            last_player = state.move_history[-1][2]
            value = 1 if state.winner == last_player else -1
            return value, True

        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        """Perspective is handled in encoding — return state as-is."""
        return state

    def get_encoded_state(self, state):
        """Encode the board as a (3, H, W) float32 array.

        Channel 0: current player's stones
        Channel 1: opponent's stones
        Channel 2: empty cells (1 where empty, 0 where occupied)

        H and W are determined by the dynamic bounding box.
        """
        oq, orr, h, w = self.get_grid_bounds(state)
        encoded = np.zeros((3, h, w), dtype=np.float32)
        player = state.current_player

        for q, r in state.get_stones(player):
            gq, gr = self._axial_to_grid(oq, orr, q, r)
            encoded[0, gq, gr] = 1.0

        for q, r in state.get_stones(-player):
            gq, gr = self._axial_to_grid(oq, orr, q, r)
            encoded[1, gq, gr] = 1.0

        encoded[2] = 1.0 - encoded[0] - encoded[1]
        return encoded

    def get_encoded_state_for_player(self, state, player):
        """Encode from a specific player's perspective (for training data)."""
        oq, orr, h, w = self.get_grid_bounds(state)
        encoded = np.zeros((3, h, w), dtype=np.float32)

        for q, r in state.get_stones(player):
            gq, gr = self._axial_to_grid(oq, orr, q, r)
            encoded[0, gq, gr] = 1.0

        for q, r in state.get_stones(-player):
            gq, gr = self._axial_to_grid(oq, orr, q, r)
            encoded[1, gq, gr] = 1.0

        encoded[2] = 1.0 - encoded[0] - encoded[1]
        return encoded

    # -------------------------------------------------------------------------
    # Turn structure helpers
    # -------------------------------------------------------------------------

    def get_moves_left_in_turn(self, state):
        """How many more placements the current player has this turn."""
        return state.moves_left_in_turn

    # -------------------------------------------------------------------------
    # Coordinate conversion (for human input / display)
    # -------------------------------------------------------------------------

    def axial_to_action(self, state, q, r):
        """Convert axial coordinates to a flat action index."""
        oq, orr, h, w = self.get_grid_bounds(state)
        gq, gr = self._axial_to_grid(oq, orr, q, r)
        if 0 <= gq < h and 0 <= gr < w:
            return gq * w + gr
        return -1  # out of grid

    def action_to_axial(self, state, action):
        """Convert a flat action index to axial coordinates."""
        oq, orr, h, w = self.get_grid_bounds(state)
        gq = action // w
        gr = action % w
        return self._grid_to_axial(oq, orr, gq, gr)

    # -------------------------------------------------------------------------
    # Core game logic
    # -------------------------------------------------------------------------

    def _get_legal_axial(self, state):
        """Return the set of legal (q, r) placements for the current player."""
        if not state.all_stones:
            return {(0, 0)}

        candidates = set()
        for sq, sr in state.all_stones:
            for cell in cells_within_distance(sq, sr, self.max_placement_dist):
                if cell not in state.all_stones:
                    candidates.add(cell)

        return candidates

    def _place_stone(self, state, q, r, player):
        """Place stone at (q, r) for player and update turn structure."""
        state.get_stones(player).add((q, r))
        state.all_stones.add((q, r))
        state.move_history.append((q, r, player))
        state.total_moves += 1
        state.moves_left_in_turn -= 1

        if check_line_through(state.get_stones(player), q, r, self.win_length):
            state.winner = player

        if state.moves_left_in_turn == 0:
            state.current_player = -player
            state.moves_left_in_turn = 2

    def is_valid_placement(self, state, q, r):
        """Check if (q, r) is a valid placement in the current state."""
        if (q, r) in state.all_stones:
            return False
        if not state.all_stones:
            return True
        for sq, sr in state.all_stones:
            if hex_distance(q, r, sq, sr) <= self.max_placement_dist:
                return True
        return False

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def render(self, state, highlight=None):
        """Render the board as a string for terminal display."""
        if not state.all_stones:
            return "(empty board)"

        highlight = highlight or set()
        min_q, max_q, min_r, max_r = bounding_box(state.all_stones)
        min_q -= 1; max_q += 1; min_r -= 1; max_r += 1

        lines = []
        header = "     "
        for r in range(min_r, max_r + 1):
            header += f"{r:>3} "
        lines.append(header)

        for q in range(min_q, max_q + 1):
            indent = " " * ((q - min_q) % 2 * 2)
            row_str = f"{q:>3}  {indent}"
            for r in range(min_r, max_r + 1):
                pos = (q, r)
                if pos in state.stones_p1:
                    ch = "[X]" if pos in highlight else " X "
                elif pos in state.stones_p2:
                    ch = "[O]" if pos in highlight else " O "
                else:
                    ch = " . "
                row_str += f"{ch} "
            lines.append(row_str)

        p = "P1 (X)" if state.current_player == 1 else "P2 (O)"
        oq, orr, h, w = self.get_grid_bounds(state)
        lines.append(f"\nTurn: {p} | Moves left this turn: {state.moves_left_in_turn}"
                      f" | Total placements: {state.total_moves}"
                      f" | Grid: {h}x{w}")
        if state.winner != 0:
            w_name = "P1 (X)" if state.winner == 1 else "P2 (O)"
            lines.append(f"*** {w_name} WINS! ***")

        return "\n".join(lines)
