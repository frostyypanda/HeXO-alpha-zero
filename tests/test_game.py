"""Unit tests for HeXO game logic."""

import numpy as np
import pytest
from hexo.hex_utils import hex_distance, check_line_through, count_in_direction, hex_neighbors
from hexo.game import HeXOGame, HeXOState


# ---------------------------------------------------------------------------
# hex_utils tests
# ---------------------------------------------------------------------------

class TestHexUtils:
    def test_hex_distance_same(self):
        assert hex_distance(0, 0, 0, 0) == 0

    def test_hex_distance_neighbors(self):
        for nq, nr in hex_neighbors(0, 0):
            assert hex_distance(0, 0, nq, nr) == 1

    def test_hex_distance_farther(self):
        assert hex_distance(0, 0, 3, 0) == 3
        assert hex_distance(0, 0, 3, -3) == 3
        assert hex_distance(0, 0, 2, 2) == 4

    def test_count_in_direction(self):
        stones = {(0, 0), (1, 0), (2, 0), (3, 0)}
        assert count_in_direction(stones, 0, 0, 1, 0) == 3
        assert count_in_direction(stones, 3, 0, -1, 0) == 3
        assert count_in_direction(stones, 0, 0, 0, 1) == 0

    def test_check_line_through_5_not_enough(self):
        stones = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}
        assert check_line_through(stones, 2, 0) is False

    def test_check_line_through_6_wins(self):
        stones = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)}
        assert check_line_through(stones, 3, 0) is True
        assert check_line_through(stones, 0, 0) is True
        assert check_line_through(stones, 5, 0) is True

    def test_check_line_r_axis(self):
        stones = {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)}
        assert check_line_through(stones, 0, 3) is True

    def test_check_line_s_axis(self):
        stones = {(0, 0), (1, -1), (2, -2), (3, -3), (4, -4), (5, -5)}
        assert check_line_through(stones, 2, -2) is True


# ---------------------------------------------------------------------------
# Turn structure tests
# ---------------------------------------------------------------------------

class TestTurnStructure:
    def setup_method(self):
        self.game = HeXOGame()

    def test_initial_state(self):
        state = self.game.get_initial_state()
        assert state.current_player == 1
        assert state.moves_left_in_turn == 1
        assert state.total_moves == 0
        assert state.winner == 0

    def test_first_move_switches_to_p2_with_2_moves(self):
        state = self.game.get_initial_state()
        action = self.game.axial_to_action(state, 0, 0)
        state = self.game.get_next_state(state, action, 1)
        assert state.current_player == -1
        assert state.moves_left_in_turn == 2
        assert state.total_moves == 1

    def test_p2_two_moves_then_p1_gets_two(self):
        state = self.game.get_initial_state()
        action = self.game.axial_to_action(state, 0, 0)
        state = self.game.get_next_state(state, action, 1)

        action = self.game.axial_to_action(state, 1, 0)
        state = self.game.get_next_state(state, action, -1)
        assert state.current_player == -1
        assert state.moves_left_in_turn == 1

        action = self.game.axial_to_action(state, 0, 1)
        state = self.game.get_next_state(state, action, -1)
        assert state.current_player == 1
        assert state.moves_left_in_turn == 2

    def test_full_turn_cycle(self):
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 0, 0), 1)
        assert (state.current_player, state.moves_left_in_turn) == (-1, 2)

        state = self.game.get_next_state(state, self.game.axial_to_action(state, 1, 0), -1)
        assert (state.current_player, state.moves_left_in_turn) == (-1, 1)
        state = self.game.get_next_state(state, self.game.axial_to_action(state, -1, 0), -1)
        assert (state.current_player, state.moves_left_in_turn) == (1, 2)

        state = self.game.get_next_state(state, self.game.axial_to_action(state, 0, 1), 1)
        assert (state.current_player, state.moves_left_in_turn) == (1, 1)
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 0, -1), 1)
        assert (state.current_player, state.moves_left_in_turn) == (-1, 2)


# ---------------------------------------------------------------------------
# Move validation tests
# ---------------------------------------------------------------------------

class TestMoveValidation:
    def setup_method(self):
        self.game = HeXOGame()

    def test_first_move_valid_at_origin(self):
        state = self.game.get_initial_state()
        assert self.game.is_valid_placement(state, 0, 0) is True

    def test_cannot_place_on_occupied(self):
        state = self.game.get_initial_state()
        action = self.game.axial_to_action(state, 0, 0)
        state = self.game.get_next_state(state, action, 1)
        assert self.game.is_valid_placement(state, 0, 0) is False

    def test_within_distance_8(self):
        state = self.game.get_initial_state()
        action = self.game.axial_to_action(state, 0, 0)
        state = self.game.get_next_state(state, action, 1)
        assert self.game.is_valid_placement(state, 8, 0) is True
        assert self.game.is_valid_placement(state, 9, 0) is False

    def test_valid_moves_first_turn(self):
        state = self.game.get_initial_state()
        valid = self.game.get_valid_moves(state)
        assert valid.sum() == 1

    def test_valid_moves_dtype(self):
        state = self.game.get_initial_state()
        valid = self.game.get_valid_moves(state)
        assert valid.dtype == np.uint8


# ---------------------------------------------------------------------------
# Dynamic grid tests
# ---------------------------------------------------------------------------

class TestDynamicGrid:
    def setup_method(self):
        self.game = HeXOGame()

    def test_empty_board_grid_size(self):
        state = self.game.get_initial_state()
        oq, orr, h, w = self.game.get_grid_bounds(state)
        # 1 cell + 8 padding each side = 17
        assert h == 17
        assert w == 17
        assert oq == -8
        assert orr == -8

    def test_grid_grows_with_pieces(self):
        state = self.game.get_initial_state()
        _, _, h0, w0 = self.game.get_grid_bounds(state)

        # Place at origin
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 0, 0), 1)
        _, _, h1, w1 = self.game.get_grid_bounds(state)
        assert h1 == 17  # single piece: 1 + 8*2 = 17
        assert w1 == 17

        # Place far away — grid should grow
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 7, 0), -1)
        _, _, h2, w2 = self.game.get_grid_bounds(state)
        assert h2 == 7 + 2 * 8 + 1  # spread of 7 + padding
        assert h2 == 24

    def test_all_pieces_always_in_grid(self):
        """Place pieces in a spread pattern — all must be within grid bounds."""
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 0, 0), 1)
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 7, 0), -1)
        state = self.game.get_next_state(state, self.game.axial_to_action(state, -5, 5), -1)

        oq, orr, h, w = self.game.get_grid_bounds(state)
        for q, r in state.all_stones:
            gq, gr = self.game._axial_to_grid(oq, orr, q, r)
            assert 0 <= gq < h, f"Piece ({q},{r}) -> grid ({gq},{gr}) out of bounds h={h}"
            assert 0 <= gr < w, f"Piece ({q},{r}) -> grid ({gq},{gr}) out of bounds w={w}"

    def test_all_legal_moves_in_grid(self):
        """Every legal move must map to a valid grid index."""
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 0, 0), 1)
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 5, -3), -1)

        oq, orr, h, w = self.game.get_grid_bounds(state)
        legal = self.game._get_legal_axial(state)
        for q, r in legal:
            gq, gr = self.game._axial_to_grid(oq, orr, q, r)
            assert 0 <= gq < h, f"Legal move ({q},{r}) out of grid h"
            assert 0 <= gr < w, f"Legal move ({q},{r}) out of grid w"

    def test_action_roundtrip(self):
        """axial -> action -> axial should be identity."""
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 0, 0), 1)

        for q, r in [(1, 0), (-1, 0), (0, 1), (3, -2), (5, 5)]:
            action = self.game.axial_to_action(state, q, r)
            if action != -1:
                q2, r2 = self.game.action_to_axial(state, action)
                assert (q, r) == (q2, r2), f"Roundtrip failed for ({q},{r})"

    def test_valid_moves_size_matches_grid(self):
        """Valid moves array should be H*W for the current grid."""
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 0, 0), 1)
        valid = self.game.get_valid_moves(state)
        _, _, h, w = self.game.get_grid_bounds(state)
        assert valid.shape == (h * w,)


# ---------------------------------------------------------------------------
# Win detection tests
# ---------------------------------------------------------------------------

class TestWinDetection:
    def setup_method(self):
        self.game = HeXOGame()

    def test_no_win_with_5(self):
        state = HeXOState()
        state.stones_p1 = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}
        state.all_stones = set(state.stones_p1)
        assert state.winner == 0

    def test_win_q_axis(self):
        stones = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)}
        assert check_line_through(stones, 5, 0) is True

    def test_win_r_axis(self):
        stones = {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)}
        assert check_line_through(stones, 0, 5) is True

    def test_win_s_axis(self):
        stones = {(0, 0), (1, -1), (2, -2), (3, -3), (4, -4), (5, -5)}
        assert check_line_through(stones, 5, -5) is True

    def test_win_detected_via_place(self):
        game = self.game
        state = game.get_initial_state()

        state.stones_p1 = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}
        state.all_stones = set(state.stones_p1)
        state.stones_p2 = set()
        state.move_history = [(q, 0, 1) for q in range(5)]
        state.total_moves = 5
        state.current_player = 1
        state.moves_left_in_turn = 1

        action = game.axial_to_action(state, 5, 0)
        state = game.get_next_state(state, action, 1)
        assert state.winner == 1

        value, terminated = game.get_value_and_terminated(state, action)
        assert terminated is True
        assert value == 1


# ---------------------------------------------------------------------------
# Encoding tests
# ---------------------------------------------------------------------------

class TestEncoding:
    def setup_method(self):
        self.game = HeXOGame()

    def test_encoded_shape_empty(self):
        state = self.game.get_initial_state()
        encoded = self.game.get_encoded_state(state)
        assert encoded.shape == (3, 17, 17)
        assert encoded.dtype == np.float32

    def test_encoded_shape_grows(self):
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 0, 0), 1)
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 7, 0), -1)
        encoded = self.game.get_encoded_state(state)
        _, _, h, w = self.game.get_grid_bounds(state)
        assert encoded.shape == (3, h, w)

    def test_encoded_empty_board(self):
        state = self.game.get_initial_state()
        encoded = self.game.get_encoded_state(state)
        assert encoded[0].sum() == 0
        assert encoded[1].sum() == 0
        assert encoded[2].sum() == 17 * 17

    def test_encoded_one_stone(self):
        state = self.game.get_initial_state()
        action = self.game.axial_to_action(state, 0, 0)
        state = self.game.get_next_state(state, action, 1)
        encoded = self.game.get_encoded_state(state)
        # Current player is now -1, so P1's stone is in channel 1 (opponent)
        assert encoded[1].sum() == 1
        assert encoded[0].sum() == 0

    def test_encoded_channels_sum_to_one(self):
        state = self.game.get_initial_state()
        action = self.game.axial_to_action(state, 0, 0)
        state = self.game.get_next_state(state, action, 1)
        encoded = self.game.get_encoded_state(state)
        _, _, h, w = self.game.get_grid_bounds(state)
        total = encoded[0] + encoded[1] + encoded[2]
        np.testing.assert_array_equal(total, np.ones((h, w)))

    def test_all_stones_present_in_encoding(self):
        """Every placed stone must appear in exactly one of channels 0 or 1."""
        state = self.game.get_initial_state()
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 0, 0), 1)
        state = self.game.get_next_state(state, self.game.axial_to_action(state, 3, 0), -1)
        state = self.game.get_next_state(state, self.game.axial_to_action(state, -3, 3), -1)

        encoded = self.game.get_encoded_state(state)
        oq, orr, h, w = self.game.get_grid_bounds(state)

        for q, r in state.all_stones:
            gq, gr = self.game._axial_to_grid(oq, orr, q, r)
            stone_val = encoded[0, gq, gr] + encoded[1, gq, gr]
            assert stone_val == 1.0, f"Stone ({q},{r}) not found in encoding"


# ---------------------------------------------------------------------------
# State copy tests
# ---------------------------------------------------------------------------

class TestStateCopy:
    def test_copy_is_independent(self):
        game = HeXOGame()
        state = game.get_initial_state()
        action = game.axial_to_action(state, 0, 0)
        state = game.get_next_state(state, action, 1)

        copy = state.copy()
        copy.stones_p1.add((99, 99))
        assert (99, 99) not in state.stones_p1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
