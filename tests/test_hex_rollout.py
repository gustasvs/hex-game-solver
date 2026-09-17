import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from game_logic.hex import HexBoardState


def board_of_size(size):
    state = HexBoardState()
    state.size = size
    state.p1 = [[0 for _ in range(size)] for _ in range(size)]
    state.p2 = [[0 for _ in range(size)] for _ in range(size)]
    return state


class HexRolloutTest(unittest.TestCase):
    def test_player_one_gets_first_and_extra_move_on_odd_board(self):
        state = board_of_size(3)

        with patch("game_logic.hex.np.random.shuffle", lambda moves: None):
            self.assertEqual(state.fast_random_rollout(), 1)

    def test_player_two_moves_next_when_player_one_has_extra_stone(self):
        state = board_of_size(2)
        state.p1[1][0] = 1
        state.p1[1][1] = 1
        state.p2[0][1] = 1

        with patch("game_logic.hex.np.random.shuffle", lambda moves: None):
            self.assertEqual(state.fast_random_rollout(), -1)

    def test_terminal_position_returns_without_adding_moves(self):
        state = board_of_size(2)
        state.p1[0][0] = 1
        state.p1[1][0] = 1

        self.assertEqual(state.model_based_rollout(None, None), (1, None))


if __name__ == "__main__":
    unittest.main()
