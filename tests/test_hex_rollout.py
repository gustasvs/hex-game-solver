import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from game_logic.hex import HexBoardState
from mcts.mcts import Node
from settings import HEX_BOARD_SIZE


class HexRolloutTest(unittest.TestCase):
    def test_player_one_gets_first_and_extra_move_on_odd_board(self):
        state = HexBoardState()
        captured = []
        def connection(bits, start, end):
            captured.append(bits)
            return True
        with patch("game_logic.hex.random.shuffle", lambda moves: None), patch(
            "game_logic.hex.has_bit_connection", connection
        ):
            self.assertEqual(state.fast_random_rollout_bits(0, 0, True), 1)
        self.assertEqual(captured[0], sum(1 << i for i in range(0, HEX_BOARD_SIZE**2, 2)))

    def test_player_two_moves_next_when_player_one_has_extra_stone(self):
        state = HexBoardState()
        captured = []
        def connection(bits, start, end):
            captured.append(bits)
            return len(captured) == 2
        with patch("game_logic.hex.random.shuffle", lambda moves: None), patch(
            "game_logic.hex.has_bit_connection", connection
        ):
            self.assertEqual(state.fast_random_rollout_bits(1, 0, False), -1)
        self.assertEqual(captured[1], sum(1 << i for i in range(1, HEX_BOARD_SIZE**2, 2)))

    def test_terminal_position_returns_without_adding_moves(self):
        p1 = sum(1 << (row * HEX_BOARD_SIZE) for row in range(HEX_BOARD_SIZE))
        p2 = sum(1 << (row * HEX_BOARD_SIZE + 2) for row in range(HEX_BOARD_SIZE - 1))
        node = Node(HexBoardState.from_bitboards(p1, p2), move_index=2 * HEX_BOARD_SIZE - 1)
        with patch.object(HexBoardState, "fast_random_rollout_bits", side_effect=AssertionError("terminal rollout")):
            self.assertEqual(node.rollout(None, None), (1, None))
        self.assertEqual((node.p1_bits, node.p2_bits), (p1, p2))


if __name__ == "__main__":
    unittest.main()
