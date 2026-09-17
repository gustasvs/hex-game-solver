import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from game_logic.hex import HexBoardState
from mcts.mcts import Node


class NodeLegalMovesTest(unittest.TestCase):
    def test_children_derive_complete_legal_moves_from_parent(self):
        root = Node(HexBoardState())

        first_move = root.legal_moves[0]
        second_move = root.legal_moves[1]
        first = root.create_child(first_move)
        second = root.create_child(second_move)

        self.assertNotIn(first_move, first.legal_moves)
        self.assertIn(second_move, first.legal_moves)
        self.assertIn(first_move, second.legal_moves)
        self.assertNotIn(second_move, second.legal_moves)
        self.assertEqual(set(first.legal_moves), set(first.state.get_legal_moves()))
        self.assertEqual(set(second.legal_moves), set(second.state.get_legal_moves()))

    def test_each_legal_move_is_expanded_once(self):
        root = Node(HexBoardState())

        for move in root.legal_moves:
            root.create_child(move)

        self.assertEqual(len(root.children), root.state.size ** 2)
        self.assertEqual(
            {child.move_from_parent[1:] for child in root.children},
            set(root.legal_moves),
        )
        occupied_cells = {
            next(
                (row, col)
                for row in range(child.state.size)
                for col in range(child.state.size)
                if child.state.p1[row][col]
            )
            for child in root.children
        }
        self.assertEqual(occupied_cells, set(root.legal_moves))

    def test_explicit_out_of_order_expansion_stays_correct(self):
        root = Node(HexBoardState())
        selected_move = root.legal_moves[-1]

        child = root.create_child(selected_move)

        self.assertNotIn(selected_move, child.legal_moves)
        self.assertEqual(set(child.legal_moves), set(child.state.get_legal_moves()))
        self.assertIn(selected_move, root.legal_moves)


if __name__ == "__main__":
    unittest.main()
