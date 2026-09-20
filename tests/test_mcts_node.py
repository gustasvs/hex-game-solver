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
        self.assertEqual(
            {move.get_idx() for move in first.legal_moves},
            {move.get_idx() for move in first.state.get_legal_moves()},
        )
        self.assertEqual(
            {move.get_idx() for move in second.legal_moves},
            {move.get_idx() for move in second.state.get_legal_moves()},
        )

    def test_each_legal_move_is_expanded_once(self):
        root = Node(HexBoardState())

        for move in root.legal_moves:
            root.create_child(move)

        self.assertEqual(len(root.children), root.state.size ** 2)
        self.assertEqual(
            {edge.move_from_parent[1:] for edge in root.children},
            {(move.x, move.y) for move in root.legal_moves},
        )
        occupied_cells = {
            next(
                (row, col)
                for row in range(edge.child.state.size)
                for col in range(edge.child.state.size)
                if edge.child.state.p1[row][col]
            )
            for edge in root.children
        }
        self.assertEqual(
            occupied_cells,
            {(move.x, move.y) for move in root.legal_moves},
        )

    def test_explicit_out_of_order_expansion_stays_correct(self):
        root = Node(HexBoardState())
        selected_move = root.legal_moves[-1]

        child = root.create_child(selected_move)

        self.assertNotIn(selected_move, child.legal_moves)
        self.assertEqual(
            {move.get_idx() for move in child.legal_moves},
            {move.get_idx() for move in child.state.get_legal_moves()},
        )
        self.assertIn(selected_move, root.legal_moves)

    def test_transpositions_share_nodes_but_keep_distinct_incoming_edges(self):
        root = Node(HexBoardState())

        def descend(indices):
            node = root
            last_edge = None
            for index in indices:
                move = next(move for move in node.legal_moves if move.idx == index)
                last_edge = node.create_edge(move)
                node = last_edge.child
            return node, last_edge

        first_node, first_edge = descend((0, 1, 2, 3))
        second_node, second_edge = descend((2, 3, 0, 1))

        self.assertIs(first_node, second_node)
        self.assertIsNot(first_edge, second_edge)

        first_edge.backpropagate(1)
        self.assertEqual(first_edge.N, 1)
        self.assertEqual(second_edge.N, 0)


if __name__ == "__main__":
    unittest.main()
