import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from game_logic.hex import HexBoardState
from mcts.mcts import Node


class NodeLegalMovesTest(unittest.TestCase):
    def test_node_and_edge_use_slots(self):
        root = Node(HexBoardState())
        edge = root.create_edge(0)

        self.assertFalse(hasattr(root, "__dict__"))
        self.assertFalse(hasattr(edge, "__dict__"))
        self.assertNotIn("legal_moves", Node.__slots__)
        self.assertEqual(edge.move_idx, 0)

    def test_children_derive_legal_masks_without_eager_move_lists(self):
        root = Node(HexBoardState())

        first = root.create_edge(0).child
        second = root.create_edge(1).child

        self.assertIsNone(first.ordered_move_indices)
        self.assertIsNone(second.ordered_move_indices)
        self.assertNotIn(0, first.iter_legal_move_indices())
        self.assertIn(1, first.iter_legal_move_indices())
        self.assertIn(0, second.iter_legal_move_indices())
        self.assertNotIn(1, second.iter_legal_move_indices())
        self.assertEqual(
            set(first.iter_legal_move_indices()),
            {move.get_idx() for move in first.state.get_legal_moves()},
        )
        self.assertEqual(
            set(second.iter_legal_move_indices()),
            {move.get_idx() for move in second.state.get_legal_moves()},
        )

    def test_each_legal_move_is_expanded_once(self):
        root = Node(HexBoardState())

        for move_idx in list(root.iter_legal_move_indices()):
            root.create_edge(move_idx)

        self.assertEqual(len(root.children), root.state.size ** 2)
        self.assertEqual(
            {edge.move_idx for edge in root.children},
            set(root.iter_legal_move_indices()),
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
            {
                (move.idx // root.state.size, move.idx % root.state.size)
                for move in root.state.get_legal_moves()
            },
        )

    def test_deferred_policy_orders_integer_indices_like_legacy_moves(self):
        root = Node(HexBoardState())
        logits = torch.arange(root.state.size ** 2, dtype=torch.float32)

        root.store_policy(logits)

        self.assertIsNone(root.ordered_move_indices)
        root._prepare_policy()
        expected = sorted(
            root.iter_legal_move_indices(),
            key=lambda move_idx: (root.stored_policy[move_idx], -move_idx),
        )
        self.assertEqual(root.ordered_move_indices, expected)

    def test_explicit_out_of_order_expansion_stays_correct(self):
        root = Node(HexBoardState())
        selected_move_idx = root.state.size ** 2 - 1

        child = root.create_edge(selected_move_idx).child

        self.assertNotIn(selected_move_idx, child.iter_legal_move_indices())
        self.assertEqual(
            set(child.iter_legal_move_indices()),
            {move.get_idx() for move in child.state.get_legal_moves()},
        )
        self.assertIn(selected_move_idx, root.iter_legal_move_indices())

    def test_transpositions_share_nodes_but_keep_distinct_incoming_edges(self):
        root = Node(HexBoardState())

        def descend(indices):
            node = root
            last_edge = None
            for index in indices:
                last_edge = node.create_edge(index)
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
