import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from game import play_game
from game_logic.hex import HexBoardState
from mcts.mcts import MCTS, Node
from pytorch_model import CustomNNUE
from settings import DEVICE, HEX_BOARD_SIZE


def state_after(moves):
    state = HexBoardState()
    for move in moves:
        state.make_move(move)
    return state


class AccumulatorTest(unittest.TestCase):
    def test_cached_inference_parameters_match_default_forward(self):
        torch.manual_seed(9)
        model = CustomNNUE().to(DEVICE)
        accumulator = torch.randn(
            model.cached_embeddings.embedding_dim, device=DEVICE
        )

        expected = model(accumulator)
        actual = model(accumulator, model.get_inference_parameters())

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_full_reconstruction_equals_path_updated_accumulator(self):
        torch.manual_seed(7)
        model = CustomNNUE().to(DEVICE)
        moves = [
            (True, 0, 0),
            (False, 1, 0),
            (True, 0, 1),
            (False, 2, 0),
        ]

        path_accumulator = model.calculate_accumulator(
            HexBoardState().get_state(), is_p1_move_now=True
        ).clone()
        state = HexBoardState()
        for move in moves:
            path_accumulator = model.advance_accumulator(
                path_accumulator, move[0], move
            )
            state.make_move(move)
            reconstructed = model.calculate_accumulator(
                state.get_state(), is_p1_move_now=not move[0]
            )
            torch.testing.assert_close(path_accumulator, reconstructed)

    def test_precomputed_move_deltas_match_incremental_updates(self):
        torch.manual_seed(7)
        model = CustomNNUE().to(DEVICE)
        move_deltas = model.calculate_move_deltas()
        root_accumulator = model.calculate_accumulator(
            HexBoardState().get_state(), is_p1_move_now=True
        )

        for is_p1_move, row, col in ((True, 2, 3), (False, 4, 1)):
            move = (is_p1_move, row, col)
            expected = model.advance_accumulator(
                root_accumulator.clone(), is_p1_move, move
            )
            player_index = 0 if is_p1_move else 1
            actual = root_accumulator + move_deltas[
                player_index, row * HEX_BOARD_SIZE + col
            ]
            torch.testing.assert_close(actual, expected)

    def test_mcts_stored_accumulators_do_not_retain_autograd_graphs(self):
        model = CustomNNUE().to(DEVICE)
        root = Node(HexBoardState())
        search = MCTS(root, model, 1)

        root.stored_policy = [1.0] * (HEX_BOARD_SIZE * HEX_BOARD_SIZE)
        child, _ = search.select(root)

        self.assertFalse(root.stored_accumulator.requires_grad)
        self.assertIsNone(root.stored_accumulator.grad_fn)
        for player_deltas in search.move_delta_views:
            for move_delta in player_deltas:
                self.assertFalse(move_delta.requires_grad)
                self.assertIsNone(move_delta.grad_fn)
        self.assertFalse(child.stored_accumulator.requires_grad)
        self.assertIsNone(child.stored_accumulator.grad_fn)


class PolicyTest(unittest.TestCase):
    def test_policy_is_normalized_over_legal_moves_and_zero_elsewhere(self):
        state = state_after([(True, 0, 0), (False, 1, 1)])
        node = Node(state, move_index=2)
        logits = torch.linspace(-2.0, 2.0, HEX_BOARD_SIZE * HEX_BOARD_SIZE)

        node.store_policy(logits)

        legal_indices = {move.get_idx() for move in node.legal_moves}
        legal_total = sum(node.stored_policy[index] for index in legal_indices)
        self.assertAlmostEqual(legal_total, 1.0, places=6)
        for index, probability in enumerate(node.stored_policy):
            if index not in legal_indices:
                self.assertEqual(probability, 0.0)


class TreeTest(unittest.TestCase):
    def test_lazy_child_state_matches_bitboards_without_mutating_parent(self):
        root = Node(HexBoardState())
        child = root.create_child(root.legal_moves[0])
        grandchild = child.create_child(child.legal_moves[0])

        self.assertIsNone(grandchild._state)
        self.assertEqual(sum(map(sum, grandchild.state.p1)), 1)
        self.assertEqual(sum(map(sum, grandchild.state.p2)), 1)
        self.assertEqual(sum(map(sum, root.state.p1)), 0)
        self.assertEqual(sum(map(sum, root.state.p2)), 0)

    def test_transposed_move_orders_have_identical_bitboard_keys(self):
        root = Node(HexBoardState())

        def descend(indices):
            node = root
            for index in indices:
                node = node.create_child(next(move for move in node.legal_moves if move.idx == index))
            return node

        self.assertIs(descend((0, 1, 2, 3)), descend((2, 3, 0, 1)))

    def test_reusing_tree_after_model_update_is_rejected(self):
        model = CustomNNUE().to(DEVICE)
        root = Node(HexBoardState())
        MCTS(root, model, 1)

        with torch.no_grad():
            model.cached_embeddings.weight.add_(1)

        with self.assertRaisesRegex(RuntimeError, "model weights change"):
            MCTS(root, model, 1)

    def test_promoted_child_has_no_parent(self):
        promoted_roots = []

        class OneMoveWinningMCTS:
            def __init__(self, root, model, max_iterations):
                self.root = root

            def run(self):
                edge = self.root.create_edge(self.root.legal_moves[0])
                child = edge.child
                for row in range(child.state.size):
                    child.state.p1[row][0] = 1
                edge.N = 1
                promoted_roots.append(child)
                return self.root

        with (
            patch("game.MCTS", OneMoveWinningMCTS),
        ):
            play_game(model=object())

        promoted_root = promoted_roots[-1]
        self.assertIsInstance(promoted_root, Node)


class TerminalTest(unittest.TestCase):
    class FailIfEvaluatedModel:
        def forward(self, accumulator):
            raise AssertionError("terminal leaves must not call the neural network")

    def test_terminal_leaf_never_calls_nn_and_propagates_exactly_one(self):
        self._assert_terminal_result(player=1, expected=1)

    def test_terminal_leaf_never_calls_nn_and_propagates_exactly_minus_one(self):
        self._assert_terminal_result(player=2, expected=-1)

    def _assert_terminal_result(self, player, expected):
        state = HexBoardState()
        board = state.p1 if player == 1 else state.p2
        if player == 1:
            for row in range(state.size - 1):
                board[row][0] = 1
            winning_move = (state.size - 1, 0)
        else:
            for col in range(state.size - 1):
                board[0][col] = 1
            winning_move = (0, state.size - 1)

        root = Node(state, move_index=2 * state.size - 2 if player == 1 else 2 * state.size - 1)
        terminal_edge = root.create_edge(
            next(move for move in root.legal_moves if (move.x, move.y) == winning_move)
        )
        terminal_leaf = terminal_edge.child
        outcome, policy = terminal_leaf.rollout(
            self.FailIfEvaluatedModel(), shared_accumulator=torch.zeros(1)
        )
        terminal_edge.backpropagate(outcome)

        self.assertIsNone(policy)
        self.assertEqual(outcome, expected)
        self.assertEqual(terminal_edge.N, 1)
        self.assertEqual(terminal_edge.W, expected)
        self.assertEqual(root.total_edge_visits, 1)
        self.assertEqual(terminal_edge.Q(), expected)


class PerspectiveTest(unittest.TestCase):
    def test_same_q_player_one_prefers_larger_and_player_two_smaller(self):
        self.assertEqual(self._selected_q(move_index=0), 0.75)
        self.assertEqual(self._selected_q(move_index=1), -0.25)

    @staticmethod
    def _selected_q(move_index):
        parent = Node(HexBoardState(), move_index=move_index)
        lower = parent.create_edge(parent.legal_moves[0])
        higher = parent.create_edge(parent.legal_moves[1])
        lower.N = higher.N = 1
        lower.W = -0.25
        higher.W = 0.75
        parent.total_edge_visits = 2

        return parent.best_child_using_uct(c=0).Q()


class PUCTTest(unittest.TestCase):
    def test_sparse_selection_matches_exhaustive_selection(self):
        random.seed(11)
        parent = Node(HexBoardState())
        parent.store_policy(torch.randn(HEX_BOARD_SIZE * HEX_BOARD_SIZE))
        parent.total_edge_visits = 40
        for move in random.sample(parent.legal_moves, 8):
            edge = parent.create_edge(move)
            edge.N = random.randint(1, 8)
            edge.W = random.uniform(-edge.N, edge.N)

        expected = None
        expected_score = -float("inf")
        exploration_factor = 1.4 * parent.total_edge_visits ** 0.5
        for move in sorted(parent.legal_moves, key=lambda item: item.idx):
            edge = parent.move_edge_map.get(move.idx)
            n = 0 if edge is None else edge.N
            q = 0 if edge is None else edge.Q()
            score = q + exploration_factor * parent.stored_policy[move.idx] / (1 + n)
            if score > expected_score:
                expected = move
                expected_score = score

        self.assertEqual(parent.get_best_action_using_puct(c=1.4), expected)

    def test_higher_policy_prior_is_selected_when_q_and_n_are_equal(self):
        parent = Node(HexBoardState())
        low_prior_move, high_prior_move = parent.legal_moves[:2]
        low_prior_edge = parent.create_edge(low_prior_move)
        high_prior_edge = parent.create_edge(high_prior_move)
        parent.total_edge_visits = 10
        for edge in (low_prior_edge, high_prior_edge):
            edge.N = 3
            edge.W = 1.5

        policy = [0.0] * (HEX_BOARD_SIZE * HEX_BOARD_SIZE)
        policy[low_prior_move.get_idx()] = 0.2
        policy[high_prior_move.get_idx()] = 0.8
        parent._install_policy(policy)

        self.assertEqual(parent.get_best_action_using_puct(c=1.0), high_prior_move)


if __name__ == "__main__":
    unittest.main()
