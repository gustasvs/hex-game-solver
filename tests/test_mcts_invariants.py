import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from game import play_game
from game_logic.hex import HexBoardState
from mcts.mcts import Node
from pytorch_model import CustomNNUE
from settings import DEVICE, HEX_BOARD_SIZE


def state_after(moves):
    state = HexBoardState()
    for move in moves:
        state.make_move(move)
    return state


class AccumulatorTest(unittest.TestCase):
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


class PolicyTest(unittest.TestCase):
    def test_policy_is_normalized_over_legal_moves_and_zero_elsewhere(self):
        state = state_after([(True, 0, 0), (False, 1, 1)])
        node = Node(state, move_index=2)
        logits = torch.linspace(-2.0, 2.0, HEX_BOARD_SIZE * HEX_BOARD_SIZE)

        node.store_policy(logits)

        legal_indices = {
            row * HEX_BOARD_SIZE + col for row, col in node.legal_moves
        }
        legal_total = sum(node.stored_policy[index] for index in legal_indices)
        self.assertAlmostEqual(legal_total, 1.0, places=6)
        for index, probability in enumerate(node.stored_policy):
            if index not in legal_indices:
                self.assertEqual(probability, 0.0)


class TreeTest(unittest.TestCase):
    def test_promoted_child_has_no_parent(self):
        displayed_roots = []

        class OneMoveWinningMCTS:
            def __init__(self, root, model):
                self.root = root

            def run(self):
                child = self.root.create_child(self.root.legal_moves[0])
                for row in range(child.state.size):
                    child.state.p1[row][0] = 1
                child.N = 1
                return self.root

        with (
            patch("game.MCTS", OneMoveWinningMCTS),
            patch("game.display_game", lambda state, root: displayed_roots.append(root)),
        ):
            play_game(model=object())

        promoted_root = displayed_roots[-1]
        self.assertIsNone(promoted_root.parent)


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

        root = Node(state, move_index=0 if player == 1 else 1)
        terminal_leaf = root.create_child(winning_move)
        outcome, policy = terminal_leaf.rollout(
            self.FailIfEvaluatedModel(), shared_accumulator=torch.zeros(1)
        )
        terminal_leaf.backpropagate(outcome)

        self.assertIsNone(policy)
        self.assertEqual(outcome, expected)
        self.assertEqual(terminal_leaf.N, 1)
        self.assertEqual(terminal_leaf.W, expected)
        self.assertEqual(root.N, 1)
        self.assertEqual(root.W, expected)
        self.assertEqual(root.Q(), expected)


class PerspectiveTest(unittest.TestCase):
    def test_same_q_player_one_prefers_larger_and_player_two_smaller(self):
        self.assertEqual(self._selected_q(move_index=0), 0.75)
        self.assertEqual(self._selected_q(move_index=1), -0.25)

    @staticmethod
    def _selected_q(move_index):
        parent = Node(HexBoardState(), move_index=move_index)
        parent.N = 2
        lower = parent.create_child(parent.legal_moves[0])
        higher = parent.create_child(parent.legal_moves[0])
        lower.N = higher.N = 1
        lower.W = -0.25
        higher.W = 0.75

        return parent.best_child_using_uct(c=0).Q()


class PUCTTest(unittest.TestCase):
    def test_higher_policy_prior_is_selected_when_q_and_n_are_equal(self):
        parent = Node(HexBoardState())
        low_prior_move, high_prior_move = parent.legal_moves[:2]
        low_prior_child = parent.create_child(low_prior_move)
        high_prior_child = parent.create_child(high_prior_move)
        parent.N = 10
        for child in (low_prior_child, high_prior_child):
            child.N = 3
            child.W = 1.5

        parent.stored_policy = [0.0] * (HEX_BOARD_SIZE * HEX_BOARD_SIZE)
        parent.stored_policy[low_prior_move[0] * HEX_BOARD_SIZE + low_prior_move[1]] = 0.2
        parent.stored_policy[high_prior_move[0] * HEX_BOARD_SIZE + high_prior_move[1]] = 0.8

        self.assertEqual(parent.get_best_action_using_puct(c=1.0), high_prior_move)


if __name__ == "__main__":
    unittest.main()
