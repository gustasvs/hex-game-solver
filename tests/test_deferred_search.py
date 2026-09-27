import math
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from game_logic.hex import HexBoardState
from mcts.bitboard_helpers import rotate_bitboard_180, rotate_policy_180
from mcts.mcts import MCTS, Node, best_expanded_edge, puct_exploration_factor
from mcts.policy import DeferredPolicy
from settings import HEX_BOARD_SIZE, MCTS_FORCE_EXPAND_ALL


class DeferredPolicyTest(unittest.TestCase):
    def test_softmax_waits_for_first_use_and_is_not_repeated(self):
        node = Node(HexBoardState())
        with patch("mcts.policy.math.exp", wraps=math.exp) as exponential:
            node.store_policy(torch.arange(HEX_BOARD_SIZE**2, dtype=torch.float64))
            rotated = rotate_policy_180(node.stored_policy)
            self.assertEqual(exponential.call_count, 0)
            self.assertIs(rotate_policy_180(rotated), node.stored_policy)
            self.assertAlmostEqual(sum(rotated), 1)
            self.assertEqual(exponential.call_count, HEX_BOARD_SIZE**2)
            self.assertEqual(list(rotated), list(reversed(node.stored_policy)))
            self.assertEqual(exponential.call_count, HEX_BOARD_SIZE**2)

    def test_mask_and_original_float64_summation_order_are_preserved(self):
        rng = random.Random(104)
        for _ in range(50):
            logits = [rng.uniform(-100, 100) for _ in range(25)]
            indices = rng.sample(range(25), rng.randrange(1, 26))
            maximum = max(logits[i] for i in indices)
            expected = [0.0] * 25
            total = 0.0
            for i in indices:
                expected[i] = math.exp(logits[i] - maximum)
                total += expected[i]
            for i in indices:
                expected[i] /= total
            policy = DeferredPolicy(logits, indices)
            self.assertEqual(list(policy), expected)
            self.assertEqual(list(rotate_policy_180(policy)), expected[::-1])

    def test_direct_edge_creation_prepares_order_and_priors(self):
        node = Node(HexBoardState())
        move = node.legal_moves[0]
        node.store_policy(torch.arange(HEX_BOARD_SIZE**2, dtype=torch.float32))
        edge = node.create_edge(move)
        self.assertEqual(edge.prior, node.stored_policy[move.idx])
        self.assertNotIn(move.idx, edge.child.iter_legal_move_indices())
        self.assertIsNone(node.unexpanded_count)  # An out-of-order expansion.
        self.assertEqual(node.get_best_action_using_puct(1.4), HEX_BOARD_SIZE**2 - 1)

    def test_install_on_existing_edges_updates_prior_and_puct_weight(self):
        node = Node(HexBoardState())
        edge = node.create_edge(node.legal_moves[0])
        edge.backpropagate(0.25)
        edge.backpropagate(-0.5)
        node.store_policy(torch.zeros(HEX_BOARD_SIZE**2))
        self.assertAlmostEqual(edge.prior, 1 / HEX_BOARD_SIZE**2)
        self.assertAlmostEqual(
            edge.puct_weight,
            edge.prior / (1 + edge.N),
        )

        replacement = [0.0] * (HEX_BOARD_SIZE**2)
        replacement[edge.move_idx] = 0.7
        node._install_policy(replacement)
        self.assertEqual(edge.prior, 0.7)
        self.assertAlmostEqual(edge.puct_weight, 0.7 / (1 + edge.N))
        self.assertIsInstance(node.stored_policy, list)

    def test_legal_indices_larger_than_one_byte_are_supported(self):
        policy = DeferredPolicy([0.0] * 289, [0, 288])
        self.assertEqual(len(policy), 289)
        self.assertEqual(policy[0], 0.5)
        self.assertEqual(policy[288], 0.5)
        self.assertEqual(sum(policy), 1)


class SelectionTest(unittest.TestCase):
    def test_measured_default_allows_early_revisits(self):
        self.assertFalse(MCTS_FORCE_EXPAND_ALL)
        self.assertFalse(MCTS(Node(HexBoardState()), None, 1).force_expand_all)

    def test_fully_expanded_puct_returns_and_selects_the_winning_edge(self):
        rng = random.Random(50)
        for turn in (0, 1):
            node = Node(HexBoardState(), move_index=turn)
            node.store_policy(torch.zeros(HEX_BOARD_SIZE**2))
            node._prepare_policy()
            while node.unexpanded_count:
                node.create_edge(node.ordered_move_indices[node.unexpanded_count - 1])

            rng.shuffle(node.children)
            self.assertEqual(
                best_expanded_edge(node, 1.4).move_idx,
                0,
            )
            for _ in range(100):
                rng.choice(node.children).backpropagate(rng.uniform(-1, 1))

            expected_action = node.get_best_action_using_puct(1.4)
            expected_edge = node.edge_for_move(expected_action)
            self.assertIs(
                best_expanded_edge(node, 1.4),
                expected_edge,
            )

            self.assertFalse(hasattr(node, "move_edge_map"))
            selected, path = MCTS(node, None, 1).select(node)
            self.assertEqual(path, [expected_edge])
            self.assertIs(selected, expected_edge.child)

    def test_public_select_path_is_not_mutated_by_later_selection(self):
        root = Node(HexBoardState())
        root._install_policy([1 / (HEX_BOARD_SIZE**2)] * (HEX_BOARD_SIZE**2))
        search = MCTS(root, None, 1)

        _, first_path = search.select(root)
        first_snapshot = list(first_path)
        search.select(root)

        self.assertEqual(first_path, first_snapshot)
        self.assertIsNot(first_path, search._selection_path)

    def test_partial_selection_does_not_lookup_known_unexpanded_move(self):
        root = Node(HexBoardState())
        root._install_policy([1 / (HEX_BOARD_SIZE**2)] * (HEX_BOARD_SIZE**2))
        root._prepare_policy()
        self.assertFalse(hasattr(root, "move_edge_map"))

        selected, path = MCTS(root, None, 1, force_expand_all=False).select(root)

        self.assertEqual(len(path), 1)
        self.assertIs(selected, path[0].child)

    def test_signed_q_and_puct_weight_follow_every_backpropagation(self):
        for move_index in (0, 1):
            node = Node(HexBoardState(), move_index=move_index)
            edge = node.create_edge(node.legal_moves[0])
            total = 0.0
            for n, result in enumerate((0.75, -1, 0.25, 1), 1):
                edge.backpropagate(result)
                total += result
                self.assertEqual(edge.Q(), total / n)
                self.assertEqual(edge.selection_q, (total / n) * (1 if node.p1_turn else -1))
                self.assertEqual(edge.puct_weight, edge.prior / (1 + n))

    def test_puct_sqrt_lookup_matches_reference_for_arbitrary_constants(self):
        for c in (0, 0.25, 1.0, 1.4, 3.75):
            for visits in (0, 1, 2, 7, 25, 257):
                self.assertEqual(
                    puct_exploration_factor(c, visits),
                    c * max(1, visits) ** 0.5,
                )

    def test_sparse_puct_matches_exhaustive_for_both_players(self):
        rng = random.Random(72)
        for turn in (0, 1):
            for _ in range(100):
                node = Node(HexBoardState(), move_index=turn)
                node.store_policy(torch.tensor([rng.random() for _ in range(25)]))
                node._prepare_policy()
                for _ in range(rng.randrange(1, 26)):
                    edge = node.create_edge(
                        node.ordered_move_indices[node.unexpanded_count - 1]
                    )
                    for _ in range(rng.randrange(1, 8)):
                        edge.backpropagate(rng.uniform(-1, 1))
                factor = 1.4 * max(1, node.total_edge_visits) ** 0.5
                def score(move_idx):
                    edge = node.edge_for_move(move_idx)
                    q = (edge.W / edge.N) if edge else 0
                    if not node.p1_turn:
                        q = -q
                    return q + factor * node.stored_policy[move_idx] / (1 + (edge.N if edge else 0))
                expected = max(sorted(node.iter_legal_move_indices()), key=score)
                self.assertEqual(node.get_best_action_using_puct(1.4), expected)

    def test_puct_can_revisit_a_good_move_before_all_moves_expand(self):
        root = Node(HexBoardState())
        policy = [0.001] * 25
        policy[0] = 0.976
        root._install_policy(policy)
        good = root.create_edge(root.legal_moves[0])
        good.backpropagate(0.8)
        forced, _ = MCTS(root, None, 1, force_expand_all=True).select(root)
        focused, _ = MCTS(root, None, 1, force_expand_all=False).select(root)
        self.assertIsNot(forced, good.child)
        self.assertIs(focused, good.child)
        self.assertLess(len(root.children), root.legal_mask.bit_count())

    def test_puct_still_explores_when_visited_move_is_poor(self):
        root = Node(HexBoardState())
        root._install_policy([1 / 25] * 25)
        first = root.create_edge(root.legal_moves[0])
        first.backpropagate(-1)
        selected, _ = MCTS(root, None, 1, force_expand_all=False).select(root)
        self.assertIsNot(selected, first.child)

    def test_equal_scores_choose_smallest_move_index(self):
        root = Node(HexBoardState())
        root._install_policy([1 / 25] * 25)
        self.assertEqual(root.get_best_action_using_puct(1.4), 0)


class RotationTest(unittest.TestCase):
    def test_lookup_rotation_matches_direct_bit_mapping(self):
        count = HEX_BOARD_SIZE**2
        rng = random.Random(18)
        for bits in [0, (1 << count)-1] + [1 << i for i in range(count)] + [rng.getrandbits(count) for _ in range(500)]:
            expected = sum(((bits >> i) & 1) << (count - i - 1) for i in range(count))
            self.assertEqual(rotate_bitboard_180(bits), expected)
            self.assertEqual(rotate_bitboard_180(expected), bits)

    def test_rotation_fallback_for_boards_larger_than_32_bits(self):
        source = (Path(__file__).resolve().parents[1] / "src/mcts/bitboard_helpers.py").read_text()
        namespace = {}
        with patch("settings.HEX_BOARD_SIZE", 6):
            exec(compile(source, "six_by_six_bitboards", "exec"), namespace)
        rng = random.Random(46)
        for bits in [0, (1 << 36)-1] + [rng.getrandbits(36) for _ in range(100)]:
            expected = sum(((bits >> i) & 1) << (35 - i) for i in range(36))
            self.assertEqual(namespace["rotate_bitboard_180"](bits), expected)


if __name__ == "__main__":
    unittest.main()
