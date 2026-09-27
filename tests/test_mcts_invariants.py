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
from mcts.mcts import EvaluationCache, MCTS, Node, NodeTable, TreeContext
from mcts.bitboard_helpers import (
    canonical_evaluation_key,
    rotate_bitboard_180,
    rotate_policy_180,
)
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
        root = Node(HexBoardState(), node_cache=NodeTable())
        search = MCTS(root, model, 2)
        search.run()
        child = root.children[0].child

        self.assertFalse(root.stored_accumulator.requires_grad)
        self.assertIsNone(root.stored_accumulator.grad_fn)
        for player_deltas in search.move_delta_views:
            for move_delta in player_deltas:
                self.assertFalse(move_delta.requires_grad)
                self.assertIsNone(move_delta.grad_fn)
        self.assertFalse(child.stored_accumulator.requires_grad)
        self.assertIsNone(child.stored_accumulator.grad_fn)

    def test_shared_cache_hit_does_not_materialize_child_accumulator(self):
        model = CustomNNUE().to(DEVICE)
        evaluation_cache = {}
        root = Node(HexBoardState(), evaluation_cache=evaluation_cache)
        child = root.create_edge(0).child
        root_policy = [0.0] * (HEX_BOARD_SIZE * HEX_BOARD_SIZE)
        root_policy[0] = 1.0
        root._install_policy(root_policy)
        child_key, _ = canonical_evaluation_key(child.p1_bits, child.p2_bits)
        evaluation_cache[child_key] = (0.25, [1 / 24] * 25)
        forward_calls = 0

        def counted_forward(accumulator, inference_parameters=None):
            nonlocal forward_calls
            forward_calls += 1
            return torch.tensor([0.0], device=DEVICE), torch.zeros(
                HEX_BOARD_SIZE * HEX_BOARD_SIZE,
                device=DEVICE,
            )

        model.forward = counted_forward
        search = MCTS(root, model, 1)
        search.run()

        self.assertEqual(forward_calls, 0)
        self.assertEqual(search.cross_game_cache_hits, 1)
        self.assertEqual(search.accumulator_creations, 1)  # root only
        self.assertIsNone(child.stored_accumulator)

    def test_cache_hit_then_expand_materializes_internal_hit_once(self):
        model = CustomNNUE().to(DEVICE)
        evaluation_cache = {}
        root = Node(HexBoardState(), evaluation_cache=evaluation_cache)
        child = root.create_edge(0).child
        root_policy = [0.0] * (HEX_BOARD_SIZE * HEX_BOARD_SIZE)
        root_policy[0] = 1.0
        root._install_policy(root_policy)
        child_key, _ = canonical_evaluation_key(child.p1_bits, child.p2_bits)
        evaluation_cache[child_key] = (0.25, [1 / 24] * 25)
        forward_calls = 0
        forwarded_accumulators = []
        original_forward = model.forward

        def counted_forward(accumulator, inference_parameters=None):
            nonlocal forward_calls
            forward_calls += 1
            forwarded_accumulators.append(accumulator.detach().clone())
            return original_forward(accumulator, inference_parameters)

        model.forward = counted_forward
        MCTS(root, model, 1).run()
        self.assertIsNone(child.stored_accumulator)

        search = MCTS(root, model, 1)
        search.run()
        grandchild = child.children[0].child

        self.assertEqual(forward_calls, 1)
        self.assertEqual(search.accumulator_creations, 2)
        self.assertIsNotNone(child.stored_accumulator)
        self.assertIsNotNone(grandchild.stored_accumulator)
        child_reconstructed = model.calculate_accumulator(
            child.state.get_state(),
            child.is_p1_turn(),
        ).detach()
        torch.testing.assert_close(child.stored_accumulator, child_reconstructed)
        reconstructed = model.calculate_accumulator(
            grandchild.state.get_state(),
            grandchild.is_p1_turn(),
        ).detach()
        torch.testing.assert_close(grandchild.stored_accumulator, reconstructed)
        expected_value, expected_policy = original_forward(
            reconstructed,
            search.inference_parameters,
        )
        actual_value, actual_policy = original_forward(
            forwarded_accumulators[0],
            search.inference_parameters,
        )
        torch.testing.assert_close(actual_value, expected_value)
        torch.testing.assert_close(actual_policy, expected_policy)

    def test_accumulator_creation_counter_includes_root_and_new_child(self):
        model = CustomNNUE().to(DEVICE)
        search = MCTS(Node(HexBoardState()), model, 2)

        search.run()

        self.assertEqual(search.accumulator_creations, 2)

    def test_promoted_child_reuses_model_search_context(self):
        model = CustomNNUE().to(DEVICE)
        shared_evaluation_cache = {}
        root = Node(HexBoardState(), evaluation_cache=shared_evaluation_cache)

        with (
            patch.object(
                model,
                "calculate_move_deltas",
                wraps=model.calculate_move_deltas,
            ) as move_deltas,
            patch.object(
                model,
                "get_inference_parameters",
                wraps=model.get_inference_parameters,
            ) as inference_parameters,
        ):
            first_search = MCTS(root, model, 2)
            first_search.run()
            promoted = root.children[0].child
            second_search = MCTS(promoted, model, 1)
            second_search.run()
            next_game_search = MCTS(
                Node(HexBoardState(), evaluation_cache=shared_evaluation_cache),
                model,
                1,
            )

        self.assertIs(promoted.search_context, first_search.search_context)
        self.assertIs(second_search.search_context, first_search.search_context)
        self.assertIs(next_game_search.search_context, first_search.search_context)
        self.assertEqual(move_deltas.call_count, 1)
        self.assertEqual(inference_parameters.call_count, 1)


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


class EvaluationCacheSymmetryTest(unittest.TestCase):
    def test_rotated_positions_have_the_same_canonical_key(self):
        p1_bits = (1 << 0) | (1 << 7)
        p2_bits = (1 << 3) | (1 << 12)
        rotated_p1 = rotate_bitboard_180(p1_bits)
        rotated_p2 = rotate_bitboard_180(p2_bits)

        key, _ = canonical_evaluation_key(p1_bits, p2_bits)
        rotated_key, _ = canonical_evaluation_key(rotated_p1, rotated_p2)

        self.assertEqual(key, rotated_key)
        self.assertEqual(
            rotate_bitboard_180(rotated_p1),
            p1_bits,
        )
        self.assertEqual(
            rotate_bitboard_180(rotated_p2),
            p2_bits,
        )

    def test_policy_rotation_is_its_own_inverse(self):
        policy = list(range(HEX_BOARD_SIZE * HEX_BOARD_SIZE))
        self.assertEqual(
            rotate_policy_180(rotate_policy_180(policy)),
            policy,
        )

    def test_rotated_position_reuses_evaluation_and_rotates_policy(self):
        evaluation_cache = {}
        model = CustomNNUE().to(DEVICE)
        forward_calls = 0
        policy_logits = torch.arange(
            HEX_BOARD_SIZE * HEX_BOARD_SIZE,
            dtype=torch.float32,
            device=DEVICE,
        )

        def counted_forward(accumulator, inference_parameters=None):
            nonlocal forward_calls
            forward_calls += 1
            return torch.tensor([0.25], device=DEVICE), policy_logits

        model.forward = counted_forward

        state = state_after([(True, 0, 1), (False, 1, 3)])
        first = Node(state, move_index=2, evaluation_cache=evaluation_cache)
        first_search = MCTS(first, model, 1)
        first_search.run()

        rotated_state = HexBoardState.from_bitboards(
            rotate_bitboard_180(first.p1_bits),
            rotate_bitboard_180(first.p2_bits),
        )
        rotated = Node(
            rotated_state,
            move_index=2,
            evaluation_cache=evaluation_cache,
        )
        rotated_search = MCTS(rotated, model, 1)
        rotated_search.run()

        self.assertEqual(forward_calls, 1)
        self.assertEqual(first_search.cross_game_cache_misses, 1)
        self.assertEqual(first_search.cross_game_cache_misses_by_move_index[2], 1)
        self.assertEqual(first_search.cross_game_cache_hits, 0)
        self.assertEqual(rotated_search.evaluation_cache_hits, 1)
        self.assertEqual(rotated_search.cross_game_cache_hits, 1)
        self.assertEqual(rotated_search.cross_game_cache_hits_by_move_index[2], 1)
        self.assertEqual(rotated_search.cross_game_cache_misses, 0)
        self.assertEqual(len(evaluation_cache), 1)
        self.assertEqual(
            rotated.stored_policy,
            rotate_policy_180(first.stored_policy),
        )
        self.assertEqual(rotated.stored_value, first.stored_value)

    def test_node_value_hits_are_not_shared_cache_hits(self):
        model = CustomNNUE().to(DEVICE)
        root = Node(HexBoardState())
        root.stored_value = 0.125

        search = MCTS(root, model, 1)
        search.run()

        self.assertEqual(search.node_value_hits, 1)
        self.assertEqual(search.cross_game_cache_hits, 0)
        self.assertEqual(search.cross_game_cache_misses, 0)
        self.assertEqual(search.evaluation_cache_hits, 1)


class TreeTest(unittest.TestCase):
    def test_descendants_share_one_tree_context_and_ply_bucketed_table(self):
        root = Node(HexBoardState(), node_cache=NodeTable())
        child = root.create_edge(0).child
        grandchild = child.create_edge(1).child

        self.assertIs(root.tree_context, child.tree_context)
        self.assertIs(child.tree_context, grandchild.tree_context)
        self.assertIsInstance(root.node_cache, NodeTable)
        self.assertEqual(root.node_cache.sizes_by_ply[:3], [1, 1, 1])

    def test_bucketed_evaluation_cache_retains_opening_and_drops_deep_entries(self):
        cache = EvaluationCache()
        opening = (1, 0)
        deep = ((1 << 8) - 1, ((1 << 16) - 1) ^ ((1 << 8) - 1))
        cache[opening] = "opening"
        cache[deep] = "deep"

        self.assertEqual(cache.get(opening), "opening")
        self.assertIsNone(cache.get(deep))
        self.assertEqual(cache.hits_by_ply[1], 1)
        self.assertEqual(cache.misses_by_ply[16], 1)

    def test_root_promotion_prunes_only_incompatible_positions(self):
        table = {}
        context = TreeContext(evaluation_cache={}, node_cache=table)
        root = Node(HexBoardState(), tree_context=context)
        kept = root.create_edge(0).child
        dropped = root.create_edge(1).child

        pruned = kept.promote_to_root(threshold=0)

        self.assertEqual(pruned, 2)
        self.assertIn((kept.p1_bits, kept.p2_bits), table)
        self.assertNotIn((dropped.p1_bits, dropped.p2_bits), table)

    def test_node_cache_uses_distinct_board_tuple_keys(self):
        pairs = (
            (0, 0),
            (1 << 0, 1 << 1),
            ((1 << 3) | (1 << 17), (1 << 2) | (1 << 24)),
            ((1 << 24), (1 << 0) | (1 << 12)),
        )
        node_cache = {}
        nodes = []
        for p1_bits, p2_bits in pairs:
            nodes.append(
                Node(
                    None,
                    p1_bits=p1_bits,
                    p2_bits=p2_bits,
                    node_cache=node_cache,
                )
            )

        self.assertEqual(len(node_cache), len(pairs))
        for pair, node in zip(pairs, nodes):
            self.assertIs(node_cache[pair], node)

    def test_lazy_child_state_matches_bitboards_without_mutating_parent(self):
        root = Node(HexBoardState())
        child = root.create_edge(root.legal_moves[0]).child
        grandchild = child.create_edge(child.legal_moves[0]).child

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
                node = node.create_edge(next(move for move in node.legal_moves if move.idx == index)).child
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
                child.stored_terminal = True
                child.stored_terminal_outcome = 1
                edge.backpropagate(1)
                promoted_roots.append(child)
                return self.root

        with (
            patch("game.MCTS", OneMoveWinningMCTS),
        ):
            play_game(model=object())

        promoted_root = promoted_roots[-1]
        self.assertEqual(len(promoted_roots), 1)
        self.assertIsInstance(promoted_root, Node)
        self.assertFalse(hasattr(promoted_root, "parent"))


class TerminalTest(unittest.TestCase):
    class FailIfEvaluatedModel:
        def forward(self, accumulator):
            raise AssertionError("terminal leaves must not call the neural network")

    def test_terminal_leaf_never_calls_nn_and_propagates_exactly_one(self):
        self._assert_terminal_result(player=1, expected=1)

    def test_terminal_leaf_never_calls_nn_and_propagates_exactly_minus_one(self):
        self._assert_terminal_result(player=2, expected=-1)

    def test_edge_terminal_check_is_default_and_last_move_is_opt_in(self):
        self.assertFalse(
            MCTS(Node(HexBoardState()), None, 1).use_last_move_terminal_check
        )
        self.assertTrue(
            MCTS(
                Node(HexBoardState()),
                None,
                1,
                use_last_move_terminal_check=True,
            ).use_last_move_terminal_check
        )

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

    def test_last_move_terminal_check_matches_reference_on_legal_extensions(self):
        """Exhaust every legal next move from varied legal non-terminal paths."""
        generator = random.Random(83)
        for _ in range(12):
            node = Node(HexBoardState())
            while not node.is_terminal(use_last_move_terminal_check=False):
                legal_indices = list(node.iter_legal_move_indices())
                for move_idx in legal_indices:
                    move_bit = 1 << move_idx
                    p1_bits = (
                        node.p1_bits | move_bit if node.p1_turn else node.p1_bits
                    )
                    p2_bits = (
                        node.p2_bits if node.p1_turn else node.p2_bits | move_bit
                    )
                    reference = Node(
                        None,
                        move_index=node.move_index + 1,
                        p1_bits=p1_bits,
                        p2_bits=p2_bits,
                    )
                    experimental = Node(
                        None,
                        move_index=node.move_index + 1,
                        p1_bits=p1_bits,
                        p2_bits=p2_bits,
                    )
                    self.assertEqual(
                        experimental.is_terminal(
                            last_move_idx=move_idx,
                            use_last_move_terminal_check=True,
                        ),
                        reference.is_terminal(use_last_move_terminal_check=False),
                    )
                    self.assertEqual(
                        experimental.stored_terminal_outcome,
                        reference.stored_terminal_outcome,
                    )

                move_idx = generator.choice(legal_indices)
                node = node.create_edge(move_idx).child


class PerspectiveTest(unittest.TestCase):
    def test_run_inline_backpropagation_maintains_edge_caches(self):
        root = Node(HexBoardState())
        MCTS(root, None, 2).run()

        self.assertEqual(root.total_edge_visits, 1)
        self.assertEqual(len(root.children), 1)
        edge = root.children[0]
        self.assertEqual(edge.N, 1)
        self.assertEqual(edge.Q(), edge.W)
        self.assertEqual(edge.selection_q, edge.W)
        self.assertEqual(edge.puct_weight, edge.prior / 2)

    def test_same_q_player_one_prefers_larger_and_player_two_smaller(self):
        self.assertEqual(self._selected_q(move_index=0), 0.75)
        self.assertEqual(self._selected_q(move_index=1), -0.25)

    @staticmethod
    def _selected_q(move_index):
        parent = Node(HexBoardState(), move_index=move_index)
        lower = parent.create_edge(parent.legal_moves[0])
        higher = parent.create_edge(parent.legal_moves[1])
        lower.backpropagate(-0.25)
        higher.backpropagate(0.75)
        parent._install_policy([1 / 25] * 25)

        action = parent.get_best_action_using_puct(c=0)
        return parent.edge_for_move(action).Q()


class PUCTTest(unittest.TestCase):
    def test_sparse_selection_matches_exhaustive_selection(self):
        random.seed(11)
        parent = Node(HexBoardState())
        parent.store_policy(torch.randn(HEX_BOARD_SIZE * HEX_BOARD_SIZE))
        for move in random.sample(parent.legal_moves, 8):
            edge = parent.create_edge(move)
            for _ in range(random.randint(1, 8)):
                edge.backpropagate(random.uniform(-1, 1))
        parent.total_edge_visits = 40

        expected = None
        expected_score = -float("inf")
        exploration_factor = 1.4 * parent.total_edge_visits ** 0.5
        for move_idx in sorted(parent.iter_legal_move_indices()):
            edge = parent.edge_for_move(move_idx)
            n = 0 if edge is None else edge.N
            q = 0 if edge is None else edge.Q()
            score = q + exploration_factor * parent.stored_policy[move_idx] / (1 + n)
            if score > expected_score:
                expected = move_idx
                expected_score = score

        self.assertEqual(parent.get_best_action_using_puct(c=1.4), expected)

    def test_higher_policy_prior_is_selected_when_q_and_n_are_equal(self):
        parent = Node(HexBoardState())
        low_prior_move, high_prior_move = parent.legal_moves[:2]
        low_prior_edge = parent.create_edge(low_prior_move)
        high_prior_edge = parent.create_edge(high_prior_move)
        for edge in (low_prior_edge, high_prior_edge):
            for _ in range(3):
                edge.backpropagate(0.5)
        parent.total_edge_visits = 10

        policy = [0.0] * (HEX_BOARD_SIZE * HEX_BOARD_SIZE)
        policy[low_prior_move.get_idx()] = 0.2
        policy[high_prior_move.get_idx()] = 0.8
        parent._install_policy(policy)

        self.assertEqual(
            parent.get_best_action_using_puct(c=1.0),
            high_prior_move.idx,
        )


if __name__ == "__main__":
    unittest.main()
