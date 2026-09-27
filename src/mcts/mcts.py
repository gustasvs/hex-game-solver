import numpy as np
import random
import math
import weakref
from collections.abc import Iterable, MutableMapping

import time

import torch

from game_logic.hex import HexBoardState
from settings import HEX_BOARD_SIZE, MCTS_FORCE_EXPAND_ALL
from mcts.policy import DeferredPolicy
from mcts.bitboard_helpers import BOARD_MASK

from pytorch_model import CustomNNUE
from game_logic.classes import Move

np.random.seed(42)

from mcts.bitboard_helpers import (
    BOTTOM_EDGE_MASK,
    LEFT_EDGE_MASK,
    RIGHT_EDGE_MASK,
    TOP_EDGE_MASK,
    canonical_evaluation_key,
    has_bit_connection,
    has_bit_connection_from,
    rotate_policy_180,
)


# Keep ordinary standalone Node callers fast too.  MCTS extends this table once
# at construction for the full search budget, so selectors only list-index it.
NUMBERS_CACHE_SIZE = 16_385

_PUCT_SQRT_VISITS = [math.sqrt(index) for index in range(NUMBERS_CACHE_SIZE)]
_EXPLORATION_FACTORS = [1.4 * math.sqrt(index) for index in range(NUMBERS_CACHE_SIZE)]
_INV_VISITS = [0.0] + [1.0 / index for index in range(1, NUMBERS_CACHE_SIZE)]

# Edge-seeded connectivity is the measured production default.  The
# last-move-seeded equivalent remains available as an explicit experiment for
# comparison, but was slower in both full benchmark scenarios.
USE_LAST_MOVE_TERMINAL_CONNECTIVITY = False

class TreeContext:
    """One game's tables; nodes retain this single ownership handle."""

    __slots__ = ("evaluation_cache", "node_cache", "nodes_pruned")

    def __init__(self, evaluation_cache=None, node_cache=None):
        self.evaluation_cache = {} if evaluation_cache is None else evaluation_cache
        self.node_cache = {} if node_cache is None else node_cache
        self.nodes_pruned = 0

    def promote(self, root, threshold=50_000):
        """Drop unreachable TT entries only when a scan can repay its cost."""
        table = self.node_cache
        if len(table) < threshold:
            return 0
        root_p1, root_p2 = root.p1_bits, root.p2_bits
        doomed = [
            key for key in table
            if (key[0] & root_p1) != root_p1 or (key[1] & root_p2) != root_p2
        ]
        for key in doomed:
            del table[key]
        count = len(doomed)
        self.nodes_pruned += count
        return count


class SearchContext:
    """Model-generation state shared by every MCTS tree using one cache."""

    __slots__ = (
        "_model_ref",
        "model_signature",
        "move_delta_views",
        "inference_parameters",
        "evaluation_cache",
    )

    def __init__(self, model: CustomNNUE, evaluation_cache: dict):
        self._model_ref = weakref.ref(model)
        self.model_signature = tuple(parameter._version for parameter in model.parameters())
        move_deltas = model.calculate_move_deltas().detach()
        self.move_delta_views = (tuple(move_deltas[0]), tuple(move_deltas[1]))
        self.inference_parameters = tuple(
            parameter.detach() for parameter in model.get_inference_parameters()
        )
        self.evaluation_cache = evaluation_cache

    def matches_current_model(self, model: CustomNNUE) -> bool:
        return (
            self._model_ref() is model
            and self.model_signature == tuple(parameter._version for parameter in model.parameters())
        )


# Context values hold only weak references to their model, so this registry does
# not keep retired model generations alive.  Distinct external cache dictionaries
# deliberately receive distinct contexts.
_MODEL_CONTEXTS = weakref.WeakKeyDictionary()


def get_search_context(model: CustomNNUE, evaluation_cache: dict) -> SearchContext:
    contexts = _MODEL_CONTEXTS.setdefault(model, [])
    for context in tuple(contexts):
        if context.evaluation_cache is evaluation_cache:
            if context.matches_current_model(model):
                return context
            contexts.remove(context)
            break
    context = SearchContext(model, evaluation_cache)
    contexts.append(context)
    return context


def ensure_visit_table_capacity(maximum_visits):
    """Extend arithmetic tables once, outside the selection/backprop loops."""
    if maximum_visits >= len(_PUCT_SQRT_VISITS):
        first_new_index = len(_PUCT_SQRT_VISITS)
        _PUCT_SQRT_VISITS.extend(
            math.sqrt(index)
            for index in range(first_new_index, maximum_visits + 1)
        )
        _INV_VISITS.extend(
            1.0 / index
            for index in range(first_new_index, maximum_visits + 1)
        )


def puct_exploration_factor(c, total_edge_visits):
    """Reference helper for callers outside MCTS's pre-sized selector paths."""
    visits = max(1, total_edge_visits)
    ensure_visit_table_capacity(visits)
    return c * _PUCT_SQRT_VISITS[visits]


class Edge:
    __slots__ = (
        "parent",
        "child",
        "move_idx",
        "is_transposition",
        "selection_q",
        "N",
        "W",
        "prior",
        "puct_weight",
    )

    def __init__(self, parent: Node, child: Node, move_idx: int, prior=0, is_transposition=False):
        self.parent = parent
        self.child = child
        self.move_idx = move_idx
        self.is_transposition = is_transposition

        # cached values for efficiency
        self.selection_q = 0.0

        # Update statistics via backpropagate so all selection caches stay valid.
        self.N = 0 # visit count
        self.W = 0 # accumulated value
        self.prior = prior
        self.puct_weight = prior

    def Q(self):
        if self.N <= 0:
            return 0
        ensure_visit_table_capacity(self.N)
        return self.W * _INV_VISITS[self.N]

    def backpropagate(self, result):
        self.N += 1
        # since score is kept from p1's perspective, we add the result directly
        self.W += result
        self.parent.total_edge_visits += 1
        ensure_visit_table_capacity(self.N + 1)
        q = self.W * _INV_VISITS[self.N]
        self.selection_q = q if self.parent.p1_turn else -q
        self.puct_weight = self.prior * _INV_VISITS[self.N + 1]

class Node:
    __slots__ = (
        "_state",
        "p1_bits",
        "p2_bits",
        "tree_context",
        "search_context",
        "legal_mask",
        "ordered_move_indices",
        "children",
        "stored_policy",
        "stored_value",
        "stored_terminal",
        "stored_terminal_outcome",
        "stored_accumulator",
        "unexpanded_count",
        "total_edge_visits",
        "move_index",
        "p1_turn",
    )

    def __init__(self, state: HexBoardState, move_index=0, legal_moves: Iterable[Move] | None = None, p1_bits=None, p2_bits=None, evaluation_cache=None, search_context=None, node_cache=None, tree_context=None):
        self._state = state
        if p1_bits is None or p2_bits is None:
            self.p1_bits = sum(
                self.state.p1[row][col] << (row * HEX_BOARD_SIZE + col)
                for row in range(HEX_BOARD_SIZE)
                for col in range(HEX_BOARD_SIZE)
            )
            self.p2_bits = sum(
                self.state.p2[row][col] << (row * HEX_BOARD_SIZE + col)
                for row in range(HEX_BOARD_SIZE)
                for col in range(HEX_BOARD_SIZE)
            )
        else:
            self.p1_bits = p1_bits
            self.p2_bits = p2_bits
        if tree_context is None:
            tree_context = TreeContext(evaluation_cache, node_cache)
        elif evaluation_cache is not None or node_cache is not None:
            raise ValueError("pass tree_context or individual caches, not both")
        self.tree_context = tree_context
        if search_context is not None:
            if (
                evaluation_cache is not None
                and evaluation_cache is not search_context.evaluation_cache
            ):
                raise ValueError("evaluation_cache must match search_context")
            self.search_context = search_context
            if self.evaluation_cache is not search_context.evaluation_cache:
                raise ValueError("tree_context cache must match search_context")
        else:
            self.search_context = None
        self.node_cache[(self.p1_bits, self.p2_bits)] = self
        self.legal_mask = BOARD_MASK & ~(self.p1_bits | self.p2_bits)
        if legal_moves is not None:
            supplied_mask = 0
            for move in legal_moves:
                move_idx = move.idx if isinstance(move, Move) else int(move)
                supplied_mask |= 1 << move_idx
            if supplied_mask != self.legal_mask:
                raise ValueError("legal_moves does not match the node bitboards")
        self.ordered_move_indices = None
        self.children = []
        self.stored_policy = None
        self.stored_value = None
        self.stored_terminal = None
        self.stored_terminal_outcome = None
        self.stored_accumulator = None
        self.unexpanded_count = None
        self.total_edge_visits = 0
        
        # on initial empty game its 0, first p1 move is 1, then p2 move is 2, and so on
        self.move_index = move_index
        self.p1_turn = move_index % 2 == 0
        
    def is_p1_turn(self):
        return self.p1_turn

    @property
    def state(self):
        # Search positions are immutable. Materialize only for callers which need
        # the nested-list representation; search itself uses the bitboards.
        if self._state is None:
            self._state = HexBoardState.from_bitboards(self.p1_bits, self.p2_bits)
        return self._state

    @property
    def evaluation_cache(self):
        return self.tree_context.evaluation_cache

    @property
    def node_cache(self):
        return self.tree_context.node_cache

    def promote_to_root(self, threshold=50_000):
        return self.tree_context.promote(self, threshold)

    def iter_legal_move_indices(self):
        mask = self.legal_mask
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    @property
    def legal_moves(self):
        """Compatibility/UI boundary; search itself stores only integer moves."""
        return [
            Move(move_idx // HEX_BOARD_SIZE, move_idx % HEX_BOARD_SIZE)
            for move_idx in self.iter_legal_move_indices()
        ]

    def create_edge(self, move) -> Edge:
        move_idx = move.idx if isinstance(move, Move) else int(move)
        existing_edge = self.edge_for_move(move_idx)
        if existing_edge is not None:
            return existing_edge

        return self._create_edge(move_idx)

    def edge_for_move(self, move_idx: int) -> Edge | None:
        """Compatibility lookup used outside the normal direct-selection path."""
        for edge in self.children:
            if edge.move_idx == move_idx:
                return edge
        return None

    def _create_edge(self, move_idx: int) -> Edge:
        """Create a known-unexpanded edge without a redundant map lookup."""
        if not (self.legal_mask & (1 << move_idx)):
            raise ValueError(f"move index {move_idx} is not legal")

        if self.unexpanded_count is not None and self.unexpanded_count < 0:
            self._prepare_policy()

        move_bit = 1 << move_idx
        child_p1_bits = self.p1_bits | move_bit if self.p1_turn else self.p1_bits
        child_p2_bits = self.p2_bits if self.p1_turn else self.p2_bits | move_bit
        child_position_key = (child_p1_bits, child_p2_bits)
        child_node = self.node_cache.get(child_position_key)
        is_transposition = child_node is not None
        if child_node is None:
            child_node = Node(
                None,
                move_index=self.move_index + 1,
                p1_bits=child_p1_bits,
                p2_bits=child_p2_bits,
                search_context=self.search_context,
                tree_context=self.tree_context,
            )
        prior = 0 if self.stored_policy is None else self.stored_policy[move_idx]
        edge = Edge(self, child_node, move_idx, prior, is_transposition)
        self.children.append(edge)
        if self.unexpanded_count is not None:
            if self.ordered_move_indices[self.unexpanded_count - 1] == move_idx:
                self.unexpanded_count -= 1
            else:
                self.unexpanded_count = None
        return edge
    
    def get_best_action_using_puct(self, c) -> int:
        if self.unexpanded_count is not None and self.unexpanded_count < 0:
            self._prepare_policy()
        if self.unexpanded_count is not None:
            edge, move_idx = self._select_partially_expanded_puct(c)
            return edge.move_idx if edge is not None else move_idx

        q = None
        best_score = -float('inf')

        assert self.stored_policy is not None
        assert self.legal_mask  # important invariant

        best_move_idx = None
        p1_turn = self.is_p1_turn()
        # exploration_factor = c * _PUCT_SQRT_VISITS[max(1, self.total_edge_visits)]
        exploration_factor = _EXPLORATION_FACTORS[max(1, self.total_edge_visits)]
        stored_policy = self.stored_policy
        move_edge_map = {edge.move_idx: edge for edge in self.children}

        for move_idx in self.iter_legal_move_indices():
            prior = stored_policy[move_idx]
            matching_edge = move_edge_map.get(move_idx)

            if matching_edge is not None:
                q = matching_edge.Q()
                puct_weight = matching_edge.puct_weight
            else:
                q = 0
                puct_weight = prior

            if not p1_turn:
                q = -q

            exploration = exploration_factor * puct_weight
            score = q + exploration
            if best_move_idx is None or score > best_score:
                best_score = score
                best_move_idx = move_idx

        assert best_move_idx is not None
        return best_move_idx

    def _select_partially_expanded_puct(self, c) -> tuple[Edge | None, int | None]:
        """Return the winning existing edge or a known-unexpanded move index."""
        assert self.stored_policy is not None
        assert self.unexpanded_count is not None

        # exploration_factor = c * _PUCT_SQRT_VISITS[max(1, self.total_edge_visits)]
        exploration_factor = _EXPLORATION_FACTORS[max(1, self.total_edge_visits)]
        best_edge = None
        best_move_idx = None
        best_score = -float('inf')

        if self.unexpanded_count:
            best_move_idx = self.ordered_move_indices[self.unexpanded_count - 1]
            best_score = exploration_factor * self.stored_policy[best_move_idx]

        for edge in self.children:
            q = edge.selection_q
            score = q + exploration_factor * edge.puct_weight
            if (
                best_move_idx is None
                or score > best_score
                or (score == best_score and edge.move_idx < best_move_idx)
            ):
                best_score = score
                best_edge = edge
                best_move_idx = edge.move_idx

        assert best_move_idx is not None
        return best_edge, None if best_edge is not None else best_move_idx

    def store_policy(self, policy):
        self._install_policy(DeferredPolicy(policy.tolist(), self.iter_legal_move_indices()))

    def _install_policy(self, probabilities):
        if self.children and isinstance(probabilities, DeferredPolicy):
            probabilities = probabilities.materialize()
        self.stored_policy = probabilities
        for edge in self.children:
            edge.prior = probabilities[edge.move_idx]
            edge.puct_weight = edge.prior * _INV_VISITS[edge.N + 1]
        if not self.children:
            # A negative count means ordering/normalization is still deferred.
            self.unexpanded_count = -self.legal_mask.bit_count()

    def _prepare_policy(self):
        if isinstance(self.stored_policy, DeferredPolicy):
            self.stored_policy = self.stored_policy.materialize()
        probabilities = self.stored_policy
        self.ordered_move_indices = list(self.iter_legal_move_indices())
        self.ordered_move_indices.sort(
            key=lambda move_idx: (probabilities[move_idx], -move_idx)
        )
        self.unexpanded_count = len(self.ordered_move_indices)
    
    def rollout(
        self,
        model,
        shared_accumulator,
        inference_parameters=None,
        *,
        last_move_idx=None,
        use_last_move_terminal_check=USE_LAST_MOVE_TERMINAL_CONNECTIVITY,
    ):
        if self.stored_terminal is None:
            self.is_terminal(
                last_move_idx=last_move_idx,
                use_last_move_terminal_check=use_last_move_terminal_check,
            )

        if self.stored_terminal:
            return self.stored_terminal_outcome, None

        if shared_accumulator is None:
            return self.state.fast_random_rollout_bits(self.p1_bits, self.p2_bits, self.p1_turn), None

        value, policy = model.forward(shared_accumulator, inference_parameters)
        return value.item(), policy
    
    def is_terminal(
        self,
        last_move_idx=None,
        *,
        use_last_move_terminal_check=USE_LAST_MOVE_TERMINAL_CONNECTIVITY,
    ):
        
        if self.stored_terminal is not None:
            return self.stored_terminal

        # earliest possible win, no need to check earlier
        if self.move_index < 2 * HEX_BOARD_SIZE - 1:
            self.stored_terminal = False
            return False
        
        won = False

        # P1 just moved, so only P1 could have created a new win.
        if not self.p1_turn:
            last_move_bit = 0 if last_move_idx is None else 1 << last_move_idx
            if (
                use_last_move_terminal_check
                and last_move_bit & self.p1_bits
            ):
                won = has_bit_connection_from(
                    self.p1_bits,
                    last_move_bit,
                    TOP_EDGE_MASK,
                    BOTTOM_EDGE_MASK,
                )
            else:
                touches_both_edges = (
                    (self.p1_bits & TOP_EDGE_MASK) != 0
                    and (self.p1_bits & BOTTOM_EDGE_MASK) != 0
                )

                if touches_both_edges:
                    won = has_bit_connection(
                        self.p1_bits,
                        TOP_EDGE_MASK,
                        BOTTOM_EDGE_MASK,
                    )

            if won:
                self.stored_terminal_outcome = 1

        # P2 just moved, so only P2 could have created a new win.
        else:
            last_move_bit = 0 if last_move_idx is None else 1 << last_move_idx
            if (
                use_last_move_terminal_check
                and last_move_bit & self.p2_bits
            ):
                won = has_bit_connection_from(
                    self.p2_bits,
                    last_move_bit,
                    LEFT_EDGE_MASK,
                    RIGHT_EDGE_MASK,
                )
            else:
                touches_both_edges = (
                    (self.p2_bits & LEFT_EDGE_MASK) != 0
                    and (self.p2_bits & RIGHT_EDGE_MASK) != 0
                )

                if touches_both_edges:
                    won = has_bit_connection(
                        self.p2_bits,
                        LEFT_EDGE_MASK,
                        RIGHT_EDGE_MASK,
                    )

            if won:
                self.stored_terminal_outcome = -1

        board_full = (self.p1_bits | self.p2_bits) == BOARD_MASK
        self.stored_terminal = won or board_full

        if self.stored_terminal and self.stored_terminal_outcome is None:
            self.stored_terminal_outcome = 0

        return self.stored_terminal
    
    def is_p1_win(self):
        if self.stored_terminal is None:
            self.is_terminal()
        return self.stored_terminal_outcome == 1
    
    # LOGGING
    
    def print_visits_and_score(self):
        print(f"Visits: {self.total_edge_visits}")
        
    def get_total_children(self):
        # returns unique created position nodes
        return max(0, len(self.node_cache) - 1)
        
                

def best_expanded_edge(node: Node, c) -> Edge:
    """Return the PUCT winner directly when every move is expanded."""
    # exploration_factor = c * _PUCT_SQRT_VISITS[max(1, node.total_edge_visits)]
    exploration_factor = _EXPLORATION_FACTORS[max(1, node.total_edge_visits)]
    best_edge = None
    best_score = -float('inf')

    for edge in node.children:
        score = (
            edge.selection_q
            + exploration_factor * edge.puct_weight
        )
        if (
            best_edge is None
            or score > best_score
            or (
                score == best_score
                and edge.move_idx < best_edge.move_idx
            )
        ):
            best_score = score
            best_edge = edge

    assert best_edge is not None
    return best_edge


class MCTS:
    
    def __init__(
        self,
        root: Node,
        model: CustomNNUE | None,
        max_iterations: int,
        *,
        force_expand_all: bool = MCTS_FORCE_EXPAND_ALL,
        use_last_move_terminal_check: bool = USE_LAST_MOVE_TERMINAL_CONNECTIVITY,
    ):
        # root is an empty Node representing a new game
        self.root = root
        self.model = model
        self.force_expand_all = force_expand_all
        self.use_last_move_terminal_check = use_last_move_terminal_check
        
        self.c = 1.4 # exploration constant
        assert max_iterations > 0, "max_iterations must be positive"
        self.max_iterations = max_iterations # maximum number of iterations to perform
        ensure_visit_table_capacity(
            self.root.total_edge_visits + self.max_iterations + 1
        )
        
        # DEBUG TIMERS
        self.total_selection_time = 0
        self.total_rollout_time = 0
        self.total_backpropagation_time = 0
        self.terminal_check_time = 0
        self.untested_child_check_time = 0
        self.calls_to_best_child_by_uct = 0
        self.total_uct_children_examined = 0
        # Legacy aggregate counters remain for existing callers.  The split
        # counters below distinguish node-local reuse from shared-cache lookup.
        self.evaluation_cache_hits = 0
        self.evaluation_cache_misses = 0
        self.node_value_hits = 0
        self.cross_game_cache_hits = 0
        self.cross_game_cache_misses = 0
        self.cross_game_cache_hits_by_move_index = [0] * (HEX_BOARD_SIZE ** 2 + 1)
        self.cross_game_cache_misses_by_move_index = [0] * (HEX_BOARD_SIZE ** 2 + 1)
        self.accumulator_creations = 0
        self._root_accumulator_created = False
        self.edges_created = 0
        self.transposition_hits = 0
        self.transposition_hits_by_move_index = [0] * (HEX_BOARD_SIZE ** 2 + 1)
        self.transposition_misses_by_move_index = [0] * (HEX_BOARD_SIZE ** 2 + 1)
        self.unique_nodes_created = 0
        self.nn_evaluations = 0
        self.nn_evaluations_cached = 0
        
        self.move_delta_views = None
        self.inference_parameters = None
        self.search_context = None
        self._selection_path: list[Edge] = []

        if model is not None:
            if self.root.search_context is None:
                self.root.search_context = get_search_context(
                    model,
                    self.root.evaluation_cache,
                )
            elif not self.root.search_context.matches_current_model(model):
                raise RuntimeError(
                    "Cannot reuse an MCTS tree after model weights change"
                )
            if self.root.evaluation_cache is not self.root.search_context.evaluation_cache:
                raise RuntimeError("MCTS node cache does not match its search context")

            self.search_context = self.root.search_context

            if self.root.stored_accumulator is None:
                self.root.stored_accumulator = (
                    model.calculate_accumulator(
                        self.root.state.get_state(),
                        self.root.is_p1_turn()
                    ).detach()
                )
                self._root_accumulator_created = True

            self.move_delta_views = self.search_context.move_delta_views
            self.inference_parameters = self.search_context.inference_parameters
            
    
    def run(self) -> Node:
        start_time = time.time()
        self.total_selection_time = 0
        self.total_rollout_time = 0
        self.total_backpropagation_time = 0
        self.terminal_check_time = 0
        self.untested_child_check_time = 0
        self.score_calc_time = 0
        self.calls_to_best_child_by_uct = 0
        self.total_uct_children_examined = 0
        self.evaluation_cache_hits = 0
        self.evaluation_cache_misses = 0
        self.node_value_hits = 0
        self.cross_game_cache_hits = 0
        self.cross_game_cache_misses = 0
        self.cross_game_cache_hits_by_move_index = [0] * (HEX_BOARD_SIZE ** 2 + 1)
        self.cross_game_cache_misses_by_move_index = [0] * (HEX_BOARD_SIZE ** 2 + 1)
        self.accumulator_creations = int(self._root_accumulator_created)
        self._root_accumulator_created = False
        self.edges_created = 0
        self.transposition_hits = 0
        self.transposition_hits_by_move_index = [0] * (HEX_BOARD_SIZE ** 2 + 1)
        self.transposition_misses_by_move_index = [0] * (HEX_BOARD_SIZE ** 2 + 1)
        self.unique_nodes_created = 0
        self.nn_evaluations = 0
        self.nn_evaluations_cached = 0
        inv_visits = _INV_VISITS
        
        for _ in range(self.max_iterations):
            
            # 1: Selection phase
            # at this point this loop doesnt care how, but we get a good candidate to analyse
            selection_start = time.time()
            selected, selected_path = self._select(self.root)
            selection_end = time.time()
            self.total_selection_time += selection_end - selection_start
            if selected is None:
                print("No valid child found during selection.")
                break
            
            # 2: Rollout phase
            rollout_start = time.time()
            if self.model is None:
                # ROLLOUT FOR NO MODEL
                outcome, policy = selected.rollout(
                    None,
                    None,
                    None,
                    last_move_idx=(
                        selected_path[-1].move_idx if selected_path else None
                    ),
                    use_last_move_terminal_check=self.use_last_move_terminal_check,
                )

                if not selected.stored_terminal and selected.stored_policy is None:
                    probability = 1.0 / selected.legal_mask.bit_count()

                    uniform_policy = [0.0] * (HEX_BOARD_SIZE ** 2)

                    for move_idx in selected.iter_legal_move_indices():
                        uniform_policy[move_idx] = probability

                    selected._install_policy(uniform_policy)

            else:
                # TRADITIONAL PUCT WHEN MODEL EXISTS
                if selected.stored_value is not None:
                    outcome = selected.stored_value
                    policy = None
                    self.node_value_hits += 1
                    self.evaluation_cache_hits += 1
                    if selected.stored_policy is not None:
                        self.nn_evaluations_cached += 1
                else:
                    evaluation_key, position_is_rotated = canonical_evaluation_key(
                        selected.p1_bits,
                        selected.p2_bits,
                    )
                    cached_evaluation = selected.evaluation_cache.get(evaluation_key)
                    if cached_evaluation is None:
                        last_move_idx = (
                            selected_path[-1].move_idx if selected_path else None
                        )
                        if selected.stored_terminal is None:
                            selected.is_terminal(
                                last_move_idx=last_move_idx,
                                use_last_move_terminal_check=(
                                    self.use_last_move_terminal_check
                                ),
                            )
                        if selected.stored_terminal:
                            outcome, policy = selected.stored_terminal_outcome, None
                        else:
                            outcome, policy = selected.rollout(
                                self.model,
                                self._materialize_selected_accumulator(
                                    selected,
                                    selected_path,
                                ),
                                self.inference_parameters,
                                last_move_idx=last_move_idx,
                                use_last_move_terminal_check=(
                                    self.use_last_move_terminal_check
                                ),
                            )
                        selected.stored_value = outcome
                        self.cross_game_cache_misses += 1
                        self.cross_game_cache_misses_by_move_index[selected.move_index] += 1
                        self.evaluation_cache_misses += 1
                        if policy is not None:
                            self.nn_evaluations += 1
                    else:
                        outcome, cached_policy = cached_evaluation
                        selected.stored_value = outcome
                        if position_is_rotated:
                            cached_policy = rotate_policy_180(cached_policy)
                        selected._install_policy(cached_policy)
                        policy = None
                        self.cross_game_cache_hits += 1
                        self.cross_game_cache_hits_by_move_index[selected.move_index] += 1
                        self.evaluation_cache_hits += 1
                        self.nn_evaluations_cached += 1
            rollout_end = time.time()
            self.total_rollout_time += rollout_end - rollout_start
            
            # 3: Backpropagation phase
            backpropagation_start = time.time()
            if policy is not None:
                selected.store_policy(policy)
                selected.stored_value = outcome
                cached_policy = selected.stored_policy
                if position_is_rotated:
                    cached_policy = rotate_policy_180(cached_policy)
                selected.evaluation_cache[evaluation_key] = (
                    outcome, cached_policy
                )
            # Keep this tiny update inline: it runs once for every traversed
            # edge and mirrors Edge.backpropagate for external callers/tests.
            for edge in reversed(selected_path):
                edge.N += 1
                edge.W += outcome
                parent = edge.parent
                parent.total_edge_visits += 1
                inv_n = inv_visits[edge.N]
                q = edge.W * inv_n
                edge.selection_q = q if parent.p1_turn else -q
                edge.puct_weight = edge.prior * inv_visits[edge.N + 1]
            backpropagation_end = time.time()
            self.total_backpropagation_time += backpropagation_end - backpropagation_start

        # print("*" * 20)
        # print(f"TOTAL TIME: {time.time() - start_time}")
        # print(f"Total selection time: {self.total_selection_time}")
        # print(f"    > Total terminal check time: {self.terminal_check_time}")
        # print(f"    > Total untested child check time: {self.untested_child_check_time}")
        # print(f"    > Total score calculation time: {self.score_calc_time}")
        # print(f"Total rollout time: {self.total_rollout_time}")
        # print(f"Total backpropagation time: {self.total_backpropagation_time}")
        # print(" > COUNTS:")
        # explored_children = self.root.get_total_children()
        # print(f"    > Total explored children: {explored_children} / {self.max_iterations}")
        # print(f"    > Calls to best child by UCT: {self.calls_to_best_child_by_uct}")
        # print(f"    > Total UCT children examined: {self.total_uct_children_examined}")
        # print(f"    > Evaluation cache hits: {self.evaluation_cache_hits}")
        # print(f"    > Evaluation cache misses: {self.evaluation_cache_misses}")
        # print(f"    > Edges created: {self.edges_created}")
        # print(f"    > Transposition hits: {self.transposition_hits}")
        # print(f"    > Unique nodes created: {self.unique_nodes_created}")
        # print(f"    > NN evaluations: {self.nn_evaluations}")
        # print(f"    > NN evaluations cached: {self.nn_evaluations_cached}")
        return self.root
    
    def select(self, node: Node) -> tuple[Node, list[Edge]]:
        """Select for external callers, returning a stable path snapshot.

        MCTS.run uses _select directly to reuse its private path list.  A copy
        here prevents callers holding a prior select() result from observing a
        later selection's clear()/append mutations.
        """
        selected, selected_path = self._select(node)
        return selected, list(selected_path)

    def _select(self, node: Node) -> tuple[Node, list[Edge]]:
        """Hot selection path; valid only until the next _select call."""
        selected_path = self._selection_path
        selected_path.clear()
        incoming_move_idx = None
        while True:
            is_terminal = node.stored_terminal
            if is_terminal is None:
                is_terminal = node.is_terminal(
                    last_move_idx=incoming_move_idx,
                    use_last_move_terminal_check=self.use_last_move_terminal_check,
                )
            if is_terminal or node.stored_policy is None:
                break

            # A cache hit deliberately leaves a leaf without an accumulator.
            # Once that same node is traversed as an internal node, materialize
            # it once so its new child needs only one final move delta.
            if self.model is not None and node.stored_accumulator is None:
                self._materialize_selected_accumulator(node, selected_path)
            
            if node.unexpanded_count == 0:
                # The PUCT scan already has the winning edge. Keep it instead of
                # returning its action and looking the same edge up again.
                edge = best_expanded_edge(node, self.c)
                created_edge = False
            else:
                if node.unexpanded_count is not None:
                    if node.unexpanded_count < 0:
                        node._prepare_policy()
                    if self.force_expand_all:
                        # The ordered tail is guaranteed not to have an edge.
                        edge = None
                        move_idx = node.ordered_move_indices[node.unexpanded_count - 1]
                    else:
                        edge, move_idx = node._select_partially_expanded_puct(self.c)
                else:
                    move_idx = node.get_best_action_using_puct(self.c)
                    edge = node.edge_for_move(move_idx)
                created_edge = edge is None

            self.calls_to_best_child_by_uct += 1
            self.total_uct_children_examined += len(node.children)

            if created_edge:
                edge = node._create_edge(move_idx)
                self.edges_created += 1
                if edge.is_transposition:
                    self.transposition_hits += 1
                    self.transposition_hits_by_move_index[node.move_index + 1] += 1
                else:
                    self.unique_nodes_created += 1
                    self.transposition_misses_by_move_index[node.move_index + 1] += 1

            selected_path.append(edge)
            node = edge.child
            incoming_move_idx = edge.move_idx
            if created_edge:
                break

        return node, selected_path

    def _materialize_selected_accumulator(
        self,
        selected: Node,
        selected_path: list[Edge],
    ):
        """Materialize a node from the closest accumulated path prefix.

        Cache-hit leaves intentionally remain without an accumulator.  A hit
        that later becomes an internal selection node calls this helper before
        it descends; newly selected cache-miss leaves can then take the fast
        path below and apply only their final move delta.
        """
        if selected.stored_accumulator is not None:
            return selected.stored_accumulator

        assert self.model is not None
        assert self.root.stored_accumulator is not None

        # Normal selection has just descended from a materialized internal
        # node.  Avoid another full path scan for the cache-miss leaf.
        if selected_path:
            final_edge = selected_path[-1]
            if final_edge.child is selected:
                parent_accumulator = final_edge.parent.stored_accumulator
                if parent_accumulator is not None:
                    player_index = 0 if final_edge.parent.p1_turn else 1
                    selected.stored_accumulator = (
                        parent_accumulator
                        + self.move_delta_views[player_index][final_edge.move_idx]
                    )
                    self.accumulator_creations += 1
                    return selected.stored_accumulator

        accumulator = self.root.stored_accumulator
        first_unmaterialized_edge = 0
        for path_index, edge in enumerate(selected_path):
            child_accumulator = edge.child.stored_accumulator
            if child_accumulator is not None:
                accumulator = child_accumulator
                first_unmaterialized_edge = path_index + 1

        for edge in selected_path[first_unmaterialized_edge:]:
            player_index = 0 if edge.parent.p1_turn else 1
            accumulator = (
                accumulator
                + self.move_delta_views[player_index][edge.move_idx]
            )

        selected.stored_accumulator = accumulator
        self.accumulator_creations += 1
        return accumulator
