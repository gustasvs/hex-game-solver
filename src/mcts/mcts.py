import numpy as np
import math
import random

import time

import torch

from game_logic.hex import HexBoardState
from settings import HEX_BOARD_SIZE

from pytorch_model import CustomNNUE
from game_logic.classes import Move
from collections.abc import Iterable

np.random.seed(42)

PathMove = tuple[bool, int, int]
from mcts.bitboard_helpers import (
    BOTTOM_EDGE_MASK,
    LEFT_EDGE_MASK,
    RIGHT_EDGE_MASK,
    TOP_EDGE_MASK,
    canonical_evaluation_key,
    has_bit_connection,
    rotate_policy_180,
)

class Edge:
    def __init__(self, parent: Node, child: Node, action: Move, prior=0, is_transposition=False):
        self.parent = parent
        self.child = child
        self.action = action
        self.move_from_parent: PathMove = (
            parent.is_p1_turn(), action.x, action.y
        )
        self.prior = prior
        self.is_transposition = is_transposition

        # cached values for efficiency
        self.stored_uct_factor = None
        self.stored_puct_factor = 1.0
        self.stored_q = None

        self.N = 0 # visit count
        self.W = 0 # accumulated value

    def Q(self):
        if self.stored_q is not None:
            return self.stored_q

        self.stored_q = self.W / self.N if self.N > 0 else 0
        return self.stored_q

    def backpropagate(self, result):
        self.N += 1
        # since score is kept from p1's perspective, we add the result directly
        self.W += result
        self.parent.total_edge_visits += 1
        self.stored_uct_factor = None
        self.stored_puct_factor = 1 / (1 + self.N)
        self.stored_q = None

class Node:
    def __init__(self, state: HexBoardState, move_index=0, legal_moves: Iterable[Move] | None = None, p1_bits=None, p2_bits=None, evaluation_cache=None, model_signature=None, node_cache=None):
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
        self.evaluation_cache = {} if evaluation_cache is None else evaluation_cache
        self.node_cache = {} if node_cache is None else node_cache
        self.node_cache[(self.p1_bits, self.p2_bits)] = self
        self.model_signature = model_signature
        self.legal_moves: list[Move] = (
            self.state.get_legal_moves()
            if legal_moves is None
            else list(legal_moves)
        )
        self.children = []
        self.move_edge_map: dict[int, Edge] = {}
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

    def create_edge(self, move) -> Edge:
        existing_edge = self.move_edge_map.get(move.idx)
        if existing_edge is not None:
            return existing_edge

        child_move: PathMove = (self.is_p1_turn(), move.x, move.y)
        move_bit = 1 << move.idx
        child_p1_bits = self.p1_bits | move_bit if child_move[0] else self.p1_bits
        child_p2_bits = self.p2_bits if child_move[0] else self.p2_bits | move_bit
        position_key = (child_p1_bits, child_p2_bits)
        child_node = self.node_cache.get(position_key)
        is_transposition = child_node is not None
        if child_node is None:
            child_node = Node(
                None,
                move_index=self.move_index + 1,
                legal_moves=[m for m in self.legal_moves if m.idx != move.idx],
                p1_bits=child_p1_bits,
                p2_bits=child_p2_bits,
                evaluation_cache=self.evaluation_cache,
                model_signature=self.model_signature,
                node_cache=self.node_cache,
            )
        prior = 0 if self.stored_policy is None else self.stored_policy[move.idx]
        edge = Edge(self, child_node, move, prior, is_transposition)
        self.children.append(edge)
        self.move_edge_map[move.idx] = edge
        if self.unexpanded_count is not None:
            if self.legal_moves[self.unexpanded_count - 1] is move:
                self.unexpanded_count -= 1
            else:
                self.unexpanded_count = None
        return edge
    
    def get_best_action_using_puct(self, c) -> Move:
        if self.unexpanded_count is not None:
            return self._get_best_action_from_expanded_children(c)

        q = None
        best_score = -float('inf')

        assert self.stored_policy is not None
        assert self.legal_moves  # important invariant

        best_move = None
        p1_turn = self.is_p1_turn()
        exploration_factor = c * max(1, self.total_edge_visits) ** 0.5
        stored_policy = self.stored_policy
        move_edge_map = self.move_edge_map

        for move in self.legal_moves:
            move_index = move.idx
            prior = stored_policy[move_index]
            matching_edge = move_edge_map.get(move_index)

            if matching_edge is not None:
                n = matching_edge.N
                q = matching_edge.stored_q if matching_edge.stored_q is not None else matching_edge.Q()
            else:
                n = 0
                q = 0

            if not p1_turn:
                q = -q

            exploration = exploration_factor * prior / (1 + n)
            score = q + exploration
            if best_move is None or score > best_score:
                best_score = score
                best_move = move

        assert best_move is not None
        return best_move

    def _get_best_action_from_expanded_children(self, c) -> Move:
        assert self.stored_policy is not None

        exploration_factor = c * max(1, self.total_edge_visits) ** 0.5
        p1_turn = self.is_p1_turn()
        best_move = None
        best_score = -float('inf')

        if self.unexpanded_count:
            best_move = self.legal_moves[self.unexpanded_count - 1]
            best_score = exploration_factor * self.stored_policy[best_move.idx]

        for edge in self.children:
            move = edge.action
            q = edge.stored_q if edge.stored_q is not None else edge.Q()
            if not p1_turn:
                q = -q
            score = q + exploration_factor * edge.prior * edge.stored_puct_factor
            if (
                best_move is None
                or score > best_score
                or (score == best_score and move.idx < best_move.idx)
            ):
                best_score = score
                best_move = move

        assert best_move is not None
        return best_move
        
                
    def store_policy(self, policy):
        policy_values = policy.tolist()
        legal_indices = [move.idx for move in self.legal_moves]
        max_value = max([policy_values[index] for index in legal_indices])
        probabilities = [0.0] * len(policy_values)
        total = 0.0

        for index in legal_indices:
            probability = math.exp(policy_values[index] - max_value)
            probabilities[index] = probability
            total += probability

        for index in legal_indices:
            probabilities[index] /= total

        self._install_policy(probabilities)

    def _install_policy(self, probabilities):
        self.stored_policy = probabilities
        for edge in self.children:
            edge.prior = probabilities[edge.action.idx]
        if not self.children:
            self.legal_moves.sort(
                key=lambda move: (probabilities[move.idx], -move.idx)
            )
            self.unexpanded_count = len(self.legal_moves)
    
    def rollout(self, model, shared_accumulator, inference_parameters=None):
        if self.stored_terminal is None:
            self.is_terminal()

        if self.stored_terminal:
            return self.stored_terminal_outcome, None

        if shared_accumulator is None:
            return self.state.fast_random_rollout_bits(self.p1_bits, self.p2_bits, self.p1_turn), None

        value, policy = model.forward(shared_accumulator, inference_parameters)
        return value.item(), policy
    
    def is_terminal(self):
        
        if self.stored_terminal is not None:
            return self.stored_terminal

        # earliest possible win, no need to check earlier
        if self.move_index < 2 * HEX_BOARD_SIZE - 1:
            self.stored_terminal = False
            return False
        
        won = False

        # P1 just moved, so only P1 could have created a new win.
        if not self.p1_turn:
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

        self.stored_terminal = won or not self.legal_moves

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
        
                

class MCTS:
    
    def __init__(self, root: Node, model: CustomNNUE | None, max_iterations: int):
        # root is an empty Node representing a new game
        self.root = root
        self.model = model
        
        self.c = 1.4 # exploration constant
        assert max_iterations > 0, "max_iterations must be positive"
        self.max_iterations = max_iterations # maximum number of iterations to perform
        
        # DEBUG TIMERS
        self.total_selection_time = 0
        self.total_rollout_time = 0
        self.total_backpropagation_time = 0
        self.terminal_check_time = 0
        self.untested_child_check_time = 0
        self.calls_to_best_child_by_uct = 0
        self.total_uct_children_examined = 0
        self.evaluation_cache_hits = 0
        self.evaluation_cache_misses = 0
        self.edges_created = 0
        self.transposition_hits = 0
        self.unique_nodes_created = 0
        self.nn_evaluations = 0
        self.nn_evaluations_cached = 0
        
        self.move_delta_views = None
        self.inference_parameters = None

        if model is not None:
            model_signature = tuple(
                parameter._version for parameter in model.parameters()
            )

            if self.root.model_signature is None:
                self.root.model_signature = model_signature
            elif self.root.model_signature != model_signature:
                raise RuntimeError(
                    "Cannot reuse an MCTS tree after model weights change"
                )

            if self.root.stored_accumulator is None:
                self.root.stored_accumulator = (
                    model.calculate_accumulator(
                        self.root.state.get_state(),
                        self.root.is_p1_turn()
                    ).detach()
                )

            move_deltas = model.calculate_move_deltas().detach()
            self.move_delta_views = (
                tuple(move_deltas[0]),
                tuple(move_deltas[1]),
            )

            self.inference_parameters = model.get_inference_parameters()
            
    
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
        self.edges_created = 0
        self.transposition_hits = 0
        self.unique_nodes_created = 0
        self.nn_evaluations = 0
        self.nn_evaluations_cached = 0
        
        for _ in range(self.max_iterations):
            
            # 1: Selection phase
            # at this point this loop doesnt care how, but we get a good candidate to analyse
            selection_start = time.time()
            selected, selected_path = self.select(self.root)
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
                )

                if not selected.stored_terminal and selected.stored_policy is None:
                    probability = 1.0 / len(selected.legal_moves)

                    uniform_policy = [0.0] * (HEX_BOARD_SIZE ** 2)

                    for move in selected.legal_moves:
                        uniform_policy[move.idx] = probability

                    selected._install_policy(uniform_policy)

            else:
                # TRADITIONAL PUCT WHEN MODEL EXISTS
                evaluation_key, position_is_rotated = canonical_evaluation_key(
                    selected.p1_bits,
                    selected.p2_bits,
                )
                if selected.stored_value is not None:
                    outcome = selected.stored_value
                    policy = None
                    self.evaluation_cache_hits += 1
                    if selected.stored_policy is not None:
                        self.nn_evaluations_cached += 1
                else:
                    cached_evaluation = selected.evaluation_cache.get(evaluation_key)
                    if cached_evaluation is None:
                        outcome, policy = selected.rollout(
                            self.model,
                            selected.stored_accumulator,
                            self.inference_parameters,
                        )
                        selected.stored_value = outcome
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
            for edge in reversed(selected_path):
                edge.backpropagate(outcome)
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
        selected_path = []
        while True:
            is_terminal = node.stored_terminal
            if is_terminal is None:
                is_terminal = node.is_terminal()
            if is_terminal or node.stored_policy is None:
                break
            
            if node.unexpanded_count is not None and node.unexpanded_count > 0:
                # expand each child atleast once
                best_action = node.legal_moves[node.unexpanded_count - 1]
                
            else:
                best_action = node.get_best_action_using_puct(self.c)

            self.calls_to_best_child_by_uct += 1
            self.total_uct_children_examined += len(node.children)

            edge: Edge | None = node.move_edge_map.get(best_action.idx)
            created_edge = edge is None
            if created_edge:
                edge = node.create_edge(best_action)
                self.edges_created += 1
                if edge.is_transposition:
                    self.transposition_hits += 1
                else:
                    self.unique_nodes_created += 1

            child = edge.child
            child_move = edge.move_from_parent
            if self.model is not None and child.stored_accumulator is None:
                assert node.stored_accumulator is not None

                player_index = 0 if child_move[0] else 1

                child.stored_accumulator = (
                    node.stored_accumulator
                    + self.move_delta_views[player_index][best_action.idx]
                )
            selected_path.append(edge)
            node = child
            if created_edge:
                break

        return node, selected_path
