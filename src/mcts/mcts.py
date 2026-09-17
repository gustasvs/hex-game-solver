import numpy as np
import math
import random

import time

import torch
from tqdm import tqdm

from game_logic.hex import HexBoardState

from pytorch_model import CustomNNUE
from game_logic.classes import Move
from collections.abc import Iterable

np.random.seed(42)

PathMove = tuple[bool, int, int]

class Node:
    def __init__(self, state: HexBoardState, parent=None, move_from_parent: PathMove | None = None, move_index=0, legal_moves: Iterable[Move] | None = None,):
        self.state = state
        self.legal_moves: tuple[Move, ...] = (
            tuple(self.state.get_legal_moves())
            if legal_moves is None
            else tuple(legal_moves)
        )
        self.parent = parent
        self.children = []
        self.move_child_map: dict[int, Node | None] = {move.get_idx(): None for move in self.legal_moves}
        self.stored_policy = None
        self.move_from_parent = move_from_parent
        
        # cached values for efficiency
        self.stored_uct_factor = None
        self.stored_q = None
        
        
        # on initial empty game its 0, first p1 move is 1, then p2 move is 2, and so on
        self.move_index = move_index
        
        self.N = 0 # visit count
        self.W = 0 # accumulated value
        
    def is_p1_turn(self):
        return self.move_index % 2 == 0
        
    def Q(self):
        if self.stored_q is not None:
            return self.stored_q
        
        self.stored_q = self.W / self.N if self.N > 0 else 0
        return self.stored_q
    
            
    def create_child(self, move) -> Node:
        child_move: PathMove = (self.is_p1_turn(), move.x, move.y)
        child_state = HexBoardState(state=self.state.get_state(), move=child_move)
        child_node = Node(
            child_state,
            parent=self,
            move_from_parent=child_move,
            move_index=self.move_index + 1,
            legal_moves=[m for m in self.legal_moves if m.get_idx() != move.get_idx()],
        )
        self.children.append(child_node)
        self.move_child_map[move.get_idx()] = child_node
        return child_node
                
    def best_child_using_uct(self, c) -> Node | None:


        best_score = -float('inf')
        best_child = None
        log_self_N = math.log(self.N)
        parent_exploration_factor = c * math.sqrt(log_self_N)
        p1_turn = self.is_p1_turn()
        for child in self.children:
            # `UCT = W / N + C sqrt(ln(parent.N) / N)`
            # at this point all children will have N
            # uct = child.Q() + c * np.sqrt(np.log(self.N + 1) / (child.N + 1))
            exploitation = child.stored_q if child.stored_q is not None else child.Q()
            if not p1_turn:
                exploitation = -exploitation
            
            child_factor = None
            if child.stored_uct_factor is not None:
                child_factor = child.stored_uct_factor
            else:
                child_factor = 1 / math.sqrt(child.N)
                child.stored_uct_factor = child_factor
                
            exploration = parent_exploration_factor * child_factor
            
            uct = exploitation + exploration
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child
    
    def get_best_action_using_puct(self, c) -> Move:
        n = None
        q = None
        best_score = -float('inf')
        
        assert self.stored_policy is not None
        assert self.legal_moves  # important invariant
        
        best_move = None
        
        for move in self.legal_moves:
            prior = self.stored_policy[move.get_idx()]
            matching_child = self.move_child_map.get(move.get_idx())
            
            if matching_child is not None:
                n = matching_child.N
                q = matching_child.stored_q if matching_child.stored_q is not None else matching_child.Q()
            else:
                n = 0
                q = 0
            
            if not self.is_p1_turn():
                q = -q
            
            exploration = c * prior * math.sqrt(self.N) / (1 + n)
            score = q + exploration
            if best_move is None or score > best_score:
                best_score = score
                best_move = move

        assert best_move is not None
        return best_move
        
                
    def backpropagate(self, result):
        self.N += 1
        # since score is kept from p1's perspective, we add the result directly
        self.W += result
        self.stored_uct_factor = None
        self.stored_q = None
        if self.parent:
            self.parent.backpropagate(result)
            
    def store_policy(self, policy):
        
        masked_policy = torch.full_like(policy, float('-inf'))
        for move in self.legal_moves:
            index = move.get_idx()
            masked_policy[index] = policy[index]
        
        probabilities = torch.softmax(masked_policy, dim=0)
        self.stored_policy = probabilities.tolist()
    
    def rollout(self, model, shared_accumulator):
        return self.state.model_based_rollout(model, shared_accumulator)
    
    def is_terminal(self):
        return self.state.p1_win() or self.state.p2_win() or not self.legal_moves
    
    # LOGGING
    
    def print_visits_and_score(self):
        print(f"Visits: {self.N}, Score: {self.W}")
        
    def get_total_children(self):
        # returns created "Node" objects
        children = 0
        if self.children and len(self.children):
            children += len(self.children)
        for child in self.children or []:
            children += child.get_total_children()
        return children        
        
                

class MCTS:
    
    def __init__(self, root: Node, model: CustomNNUE):
        # root is an empty Node representing a new game
        self.root = root
        self.model = model
        self.c = 1.4 # exploration constant
        self.max_iterations = 40000 # maximum number of iterations to perform
        
        # DEBUG TIMERS
        self.total_selection_time = 0
        self.total_rollout_time = 0
        self.total_backpropagation_time = 0
        self.terminal_check_time = 0
        self.untested_child_check_time = 0
        self.calls_to_best_child_by_uct = 0
        self.total_uct_children_examined = 0
        
        self.shared_accumulator = self.model.calculate_accumulator(self.root.state.get_state(), self.root.is_p1_turn())
            
    
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
        
        pbar = tqdm(range(self.max_iterations))
        for _ in pbar:
            
            # 1: Selection phase
            # at this point this loop doesnt care how, but we get a good candidate to analyse
            selection_start = time.time()
            selected, path = self.select(self.root)
            working_acc = self.shared_accumulator.clone()
            for move in path:
                working_acc = self.model.advance_accumulator(working_acc, move[0], move)
            selection_end = time.time()
            self.total_selection_time += selection_end - selection_start
            if selected is None:
                print("No valid child found during selection.")
                break
            
            # 2: Rollout phase
            rollout_start = time.time()
            outcome, policy = selected.rollout(self.model, working_acc)
            rollout_end = time.time()
            self.total_rollout_time += rollout_end - rollout_start
            
            # 3: Backpropagation phase
            backpropagation_start = time.time()
            if policy is not None:
                selected.store_policy(policy)
            selected.backpropagate(outcome)
            backpropagation_end = time.time()
            self.total_backpropagation_time += backpropagation_end - backpropagation_start

        print(f"TOTAL TIME: {time.time() - start_time}")
        print(f"Total selection time: {self.total_selection_time}")
        print(f"    > Total terminal check time: {self.terminal_check_time}")
        print(f"    > Total untested child check time: {self.untested_child_check_time}")
        print(f"    > Total score calculation time: {self.score_calc_time}")
        print(f"Total rollout time: {self.total_rollout_time}")
        print(f"Total backpropagation time: {self.total_backpropagation_time}")
        print(" > COUNTS:")
        explored_children = self.root.get_total_children()
        print(f"    > Total explored children: {explored_children} / {self.max_iterations}")
        print(f"    > Calls to best child by UCT: {self.calls_to_best_child_by_uct}")
        print(f"    > Total UCT children examined: {self.total_uct_children_examined}")
        return self.root
    
    def select(self, node: Node) -> tuple[Node | None, list[PathMove]]:
        
        # terminal? stop
        if node.is_terminal():
            return node, []
        
        # node has no stored policy yet? stop
        if node.stored_policy is None:
            return node, []
        
        best_action = node.get_best_action_using_puct(self.c)

        self.calls_to_best_child_by_uct += 1
        self.total_uct_children_examined += len(node.children)

        child = node.move_child_map.get(best_action.get_idx())

        if child is None:
            child = node.create_child(best_action)

            child_move = child.move_from_parent
            assert child_move is not None

            return child, [child_move]

        selected, path = self.select(child)

        child_move = child.move_from_parent
        assert child_move is not None

        return selected, [child_move] + path
