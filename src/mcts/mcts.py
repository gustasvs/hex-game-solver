import numpy as np
import math
import random

import time

from game_logic.hex import HexBoardState

np.random.seed(42)

class Move:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Node:
    def __init__(self, state: HexBoardState, parent=None, move_index=0):
        self.state = state
        self.untested_legal_moves = set(self.state.get_legal_moves())
        self.parent = parent
        self.children = []
        
        
        # on initial empty game its 0, first p1 move is 1, then p2 move is 2, and so on
        self.move_index = move_index
        
        self.N = 0 # visit count
        self.W = 0 # accumulated value
        
    def is_p1_turn(self):
        return self.move_index % 2 == 0
        
    def Q(self):
        return self.W / self.N if self.N > 0 else 0
    
    def ensure_children_exist(self):
        if not self.children:
            children = []
            for move in self.state_legal_moves:
                child_move = (self.is_p1_turn(), move[0], move[1])
                child_state = HexBoardState(state=self.state.get_state(), move=child_move)
                children.append(Node(child_state, parent=self, move_index=self.move_index + 1))
            self.children = children
            
    def create_child(self, move):
        child_move = (self.is_p1_turn(), move[0], move[1])
        child_state = HexBoardState(state=self.state.get_state(), move=child_move)
        child_node = Node(child_state, parent=self, move_index=self.move_index + 1)
        self.children.append(child_node)
        self.untested_legal_moves.discard(move)
        return child_node
            
    def find_and_yield_untested_child(self):
        # self.ensure_children_exist()
        
        # for child in self.children:
        #     if child.N == 0:
        #         yield child
        
        for move in  self.untested_legal_moves:
            child_node = self.create_child(move)
            yield child_node
                
    def best_child_using_uct(self, c) -> Node | None:        
        if not self.children:
            return None
        
        # return self.children[np.random.randint(len(self.children))]
        best_score = -float('inf')
        best_child = None
        log_self_N = math.log(self.N + 1)
        p1_turn = self.is_p1_turn()
        for child in self.children:
            # `UCT = W / N + C sqrt(ln(parent.N) / N)`
            # at this point all children will have N
            # uct = child.Q() + c * np.sqrt(np.log(self.N + 1) / (child.N + 1))
            exploitation = child.Q()
            if not p1_turn:
                exploitation = -exploitation
                
            # exploration = c * np.sqrt(
            #     np.log(self.N) / child.N
            # )
            exploration = c * math.sqrt(
                log_self_N / (child.N + 1)
            )

            uct = exploitation + exploration
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child
                
    def backpropagate(self, result):
        self.N += 1
        # since score is kept from p1's perspective, we add the result directly
        self.W += result
        if self.parent:
            self.parent.backpropagate(result)
    
    def rollout(self):
        return self.state.rollout()
    
    def is_terminal(self):
        return self.state.is_terminal()
    
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
    
    def __init__(self, root):
        # root is an empty Node representing a new game
        self.root = root
        self.c = 1.4 # exploration constant
        self.iterations = 0 # number of iterations performed
        self.max_iterations = 8000 # maximum number of iterations to perform
        
        
        self.total_selection_time = 0
        self.total_rollout_time = 0
        self.total_backpropagation_time = 0
        self.terminal_check_time = 0
        self.untested_child_check_time = 0
        self.calls_to_best_child_by_uct = 0
            
    
    def run(self) -> Node:
        start_time = time.time()
        self.total_selection_time = 0
        self.total_rollout_time = 0
        self.total_backpropagation_time = 0
        self.terminal_check_time = 0
        self.untested_child_check_time = 0
        self.score_calc_time = 0
        self.calls_to_best_child_by_uct = 0
        
        for _ in range(self.max_iterations):
            self.iterations += 1
            
            # 1: Selection phase
            # at this point this loop doesnt care how, but we get a good candidate to analyse
            selection_start = time.time()
            selected = self.select(self.root)
            selection_end = time.time()
            self.total_selection_time += selection_end - selection_start
            if selected is None:
                print("No valid child found during selection.")
                break
            
            # 2: Rollout phase
            rollout_start = time.time()
            outcome = selected.rollout()
            rollout_end = time.time()
            self.total_rollout_time += rollout_end - rollout_start
            
            # 3: Backpropagation phase
            backpropagation_start = time.time()
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
        return self.root
    
    def select(self, node: Node) -> Node | None:
        
        terminal_check_start = time.time()
        if node.is_terminal():
            terminal_check_end = time.time()
            self.terminal_check_time += terminal_check_end - terminal_check_start
            return node
        terminal_check_end = time.time()
        self.terminal_check_time += terminal_check_end - terminal_check_start
        
        # find a leaf node to expand
        untested_child_check_start = time.time()
        untested_child = next(node.find_and_yield_untested_child(), None)
        if untested_child is not None:
            # prioritize untested children as leaf nodes
            untested_child_check_end = time.time()
            self.untested_child_check_time = getattr(self, 'untested_child_check_time', 0) + (untested_child_check_end - untested_child_check_start)
            return untested_child
        
        untested_child_check_end = time.time()
        self.untested_child_check_time = getattr(self, 'untested_child_check_time', 0) + (untested_child_check_end - untested_child_check_start)

        # if we get here then the choosen child has already been once explored (because of the untested child yielding above)
        score_calc_start = time.time()
        best_child_by_score = node.best_child_using_uct(self.c)
        self.calls_to_best_child_by_uct += 1
        if best_child_by_score is None:
            score_calc_end = time.time()
            self.score_calc_time += score_calc_end - score_calc_start
            return None

        score_calc_end = time.time()
        self.score_calc_time += score_calc_end - score_calc_start
        return self.select(best_child_by_score)
        
