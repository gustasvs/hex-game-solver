
import math
import sys
import random

import torch

from settings import DEVICE, HEX_BOARD_SIZE
# def softmax(values):
#     max_val = max(values)
#     exps = [math.exp(v - max_val) for v in values]
#     total = sum(exps)
#     return [e / total for e in exps]

from game_logic.hex import HexBoardState
from pytorch_model import CustomNNUE
from mcts.mcts import MCTS, Node
from utils.display_game import display_game

def play_game(model: CustomNNUE) -> tuple:
    
    # state = HexBoardState(move=(1, 4, 1))
    # root = Node(state, move_index=1)
    state = HexBoardState()
    root = Node(state)
    
    states = []
    policies = []
    values = []
    with torch.inference_mode():
        while not state.is_terminal():
            
            move_tree = MCTS(root, model)
            root.parent = None
                    
            mtcs_results_node = move_tree.run()        
            # best_child = max(mtcs_results_node.children, key=lambda c: c.N, default=None)
            # for training always random with probability proportional to visit count
            best_child = random.choices(
                mtcs_results_node.children,
                weights=[c.N for c in mtcs_results_node.children],
                k=1
            )[0] if mtcs_results_node.children else None
            
            total_visits = sum(c.N for c in mtcs_results_node.children)
            policy_target = [0.0] * (HEX_BOARD_SIZE ** 2)
            if total_visits > 0:
                for child in mtcs_results_node.children:
                    _, row, col = child.move_from_parent
                    idx = row * HEX_BOARD_SIZE + col
                    policy_target[idx] = child.N / total_visits

            policies.append(policy_target)            
            states.append((
                [
                    [row[:] for row in state.p1],
                    [row[:] for row in state.p2],
                ],
                root.is_p1_turn()
            ))
            
            display_game(state, root)
            
            state = best_child.state if best_child else state
            root = best_child if best_child else root
            root.parent = None
        
        
        # for child in result.children:
        #     if child.N > 0:
        #         state = child.state
        #         break
    display_game(state, root)
    
    winner_value = +1 if state.p1_win() else -1
    values = [winner_value for _ in range(len(states))]

    return states, policies, values

if __name__ == "__main__":
    model = CustomNNUE().to(DEVICE)
    try:
        play_game(model)
    except KeyboardInterrupt:
        print("Game interrupted by user.")
        sys.exit(0)
        
