
import math
import os
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

def play_game(model: CustomNNUE | None, eval_cache: dict | None = None, display: bool = False, deterministic: bool = False) -> tuple:
    
    # state = HexBoardState(move=(1, 4, 1))
    # root = Node(state, move_index=1)
    state = HexBoardState()
    root = Node(state, evaluation_cache=eval_cache)
    
    states = []
    policies = []
    values = []
    moves_played = 0
    with torch.inference_mode():
        while not state.is_terminal():
            
            move_tree = MCTS(root, model, 20_000 if model is None else 5_000)
            mtcs_results_node = move_tree.run()
            
            if not mtcs_results_node.children:
                raise RuntimeError(
                    "MCTS produced no children from a non-terminal state"
                )
            
            if root.move_index > 10 or deterministic:
                # deterministic later in the game
                best_edge = max(mtcs_results_node.children, key=lambda edge: edge.N)
            else:
                best_edge = random.choices(
                    mtcs_results_node.children,
                    weights=[edge.N for edge in mtcs_results_node.children],
                    k=1
                )[0]
            # print([edge.N for edge in mtcs_results_node.children])
            # print(f"Best child move: {best_edge.move_from_parent}")
            
            total_visits = sum(edge.N for edge in mtcs_results_node.children)
            policy_target = [0.0] * (HEX_BOARD_SIZE ** 2)
            if total_visits > 0:
                for edge in mtcs_results_node.children:
                    _, row, col = edge.move_from_parent
                    idx = row * HEX_BOARD_SIZE + col
                    policy_target[idx] = edge.N / total_visits

            policies.append(policy_target)            
            states.append((
                [
                    [row[:] for row in state.p1],
                    [row[:] for row in state.p2],
                ],
                root.is_p1_turn()
            ))
            
            if display:
                display_game(state, root)
            
            if best_edge is None:
                raise RuntimeError(
                    "MCTS returned no child for a non-terminal position"
                )
            
            best_child: Node = best_edge.child
            state = best_child.state
            root = best_child
            moves_played += 1

            # Optimisation benchmark: only process the first three moves.
            # if moves_played == 3:
            #     return states, policies, values
            
        
        
        # for child in result.children:
        #     if child.N > 0:
        #         state = child.state
        #         break
    if display:
        display_game(state, root)
    
    winner_value = +1 if state.p1_win() else -1
    values = [winner_value for _ in range(len(states))]

    return states, policies, values

if __name__ == "__main__":
    # random.seed(42)
    # torch.manual_seed(42)
    
    
    weights_path = f"weights/board_size{HEX_BOARD_SIZE}/model_weights.pt"

    weights = None

    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        weights = torch.load(
            weights_path,
            map_location=DEVICE,
            weights_only=True,
        )

    model = CustomNNUE(weights=weights).to(DEVICE)
    
    try:
        # play_game(None, {}, True, True)
        play_game(model, {}, True, True)
    except KeyboardInterrupt:
        print("Game interrupted by user.")
        sys.exit(0)
        
