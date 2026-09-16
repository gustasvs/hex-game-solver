
import sys

from game_logic.hex import HexBoardState
from mcts.mcts import MCTS, Node
from utils.display_game import display_game

def play_game():
    
    state = HexBoardState(move=(1, 4, 1))
    root = Node(state, move_index=1)
    print(state.get_legal_moves())
    while not state.is_terminal():
        
        new_game = MCTS(root)
        
        mtcs_results_node = new_game.run()
        
        best_child = max(mtcs_results_node.children, key=lambda c: c.N, default=None)
        # if best_child: 
        #     best_child.print_visits_and_score()
        display_game(state, root)
        state = best_child.state if best_child else state
        root = best_child if best_child else root
        
        
        # for child in result.children:
        #     if child.N > 0:
        #         state = child.state
        #         break
    display_game(state, root)

if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("Game interrupted by user.")
        sys.exit(0)
        
