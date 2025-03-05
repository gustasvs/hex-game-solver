import numpy as np
import copy

from game_logic.game_base import Game
from mcts.mcts_search import MCTS, MockNetwork

def test_single_player_game():
    game = Game(7, 6)
    game.reset()
    while not game.game_over:
        print(game.get_human_readable_board())
        print("Player", game.current_player, "'s turn")
        manual_move = input("Enter column: ")
        if manual_move == "q":
            return None
        is_won = game.make_move(int(manual_move), game.current_player)
        if is_won:
            print(game.get_human_readable_board())
            print("Player", game.current_player, "won!")
            return game.winner

    return game.winner


def self_play(game, mcts):
    recorded_game = []
    
    current_move = 0
    while not game.game_over:
        action_probabilities, visit_counts = mcts.run(game)
        # action_probabilities, visit_counts = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.4}, [1, 1, 1, 1, 1, 1, 1]

        available_moves = game.available_moves()

        # print("AVAIALLBALE MOVES", available_moves)

        for i in range(len(visit_counts)):
            if i not in available_moves:
                visit_counts[i] = 0
            if current_move > 15 and not (action_probabilities[i] == np.max(action_probabilities)):
                visit_counts[i] = 0

        # print("VISIT COUNTS", visit_counts, "PROBABILITIES", action_probabilities)

        if np.sum(visit_counts) == 0:
            print("No valid moves left")

            action = np.random.choice(available_moves)
        else:
            action = np.random.choice([i for i in range(7)], p=visit_counts / np.sum(visit_counts))

        recorded_game.append((game.get_hash(), action_probabilities, 0))

        is_won = game.make_move(action, game.current_player)

        # print(game.get_human_readable_board(), action_probabilities, action)
        
        if is_won:
            current_player = game.current_player
            print("Player", current_player, "won!")
            print(game.get_human_readable_board())
            
            result_for_first_player = 1 if current_player == 1 else -1
            result_for_second_player = 1 if current_player == 2 else -1

            for i in range(len(recorded_game)):
                if i % 2 == 0:
                    recorded_game[i] = (recorded_game[i][0], recorded_game[i][1], result_for_first_player)
                else:
                    recorded_game[i] = (recorded_game[i][0], recorded_game[i][1], result_for_second_player)

            return recorded_game
            
        current_move += 1

    return recorded_game


if __name__ == "__main__":
    # test_single_player_game()
    game = Game(7, 6)
    net = MockNetwork()
    mcts = MCTS(network=net, c_puct=1.0, num_simulations=25)
    
    self_play_game_count = 1

    games = []
    
    for i in range(self_play_game_count):
        game.reset()
        recorded_game = self_play(game, mcts)
        games.append(recorded_game)
        print("RECORDED GAME", recorded_game)

    
    

