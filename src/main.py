import numpy as np
import copy
import random
import pickle

from game_logic.game_base import Game
from mcts.mcts_search import MCTS, MockNetwork
from model import Model

from utils.clean_up_console_logs import clean_up_console_logs


from settings import *

clean_up_console_logs()


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
       
        available_moves = game.available_moves()

        # print("AVAIALLBALE MOVES", available_moves)
        print("VISIT COUNTS", visit_counts, "PROBABILITIES", action_probabilities)
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

        board_state, _ = game.get_board_state_for_nn()
        recorded_game.append((board_state, action_probabilities, 0))

        is_won = game.make_move(action, game.current_player)

        # print(game.get_human_readable_board(), action_probabilities, action)
        
        if is_won:
            current_player = game.current_player
            print("Player", current_player, "won!")
            print(game.get_human_readable_board())
            print("calculations saved using hash map", mcts.network.times_used_hash)
            print("total predictions made", mcts.network.total_predict_calls)
            
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


def play_and_save_games():

    game = Game(7, 6)
    net = Model(7, 6)
    mcts = MCTS(network=net, c_puct=1.0, num_simulations=MCTS_SIMULATIONS_PER_MOVE)

    games = []
    
    for i in range(SELF_PLAY_GAME_COUNT):
        game.reset()
        recorded_game = self_play(game, mcts)
        games.append(recorded_game)
        # print("RECORDED GAME", recorded_game)

    with open("games.pkl", "wb") as f:
        pickle.dump(games, f)



def train_model():
    net = Model(7, 6)
    net.model.summary()
    with open("games.pkl", "rb") as f:
         games = pickle.load(f)



    actions = [move for game in games for move in game]
    random.shuffle(actions)
    batch_size = 32
    batches = [actions[i:i+batch_size] for i in range(0, len(actions), batch_size)]

    for batch in batches:
        print("Training batch",  batch)
        target_data = []
        for move in batch:
            action_probabilities = move[1]
            value = move[2]
            target_data.append((action_probabilities, value))
        target_data = list(zip(*target_data))
        target_data = [np.array(target_data[0]), np.array(target_data[1])]
        net.train(input_data, target_data)

    net.save()

if __name__ == "__main__":

    # play_and_save_games()

    train_model()
    
    


    
    

