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


def play_against_model(network):
    game = Game(7, 6)
    mcts = MCTS(network=network, c_puct=1.0, num_simulations=MCTS_SIMULATIONS_PER_MOVE)
    game.reset()
    while not game.game_over:
        print(game.get_human_readable_board())
        print("Player", game.current_player, "'s turn")
        manual_move = input("Enter column: ")
        
        if manual_move == "q":
            return None

        is_finished = game.make_move(int(manual_move), game.current_player)
        if is_finished:
            print(game.get_human_readable_board())
            print("Human player won!")
            return game.winner
        
        action_probabilities, _ = mcts.run(game)
        action = np.argmax(action_probabilities)

        is_finished = game.make_move(action, game.current_player)
        if is_finished:
            print(game.get_human_readable_board())
            print("AI player won!")
        
    return game.winner


def self_play(game, mcts):
    recorded_game = []
    
    current_move = 0
    while not game.game_over:
        action_probabilities, visit_counts = mcts.run(game)
       
        available_moves = game.available_moves()

        # print("AVAIALLBALE MOVES", available_moves)
        # print("VISIT COUNTS", visit_counts, "PROBABILITIES", action_probabilities)
        for i in range(len(visit_counts)):
            if i not in available_moves:
                visit_counts[i] = 0
            if current_move > 15 and not (action_probabilities[i] == np.max(action_probabilities)):
                visit_counts[i] = 0

        # print("VISIT COUNTS", visit_counts, "PROBABILITIES", action_probabilities)

        if np.sum(visit_counts) == 0:
            print("-" * 10)
            print(game.get_human_readable_board())
            print("available moves", available_moves)
            print("action probabilities", action_probabilities)
            print("visit counts", visit_counts)
            print("No valid moves left")

            return recorded_game
        else:
            action = np.random.choice([i for i in range(7)], p=visit_counts / np.sum(visit_counts))

            board_state, _ = game.get_board_state_for_nn()
            recorded_game.append((board_state, action_probabilities, 0))


        is_finished = game.make_move(action, game.current_player)

        # print(game.get_human_readable_board(), action_probabilities, action)
        
        if is_finished:

            if game.winner == 0:
                print("Draw!")
                print(game.get_human_readable_board())

                # we can just return the current recorded game as the value targets are already 0 from initialization
                return recorded_game

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


def play_and_save_games(game, mcts):

    games = []
    
    for i in range(SELF_PLAY_GAME_COUNT):
        print("playing game", i)
        game.reset()
        recorded_game = self_play(game, mcts)
        games.append(recorded_game)
        # print("RECORDED GAME", recorded_game)

    with open("games.pkl", "wb") as f:
        pickle.dump(games, f)



def train_model(net):

    # net.model.summary()
    
    with open("games.pkl", "rb") as f:
         games = pickle.load(f)


    actions = []
    action_count = 0

    for game in games[:PAST_GAMES_TO_KEEP]:
        for action in game:
            actions.append(action)
            action_count += 1

    print("Total actions after keeping only the last", PAST_GAMES_TO_KEEP, "games", action_count)

    random.shuffle(actions)

    input_data = []
    policy_targets = []
    value_targets = []

    for action in actions:
        
        policy_target = action[1]
        value_target = action[2]
        board_state = action[0]         
           
        input_data.append(np.array(board_state[0]))
        policy_targets.append(np.array(policy_target))
        value_targets.append(np.array(value_target))

    input_data = np.array(input_data)
    policy_targets = np.array(policy_targets)
    value_targets = np.array(value_targets)

    print("Input data shape", input_data.shape)
    print("Policy targets shape", policy_targets.shape)
    print("Value targets shape", value_targets.shape)

    net.train(input_data, [policy_targets, value_targets])
                  
    net.save()

def main():

    game = Game(7, 6)
    net = Model(7, 6)

    net.load_pretrained_model()

    # play_against_model(net)
    # exit()

    mcts = MCTS(network=net, c_puct=1.0, num_simulations=MCTS_SIMULATIONS_PER_MOVE)


    for iteration in range(ITERATIONS):
        print("-" * 10, "ITERATION", iteration, "-" * 10)
        play_and_save_games(game, mcts)

        train_model(net)
        # mcts.network.clear_hash_map()
        net.clear_hash_map()
    
    

    

if __name__ == "__main__":
    main()


    
    

