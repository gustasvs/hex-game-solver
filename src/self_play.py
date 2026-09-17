from game import play_game

LEARNING_ITERATIONS = 10
SELF_PLAY_GAMES = 100
from pytorch_model import CustomNNUE
from settings import DEVICE


def play_and_record_games(model) -> list:
    
    game_data = []
    
    for game_index in range(SELF_PLAY_GAMES):
        
        states, policies, values = play_game(model)
        game_data.append((states, policies, values))
        print(f"Completed self-play game {game_index + 1}/{SELF_PLAY_GAMES}")
        
        
        
        
    return game_data


def self_play():
    
    model = CustomNNUE().to(DEVICE)
    
    for iteration in range(LEARNING_ITERATIONS):
        print(f"Starting learning iteration {iteration + 1}/{LEARNING_ITERATIONS}")
        data = play_and_record_games(model)
        
        
        
    

    