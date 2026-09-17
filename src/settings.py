import torch

TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

HEX_BOARD_SIZE = 7

ITERATIONS = 20

MCTS_SIMULATIONS_PER_MOVE = 30
SELF_PLAY_GAME_COUNT = 20

MAX_PAST_GAMES_TO_KEEP = 150
MAX_PAST_ACTIONS_TO_KEEP = 1200
