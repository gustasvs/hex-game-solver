import torch

TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

HEX_BOARD_SIZE = 5

# Let unvisited moves compete with visited moves under PUCT immediately.  A
# paired 40-game, fixed-opening/color-swapped gate favored this setting 33-7 while
# it also roughly halved child scans and neural evaluations.
MCTS_FORCE_EXPAND_ALL = False

ITERATIONS = 20

