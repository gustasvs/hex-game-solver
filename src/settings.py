import torch

TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

HEX_BOARD_SIZE = 5

ITERATIONS = 20

