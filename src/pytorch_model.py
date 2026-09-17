
import torch
import torch.nn as nn

from settings import HEX_BOARD_SIZE, DEVICE

EMBEDDING_DIM = 128
FIRST_LAYER = 64
SECOND_LAYER = 32

P1_OFFSET = 0
P2_OFFSET = HEX_BOARD_SIZE * HEX_BOARD_SIZE
P1_TURN = 2 * HEX_BOARD_SIZE * HEX_BOARD_SIZE
P2_TURN = 2 * HEX_BOARD_SIZE * HEX_BOARD_SIZE + 1

class CustomNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # size * size (p1 moves) + size * size (p2 moves) + 2 (player to move)
        # P1 feature = cell_index
        # P2 feature = N² + cell_index
        # Player to move feature = 2 * N²
        self.cached_embeddings = nn.Embedding(2 * HEX_BOARD_SIZE * HEX_BOARD_SIZE + 2, EMBEDDING_DIM)
        
        # self.accumulator = torch.zeros(256)
        
        self.ff = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, FIRST_LAYER),
            nn.ReLU(),
            nn.Linear(FIRST_LAYER, SECOND_LAYER),
            nn.ReLU()
        )
        
        self.value_head = nn.Linear(SECOND_LAYER, 1)
        self.policy_head = nn.Linear(SECOND_LAYER, HEX_BOARD_SIZE * HEX_BOARD_SIZE)
    
    def advance_accumulator(self, accumulator, is_p1_move_now, move, move_back=False):
        if not move_back:
            # when moving forward we need to add 
            # efficiently update new move to cached accum
            accumulator += self.cached_embeddings.weight[(move[1] * HEX_BOARD_SIZE + move[2]) + (P1_OFFSET if is_p1_move_now else P2_OFFSET)]

            # The move changes whose turn is represented by the accumulator.
            accumulator += self.cached_embeddings.weight[P2_TURN if is_p1_move_now else P1_TURN]
            accumulator -= self.cached_embeddings.weight[P1_TURN if is_p1_move_now else P2_TURN]
        else:
            # when moving back the passed move is the move that made this acc, so remove it
            accumulator -= self.cached_embeddings.weight[(move[1] * HEX_BOARD_SIZE + move[2]) + (P1_OFFSET if is_p1_move_now else P2_OFFSET)]

            # Undoing a move restores the mover as the player to move.
            accumulator += self.cached_embeddings.weight[P1_TURN if is_p1_move_now else P2_TURN]
            accumulator -= self.cached_embeddings.weight[P2_TURN if is_p1_move_now else P1_TURN]
        return accumulator

    
    def forward(self, x):
        x = self.ff(x)
        value = nn.Tanh()(self.value_head(x))
        policy = self.policy_head(x)
        return value, policy
    
    # probably public fn for mcts
    def calculate_accumulator(self, state, is_p1_move_now) -> torch.Tensor:
        accumulator = torch.zeros(EMBEDDING_DIM).to(DEVICE)
        for i in range(HEX_BOARD_SIZE):
            for j in range(HEX_BOARD_SIZE):
                if state[0][i][j] == 1:
                    accumulator += self.cached_embeddings.weight[P1_OFFSET + (i * HEX_BOARD_SIZE + j)]
                if state[1][i][j] == 1:
                    accumulator += self.cached_embeddings.weight[P2_OFFSET + (i * HEX_BOARD_SIZE + j)]
        
        accumulator += self.cached_embeddings.weight[P1_TURN if is_p1_move_now else P2_TURN]
        return accumulator
    
    
