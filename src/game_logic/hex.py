
import random
import sys

import numpy as np

from settings import HEX_BOARD_SIZE
from pytorch_model import CustomNNUE
import torch

from game_logic.classes import Move
from mcts.bitboard_helpers import BOARD_MASK, BOTTOM_EDGE_MASK, LEFT_EDGE_MASK, RIGHT_EDGE_MASK, TOP_EDGE_MASK, has_bit_connection

class HexBoardState:
    _NEIGHBORS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))

    def __init__(self, state=None, move=None):
        self.size = HEX_BOARD_SIZE
        if state is None:
            self.p1 = [[0 for _ in range(HEX_BOARD_SIZE)] for _ in range(HEX_BOARD_SIZE)]
            self.p2 = [[0 for _ in range(HEX_BOARD_SIZE)] for _ in range(HEX_BOARD_SIZE)]
        else:
            self.p1 = [row[:] for row in state[0]]
            self.p2 = [row[:] for row in state[1]]
        if move is not None:
            self.make_move(move)

    @classmethod
    def from_parent(cls, parent, move):
        state = cls.__new__(cls)
        state.size = parent.size
        state.p1 = parent.p1.copy()
        state.p2 = parent.p2.copy()

        board = state.p1 if move[0] else state.p2
        board[move[1]] = board[move[1]].copy()
        board[move[1]][move[2]] = 1
        return state

    @classmethod
    def from_bitboards(cls, p1_bits, p2_bits):
        state = cls.__new__(cls)
        state.size = HEX_BOARD_SIZE
        state.p1 = [
            [
                (p1_bits >> (row * HEX_BOARD_SIZE + col)) & 1
                for col in range(HEX_BOARD_SIZE)
            ]
            for row in range(HEX_BOARD_SIZE)
        ]
        state.p2 = [
            [
                (p2_bits >> (row * HEX_BOARD_SIZE + col)) & 1
                for col in range(HEX_BOARD_SIZE)
            ]
            for row in range(HEX_BOARD_SIZE)
        ]
        return state

    def get_legal_moves_count(self):
        return sum(1 for row in range(self.size) for col in range(self.size) if self.p1[row][col] == 0 and self.p2[row][col] == 0)
    
    def get_legal_moves(self):
        return [Move(row, col) for row in range(self.size) for col in range(self.size) if self.p1[row][col] == 0 and self.p2[row][col] == 0]

    def _has_connection(self, board, vertical):
        size = self.size
        if vertical:
            stack = [(0, col) for col in range(size) if board[0][col]]
        else:
            stack = [(row, 0) for row in range(size) if board[row][0]]

        visited = [[False] * size for _ in range(size)]
        for row, col in stack:
            visited[row][col] = True

        while stack:
            row, col = stack.pop()
            if (vertical and row == size - 1) or (not vertical and col == size - 1):
                return True

            for row_delta, col_delta in self._NEIGHBORS:
                next_row = row + row_delta
                next_col = col + col_delta
                if (0 <= next_row < size and 0 <= next_col < size
                        and board[next_row][next_col]
                        and not visited[next_row][next_col]):
                    visited[next_row][next_col] = True
                    stack.append((next_row, next_col))

        return False

    def p1_win(self):
        return self._has_connection(self.p1, vertical=True)

    def p2_win(self):
        return self._has_connection(self.p2, vertical=False)

    def is_terminal(self):
        return self.p1_win() or self.p2_win() or self.get_legal_moves_count() == 0
    
    def get_state(self):
        return [self.p1, self.p2]
    
    def fast_random_rollout_bits(self, p1_bits, p2_bits, p1_to_move) -> int:
        empty = BOARD_MASK & ~(p1_bits | p2_bits)
        moves = []

        while empty:
            bit = empty & -empty
            moves.append(bit)
            empty ^= bit

        random.shuffle(moves)

        for i, bit in enumerate(moves):
            if p1_to_move == (i % 2 == 0):
                p1_bits |= bit
            else:
                p2_bits |= bit

        if has_bit_connection(p1_bits, TOP_EDGE_MASK, BOTTOM_EDGE_MASK):
            return 1
        if has_bit_connection(p2_bits, LEFT_EDGE_MASK, RIGHT_EDGE_MASK):
            return -1

        raise RuntimeError("Hex rollout ended without a winner")
    
    def make_move(self, move):
        # move[0] = 1 if p1 moves 2 if p2 moves
        # move[1] = row
        # move[2] = col
        if (move[0] is True):
            self.p1[move[1]] = self.p1[move[1]].copy()
            self.p1[move[1]][move[2]] = 1
        else:
            self.p2[move[1]] = self.p2[move[1]].copy()
            self.p2[move[1]][move[2]] = 1
        return self
