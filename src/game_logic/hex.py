
import sys

import numpy as np

from settings import HEX_BOARD_SIZE

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

    def get_legal_moves_count(self):
        return sum(1 for row in range(self.size) for col in range(self.size) if self.p1[row][col] == 0 and self.p2[row][col] == 0)
    
    def get_legal_moves(self):
        return [(row, col) for row in range(self.size) for col in range(self.size) if self.p1[row][col] == 0 and self.p2[row][col] == 0]

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
    
    def rollout(self) -> int:
        
        remaining_moves = self.get_legal_moves()
        np.random.shuffle(remaining_moves)
        p1_moves = []
        p2_moves = []
        if len(remaining_moves) % 2 == 0:
            p1_moves = remaining_moves[::2]  # p1 takes every other move starting from the first
            p2_moves = remaining_moves[1::2]  # p2 takes the remaining moves
        else:
            p1_moves = remaining_moves[1::2]  # p1 takes every other move starting from the second
            p2_moves = remaining_moves[::2]  # p2 takes the remaining moves
        
        rollout_state_p1 = [row[:] for row in self.p1]
        rollout_state_p2 = [row[:] for row in self.p2]
        for row, col in p1_moves:
            rollout_state_p1[row][col] = 1
        for row, col in p2_moves:
            rollout_state_p2[row][col] = 1
        
        p1_win =  self._has_connection(rollout_state_p1, vertical=True)
        if p1_win:
            return 1
            
        p2_win = self._has_connection(rollout_state_p2, vertical=False)
        if p2_win:
            return -1
        
        # if (p1_win and p2_win):
        #     sys.exit("Both players won in rollout ?!??! ERROR")
        # if p1_win:
        #     return 1
        # if p2_win:
        #     return -1
        sys.exit("Rollout ended without a winner ?!??! ERROR")
    
    def make_move(self, move):
        # move has x and y and p
        if (move[0] == 1):
            self.p1[move[1]][move[2]] = 1
        else:
            self.p2[move[1]][move[2]] = 1
        return self
