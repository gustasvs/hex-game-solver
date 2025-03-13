import numpy as np

# connect 4 game class
class Game:
    def __init__(self, board_width, board_height):
        self.board_width = board_width
        self.board_height = board_height

        self.board_first = np.zeros((board_height, board_width), dtype=bool)
        self.board_second = np.zeros((board_height, board_width), dtype=bool)
        
        self.current_player = 1
        self.winner = 0
        self.game_over = False

    def reset(self):
        self.board_first = np.zeros((self.board_height, self.board_width), dtype=bool)
        self.board_second = np.zeros((self.board_height, self.board_width), dtype=bool)
        self.current_player = 1
        self.winner = 0
        self.game_over = False

    def available_moves(self):
        moves = []
        for i in range(self.board_width):
            if not self.board_first[0][i] and not self.board_second[0][i]:
                moves.append(i)
        
        if len(moves) == 0:
            self.game_over = True
            self.winner = 0

        return moves
        
    def make_move(self, column, player):
        # print("called make_move", column, player)
        if self.game_over or len(self.available_moves()) == 0:
            return True
        
        assert player == self.current_player

        move_row = 0
        while move_row < self.board_height - 1 and self.board_first[move_row + 1][column] == 0 and self.board_second[move_row + 1][column] == 0:
            move_row += 1
        
        if player == 1:
            self.board_first[move_row][column] = 1
        else:
            self.board_second[move_row][column] = 1

        has_winner =  self.check_winner(column, move_row)

        # returning true earlier to not change player anymore if game has ended
        if has_winner:
            return True

        self.current_player = 3 - self.current_player

        return False

    def check_winner(self, column, row, player=None):
        # function is called after making a move to check nearby cells for a winning combination
        # returns True if the game is over and False otherwise
        if player is None:
            if self.current_player == 1:
                board = self.board_first
            else:
                board = self.board_second
        else:
            if player == 1:
                board = self.board_first
            else:
                board = self.board_second

        # check horizontal
        current_in_row = 0
        for i in range(max(0, column - 3), min(column + 4, self.board_width)):
            if board[row][i]:
                current_in_row += 1
                if current_in_row == 4:
                    self.game_over = True
                    self.winner = self.current_player
                    return True
            else:
                current_in_row = 0
        
        # check vertical
        current_in_row = 0
        for i in range(max(0, row - 3), min(row + 4, self.board_height)):
            if board[i][column]:
                current_in_row += 1
                if current_in_row == 4:
                    self.game_over = True
                    self.winner = self.current_player
                    return True
            else:
                current_in_row = 0
        
        # check diagonal slope up
        current_in_row = 0
        for i in range(-3, 4):
            if 0 <= row + i < self.board_height and 0 <= column + i < self.board_width:
                if board[row + i][column + i]:
                    current_in_row += 1
                    if current_in_row == 4:
                        self.game_over = True
                        self.winner = self.current_player
                        return True
                else:
                    current_in_row = 0
        
        # check diagonal slope down
        current_in_row = 0
        for i in range(-3, 4):
            if 0 <= row + i < self.board_height and 0 <= column - i < self.board_width:
                if board[row + i][column - i]:
                    current_in_row += 1
                    if current_in_row == 4:
                        self.game_over = True
                        self.winner = self.current_player
                        return True
                else:
                    current_in_row = 0

        return False
    
    def is_oponent_winning_next_move(self):
        """
        Checks if the opponent has a winning move on their next turn.
        Returns True if the opponent can win with their next move, otherwise False.
        """
        opponent = 3 - self.current_player  # Opponent's player number

        for column in self.available_moves():
            # Find the lowest available row in this column
            row = 0
            while row < self.board_height - 1 and not (self.board_first[row + 1][column] or self.board_second[row + 1][column]):
                row += 1

            # Temporarily place opponent's piece
            if opponent == 1:
                self.board_first[row][column] = True
            else:
                self.board_second[row][column] = True

            # Check if this move wins the game
            is_winning = self.check_winner(column, row, opponent)

            # Undo the move
            if opponent == 1:
                self.board_first[row][column] = False
            else:
                self.board_second[row][column] = False

            # Reset game state in case check_winner modified it
            self.game_over = False
            self.winner = 0

            if is_winning:
                return True  # Opponent has a winning move

        return False  # No immediate winning move for the opponent


    def get_human_readable_board(self):
        # returns string representation of the board
        board = np.zeros((self.board_height, self.board_width), dtype=int)
        board[self.board_first] = 1
        board[self.board_second] = 2

        result_str = ""
        for row in board:
            result_str += "| "
            for cell in row:
                if cell == 0:
                    result_str += " "
                elif cell == 1:
                    result_str += "X"
                else:
                    result_str += "O"

                result_str += " | "
            result_str += "\n"
        
        result_str += "-" * (self.board_width * 4 + 1) + "\n"
        for i in range(self.board_width):
            result_str += f"| {i} "
        result_str += "|\n"

        return result_str
    
    def get_hash(self):
        # returns a hash of the current game state
        return hash((self.current_player, self.winner, self.game_over, self.board_first.tobytes(), self.board_second.tobytes()))
    
    def get_board_hash(self):
        # shorter hash with only the board state
        return hash((self.board_first.tobytes(), self.board_second.tobytes()))

    def get_board_state_for_nn(self):

        board_hash = self.get_board_hash()

        if self.current_player == 1:
            return np.stack([self.board_first, self.board_second], axis=-1).reshape(1, self.board_height, self.board_width, 2), board_hash
        else:
            return np.stack([self.board_second, self.board_first], axis=-1).reshape(1, self.board_height, self.board_width, 2), board_hash

    def set_hash(self, hash):
        self.reset()
        # sets the game state from a previously saved hash
        self.current_player, self.winner, self.game_over, board_first, board_second = hash
        self.board_first = np.frombuffer(board_first, dtype=int).reshape(self.board_height, self.board_width)
        self.board_second = np.frombuffer(board_second, dtype=int).reshape(self.board_height, self.board_width)
