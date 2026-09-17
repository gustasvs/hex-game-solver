from settings import HEX_BOARD_SIZE

class Move:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def get_idx(self):
        return self.x * HEX_BOARD_SIZE + self.y