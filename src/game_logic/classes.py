from settings import HEX_BOARD_SIZE

class Move:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = x * HEX_BOARD_SIZE + y
    
    def get_idx(self):
        return self.idx
