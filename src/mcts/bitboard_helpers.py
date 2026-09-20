from settings import HEX_BOARD_SIZE


BOARD_MASK = (1 << (HEX_BOARD_SIZE ** 2)) - 1
LEFT_EDGE_MASK = sum(1 << (row * HEX_BOARD_SIZE) for row in range(HEX_BOARD_SIZE))
RIGHT_EDGE_MASK = LEFT_EDGE_MASK << (HEX_BOARD_SIZE - 1)
TOP_EDGE_MASK = (1 << HEX_BOARD_SIZE) - 1
BOTTOM_EDGE_MASK = TOP_EDGE_MASK << (HEX_BOARD_SIZE * (HEX_BOARD_SIZE - 1))


def has_bit_connection(board, start_edge, target_edge):
    connected = board & start_edge
    while connected:
        if connected & target_edge:
            return True
        expanded = (
            (connected >> HEX_BOARD_SIZE)
            | ((connected & ~RIGHT_EDGE_MASK) >> (HEX_BOARD_SIZE - 1))
            | ((connected & ~LEFT_EDGE_MASK) >> 1)
            | ((connected & ~RIGHT_EDGE_MASK) << 1)
            | ((connected & ~LEFT_EDGE_MASK) << (HEX_BOARD_SIZE - 1))
            | (connected << HEX_BOARD_SIZE)
        ) & board & BOARD_MASK
        new_connected = connected | expanded
        if new_connected == connected:
            return False
        connected = new_connected
    return False