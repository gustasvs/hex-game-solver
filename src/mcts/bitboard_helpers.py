from settings import HEX_BOARD_SIZE


BOARD_MASK = (1 << (HEX_BOARD_SIZE ** 2)) - 1
_SQUARE_COUNT = HEX_BOARD_SIZE ** 2
LEFT_EDGE_MASK = sum(1 << (row * HEX_BOARD_SIZE) for row in range(HEX_BOARD_SIZE))
RIGHT_EDGE_MASK = LEFT_EDGE_MASK << (HEX_BOARD_SIZE - 1)
TOP_EDGE_MASK = (1 << HEX_BOARD_SIZE) - 1
BOTTOM_EDGE_MASK = TOP_EDGE_MASK << (HEX_BOARD_SIZE * (HEX_BOARD_SIZE - 1))
_ROTATED_BIT_MASKS = tuple(
    1 << (_SQUARE_COUNT - 1 - index)
    for index in range(_SQUARE_COUNT)
)


def rotate_bitboard_180(board):
    """Rotate a row-major board bitset by 180 degrees."""
    rotated = 0
    remaining = board

    while remaining:
        bit = remaining & -remaining
        rotated |= _ROTATED_BIT_MASKS[bit.bit_length() - 1]
        remaining ^= bit

    return rotated


def canonical_evaluation_key(p1_bits, p2_bits):
    """Return the shared key for a position and its 180-degree rotation."""
    position_key = (p1_bits, p2_bits)
    rotated_key = (
        rotate_bitboard_180(p1_bits),
        rotate_bitboard_180(p2_bits),
    )

    if rotated_key < position_key:
        return rotated_key, True

    return position_key, False


def rotate_policy_180(policy):
    """Return policy entries in the opposite 180-degree orientation."""
    return list(reversed(policy))


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
