"""Compare board scanning with parent-derived legal moves during node creation.

Run from the repository root with:

    python benchmarks/benchmark_node_construction.py

The benchmark constructs complete 7x7 paths repeatedly.  Both strategies pay
the same cost to copy the board and apply a move; only legal-move discovery is
different.
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from game_logic.hex import HexBoardState  # noqa: E402


Move = tuple[int, int]


class _BenchmarkNode:
    """Minimal Node shape used to isolate construction throughput."""

    __slots__ = ("state", "legal_moves", "parent", "children", "next_move", "N", "W")

    def __init__(self, state, parent=None, legal_moves=None):
        self.state = state
        self.legal_moves = (
            tuple(state.get_legal_moves())
            if legal_moves is None
            else tuple(legal_moves)
        )
        self.parent = parent
        self.children = []
        self.next_move = 0
        self.N = 0
        self.W = 0


def _construct_paths(strategy: str, paths: int, move_orders: list[list[Move]]) -> int:
    nodes_created = 0
    for path_index in range(paths):
        state = HexBoardState()
        parent = _BenchmarkNode(state)

        for move_index, move in enumerate(move_orders[path_index % len(move_orders)]):
            player_one = move_index % 2 == 0
            state = HexBoardState(
                state=state.get_state(), move=(player_one, move[0], move[1])
            )

            if strategy == "scan":
                node = _BenchmarkNode(state, parent=parent)
            else:
                move_position = parent.legal_moves.index(move)
                child_legal_moves = (
                    parent.legal_moves[:move_position]
                    + parent.legal_moves[move_position + 1:]
                )
                node = _BenchmarkNode(
                    state, parent=parent, legal_moves=child_legal_moves
                )

            parent = node
            nodes_created += 1

    return nodes_created


def _measure(strategy: str, paths: int, move_orders: list[list[Move]]) -> tuple[int, float]:
    gc.collect()
    gc.disable()
    try:
        started = perf_counter()
        count = _construct_paths(strategy, paths, move_orders)
        elapsed = perf_counter() - started
    finally:
        gc.enable()
    return count, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=5_000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    board_size = HexBoardState().size
    coordinates = [
        (row, col)
        for row in range(board_size)
        for col in range(board_size)
    ]
    random_source = random.Random(args.seed)
    move_orders = []
    for _ in range(16):
        order = coordinates.copy()
        random_source.shuffle(order)
        move_orders.append(order)

    results: dict[str, list[float]] = {"scan": [], "derive": []}
    node_count = 0
    for round_index in range(args.rounds):
        # Alternate the first strategy to reduce systematic ordering effects.
        strategies = ("scan", "derive") if round_index % 2 == 0 else ("derive", "scan")
        for strategy in strategies:
            node_count, elapsed = _measure(strategy, args.paths, move_orders)
            results[strategy].append(elapsed)

    medians = {
        strategy: sorted(samples)[len(samples) // 2]
        for strategy, samples in results.items()
    }
    print(f"Board: {board_size}x{board_size}")
    print(f"Nodes per round: {node_count:,}")
    for strategy in ("scan", "derive"):
        elapsed = medians[strategy]
        print(
            f"{strategy:>6}: {node_count / elapsed:,.0f} nodes/s "
            f"(median {elapsed:.6f}s, {args.rounds} rounds)"
        )
    print(f"derive/scan throughput: {medians['scan'] / medians['derive']:.3f}x")


if __name__ == "__main__":
    main()
