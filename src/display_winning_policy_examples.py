"""Display raw model policies for positions with an immediate winning move."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import RegularPolygon

from game_logic.hex import HexBoardState
from mcts.mcts import Node
from mcts.bitboard_helpers import (
    BOTTOM_EDGE_MASK,
    LEFT_EDGE_MASK,
    RIGHT_EDGE_MASK,
    TOP_EDGE_MASK,
    has_bit_connection,
)
from pytorch_model import CustomNNUE
from settings import DEVICE, HEX_BOARD_SIZE


Coordinate = tuple[int, int]


@dataclass
class PolicyExample:
    state: HexBoardState
    p1_to_move: bool
    intended_gap: Coordinate
    winning_moves: set[Coordinate]
    policy: np.ndarray
    value: float


def load_most_recent_model() -> tuple[CustomNNUE, Path]:
    """Load the newest .pt file from src/weights."""
    weights_directory = Path(__file__).resolve().parent / "weights" / f"board_size{HEX_BOARD_SIZE}"
    weight_files = list(weights_directory.glob("*.pt"))

    if not weight_files:
        raise FileNotFoundError(
            f"No .pt model weights found in {weights_directory}"
        )

    weights_path = max(
        weight_files,
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    weights = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model = CustomNNUE(weights=weights).to(DEVICE)
    model.eval()
    return model, weights_path


def make_connecting_path(
    p1: bool,
    rng: random.Random,
    size: int = HEX_BOARD_SIZE,
) -> list[Coordinate]:
    """Make a varied connected path between the current player's two sides."""
    if size < 1:
        raise ValueError("board size must be positive")

    if p1:
        row = 0
        col = rng.randrange(size)
        path = [(row, col)]

        while row < size - 1:
            # Occasional sideways cells make examples less repetitive while
            # retaining a simple, visibly connected top-to-bottom path.
            if rng.random() < 0.55:
                sideways = [
                    candidate
                    for candidate in (col - 1, col + 1)
                    if 0 <= candidate < size
                ]
                if sideways:
                    col = rng.choice(sideways)
                    path.append((row, col))

            downward = [(row + 1, col)]
            if col > 0:
                downward.append((row + 1, col - 1))
            row, col = rng.choice(downward)
            path.append((row, col))

        return path

    row = rng.randrange(size)
    col = 0
    path = [(row, col)]

    while col < size - 1:
        if rng.random() < 0.55:
            sideways = [
                candidate
                for candidate in (row - 1, row + 1)
                if 0 <= candidate < size
            ]
            if sideways:
                row = rng.choice(sideways)
                path.append((row, col))

        rightward = [(row, col + 1)]
        if row > 0:
            rightward.append((row - 1, col + 1))
        row, col = rng.choice(rightward)
        path.append((row, col))

    return path


def node_for_position(state: HexBoardState) -> Node:
    """Create a search node with the ply inferred from the occupied cells."""
    move_index = sum(map(sum, state.p1)) + sum(map(sum, state.p2))
    return Node(state, move_index=move_index)


def position_has_winner(state: HexBoardState) -> bool:
    """Check both players, including synthetically generated positions."""
    node = node_for_position(state)
    return (
        has_bit_connection(node.p1_bits, TOP_EDGE_MASK, BOTTOM_EDGE_MASK)
        or has_bit_connection(node.p2_bits, LEFT_EDGE_MASK, RIGHT_EDGE_MASK)
    )


def immediate_winning_moves(
    state: HexBoardState,
    p1_to_move: bool,
) -> set[Coordinate]:
    """Return every legal placement that immediately wins for the mover."""
    winning_moves = set()
    root = node_for_position(state)

    if root.is_p1_turn() != p1_to_move:
        raise ValueError(
            "p1_to_move is inconsistent with the position's stone counts"
        )

    expected_outcome = 1 if p1_to_move else -1

    for move in root.legal_moves:
        child = root.create_edge(move).child
        if (
            child.is_terminal()
            and child.stored_terminal_outcome == expected_outcome
        ):
            winning_moves.add((move.x, move.y))

    return winning_moves


def generate_position(
    p1_to_move: bool,
    rng: random.Random,
    seen: set[tuple],
) -> tuple[HexBoardState, Coordinate, set[Coordinate]]:
    """Generate a balanced non-terminal position with a one-cell path gap."""
    size = HEX_BOARD_SIZE
    if size < 2:
        raise ValueError(
            "winning-policy examples require a Hex board of at least 2x2"
        )

    cell_count = size * size
    all_cells = [(row, col) for row in range(size) for col in range(size)]

    for _ in range(10_000):
        path = make_connecting_path(p1_to_move, rng, size)
        gap_candidates = path[1:-1] or path
        intended_gap = rng.choice(gap_candidates)
        current_stones = set(path)
        current_stones.remove(intended_gap)

        # Keep the board busy enough to resemble a mid-game position while
        # retaining legal move counts and at least the intended gap as empty.
        opponent_extra = 0 if p1_to_move else 1
        maximum_legal_count = (cell_count - 1 - opponent_extra) // 2
        if len(current_stones) > maximum_legal_count:
            continue

        minimum_count = max(len(current_stones), max(1, cell_count // 4))
        maximum_count = min(
            maximum_legal_count,
            max(minimum_count, cell_count // 3),
        )
        current_count = rng.randint(minimum_count, maximum_count)
        opponent_count = current_count + opponent_extra

        available = [
            cell
            for cell in all_cells
            if cell not in current_stones and cell != intended_gap
        ]
        rng.shuffle(available)

        own_extra_count = current_count - len(current_stones)
        required = own_extra_count + opponent_count
        if required > len(available):
            continue

        current_stones.update(available[:own_extra_count])
        opponent_stones = set(available[own_extra_count:required])

        if p1_to_move:
            p1_stones = current_stones
            p2_stones = opponent_stones
        else:
            p1_stones = opponent_stones
            p2_stones = current_stones

        p1 = [[0] * size for _ in range(size)]
        p2 = [[0] * size for _ in range(size)]
        for row, col in p1_stones:
            p1[row][col] = 1
        for row, col in p2_stones:
            p2[row][col] = 1

        state = HexBoardState(state=(p1, p2))
        if position_has_winner(state):
            continue

        winning_moves = immediate_winning_moves(state, p1_to_move)
        if intended_gap not in winning_moves:
            continue

        key = (
            p1_to_move,
            tuple(cell for row in p1 for cell in row),
            tuple(cell for row in p2 for cell in row),
        )
        if key in seen:
            continue

        seen.add(key)
        return state, intended_gap, winning_moves

    player = "P1" if p1_to_move else "P2"
    raise RuntimeError(f"Could not generate a unique winning position for {player}")


def model_policy(
    model: CustomNNUE,
    state: HexBoardState,
    p1_to_move: bool,
) -> tuple[float, np.ndarray]:
    """Run one network evaluation and softmax only across legal moves."""
    size = state.size
    with torch.inference_mode():
        accumulator = model.calculate_accumulator(state.get_state(), p1_to_move)
        value, logits = model(accumulator)

        if logits.numel() != size * size:
            raise ValueError(
                f"model has {logits.numel()} policy outputs for a "
                f"{size}x{size} board"
            )

        legal_indices = [move.idx for move in state.get_legal_moves()]
        legal_tensor = torch.tensor(legal_indices, device=logits.device)
        probabilities = torch.zeros_like(logits)
        probabilities[legal_tensor] = torch.softmax(logits[legal_tensor], dim=0)

    return value.item(), probabilities.reshape(
        size,
        size,
    ).cpu().numpy()


def make_examples(
    model: CustomNNUE,
    count: int,
    seed: int,
) -> list[PolicyExample]:
    rng = random.Random(seed)
    seen: set[tuple] = set()
    examples = []

    for example_index in range(count):
        p1_to_move = example_index % 2 == 0
        state, intended_gap, winning_moves = generate_position(
            p1_to_move,
            rng,
            seen,
        )
        value, policy = model_policy(model, state, p1_to_move)
        examples.append(
            PolicyExample(
                state=state,
                p1_to_move=p1_to_move,
                intended_gap=intended_gap,
                winning_moves=winning_moves,
                policy=policy,
                value=value,
            )
        )

    return examples


def board_geometry(size: int = HEX_BOARD_SIZE) -> tuple[np.ndarray, np.ndarray]:
    row, col = np.indices((size, size))
    centers = np.column_stack((
        np.sqrt(3) * (col.ravel() + row.ravel() / 2),
        -1.5 * row.ravel(),
    ))
    angles = np.pi / 6 + np.arange(6) * np.pi / 3
    corners = np.column_stack((np.cos(angles), np.sin(angles)))
    return centers, centers[:, None, :] + 0.98 * corners


def draw_player_edges(ax, hexagons: np.ndarray, size: int) -> None:
    red_edges = []
    blue_edges = []
    gap = 0.14

    for cell in hexagons[:size]:
        offset = np.array((0.0, gap))
        red_edges.extend(((cell[0] + offset, cell[1] + offset),
                          (cell[1] + offset, cell[2] + offset)))
    for cell in hexagons[-size:]:
        offset = np.array((0.0, -gap))
        red_edges.extend(((cell[3] + offset, cell[4] + offset),
                          (cell[4] + offset, cell[5] + offset)))
    for cell in hexagons[::size]:
        offset = gap * np.array((-np.sqrt(3) / 2, -0.5))
        blue_edges.extend(((cell[2] + offset, cell[3] + offset),
                           (cell[3] + offset, cell[4] + offset)))
    for cell in hexagons[size - 1::size]:
        offset = gap * np.array((np.sqrt(3) / 2, 0.5))
        blue_edges.extend(((cell[5] + offset, cell[0] + offset),
                           (cell[0] + offset, cell[1] + offset)))

    ax.add_collection(LineCollection(red_edges, colors="white", linewidths=7))
    ax.add_collection(LineCollection(blue_edges, colors="white", linewidths=7))
    ax.add_collection(LineCollection(red_edges, colors="#d92626", linewidths=4))
    ax.add_collection(LineCollection(blue_edges, colors="#2659d9", linewidths=4))


def draw_example(
    ax,
    example: PolicyExample,
    example_number: int,
    probability_norm: Normalize,
) -> None:
    size = example.state.size
    if example.policy.shape != (size, size):
        raise ValueError(
            f"policy shape {example.policy.shape} does not match "
            f"the {size}x{size} board"
        )

    centers, hexagons = board_geometry(size)
    p1 = np.asarray(example.state.p1, dtype=bool)
    p2 = np.asarray(example.state.p2, dtype=bool)
    occupied = p1 | p2
    colors = np.ones((size * size, 4))
    color_map = plt.colormaps["YlGn"]

    for row in range(size):
        for col in range(size):
            index = row * size + col
            if p1[row, col]:
                colors[index] = (0.85, 0.15, 0.15, 1.0)
            elif p2[row, col]:
                colors[index] = (0.15, 0.35, 0.85, 1.0)
            else:
                colors[index] = color_map(probability_norm(example.policy[row, col]))

    ax.add_collection(
        PolyCollection(
            list(hexagons),
            facecolors=colors,
            edgecolors="#303030",
            linewidths=0.8,
        )
    )
    draw_player_edges(ax, hexagons, size)

    for row in range(size):
        for col in range(size):
            if occupied[row, col]:
                continue

            index = row * size + col
            probability = example.policy[row, col]
            is_winner = (row, col) in example.winning_moves
            text = f"{probability:.1%}"
            if is_winner:
                text = f"WIN\n{text}"
                ax.add_patch(
                    RegularPolygon(
                        centers[index],
                        6,
                        radius=0.82,
                        orientation=np.pi / 6,
                        facecolor="none",
                        edgecolor="#ffb000",
                        linewidth=2.8,
                        zorder=5,
                    )
                )

            ax.text(
                *centers[index],
                text,
                ha="center",
                va="center",
                fontsize=5.4 if not is_winner else 5.0,
                fontweight="bold" if is_winner else "normal",
                color="#111111",
                zorder=6,
            )

    player = "P1 / red" if example.p1_to_move else "P2 / blue"
    winning_mass = sum(
        example.policy[row, col] for row, col in example.winning_moves
    )
    gap_probability = example.policy[example.intended_gap]
    ax.set_title(
        f"#{example_number}: {player} to move | value {example.value:+.3f}\n"
        f"winning policy {winning_mass:.1%} | intended gap {gap_probability:.1%}",
        fontsize=9,
    )
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")


def display_examples(
    examples: list[PolicyExample],
    weights_path: Path,
    save_path: Path | None,
) -> None:
    board_sizes = {example.state.size for example in examples}
    if len(board_sizes) != 1:
        raise ValueError("all displayed examples must use the same board size")

    size = board_sizes.pop()
    columns = min(len(examples), max(1, 25 // size))
    rows = math.ceil(len(examples) / columns)
    panel_size = min(8.0, max(4.0, 0.55 * size))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(panel_size * columns, (panel_size + 0.1) * rows),
        squeeze=False,
        layout="constrained",
    )

    largest_probability = max(
        float(example.policy.max()) for example in examples
    )
    probability_norm = Normalize(vmin=0.0, vmax=largest_probability)

    for index, example in enumerate(examples):
        draw_example(
            axes.flat[index],
            example,
            index + 1,
            probability_norm,
        )

    for unused_axis in axes.flat[len(examples):]:
        unused_axis.axis("off")

    figure.suptitle(
        "Pure model policy on one-move wins (no MCTS)\n"
        "Cell text is legal-move probability; gold outlines mark immediate wins. "
        f"Model: {weights_path.name}",
        fontsize=13,
        fontweight="bold",
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {save_path.resolve()}")

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--examples",
        type=int,
        default=10,
        help="number of examples to generate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed used to generate positions (default: 42)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="optional image path to save in addition to displaying the window",
    )
    args = parser.parse_args()
    if args.examples <= 0:
        parser.error("--examples must be positive")
    return args


def main() -> None:
    args = parse_args()
    model, weights_path = load_most_recent_model()
    print(f"Loaded most recent model: {weights_path}")
    examples = make_examples(model, args.examples, args.seed)
    display_examples(examples, weights_path, args.save)


if __name__ == "__main__":
    main()
