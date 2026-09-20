"""Utilities for comparing Hex agents in head-to-head games."""

import random
from itertools import combinations
from pathlib import Path

import torch
from tqdm import tqdm

from game_logic.classes import Move
from game_logic.hex import HexBoardState
from mcts.mcts import MCTS, Node
from pytorch_model import CustomNNUE
from settings import DEVICE, HEX_BOARD_SIZE


Competitor = tuple[str, CustomNNUE | None]


def arena(
    model_1: CustomNNUE | None = None,
    model_2: CustomNNUE | None = None,
    mcts_iterations: int = 4_000,
    games_per_pairing: int = 10,
    include_plain_mcts: bool = True,
) -> None:
    """
    Play round-robin games between up to two neural models and plain MCTS.

    Every competitor receives exactly `mcts_iterations` simulations per move.
    Games are sequential to avoid Python/PyTorch thread contention.
    Each color-swapped pair of games shares one randomly selected first move.
    Each competitor reuses its own search subtree whenever possible.
    """

    if model_1 is None and model_2 is None:
        return

    if isinstance(mcts_iterations, bool) or not isinstance(mcts_iterations, int):
        raise TypeError("mcts_iterations must be an integer")
    if isinstance(games_per_pairing, bool) or not isinstance(games_per_pairing, int):
        raise TypeError("games_per_pairing must be an integer")
    if mcts_iterations <= 0:
        raise ValueError("mcts_iterations must be positive")
    if games_per_pairing < 0:
        raise ValueError("games_per_pairing cannot be negative")

    competitors: list[Competitor] = []

    if model_1 is not None:
        competitors.append(("Model 1", model_1))

    if model_2 is not None:
        competitors.append(("Model 2", model_2))

    if include_plain_mcts:
        competitors.append((f"MCTS ({mcts_iterations:,} iterations)", None))

    if len(competitors) < 2:
        print("Arena: no pairings to play.")
        return

    pairings = list(combinations(competitors, 2))
    total_games = len(pairings) * games_per_pairing
    pairing_scores = [[0, 0] for _ in pairings]

    def play_arena_game(
        player_1: Competitor,
        player_2: Competitor,
        opening_move: Move,
    ) -> Competitor:
        state = HexBoardState()
        state.make_move((True, opening_move.x, opening_move.y))
        players = (player_1, player_2)

        # Each agent owns its own search tree because its statistics/policies
        # depend on its own evaluator.
        roots = [
            Node(state, move_index=1),
            Node(state, move_index=1),
        ]

        move_index = 1

        with torch.inference_mode():
            while True:
                player_index = move_index % 2
                _, model = players[player_index]
                root = roots[player_index]

                search = MCTS(root, model, mcts_iterations)
                searched_root = search.run()

                if not searched_root.children:
                    raise RuntimeError(
                        "MCTS produced no children from a non-terminal state"
                    )

                best_edge = max(
                    searched_root.children,
                    key=lambda edge: edge.N,
                )

                played_move = best_edge.action
                best_child = best_edge.child

                move_index += 1

                # Use the optimized bitboard terminal check rather than
                # materializing HexBoardState and running the old DFS checks.
                if best_child.is_terminal():
                    outcome = best_child.stored_terminal_outcome

                    if outcome == 1:
                        return player_1
                    if outcome == -1:
                        return player_2

                    raise RuntimeError("A terminal Hex game must have a winner")

                # The player who just searched can directly promote its chosen
                # child, exactly like training/self-play.
                roots[player_index] = best_child

                # Advance the other player's private tree through the move too.
                # If that move was already expanded in its tree, preserve the
                # entire subtree, accumulator, statistics and caches.
                other_index = 1 - player_index
                other_root = roots[other_index]
                matching_edge = other_root.move_edge_map.get(played_move.idx)

                if matching_edge is not None:
                    roots[other_index] = matching_edge.child
                else:
                    # Its tree did not contain the actual move. Build a fresh
                    # root, but preserve its position and evaluation caches.
                    state = best_child.state
                    position_key = (best_child.p1_bits, best_child.p2_bits)
                    matching_node = other_root.node_cache.get(position_key)

                    if matching_node is not None:
                        roots[other_index] = matching_node
                    else:
                        roots[other_index] = Node(
                            state,
                            move_index=move_index,
                            evaluation_cache=other_root.evaluation_cache,
                            model_signature=other_root.model_signature,
                            node_cache=other_root.node_cache,
                        )

    with tqdm(
        total=total_games,
        desc="Arena games",
        unit="game",
    ) as progress:
        for pairing_index, (competitor_a, competitor_b) in enumerate(pairings):
            opening_move = None

            for game_index in range(games_per_pairing):
                if game_index % 2 == 0:
                    player_1, player_2 = competitor_a, competitor_b
                    opening_move = random.choice(HexBoardState().get_legal_moves())
                else:
                    player_1, player_2 = competitor_b, competitor_a

                if opening_move is None:
                    raise RuntimeError("Arena game pair has no opening move")

                winner = play_arena_game(player_1, player_2, opening_move)
                score = pairing_scores[pairing_index]

                if winner is competitor_a:
                    score[0] += 1
                else:
                    score[1] += 1

                progress.set_description(
                    f"{competitor_a[0]} {score[0]}-{score[1]} {competitor_b[0]}",
                    refresh=False,
                )
                progress.update(1)

    results = [
        (competitor_a[0], competitor_b[0], *pairing_scores[pairing_index])
        for pairing_index, (competitor_a, competitor_b) in enumerate(pairings)
    ]

    headers = ("Pairing", "Games", "Wins (left)", "Wins (right)")

    rows = [
        (
            f"{name_a} vs {name_b}",
            str(games_per_pairing),
            str(a_wins),
            str(b_wins),
        )
        for name_a, name_b, a_wins, b_wins in results
    ]

    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    print("\nArena results")
    print(separator)
    print(
        "| "
        + " | ".join(
            heading.ljust(width)
            for heading, width in zip(headers, widths)
        )
        + " |"
    )
    print(separator)

    for row in rows:
        print(
            "| "
            + " | ".join(
                value.ljust(width)
                for value, width in zip(row, widths)
            )
            + " |"
        )

    print(separator)


if __name__ == "__main__":
    source_directory = Path(__file__).resolve().parent
    weights_directory = source_directory / "weights" / f"board_size{HEX_BOARD_SIZE}"
    weight_files = list(weights_directory.glob("*.pt"))

    if not weight_files:
        raise FileNotFoundError(
            f"No .pt model weights found in {weights_directory}"
        )

    newest_weights_path = max(
        weight_files,
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )

    older_weights_path = (
        source_directory
        / "weights" / f"board_size{HEX_BOARD_SIZE}"
        / newest_weights_path.name
    )

    if not older_weights_path.is_file():
        raise FileNotFoundError(
            f"Matching older model weights not found at {older_weights_path}"
        )

    newest_model = CustomNNUE(
        weights=torch.load(
            newest_weights_path,
            map_location=DEVICE,
            weights_only=True,
        )
    ).to(DEVICE)

    older_model = CustomNNUE(
        weights=torch.load(
            older_weights_path,
            map_location=DEVICE,
            weights_only=True,
        )
    ).to(DEVICE)

    newest_model.eval()
    older_model.eval()

    arena(
        newest_model,
        older_model,
        mcts_iterations=5_0,
        games_per_pairing=20,
        include_plain_mcts=True,
    )
