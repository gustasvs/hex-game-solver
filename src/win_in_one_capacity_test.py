"""Test whether the current network can learn immediate Hex wins.

This is deliberately a stand-alone experiment: it generates synthetic positions,
trains a fresh ``CustomNNUE``, and prints metrics without saving any artifacts.

Default run:
    python src/win_in_one_capacity_test.py
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mcts.bitboard_helpers import (
    BOTTOM_EDGE_MASK,
    LEFT_EDGE_MASK,
    RIGHT_EDGE_MASK,
    TOP_EDGE_MASK,
    has_bit_connection,
)
from pytorch_model import (
    FEATURE_COUNT,
    P1_OFFSET,
    P1_TURN,
    P2_OFFSET,
    P2_TURN,
    CustomNNUE,
)
from settings import HEX_BOARD_SIZE, TRAIN_DEVICE


Coordinate = tuple[int, int]


@dataclass
class PositionTensors:
    features: torch.Tensor
    policies: torch.Tensor
    values: torch.Tensor
    legal: torch.Tensor
    winning: torch.Tensor
    p1_to_move: torch.Tensor

    def __len__(self) -> int:
        return self.features.shape[0]


def make_connecting_path(p1: bool, rng: random.Random) -> list[Coordinate]:
    """Create a randomly kinked connected path between the player's sides."""
    size = HEX_BOARD_SIZE
    if p1:
        row, col = 0, rng.randrange(size)
        path = [(row, col)]
        while row < size - 1:
            if rng.random() < 0.55:
                candidates = [c for c in (col - 1, col + 1) if 0 <= c < size]
                if candidates:
                    col = rng.choice(candidates)
                    path.append((row, col))
            downward = [(row + 1, col)]
            if col > 0:
                downward.append((row + 1, col - 1))
            row, col = rng.choice(downward)
            path.append((row, col))
        return path

    row, col = rng.randrange(size), 0
    path = [(row, col)]
    while col < size - 1:
        if rng.random() < 0.55:
            candidates = [r for r in (row - 1, row + 1) if 0 <= r < size]
            if candidates:
                row = rng.choice(candidates)
                path.append((row, col))
        rightward = [(row, col + 1)]
        if row > 0:
            rightward.append((row - 1, col + 1))
        row, col = rng.choice(rightward)
        path.append((row, col))
    return path


def player_has_won(bits: int, p1: bool) -> bool:
    if p1:
        return has_bit_connection(bits, TOP_EDGE_MASK, BOTTOM_EDGE_MASK)
    return has_bit_connection(bits, LEFT_EDGE_MASK, RIGHT_EDGE_MASK)


def winning_move_bits(p1_bits: int, p2_bits: int, p1_to_move: bool) -> int:
    occupied = p1_bits | p2_bits
    winners = 0
    for index in range(HEX_BOARD_SIZE**2):
        move_bit = 1 << index
        if occupied & move_bit:
            continue
        mover_bits = (p1_bits if p1_to_move else p2_bits) | move_bit
        if player_has_won(mover_bits, p1_to_move):
            winners |= move_bit
    return winners


def generate_position(
    p1_to_move: bool,
    rng: random.Random,
    seen: set[tuple[int, int, bool]],
) -> tuple[int, int, int]:
    """Generate a legal-count, non-terminal position with an immediate win."""
    size = HEX_BOARD_SIZE
    square_count = size**2
    all_cells = [(row, col) for row in range(size) for col in range(size)]

    for _ in range(10_000):
        path = make_connecting_path(p1_to_move, rng)
        gap = rng.choice(path[1:-1] or path)
        own_stones = set(path)
        own_stones.remove(gap)

        # P2-to-move positions need one more P1 stone; P1-to-move counts match.
        opponent_extra = 0 if p1_to_move else 1
        maximum_own_count = (square_count - 1 - opponent_extra) // 2
        if len(own_stones) > maximum_own_count:
            continue

        minimum_count = max(len(own_stones), max(1, square_count // 4))
        maximum_count = min(
            maximum_own_count,
            max(minimum_count, square_count // 3),
        )
        own_count = rng.randint(minimum_count, maximum_count)
        opponent_count = own_count + opponent_extra

        available = [cell for cell in all_cells if cell not in own_stones and cell != gap]
        rng.shuffle(available)
        own_extra_count = own_count - len(own_stones)
        if own_extra_count + opponent_count > len(available):
            continue

        own_stones.update(available[:own_extra_count])
        opponent_stones = set(
            available[own_extra_count : own_extra_count + opponent_count]
        )
        p1_stones, p2_stones = (
            (own_stones, opponent_stones)
            if p1_to_move
            else (opponent_stones, own_stones)
        )

        p1_bits = sum(1 << (row * size + col) for row, col in p1_stones)
        p2_bits = sum(1 << (row * size + col) for row, col in p2_stones)
        if player_has_won(p1_bits, True) or player_has_won(p2_bits, False):
            continue

        winners = winning_move_bits(p1_bits, p2_bits, p1_to_move)
        gap_bit = 1 << (gap[0] * size + gap[1])
        if not winners & gap_bit:
            continue

        key = (p1_bits, p2_bits, p1_to_move)
        if key in seen:
            continue
        seen.add(key)
        return p1_bits, p2_bits, winners

    player = "P1" if p1_to_move else "P2"
    raise RuntimeError(f"could not generate a unique position for {player}")


def make_dataset(
    examples_per_player: int,
    rng: random.Random,
    seen: set[tuple[int, int, bool]],
) -> PositionTensors:
    square_count = HEX_BOARD_SIZE**2
    total = 2 * examples_per_player
    features = torch.zeros((total, FEATURE_COUNT), dtype=torch.float32)
    policies = torch.zeros((total, square_count), dtype=torch.float32)
    values = torch.empty((total, 1), dtype=torch.float32)
    legal = torch.empty((total, square_count), dtype=torch.bool)
    winning = torch.zeros((total, square_count), dtype=torch.bool)
    p1_turns = torch.empty(total, dtype=torch.bool)

    for example_index in range(total):
        p1_to_move = example_index % 2 == 0
        p1_bits, p2_bits, winner_bits = generate_position(p1_to_move, rng, seen)
        occupied = p1_bits | p2_bits
        winner_count = winner_bits.bit_count()

        for index in range(square_count):
            bit = 1 << index
            if p1_bits & bit:
                features[example_index, P1_OFFSET + index] = 1.0
            elif p2_bits & bit:
                features[example_index, P2_OFFSET + index] = 1.0
            if winner_bits & bit:
                winning[example_index, index] = True
                policies[example_index, index] = 1.0 / winner_count
            legal[example_index, index] = not bool(occupied & bit)

        features[example_index, P1_TURN if p1_to_move else P2_TURN] = 1.0
        values[example_index, 0] = 1.0 if p1_to_move else -1.0
        p1_turns[example_index] = p1_to_move

    permutation = torch.randperm(total)
    return PositionTensors(
        features[permutation],
        policies[permutation],
        values[permutation],
        legal[permutation],
        winning[permutation],
        p1_turns[permutation],
    )


def deployment_forward_batch(
    model: CustomNNUE,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized equivalent of calculate_accumulator() followed by forward()."""
    x = torch.relu(features @ model.cached_embeddings.weight)
    x = F.relu(F.linear(x, model.ff[0].weight, model.ff[0].bias))
    x = F.relu(F.linear(x, model.ff[2].weight, model.ff[2].bias))
    values = torch.tanh(F.linear(x, model.value_head.weight, model.value_head.bias))
    policies = F.linear(x, model.policy_head.weight, model.policy_head.bias)
    return values, policies


def subset_metrics(
    logits: torch.Tensor,
    values: torch.Tensor,
    data: PositionTensors,
    mask: torch.Tensor,
) -> tuple[float, float, float, float]:
    logits = logits[mask]
    values = values[mask]
    legal = data.legal[mask].to(logits.device)
    winners = data.winning[mask].to(logits.device)
    targets = data.values[mask].to(values.device)

    legal_logits = logits.masked_fill(~legal, -torch.inf)
    probabilities = torch.softmax(legal_logits, dim=1)
    best_moves = legal_logits.argmax(dim=1)
    top1_winner = winners.gather(1, best_moves[:, None]).float().mean().item()
    winning_mass = (probabilities * winners).sum(dim=1).mean().item()
    value_mae = (values - targets).abs().mean().item()
    value_sign = ((values[:, 0] >= 0) == (targets[:, 0] >= 0)).float().mean().item()
    return top1_winner, winning_mass, value_mae, value_sign


@torch.inference_mode()
def evaluate(
    model: CustomNNUE,
    data: PositionTensors,
    device: torch.device,
    deployment_path: bool,
) -> dict[str, tuple[float, float, float, float]]:
    model.eval()
    all_logits = []
    all_values = []
    for start in range(0, len(data), 2048):
        features = data.features[start : start + 2048].to(device)
        if deployment_path:
            values, logits = deployment_forward_batch(model, features)
        else:
            values, logits = model.forward_batch(features)
        all_logits.append(logits)
        all_values.append(values)
    logits = torch.cat(all_logits)
    values = torch.cat(all_values)
    p1 = data.p1_to_move
    return {
        "all": subset_metrics(logits, values, data, torch.ones(len(data), dtype=torch.bool)),
        "P1": subset_metrics(logits, values, data, p1),
        "P2": subset_metrics(logits, values, data, ~p1),
    }


def print_metrics(
    label: str,
    metrics: dict[str, tuple[float, float, float, float]],
) -> None:
    print(label)
    print("  split  top1-win  win-mass  value-MAE  value-sign")
    for split, (top1, mass, mae, sign) in metrics.items():
        print(f"  {split:>5}  {top1:8.2%}  {mass:8.2%}  {mae:9.4f}  {sign:10.2%}")


def train(
    model: CustomNNUE,
    train_data: PositionTensors,
    test_data: PositionTensors,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    report_every = max(1, epochs // 20)

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_data))
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for start in range(0, len(train_data), batch_size):
            indices = permutation[start : start + batch_size]
            features = train_data.features[indices].to(device)
            target_policy = train_data.policies[indices].to(device)
            target_value = train_data.values[indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            predicted_value, logits = model.forward_batch(features)
            policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(predicted_value, target_value)
            (policy_loss + value_loss).backward()
            optimizer.step()

            count = len(indices)
            total_policy_loss += policy_loss.item() * count
            total_value_loss += value_loss.item() * count

        if epoch == 1 or epoch % report_every == 0 or epoch == epochs:
            batch_metrics = evaluate(model, test_data, device, deployment_path=False)["all"]
            deploy_metrics = evaluate(model, test_data, device, deployment_path=True)["all"]
            print(
                f"epoch {epoch:3d}/{epochs} | policy {total_policy_loss / len(train_data):.4f} "
                f"value {total_value_loss / len(train_data):.4f} | "
                f"held-out top1 train-path {batch_metrics[0]:.2%} "
                f"MCTS-path {deploy_metrics[0]:.2%}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-per-player", type=int, default=5_000)
    parser.add_argument("--test-examples-per-player", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=421)
    args = parser.parse_args()
    for name in ("examples_per_player", "test_examples_per_player", "epochs", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(TRAIN_DEVICE)
    rng = random.Random(args.seed)
    seen: set[tuple[int, int, bool]] = set()
    started = time.perf_counter()
    train_data = make_dataset(args.examples_per_player, rng, seen)
    test_data = make_dataset(args.test_examples_per_player, rng, seen)
    print(
        f"Generated {len(train_data):,} training positions "
        f"({args.examples_per_player:,}/player) and {len(test_data):,} unique held-out "
        f"positions ({args.test_examples_per_player:,}/player) in "
        f"{time.perf_counter() - started:.2f}s."
    )
    print(f"Board: {HEX_BOARD_SIZE}x{HEX_BOARD_SIZE}; device: {device}; seed: {args.seed}")
    print(
        f"Mean immediate winning moves: train={train_data.winning.sum(1).float().mean():.3f}, "
        f"test={test_data.winning.sum(1).float().mean():.3f}"
    )

    model = CustomNNUE().to(device)
    print_metrics(
        "\nUntrained model, held-out data, training forward path:",
        evaluate(model, test_data, device, deployment_path=False),
    )
    print_metrics(
        "Untrained model, held-out data, actual MCTS forward path:",
        evaluate(model, test_data, device, deployment_path=True),
    )

    training_started = time.perf_counter()
    train(
        model,
        train_data,
        test_data,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        device,
    )
    elapsed = time.perf_counter() - training_started

    print_metrics(
        "\nFinal training-set metrics, training forward path:",
        evaluate(model, train_data, device, deployment_path=False),
    )
    print_metrics(
        "Final held-out metrics, training forward path:",
        evaluate(model, test_data, device, deployment_path=False),
    )
    print_metrics(
        "Final held-out metrics, actual MCTS forward path:",
        evaluate(model, test_data, device, deployment_path=True),
    )
    print(f"Training time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
