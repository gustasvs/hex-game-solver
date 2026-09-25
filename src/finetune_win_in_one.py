"""Fine-tune the archived board-size-5 checkpoint on immediate wins."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from pytorch_model import CustomNNUE
from settings import HEX_BOARD_SIZE, TRAIN_DEVICE
from win_in_one_capacity_test import make_dataset


SOURCE_DIRECTORY = Path(__file__).resolve().parent
BOARD_DIRECTORY = SOURCE_DIRECTORY / "weights" / f"board_size{HEX_BOARD_SIZE}"
ARCHIVED_WEIGHTS = (
    BOARD_DIRECTORY / "temp_pre_win_in_one_finetune" / "model_weights.pt"
)
CURRENT_WEIGHTS = BOARD_DIRECTORY / "model_weights.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-per-player", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=421)
    args = parser.parse_args()
    for name in ("examples_per_player", "epochs", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def main() -> None:
    args = parse_args()
    if HEX_BOARD_SIZE != 5:
        raise RuntimeError(
            f"this checkpoint operation is for board size 5, not {HEX_BOARD_SIZE}"
        )
    if not ARCHIVED_WEIGHTS.is_file():
        raise FileNotFoundError(f"archived checkpoint not found: {ARCHIVED_WEIGHTS}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = make_dataset(
        args.examples_per_player,
        random.Random(args.seed),
        set(),
    )
    device = torch.device(TRAIN_DEVICE)
    state_dict = torch.load(
        ARCHIVED_WEIGHTS,
        map_location=device,
        weights_only=True,
    )
    model = CustomNNUE(weights=state_dict).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Loaded: {ARCHIVED_WEIGHTS}")
    print(
        f"Fine-tuning on {len(data):,} positions "
        f"({args.examples_per_player:,} per player) for {args.epochs} epochs on {device}."
    )
    model.train()
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(data))
        total_loss = 0.0
        for start in range(0, len(data), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            features = data.features[indices].to(device)
            target_policy = data.policies[indices].to(device)
            target_value = data.values[indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            predicted_value, logits = model.forward_batch(features)
            policy_loss = -(
                target_policy * F.log_softmax(logits, dim=1)
            ).sum(dim=1).mean()
            value_loss = F.mse_loss(predicted_value, target_value)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs}: loss {total_loss / len(data):.4f}")

    CURRENT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        CURRENT_WEIGHTS,
    )
    print(f"Saved fine-tuned checkpoint: {CURRENT_WEIGHTS}")


if __name__ == "__main__":
    main()
