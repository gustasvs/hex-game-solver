import os
import random
import sys

import torch
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from game import play_game
from pytorch_model import CustomNNUE
from settings import DEVICE, HEX_BOARD_SIZE

LEARNING_ITERATIONS = 50

SELF_PLAY_GAMES = 1_000

TRAIN_NEW = False

# games to keep to nnot only trian on most recent model weak spots
GAME_BUFFER_LEN = SELF_PLAY_GAMES * 4

SELF_PLAY_WORKERS = max(1, min(19, os.cpu_count() or 1))
# print(f"Using {} self-play workers.")

WEIGHTS_PATH = f"weights/board_size{HEX_BOARD_SIZE}/model_weights.pt"


_worker_model = None
_worker_eval_cache = None

def init_self_play_worker(weights):
    global _worker_model, _worker_eval_cache
    torch.set_num_threads(1)
    
    _worker_eval_cache = {}
    
    if weights is None:
        _worker_model = None
    else:
        _worker_model = CustomNNUE(weights=weights).to(DEVICE)
        _worker_model.eval()


def play_one_game(task):
    global _worker_model, _worker_eval_cache
    
    iteration, game_index = task
    
    seed = 1000 + iteration * SELF_PLAY_GAMES + game_index
    random.seed(seed)
    torch.manual_seed(seed)

    return play_game(
        _worker_model,
        _worker_eval_cache,
        display=False,
    )


def play_and_record_games(
    model: CustomNNUE | None,
    iteration: int,
) -> list:
    weights = {key: value.detach().cpu() for key, value in model.state_dict().items()} if model is not None else None

    tasks = [(iteration, game_index) for game_index in range(SELF_PLAY_GAMES)]

    game_data = []

    with ProcessPoolExecutor(
        max_workers=SELF_PLAY_WORKERS,
        initializer=init_self_play_worker,
        initargs=(weights,),
    ) as executor:

        results = executor.map(play_one_game, tasks)

        for result in tqdm(
            results,
            total=SELF_PLAY_GAMES,
            desc="Self-play",
            unit="game",
        ):
            game_data.append(result)

    return game_data


def load_model(ignore_weights) -> CustomNNUE:
    weights = None

    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading weights from {WEIGHTS_PATH}")

        weights = torch.load(
            WEIGHTS_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

    model = CustomNNUE(
        weights=None if ignore_weights else weights,
    ).to(DEVICE)

    model.eval()

    return model


def self_play():
    model = load_model(TRAIN_NEW)

    game_buffer = []

    for iteration in range(LEARNING_ITERATIONS):
        print(
            f"\nStarting learning iteration "
            f"{iteration + 1}/"
            f"{LEARNING_ITERATIONS}"
        )

        game_data = play_and_record_games(
            # None if iteration < 2 else model,
            model,
            iteration,
        )

        game_buffer.extend(game_data)

        if len(game_buffer) > GAME_BUFFER_LEN:
            game_buffer = game_buffer[-GAME_BUFFER_LEN:]

        position_count = sum(len(game[0]) for game in game_buffer)

        print(
            f"Replay buffer: "
            f"{len(game_buffer)} games, "
            f"{position_count} positions"
        )

        model.fit_to(game_buffer)

        os.makedirs(
            os.path.dirname(WEIGHTS_PATH),
            exist_ok=True,
        )

        model.save_weights(WEIGHTS_PATH)

        print(f"Saved weights to " f"{WEIGHTS_PATH}")


if __name__ == "__main__":
    try:
        self_play()
    except KeyboardInterrupt:
        print("Self-play interrupted by user.")
        sys.exit(0)
