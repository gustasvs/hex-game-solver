import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


from settings import HEX_BOARD_SIZE, DEVICE, TRAIN_DEVICE

EMBEDDING_DIM = 128
FIRST_LAYER = 32
SECOND_LAYER = 32

P1_OFFSET = 0
P2_OFFSET = HEX_BOARD_SIZE * HEX_BOARD_SIZE
P1_TURN = 2 * HEX_BOARD_SIZE * HEX_BOARD_SIZE
P2_TURN = 2 * HEX_BOARD_SIZE * HEX_BOARD_SIZE + 1
FEATURE_COUNT = 2 * HEX_BOARD_SIZE**2 + 2


class CustomNNUE(nn.Module):
    def __init__(self, weights=None):
        super().__init__()

        # size * size (p1 moves) + size * size (p2 moves) + 2 (player to move)
        # P1 feature = cell_index
        # P2 feature = N² + cell_index
        # Player to move feature = 2 * N²
        self.cached_embeddings = nn.Embedding(
            2 * HEX_BOARD_SIZE * HEX_BOARD_SIZE + 2, EMBEDDING_DIM
        )

        # self.accumulator = torch.zeros(256)

        self.ff = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, FIRST_LAYER),
            nn.ReLU(),
            nn.Linear(FIRST_LAYER, SECOND_LAYER),
            nn.ReLU(),
        )

        self.value_head = nn.Linear(SECOND_LAYER, 1)
        self.policy_head = nn.Linear(SECOND_LAYER, HEX_BOARD_SIZE * HEX_BOARD_SIZE)

        if weights is not None:
            self.load_state_dict(weights)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def advance_accumulator(self, accumulator, is_p1_move_now, move, move_back=False):
        if not move_back:
            # when moving forward we need to add
            # efficiently update new move to cached accum
            accumulator += self.cached_embeddings.weight[
                (move[1] * HEX_BOARD_SIZE + move[2])
                + (P1_OFFSET if is_p1_move_now else P2_OFFSET)
            ]

            # The move changes whose turn is represented by the accumulator.
            accumulator += self.cached_embeddings.weight[
                P2_TURN if is_p1_move_now else P1_TURN
            ]
            accumulator -= self.cached_embeddings.weight[
                P1_TURN if is_p1_move_now else P2_TURN
            ]
        else:
            # when moving back the passed move is the move that made this acc, so remove it
            accumulator -= self.cached_embeddings.weight[
                (move[1] * HEX_BOARD_SIZE + move[2])
                + (P1_OFFSET if is_p1_move_now else P2_OFFSET)
            ]

            # Undoing a move restores the mover as the player to move.
            accumulator += self.cached_embeddings.weight[
                P1_TURN if is_p1_move_now else P2_TURN
            ]
            accumulator -= self.cached_embeddings.weight[
                P2_TURN if is_p1_move_now else P1_TURN
            ]
        return accumulator

    def calculate_move_deltas(self) -> torch.Tensor:
        square_count = HEX_BOARD_SIZE * HEX_BOARD_SIZE
        weights = self.cached_embeddings.weight
        p1_turn_delta = weights[P2_TURN] - weights[P1_TURN]
        p2_turn_delta = weights[P1_TURN] - weights[P2_TURN]

        return torch.stack(
            (
                weights[P1_OFFSET : P1_OFFSET + square_count] + p1_turn_delta,
                weights[P2_OFFSET : P2_OFFSET + square_count] + p2_turn_delta,
            )
        )

    def get_inference_parameters(self):
        return (
            self.ff[0].bias,
            self.ff[0].weight,
            self.ff[2].bias,
            self.ff[2].weight,
            self.value_head.bias,
            self.value_head.weight,
            self.policy_head.bias,
            self.policy_head.weight,
        )

    def forward(self, x, inference_parameters=None):
        if inference_parameters is None:
            inference_parameters = self.get_inference_parameters()

        (
            first_bias,
            first_weight,
            second_bias,
            second_weight,
            value_bias,
            value_weight,
            policy_bias,
            policy_weight,
        ) = inference_parameters

        x = torch.relu(x)
        x = torch.addmv(first_bias, first_weight, x)
        x.relu_()
        x = torch.addmv(second_bias, second_weight, x)
        x.relu_()
        value = torch.tanh(torch.addmv(value_bias, value_weight, x))
        policy = torch.addmv(policy_bias, policy_weight, x)
        return value, policy

    # probably public fn for mcts
    def calculate_accumulator(self, state, is_p1_move_now) -> torch.Tensor:
        accumulator = torch.zeros(EMBEDDING_DIM).to(DEVICE)
        for i in range(HEX_BOARD_SIZE):
            for j in range(HEX_BOARD_SIZE):
                if state[0][i][j] == 1:
                    accumulator += self.cached_embeddings.weight[
                        P1_OFFSET + (i * HEX_BOARD_SIZE + j)
                    ]
                if state[1][i][j] == 1:
                    accumulator += self.cached_embeddings.weight[
                        P2_OFFSET + (i * HEX_BOARD_SIZE + j)
                    ]

        accumulator += self.cached_embeddings.weight[
            P1_TURN if is_p1_move_now else P2_TURN
        ]
        return accumulator

    # TRAINING STUFF
    def fit_to(
        self,
        game_data,
        epochs=100,
        batch_size=128,
        num_workers=0,
    ):
        features = []
        policies = []
        values = []

        # Build the dataset on CPU.
        for game in game_data:
            for state, policy, value in zip(
                game[0],
                game[1],
                game[2],
            ):
                features.append(
                    self.state_to_feature_vector(
                        state[0],
                        state[1],
                    )
                )

                policies.append(
                    torch.tensor(
                        policy,
                        dtype=torch.float32,
                    )
                )

                values.append(float(value))

        if not features:
            raise ValueError("No training samples supplied")

        feature_tensor = torch.stack(features)

        policy_tensor = torch.stack(policies)

        value_tensor = torch.tensor(
            values,
            dtype=torch.float32,
        ).unsqueeze(1)

        dataset = TensorDataset(
            feature_tensor,
            policy_tensor,
            value_tensor,
        )

        train_device = torch.device(TRAIN_DEVICE)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(train_device.type == "cuda"),
        )

        # Move the network itself to the training device.
        self.to(train_device)
        self.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_samples = 0

            for (
                feature_batch,
                target_policy,
                target_value,
            ) in loader:

                feature_batch = feature_batch.to(
                    train_device,
                    non_blocking=True,
                )

                target_policy = target_policy.to(
                    train_device,
                    non_blocking=True,
                )

                target_value = target_value.to(
                    train_device,
                    non_blocking=True,
                )

                self.optimizer.zero_grad(set_to_none=True)

                predicted_value, predicted_policy = self.forward_batch(feature_batch)

                value_loss = F.mse_loss(
                    predicted_value,
                    target_value,
                )

                log_policy = F.log_softmax(
                    predicted_policy,
                    dim=1,
                )

                policy_loss = -(target_policy * log_policy).sum(dim=1).mean()

                loss = value_loss + policy_loss

                loss.backward()

                # Useful sanity check while developing.
                assert self.cached_embeddings.weight.grad is not None

                self.optimizer.step()

                current_batch_size = feature_batch.shape[0]

                total_samples += current_batch_size

                total_loss += loss.item() * current_batch_size

                total_policy_loss += policy_loss.item() * current_batch_size

                total_value_loss += value_loss.item() * current_batch_size

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {total_loss / total_samples:.4f} | "
                f"Policy: "
                f"{total_policy_loss / total_samples:.4f} | "
                f"Value: "
                f"{total_value_loss / total_samples:.4f}"
            )

        # Your MCTS inference currently expects the model on DEVICE.
        self.to(DEVICE)
        self.eval()

    def state_to_feature_vector(self, state, is_p1_move_now):
        features = torch.zeros(
            FEATURE_COUNT,
            dtype=torch.float32,
        )

        for row in range(HEX_BOARD_SIZE):
            for col in range(HEX_BOARD_SIZE):
                cell_index = row * HEX_BOARD_SIZE + col

                if state[0][row][col]:
                    features[P1_OFFSET + cell_index] = 1.0

                if state[1][row][col]:
                    features[P2_OFFSET + cell_index] = 1.0

        features[P1_TURN if is_p1_move_now else P2_TURN] = 1.0

        return features

    def forward_batch(self, feature_batch):
        # feature_batch:
        #   [batch_size, FEATURE_COUNT]
        #
        # embedding weight:
        #   [FEATURE_COUNT, EMBEDDING_DIM]
        #
        # Result:
        #   [batch_size, EMBEDDING_DIM]
        #
        # This is exactly the sum of every active embedding.
        x = feature_batch @ self.cached_embeddings.weight
        
        x = F.relu(x)

        x = F.linear(
            x,
            self.ff[0].weight,
            self.ff[0].bias,
        )
        x = F.relu(x)

        x = F.linear(
            x,
            self.ff[2].weight,
            self.ff[2].bias,
        )
        x = F.relu(x)

        value = torch.tanh(
            F.linear(
                x,
                self.value_head.weight,
                self.value_head.bias,
            )
        )

        policy = F.linear(
            x,
            self.policy_head.weight,
            self.policy_head.bias,
        )

        return value, policy

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
