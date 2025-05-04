from typing import Sequence
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        num_actions: int,
        hidden_size: int = 512,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        dummy_input = torch.zeros(1, *input_shape)
        dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)

        self.advantage = nn.Sequential(
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        a = self.advantage(x)
        v = self.value(x)
        return v + (a - a.mean(dim=1, keepdim=True))

    @property
    def device(self):
        return next(self.parameters()).device

    def get_action(self, state: np.ndarray) -> int:
        if state.dtype == np.uint8:
            state = state.astype(np.float32) / 255.0
        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        q_values = self.forward(state)
        action = q_values.argmax().item()
        return action

    def get_logits(self, state):
        if state.dtype == np.uint8:
            state = state.astype(np.float32) / 255.0
        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        q_values = self.forward(state).squeeze()
        return q_values
