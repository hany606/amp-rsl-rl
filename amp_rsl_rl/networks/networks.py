from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation

class MLP_net(nn.Sequential):
    def __init__(self, in_dim, hidden_dims, out_dim, act):
        layers = [nn.Linear(in_dim, hidden_dims[0]), act]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], out_dim))
            else:
                layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act])
        super().__init__(*layers)

class IdentityPreprocessor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class HeightConvPreprocessor(nn.Module):
    def __init__(self, height_start_idx: int, height_end_idx: int):
        super().__init__()
        self.height_start_idx = height_start_idx
        self.height_end_idx = height_end_idx
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten()
        )

        self.act = nn.ELU()
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height_data = x[:, self.height_start_idx:self.height_end_idx]  # [B, H]
        height_data = height_data.unsqueeze(1)  # [B, 1, H]
        conv_out = self.conv1d(height_data)  # [B, 4, H]
        conv_out = self.act(conv_out)
        conv_out = self.flatten(conv_out)  # [B, 4*H]
        x = torch.cat([x[:, :self.height_start_idx], conv_out, x[:, self.height_end_idx:]], dim=1)
        return x
