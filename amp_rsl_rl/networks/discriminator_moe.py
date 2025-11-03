# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch import autograd
from rsl_rl.utils import utils
from rsl_rl.utils import resolve_nn_activation
from .ac_moe import MLP_net, OrthogonalLayer
from .discriminator import Discriminator

class MoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        output_dim: int,
        num_experts: int = 4,
        gate_hidden_dims: list[int] | None = None,
        orthogonal_experts: bool = False,
        activation: str = "elu",
    ):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.orthogonal_experts = orthogonal_experts
        self.use_orthogonal = orthogonal_experts
        if self.use_orthogonal:
            print("Using orthogonal experts in MoE.")
        act = resolve_nn_activation(activation)

        # Create expert networks
        self.experts = nn.ModuleList(
            [
                MLP_net(input_dim, hidden_dims, output_dim, act)
                for _ in range(num_experts)
            ]
        )

        self.orthogonal_layer = OrthogonalLayer() if self.use_orthogonal else nn.Identity()

        # gating network
        gate_layers = []
        last_dim = input_dim
        gate_hidden_dims = gate_hidden_dims or []
        for h in gate_hidden_dims:
            gate_layers += [nn.Linear(last_dim, h), act]
            last_dim = h
        gate_layers.append(nn.Linear(last_dim, num_experts))
        self.gate = nn.Sequential(*gate_layers)
        self.softmax = nn.Softmax(dim=-1)  # kept separate for ONNX clarity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim]
        Returns:
            output: [batch, output_dim]
        """
        expert_out = torch.stack([e(x) for e in self.experts], dim=-1) # [B, D, K]
        expert_out = self.orthogonal_layer(expert_out)
        gate_logits = self.gate(x)  # [batch, K]
        weights = self.softmax(gate_logits).unsqueeze(1)  # [batch, 1, K]
        output = (expert_out * weights).sum(-1)  # weighted sum -> [batch, output_dim]
        return output

class DiscriminatorMoE(Discriminator):
    def __init__(self, 
                 *args, 
                 num_experts: int = 4,
                 orthogonal_experts: bool = False,
                 **kwargs):
        self.num_experts = num_experts
        self.orthogonal_experts = orthogonal_experts
        super().__init__(*args, **kwargs)
        print(f"Initialized DiscriminatorMoE with {self.num_experts} experts. Orthogonal experts: {self.orthogonal_experts}")
    
    def _build_trunk(self, input_dim: int, hidden_layer_sizes: list[int]) -> nn.Module:
        return MoE(
            input_dim=input_dim,
            hidden_dims=hidden_layer_sizes[:-1],  # last layer is output
            output_dim=hidden_layer_sizes[-1],
            num_experts=self.num_experts,
            orthogonal_experts=self.orthogonal_experts,
            gate_hidden_dims=hidden_layer_sizes[:-1],
            activation="elu",
        )
