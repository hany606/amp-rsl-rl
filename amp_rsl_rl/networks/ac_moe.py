from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation

from .networks import MLP_net, IdentityPreprocessor

class OrthogonalLayer(nn.Module):
    """
    Adapted from https://github.com/AhmedMagdyHendawy/MOORE/
    Orthogonalizes expert outputs along the expert dimension using a vectorized
    Gram–Schmidt (batched) procedure.

    Input:  x [B, D, E]  (batch, feature_dim, num_experts)
    Output: basis [B, D, E]  (orthonormal across E for each batch item)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x.transpose(1, 2)  # [B, E, D]
        B, E, D = x1.shape

        eps = torch.finfo(x1.dtype).eps if x1.is_floating_point() else 1e-8

        # First basis vector (normalize)
        b0 = x1[:, 0, :]  # [B, D]
        b0 = b0 / b0.norm(dim=1, keepdim=True).clamp_min(eps)
        basis = b0.unsqueeze(1)  # [B, 1, D]

        # Subsequent vectors: v - Proj_{span(basis)}(v) in one batched matmul
        for i in range(1, E):
            v = x1[:, i, :].unsqueeze(1)  # [B, 1, D]
            # projection: [B,1,D] @ [B,D,i] -> [B,1,i] ; then @ [B,i,D] -> [B,1,D]
            proj = v @ basis.transpose(2, 1) @ basis
            w = v - proj
            w = w / w.norm(dim=2, keepdim=True).clamp_min(eps)  # normalize
            basis = torch.cat([basis, w], dim=1)  # [B, i+1, D]

        # Return to [B, D, E]
        return basis.transpose(1, 2)  # [B, D, E]

class ActorMoE(nn.Module):
    """
    Mixture-of-Experts actor:  ⎡expert_1(x) … expert_K(x)⎤·softmax(gate(x))
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims,
        preprocessor: nn.Module = IdentityPreprocessor(),
        num_experts: int = 4,
        gate_hidden_dims: list[int] | None = None,
        activation="elu",
        orthogonal_experts: bool = False,
        use_magnitude_head: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_experts = num_experts
        self.use_orthogonal = orthogonal_experts
        self.use_magnitude_head = use_magnitude_head and orthogonal_experts
        if self.use_orthogonal:
            print("Using orthogonal experts in ActorMoE.")
        if self.use_magnitude_head:
            print("Using magnitude head in ActorMoE.")
        act = resolve_nn_activation(activation)
        self.preprocessor = preprocessor
        # experts
        self.experts = nn.ModuleList(
            [MLP_net(obs_dim, hidden_dims, act_dim, act) for _ in range(num_experts)]
        )

        self.orthogonal_layer = OrthogonalLayer() if self.use_orthogonal else nn.Identity()

        # gating network
        gate_layers = []
        last_dim = obs_dim
        gate_hidden_dims = gate_hidden_dims or []
        for h in gate_hidden_dims:
            gate_layers += [nn.Linear(last_dim, h), act]
            last_dim = h
        gate_layers.append(nn.Linear(last_dim, num_experts))
        self.gate = nn.Sequential(*gate_layers)
        self.softmax = nn.Softmax(dim=-1)  # kept separate for ONNX clarity

        
        if self.use_magnitude_head:
            self.magnitude_head = nn.Sequential(
                nn.Linear(obs_dim, 64),
                resolve_nn_activation(activation),
                nn.Linear(64, self.act_dim),
                nn.Softplus() 
            )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return self.preprocessor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, obs_dim]
        Returns:
            mean action: [batch, act_dim]
        """
        x = self._preprocess(x)
        expert_out = torch.stack([e(x) for e in self.experts], dim=-1) # [B, D, K]
        expert_out = self.orthogonal_layer(expert_out)
        gate_logits = self.gate(x)  # [batch, K]
        weights = self.softmax(gate_logits).unsqueeze(1)  # [batch, 1, K]
        # The weighted sum is a convex combination of orthonormal vectors.
        actions = (expert_out * weights).sum(-1)  # weighted sum -> [batch, A]
        
        if self.use_magnitude_head:
            magnitude = self.magnitude_head(x) # [B, A]
            actions = actions * magnitude
            
        return actions

class ActorCriticMoE(nn.Module):
    """Actor-critic with Mixture-of-Experts policy."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        num_experts: int = 4,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        orthogonal_experts: bool = False,
        use_magnitude_head: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                (
                    "ActorCriticGeneral.__init__ ignored unexpected arguments: "
                    + str(list(kwargs.keys()))
                )
            )
        super().__init__()
        act = resolve_nn_activation(activation)

        # Actor (Mixture-of-Experts)
        self.actor = ActorMoE(
            obs_dim=num_actor_obs,
            act_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            num_experts=num_experts,
            gate_hidden_dims=actor_hidden_dims[:-1],  # last layer is output
            activation=activation,
            orthogonal_experts=orthogonal_experts,
            use_magnitude_head=use_magnitude_head,
        )

        # Critic
        self.critic = MLP_net(num_critic_obs, critic_hidden_dims, 1, act)

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"Actor (MoE) structure:\n{self.actor}")
        print(f"Critic MLP structure:\n{self.critic}")

    def reset(self, dones=None):  # noqa: D401
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:  # "log"
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # deterministic (mean) action
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    # unchanged load_state_dict so checkpoints from the old class still load
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
