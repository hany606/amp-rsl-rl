from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation
from .networks import *
from .ac_moe import ActorMoE

class CriticMLP(nn.Module):
    """Standard MLP critic."""

    def __init__(self, obs_dim: int, hidden_dims: list[int], out_dim: int, act, preprocessor: nn.Module):
        super().__init__()
        self.mlp = MLP_net(obs_dim, hidden_dims, out_dim, act)
        self.preprocessor = preprocessor

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return self.preprocessor(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        return self.mlp(x)

class ActorCriticGeneral(nn.Module):
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
        actor_class: str = "ActorMoE",
        critic_class: str = "CriticMLP",
        actor_preprocessor_class: nn.Module = "IdentityPreprocessor",
        critic_preprocessor_class: nn.Module = "IdentityPreprocessor",
        actor_preprocessor_info: dict = {},
        critic_preprocessor_info: dict = {},
        **kwargs,
    ):
        if kwargs:
            print(
                (
                    "ActorCriticMoE.__init__ ignored unexpected arguments: "
                    + str(list(kwargs.keys()))
                )
            )
        super().__init__()
        act = resolve_nn_activation(activation)

        # Actor (Mixture-of-Experts)
        actor_class = eval(actor_class)
        actor_preprocessor = eval(actor_preprocessor_class)(actor_preprocessor_info)
        self.actor = actor_class(
            obs_dim=num_actor_obs,
            act_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            num_experts=num_experts,
            gate_hidden_dims=actor_hidden_dims[:-1],  # last layer is output
            activation=activation,
            orthogonal_experts=orthogonal_experts,
            use_magnitude_head=use_magnitude_head,
            actor_preprocessor=actor_preprocessor,
        )

        # Critic
        critic_class = eval(critic_class)
        critic_preprocessor = eval(critic_preprocessor_class)(critic_preprocessor_info)
        self.critic = critic_class(num_critic_obs, critic_hidden_dims, 1, act, critic_preprocessor)

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

