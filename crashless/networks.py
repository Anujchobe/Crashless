"""Policy and value networks shared by PPO and SAC."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(inp: int, out: int, hidden: int, act=nn.Tanh, layers: int = 2) -> nn.Sequential:
    mods, d = [], inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), act()]
        d = hidden
    mods.append(nn.Linear(d, out))
    return nn.Sequential(*mods)


class GaussianActorCritic(nn.Module):
    """PPO policy: unsquashed diagonal Gaussian with state-independent log-std.
    The env clips actions to [-1, 1]; log-probs are computed on the raw sample,
    which keeps the PPO ratio exact."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.pi = mlp(obs_dim, act_dim, hidden)
        self.v = mlp(obs_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        # Small final layer for the policy -> near-zero initial actions.
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)
        nn.init.zeros_(self.pi[-1].bias)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.pi(obs), self.log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


class SquashedGaussianActor(nn.Module):
    """SAC policy: tanh(N(mu, sigma)) with the change-of-variables correction."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.body = mlp(obs_dim, 2 * act_dim, hidden, act=nn.ReLU)
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        mu, log_std = self.body(obs).split(self.act_dim, -1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        # log N(u; mu, std) summed over action dims
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        # log|det da/du| = sum log(1 - tanh(u)^2), computed stably:
        # log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden, act=nn.ReLU)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden, act=nn.ReLU)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)
