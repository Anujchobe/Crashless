"""Soft Actor-Critic (twin Q, target networks, automatic entropy tuning)."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .env import DrivingEnv
from .logger import EpisodeTracker
from .networks import SquashedGaussianActor, TwinQ


@dataclass
class SACConfig:
    total_steps: int = 100_000
    buffer_size: int = 200_000
    batch_size: int = 256
    learning_starts: int = 5_000
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    hidden: int = 256
    init_alpha: float = 0.2
    log_every: int = 2_000


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, seed: int = 0):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.term = np.zeros(size, np.float32)
        self.size, self.ptr, self.full = size, 0, False
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, o2, terminated):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.term[i] = o, a, r, o2, terminated
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, n: int):
        idx = self.rng.integers(0, len(self), n)
        t = lambda x: torch.as_tensor(x[idx])
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.term)


class SAC:
    def __init__(self, env_cfg: dict | None, cfg: SACConfig, seed: int = 0, device: str = "cpu", env=None):
        self.cfg, self.seed = cfg, seed
        self.env = env if env is not None else DrivingEnv(env_cfg)
        torch.manual_seed(seed)
        od = self.env.obs_dim
        self.actor = SquashedGaussianActor(od, 2, cfg.hidden)
        self.q = TwinQ(od, 2, cfg.hidden)
        self.q_targ = copy.deepcopy(self.q).requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(cfg.init_alpha), dtype=torch.float32, requires_grad=True)
        self.target_entropy = -2.0
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.buf = ReplayBuffer(od, 2, cfg.buffer_size, seed)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        a, _ = self.actor(torch.as_tensor(obs, dtype=torch.float32), deterministic)
        return a.numpy()

    def update(self):
        c = self.cfg
        o, a, r, o2, term = self.buf.sample(c.batch_size)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            q_next = torch.min(*self.q_targ(o2, a2)) - alpha * logp2
            # Only true terminals cut the bootstrap; time-limit truncation does not.
            y = r + c.gamma * (1 - term) * q_next
        q1, q2 = self.q(o, a)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        self.q.requires_grad_(False)
        pi_a, logp = self.actor(o)
        pi_loss = (alpha * logp - torch.min(*self.q(o, pi_a))).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()
        self.q.requires_grad_(True)

        a_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.a_opt.zero_grad()
        a_loss.backward()
        self.a_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
                pt.mul_(1 - c.tau).add_(c.tau * p)
        return {"q_loss": q_loss.item(), "pi_loss": pi_loss.item(),
                "alpha": alpha.item(), "entropy": -logp.mean().item()}

    def train(self, log_cb=None):
        c = self.cfg
        rng = np.random.default_rng(self.seed)
        obs, _ = self.env.reset(seed=self.seed * 1000)
        tracker, stats = EpisodeTracker(1), {}
        for t in range(1, c.total_steps + 1):
            a = rng.uniform(-1, 1, 2).astype(np.float32) if t <= c.learning_starts \
                else self.act(obs, deterministic=False)
            o2, r, te, tr, info = self.env.step(a)
            self.buf.add(obs, a, r, o2, float(te))
            tracker.step(0, r, te or tr, info)
            obs = self.env.reset()[0] if (te or tr) else o2
            if t > c.learning_starts:
                stats = self.update()
            if log_cb and t % c.log_every == 0:
                log_cb(t, {**tracker.summary(), **stats})
        return self

    def save(self, path):
        torch.save({"algo": "sac", "state": self.actor.state_dict(), "q": self.q.state_dict(),
                    "q_targ": self.q_targ.state_dict(), "log_alpha": self.log_alpha.detach(),
                    "cfg": self.cfg.__dict__}, path)

    def load(self, path, full: bool = False):
        """full=True also restores critics and temperature (needed to fine-tune)."""
        ck = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ck["state"])
        if full:
            if "q" not in ck:
                raise ValueError(f"{path} was saved without critics; retrain it to fine-tune")
            self.q.load_state_dict(ck["q"])
            self.q_targ.load_state_dict(ck["q_targ"])
            with torch.no_grad():
                self.log_alpha.copy_(ck["log_alpha"])
        return self
