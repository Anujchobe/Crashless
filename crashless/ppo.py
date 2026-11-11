"""Proximal Policy Optimization (clipped objective, GAE) for continuous control."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .env import DrivingEnv
from .logger import EpisodeTracker
from .networks import GaussianActorCritic


@dataclass
class PPOConfig:
    total_steps: int = 300_000
    n_envs: int = 8
    n_steps: int = 256            # per env per rollout -> 2048 transitions per update
    epochs: int = 10
    minibatch: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    target_kl: float | None = 0.03


def compute_gae(rewards, values, next_values, terminated, ends, gamma, lam):
    """GAE over a (T, N) rollout from N parallel envs.

    next_values[t] = V(s_{t+1}) for the TRUE next state (before auto-reset).
    terminated[t]  = 1 if s_{t+1} is terminal: no bootstrap from V(s_{t+1}).
    ends[t]        = 1 if the episode ended for any reason (terminated OR
                     time-limit truncated): the advantage recursion stops here
                     so it does not leak into the next episode. A truncated
                     step still bootstraps from V(s_{t+1}), because the state
                     itself is not terminal; only the clock ran out.
    Returns (advantages, returns), both (T, N).
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros_like(rewards[0])
    for t in reversed(range(T)):
        nonterm = 1.0 - terminated[t]
        delta = rewards[t] + gamma * next_values[t] * nonterm - values[t]
        last = delta + gamma * lam * (1.0 - ends[t]) * last
        adv[t] = last
    return adv, adv + values


class PPO:
    def __init__(self, env_cfg: dict | None, cfg: PPOConfig, seed: int = 0, device: str = "cpu", envs=None):
        self.cfg, self.device, self.seed = cfg, torch.device(device), seed
        if envs is not None:            # e.g. one CARLA server = one env
            cfg.n_envs = len(envs)
        self.envs = envs if envs is not None else [DrivingEnv(env_cfg) for _ in range(cfg.n_envs)]
        torch.manual_seed(seed)
        obs_dim = self.envs[0].obs_dim
        self.net = GaussianActorCritic(obs_dim, 2, cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        d = self.net.dist(o)
        return (d.mean if deterministic else d.sample()).cpu().numpy()

    def train(self, log_cb=None):
        c, n = self.cfg, self.cfg.n_envs
        obs = np.stack([e.reset(seed=self.seed * 1000 + i)[0] for i, e in enumerate(self.envs)])
        tracker = EpisodeTracker(n)
        steps, updates = 0, max(1, c.total_steps // (n * c.n_steps))
        for u in range(updates):
            frac = 1.0 - u / updates
            for g in self.opt.param_groups:
                g["lr"] = c.lr * frac
            B = dict(obs=[], act=[], logp=[], rew=[], val=[], nval=[], term=[], end=[])
            for _ in range(c.n_steps):
                with torch.no_grad():
                    o = torch.as_tensor(obs, dtype=torch.float32)
                    d = self.net.dist(o)
                    a = d.sample()
                    logp = d.log_prob(a).sum(-1)
                    v = self.net.value(o)
                a_np = a.numpy()
                nxt, rew, term, trunc = np.empty_like(obs), np.zeros(n), np.zeros(n), np.zeros(n)
                true_next = np.empty_like(obs)
                for i, e in enumerate(self.envs):
                    o2, r, te, tr, info = e.step(a_np[i])
                    rew[i], term[i], trunc[i] = r, te, tr
                    true_next[i] = o2
                    tracker.step(i, r, te or tr, info)
                    nxt[i] = e.reset()[0] if (te or tr) else o2
                with torch.no_grad():
                    nv = self.net.value(torch.as_tensor(true_next, dtype=torch.float32)).numpy()
                for k, val in zip(B, (obs, a_np, logp.numpy(), rew, v.numpy(), nv, term,
                                      np.maximum(term, trunc))):
                    B[k].append(val)
                obs = nxt
                steps += n
            arr = {k: np.asarray(vv, dtype=np.float32) for k, vv in B.items()}
            adv, ret = compute_gae(arr["rew"], arr["val"], arr["nval"], arr["term"],
                                             arr["end"], c.gamma, c.gae_lambda)
            stats = self._update(arr, adv, ret)
            if log_cb:
                log_cb(steps, {**tracker.summary(), **stats})
        return self

    def _update(self, arr, adv, ret):
        c = self.cfg
        flat = lambda x: torch.as_tensor(x.reshape(-1, *x.shape[2:]), dtype=torch.float32)
        obs, act, old_logp = flat(arr["obs"]), flat(arr["act"]), flat(arr["logp"])
        adv_t, ret_t = flat(adv), flat(ret)
        N = obs.shape[0]
        kl = torch.tensor(0.0)
        for _ in range(c.epochs):
            perm = torch.randperm(N)
            for s in range(0, N, c.minibatch):
                idx = perm[s:s + c.minibatch]
                d = self.net.dist(obs[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = (logp - old_logp[idx]).exp()
                a = adv_t[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                vf = 0.5 * (self.net.value(obs[idx]) - ret_t[idx]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + c.vf_coef * vf - c.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - old_logp[idx])).mean()
            if c.target_kl is not None and kl > c.target_kl:
                break
        return {"pg_loss": pg.item(), "vf_loss": vf.item(), "approx_kl": kl.item(),
                "std": self.net.log_std.exp().mean().item()}

    def save(self, path):
        torch.save({"algo": "ppo", "state": self.net.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device)["state"])
        return self
