"""Deterministic evaluation on held-out road layouts."""
from __future__ import annotations

import numpy as np

from .env import DrivingEnv

EVAL_SEED_BASE = 1_000_000  # training never uses seeds this large


def evaluate(policy, env_cfg: dict | None = None, episodes: int = 100, seed_base: int = EVAL_SEED_BASE,
             env=None) -> dict:
    env = env if env is not None else DrivingEnv(env_cfg)
    outcomes = {"success": 0, "crashed": 0, "offroad": 0, "timeout": 0}
    returns, lat_abs, speeds, steer_jerk, progress = [], [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        ret, prev_steer, done = 0.0, 0.0, False
        while not done:
            a = policy.act(obs, deterministic=True)
            obs, r, te, tr, info = env.step(a)
            ret += r
            lat_abs.append(abs(info["lateral"]))
            speeds.append(info["speed"])
            steer_jerk.append(abs(info["steer"] - prev_steer))
            prev_steer = info["steer"]
            done = te or tr
        returns.append(ret)
        progress.append(min(info["progress"], 1.0))
        key = "success" if info["success"] else "crashed" if info["crashed"] else \
            "offroad" if info["offroad"] else "timeout"
        outcomes[key] += 1
    return {
        "episodes": episodes,
        "success_rate": outcomes["success"] / episodes,
        "collision_rate": outcomes["crashed"] / episodes,
        "offroad_rate": outcomes["offroad"] / episodes,
        "timeout_rate": outcomes["timeout"] / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_progress": float(np.mean(progress)),
        "mean_abs_lateral_m": float(np.mean(lat_abs)),
        "mean_speed_mps": float(np.mean(speeds)),
        "mean_abs_steer_change": float(np.mean(steer_jerk)),
    }
