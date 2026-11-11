"""Draw trajectories of the trained agents and the lane-follow baseline on held-out layouts.

    python scripts/demo.py --ppo runs/main/final/ppo_s0 --sac runs/main/final/sac_s0
Writes results/demo_trajectories.png.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from crashless.baselines import LaneFollowPolicy
from crashless.env import DrivingEnv
from crashless.evaluate import EVAL_SEED_BASE
from scripts.reeval import load_agent


def trace(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    xs, ys, done = [env.x], [env.y], False
    while not done:
        obs, _, te, tr, info = env.step(policy.act(obs))
        xs.append(env.x); ys.append(env.y)
        done = te or tr
    end = "success" if info["success"] else "crash" if info["crashed"] else "off-road" if info["offroad"] else "timeout"
    return np.array(xs), np.array(ys), end


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo", required=True)
    ap.add_argument("--sac", required=True)
    ap.add_argument("--layouts", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    env_cfg = yaml.safe_load(open(ROOT / "configs/env.yaml"))
    env = DrivingEnv(env_cfg)
    agents = {"PPO": load_agent(Path(a.ppo), env_cfg), "SAC": load_agent(Path(a.sac), env_cfg),
              "Lane-follow PD": LaneFollowPolicy(env.cfg.n_rays)}
    fig, axes = plt.subplots(len(a.layouts), 1, figsize=(12, 1.9 * len(a.layouts)))
    for ax, k in zip(np.atleast_1d(axes), a.layouts):
        seed = EVAL_SEED_BASE + k
        env.reset(seed=seed)
        x = np.linspace(0, env.cfg.road_length, 600)
        c = env._center(x)
        hw = env.cfg.half_width / np.cos(np.arctan(env._center_slope(x)))
        ax.fill_between(x, c - hw, c + hw, color="0.9")
        ax.plot(x, c, "--", color="0.6", lw=0.8)
        for ox, oy in env._obs_xy:
            ax.add_patch(plt.Circle((ox, oy), env.cfg.obstacle_radius, color="tab:red"))
        for name, pol in agents.items():
            xs, ys, end = trace(env, pol, seed)
            ax.plot(xs, ys, lw=1.6, label=f"{name} ({end})")
        ax.set_aspect("equal")
        ax.set_title(f"held-out layout {k}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out = ROOT / "results/demo_trajectories.png"
    fig.savefig(out, dpi=120)
    print(out)
