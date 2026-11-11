"""First thing to run on the CARLA machine. Checks connection, sensors, speed.

    python scripts/carla_check.py              # lidar obstacle sensing
    python scripts/carla_check.py --gt         # compare against ground-truth boxes
Drives 3 episodes with the lane-follow PD controller (it ignores obstacles, so
crashes are expected) and prints sim steps/s and lidar-vs-ground-truth ray error.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from crashless.baselines import LaneFollowPolicy
from crashless.carla_env import CarlaDrivingEnv
from train import load_carla_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--gt", action="store_true", help="use ground-truth obstacle rays")
a = ap.parse_args()

cfg = load_carla_cfg()
cfg["obstacle_sensing"] = "ground_truth" if a.gt else "lidar"
env = CarlaDrivingEnv(cfg)
pol = LaneFollowPolicy(env.cfg.n_rays)
try:
    print("map:", env.map.name, "| spawn points:", len(env._spawn_points))
    steps, t0, ray_err = 0, time.time(), []
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=ep)
        print(f"ep {ep}: route {env.route.length:.0f} m, half-width {env.route.hw.mean():.2f} m, "
              f"obstacles {len(env._obstacles)}/{env.cfg.n_obstacles}")
        done = False
        while not done:
            obs, r, te, tr, info = env.step(pol.act(obs))
            steps += 1
            done = te or tr
            if not a.gt and env._obstacles:          # how far is lidar from the exact answer?
                env.cfg.obstacle_sensing = "ground_truth"
                gt = env._obstacle_rays()
                env.cfg.obstacle_sensing = "lidar"
                ray_err.append(np.abs(env._obstacle_rays() - gt))
        end = [k for k in ("success", "crashed", "offroad") if info[k]] or ["timeout"]
        print(f"   -> {end[0]} at {info['progress']:.0%} of route, {env.steps} steps")
    print(f"\n{steps / (time.time() - t0):.1f} sim steps/s (includes resets)")
    if ray_err:
        e = np.concatenate(ray_err)
        print(f"lidar vs ground-truth ray error: mean {e.mean():.2f} m, 95th pct {np.percentile(e, 95):.2f} m")
finally:
    env.close()
